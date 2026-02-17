"""
Stage 03: Metadata Extraction.

Produces a single canonical document-level metadata record per doc_version_id
with title, publication date, and document class. Preserves strict provenance
when metadata originates from document content.

**Reads:** `document_version`, `block`, `scrape_record`.
**Writes:** `doc_metadata`, `evidence_span` (when metadata from content), `doc_stage_status(stage_03_metadata)`.
**Responsibility**:
  * Produce a single, canonical document-level metadata record per doc_version_id that downstream stages can trust for filtering/routing (taxonomy, novelty, event extraction), while preserving strict provenance when the metadata comes from the document text.
  * Candidate generation (deterministic); Normalization; Selection + confidence; Provenance enforcement (hard constraints); Document class detection
"""

import argparse
import locale
import re
import sys
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError

from open_event_intel.etl_processing.config_interface import (
    Config,
    DateFormats,
    DateValidation,
    GlobalSettings,
    MetadataAnchor,
    Publisher,
    Taxonomy,
    get_config_version,
    load_config,
)
from open_event_intel.etl_processing.database_interface import (
    BlockRow,
    DBError,
    DocMetadataRow,
    DocumentVersionRow,
    ScrapeRecordRow,
)
from open_event_intel.etl_processing.stage_03_metadata.database_stage_03_metadata import PREREQUISITE_STAGE, STAGE_NAME, Stage03DatabaseInterface
from open_event_intel.logger import get_logger

logger = get_logger(__name__)

# Module-level constants (defaults / fallbacks).
#
# These mirror config.yaml defaults so the code never silently accepts values
# outside the configured plausible window.  Override via config whenever
# possible; these exist only as a safety net.

# Date validation bounds (fallback when config is unavailable)
_FALLBACK_DATE_MIN_YEAR: int = 2010
_FALLBACK_DATE_MAX_YEAR: int = 2035

# Title extraction constants
TITLE_MIN_LENGTH: int = 3                # minimum chars to accept as a title
TITLE_CONFIDENCE_H1: float = 0.9         # base confidence for H1 headings
TITLE_CONFIDENCE_NON_H1: float = 0.7     # base confidence for non-H1 headings
TITLE_LATE_PENALTY_THRESHOLD: int = 500  # span_start beyond which confidence is reduced
TITLE_LATE_PENALTY: float = 0.2          # confidence penalty for late-appearing titles
TITLE_MIN_CONFIDENCE: float = 0.1        # floor for title confidence
TITLE_ANCHOR_BOOST: float = 0.05         # confidence boost for publisher-anchored titles
TITLE_PATTERN_MATCH_CONFIDENCE: float = 0.85  # confidence for regex-matched titles
TITLE_SOURCE_DB_CONFIDENCE: float = 0.6  # confidence for scrape_record.source_title

# Date extraction constants
DATE_SEARCH_REGION: int = 3000           # chars to scan for anchor-based date extraction
DATE_CONTENT_SEARCH_LIMIT: int = 2000    # chars to scan for generic date extraction
DATE_ANCHOR_BASE_CONFIDENCE: float = 0.90  # base confidence for anchored dates
DATE_EARLY_BONUS_THRESHOLD: int = 500    # match position below which bonus applies
DATE_EARLY_BONUS: float = 0.05           # confidence bonus for early matches (anchored)
DATE_ANCHOR_MAX_CONFIDENCE: float = 0.98 # cap for anchored date confidence
DATE_CONTENT_BASE_CONFIDENCE: float = 0.7  # base confidence for generic content dates
DATE_CONTENT_EARLY_BONUS: float = 0.1    # confidence bonus for early matches (generic)
DATE_SOURCE_DB_CONFIDENCE: float = 0.95  # confidence for scrape_record.source_published_at

# Document-class detection constants (simple keyword-matching pre-classification;
# NOT derived from taxonomy.classification_settings which serves the full pipeline)
DOC_CLASS_SEARCH_REGION: int = 3000      # chars to scan for document class keywords
DOC_CLASS_KEYWORD_WEIGHT_FALLBACK: float = 0.3  # weight per keyword match (simple scoring)
DOC_CLASS_HEADING_KEYWORD_BOOST: float = 0.15   # extra weight when keyword appears in a heading block
DOC_CLASS_POSITION_BONUS: float = 0.1    # bonus if keyword appears early
DOC_CLASS_POSITION_THRESHOLD: int = 500  # chars threshold for position bonus
DOC_CLASS_MIN_SCORE: float = 0.2         # minimum score to accept a document class
DOC_CLASS_MAX_CONFIDENCE: float = 0.9    # cap for document class confidence

# Error message truncation
ERROR_MESSAGE_MAX_LENGTH: int = 500


TitleSource = Literal[
    "content_span",
    "source_db_field(scrape_record.source_title)",
    "html_meta",
    "url",
    "unknown",
]
PublishedAtSource = Literal[
    "content_span",
    "source_db_field(scrape_record.source_published_at)",
    "html_meta",
    "url",
    "unknown",
]


@dataclass(frozen=True)
class TitleCandidate:
    """Candidate title extracted from a document."""

    text: str
    source: TitleSource
    confidence: float
    span_start: int | None = None
    span_end: int | None = None


@dataclass(frozen=True)
class DateCandidate:
    """Candidate publication date extracted from a document."""

    parsed: datetime
    raw: str
    format_used: str
    source: PublishedAtSource
    confidence: float
    span_start: int | None = None
    span_end: int | None = None


@dataclass(frozen=True)
class DocumentClassCandidate:
    """Detected document class with confidence."""

    document_class: str
    confidence: float


class MetadataResult(BaseModel):
    """Result of metadata extraction for a document."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    title: str | None
    title_source: TitleSource
    title_confidence: float | None
    title_span_start: int | None
    title_span_end: int | None

    published_at: datetime | None
    published_at_raw: str | None
    published_at_format: str | None
    published_at_source: PublishedAtSource
    published_at_confidence: float | None
    published_at_span_start: int | None
    published_at_span_end: int | None

    detected_document_class: str | None
    document_class_confidence: float | None

    extraction_log: str | None = None


# Text helpers
def normalize_text(text: str) -> str:
    """Normalize text for comparison: NFC, lowercase, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_markdown_heading(text: str) -> str:
    """Strip markdown heading markers from text."""
    stripped = text.strip()
    if stripped.startswith("#"):
        stripped = stripped.lstrip("#").strip()
    return stripped



# Block-aware search helpers
def _get_nonboilerplate_blocks(
    blocks: Sequence[BlockRow],
    char_limit: int | None = None,
) -> list[BlockRow]:
    """
    Return non-boilerplate blocks sorted by span_start, optionally capped.

    :param blocks: All blocks for the document.
    :param char_limit: If given, only include blocks whose span_start is
        below this character offset in *clean_content*.
    :return: Sorted, filtered list of blocks.
    """
    sorted_blocks = sorted(blocks, key=lambda b: b.span_start)
    result: list[BlockRow] = []
    for b in sorted_blocks:
        if b.boilerplate_flag is not None:
            continue
        if char_limit is not None and b.span_start >= char_limit:
            break
        result.append(b)
    return result


def _build_block_search_text(
    clean_content: str,
    blocks: Sequence[BlockRow],
    char_limit: int,
    exclude_boilerplate: bool = True,
) -> str:
    r"""
    Concatenate text from (non-boilerplate) blocks for keyword-style search.

    The returned string is a simple join (``\\n`` separated) of block texts and
    is intended for keyword/substring matching where character-offset provenance
    is **not** required.  For offset-accurate search, iterate over blocks
    individually using :func:`_iter_block_regions`.
    """
    filtered = (
        _get_nonboilerplate_blocks(blocks, char_limit=char_limit)
        if exclude_boilerplate
        else sorted(blocks, key=lambda b: b.span_start)
    )
    parts: list[str] = []
    for b in filtered:
        effective_end = min(b.span_end, char_limit) if char_limit else b.span_end
        text = clean_content[b.span_start : effective_end]
        if text.strip():
            parts.append(text)
    return "\n".join(parts)


def _iter_block_regions(
    clean_content: str,
    blocks: Sequence[BlockRow],
    char_limit: int,
    exclude_boilerplate: bool = True,
) -> list[tuple[int, int, str]]:
    """
    Yield (span_start, span_end, text) for each qualifying block.

    Unlike :func:`_build_block_search_text`, offsets here are **absolute**
    positions in *clean_content* so regex-match positions can be trivially
    converted back by adding ``span_start``.
    """
    filtered = (
        _get_nonboilerplate_blocks(blocks, char_limit=char_limit)
        if exclude_boilerplate
        else sorted(blocks, key=lambda b: b.span_start)
    )
    regions: list[tuple[int, int, str]] = []
    for b in filtered:
        effective_end = min(b.span_end, char_limit) if char_limit else b.span_end
        text = clean_content[b.span_start : effective_end]
        if text.strip():
            regions.append((b.span_start, effective_end, text))
    return regions


def _span_in_nonboilerplate_block(
    span_start: int,
    span_end: int,
    nonboilerplate_blocks: Sequence[BlockRow],
) -> bool:
    """Return True if [span_start, span_end) falls entirely within a non-boilerplate block."""
    for b in nonboilerplate_blocks:
        if b.span_start <= span_start and span_end <= b.span_end:
            return True
    return False


def _find_enclosing_block(
    span_start: int,
    span_end: int,
    blocks: Sequence[BlockRow],
) -> BlockRow | None:
    """Return the block that encloses [span_start, span_end), or None."""
    for b in blocks:
        if b.span_start <= span_start and span_end <= b.span_end:
            return b
    return None



# German month-name mapping (avoids system-locale dependency)
_GERMAN_MONTH_MAP: dict[str, str] = {
    "Januar": "January", "Februar": "February", "Marz": "March",
    "März": "March", "April": "April", "Mai": "May", "Juni": "June",
    "Juli": "July", "August": "August", "September": "September",
    "Oktober": "October", "November": "November", "Dezember": "December",
}

_GERMAN_MONTH_RE = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in _GERMAN_MONTH_MAP) + r")\b"
)


def _normalize_german_months(text: str) -> str:
    """Replace German month names with English equivalents for strptime."""
    return _GERMAN_MONTH_RE.sub(lambda m: _GERMAN_MONTH_MAP[m.group(0)], text)


@contextmanager
def _temporary_locale(loc: str | None):
    """
    Context manager that temporarily sets LC_TIME locale, restoring on exit.

    Falls back silently if the requested locale is unavailable.
    """
    if loc is None:
        yield
        return
    saved = locale.getlocale(locale.LC_TIME)
    try:
        locale.setlocale(locale.LC_TIME, (loc, "UTF-8"))
    except locale.Error:
        try:
            locale.setlocale(locale.LC_TIME, loc)
        except locale.Error:
            logger.debug("Locale %r unavailable, proceeding with current locale", loc)
            yield  # locale unavailable – proceed with current
            return
    try:
        yield
    finally:
        try:
            locale.setlocale(locale.LC_TIME, saved)
        except locale.Error:
            pass


# Title extraction
def extract_title_from_content(
    clean_content: str,
    blocks: Sequence[BlockRow],
) -> TitleCandidate | None:
    """
    Extract title from document content using heading blocks.

    The first H1 or the first heading block is used as the title candidate.
    """
    heading_blocks = [
        b for b in blocks
        if b.block_type == "HEADING" and b.boilerplate_flag is None
    ]

    logger.debug(
        "Title from content: %d heading blocks (of %d total blocks)",
        len(heading_blocks),
        len(blocks),
    )

    if not heading_blocks:
        logger.debug("No non-boilerplate heading blocks found for title extraction")
        return None

    heading_blocks_sorted = sorted(heading_blocks, key=lambda b: b.span_start)

    h1_blocks = [b for b in heading_blocks_sorted if b.block_level == 1]
    best_block = h1_blocks[0] if h1_blocks else heading_blocks_sorted[0]

    raw_title = clean_content[best_block.span_start : best_block.span_end].strip()
    title_text = _strip_markdown_heading(raw_title)

    if not title_text or len(title_text) < TITLE_MIN_LENGTH:
        logger.debug(
            "Title candidate too short (%d chars, min=%d): %r",
            len(title_text) if title_text else 0,
            TITLE_MIN_LENGTH,
            title_text[:50] if title_text else "",
        )
        return None

    confidence = TITLE_CONFIDENCE_H1 if best_block.block_level == 1 else TITLE_CONFIDENCE_NON_H1
    if best_block.span_start > TITLE_LATE_PENALTY_THRESHOLD:
        original_confidence = confidence
        confidence -= TITLE_LATE_PENALTY
        logger.debug(
            "Title late-penalty applied: span_start=%d > threshold=%d, confidence %.2f -> %.2f",
            best_block.span_start,
            TITLE_LATE_PENALTY_THRESHOLD,
            original_confidence,
            confidence,
        )

    confidence = max(TITLE_MIN_CONFIDENCE, confidence)

    logger.debug(
        "Title candidate from content: %r (level=%s, span=%d-%d, conf=%.2f)",
        title_text[:80],
        best_block.block_level,
        best_block.span_start,
        best_block.span_end,
        confidence,
    )

    return TitleCandidate(
        text=title_text,
        source="content_span",
        confidence=confidence,
        span_start=best_block.span_start,
        span_end=best_block.span_end,
    )


def extract_title_via_anchor(
    clean_content: str,
    blocks: Sequence[BlockRow],
    anchors: list["MetadataAnchor"],
) -> TitleCandidate | None:
    """
    Apply publisher-specific metadata_anchors for title extraction.

    Supported anchor methods:
      - first_h1: equivalent to generic H1 extraction (but publisher-prioritised).
      - pattern_match: regex against the first ``DATE_SEARCH_REGION`` chars of
        clean_content.  Matches that fall inside boilerplate blocks are skipped.
    """
    nb_blocks = _get_nonboilerplate_blocks(blocks, char_limit=DATE_SEARCH_REGION)

    for anchor in anchors:
        logger.debug("Trying title anchor method=%r pattern=%r", anchor.method, anchor.pattern)

        if anchor.method == "first_h1":
            candidate = extract_title_from_content(clean_content, blocks)
            if candidate is not None:
                boosted_confidence = min(1.0, candidate.confidence + TITLE_ANCHOR_BOOST)
                logger.debug(
                    "Title anchor first_h1 matched: %r (conf %.2f -> %.2f)",
                    candidate.text[:80],
                    candidate.confidence,
                    boosted_confidence,
                )
                return TitleCandidate(
                    text=candidate.text,
                    source=candidate.source,
                    confidence=boosted_confidence,
                    span_start=candidate.span_start,
                    span_end=candidate.span_end,
                )

        elif anchor.method == "pattern_match" and anchor.pattern:
            search_region = clean_content[:DATE_SEARCH_REGION]
            for match in re.finditer(anchor.pattern, search_region, re.MULTILINE):
                # Skip matches that fall inside boilerplate blocks
                if nb_blocks and not _span_in_nonboilerplate_block(
                    match.start(), match.end(), nb_blocks
                ):
                    logger.debug(
                        "Title anchor pattern_match: match at %d-%d in boilerplate, skipping",
                        match.start(),
                        match.end(),
                    )
                    continue

                title_text = (match.group(1) if match.lastindex else match.group(0)).strip()
                title_text = _strip_markdown_heading(title_text)
                if title_text and len(title_text) >= TITLE_MIN_LENGTH:
                    logger.debug(
                        "Title anchor pattern_match hit: %r (span=%d-%d, conf=%.2f)",
                        title_text[:80],
                        match.start(),
                        match.end(),
                        TITLE_PATTERN_MATCH_CONFIDENCE,
                    )
                    return TitleCandidate(
                        text=title_text,
                        source="content_span",
                        confidence=TITLE_PATTERN_MATCH_CONFIDENCE,
                        span_start=match.start(),
                        span_end=match.end(),
                    )

    logger.debug("No title anchor matched")
    return None


def extract_title_from_source(scrape_record: ScrapeRecordRow) -> TitleCandidate | None:
    """Extract title from scrape record's source_title field."""
    title = scrape_record.source_title.strip() if scrape_record.source_title else ""

    if not title or len(title) < TITLE_MIN_LENGTH:
        logger.debug(
            "Source title too short or empty (%d chars): %r",
            len(title),
            title[:50],
        )
        return None

    logger.debug(
        "Title from source_db: %r (conf=%.2f)",
        title[:80],
        TITLE_SOURCE_DB_CONFIDENCE,
    )

    return TitleCandidate(
        text=title,
        source="source_db_field(scrape_record.source_title)",
        confidence=TITLE_SOURCE_DB_CONFIDENCE,
    )


def select_best_title(candidates: Sequence[TitleCandidate]) -> TitleCandidate | None:
    """Select the best title candidate by confidence."""
    if not candidates:
        return None
    best = max(candidates, key=lambda c: c.confidence)
    logger.debug(
        "Selected best title from %d candidates: source=%s conf=%.2f text=%r",
        len(candidates),
        best.source,
        best.confidence,
        best.text[:80],
    )
    return best

# Date extraction
DATE_FORMATS_DEFAULT = [
    "%Y-%m-%d",
    "%d.%m.%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%d %B %Y",
    "%B %d, %Y",
    "%d. %B %Y",
    "%d %b %Y",
]


def parse_date_string(
    raw: str,
    formats: Sequence[str] | None = None,
    publisher_locale: str | None = None,
) -> tuple[datetime, str] | None:
    """
    Parse a date string using multiple formats.

    Handles German month names by translating them before parsing and
    optionally uses the publisher locale for locale-dependent formats.

    :param raw: Raw date string to parse.
    :param formats: List of datetime format strings to try.
    :param publisher_locale: Optional locale string (e.g. 'de_DE') for strptime.
    :return: Tuple of (parsed datetime, format used) or None.
    """
    if formats is None:
        formats = DATE_FORMATS_DEFAULT

    cleaned = raw.strip()
    if not cleaned:
        return None

    # First try with German-month normalisation (locale-independent)
    cleaned_en = _normalize_german_months(cleaned)

    for fmt in formats:
        for attempt in (cleaned_en, cleaned):
            try:
                parsed = datetime.strptime(attempt, fmt)
                return (parsed, fmt)
            except ValueError:
                continue

    # Last resort: try with the publisher's locale set temporarily
    if publisher_locale:
        with _temporary_locale(publisher_locale):
            for fmt in formats:
                try:
                    parsed = datetime.strptime(cleaned, fmt)
                    return (parsed, fmt)
                except ValueError:
                    continue

    return None


# Compiled patterns for the generic (non-anchor) scan
DATE_PATTERN_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
DATE_PATTERN_DMY_DOT = re.compile(r"\b(\d{1,2}\.\d{1,2}\.\d{4})\b")
DATE_PATTERN_DMY_SLASH = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
DATE_PATTERN_ENGLISH_DMY = re.compile(
    r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August"
    r"|September|October|November|December)\s+\d{4})\b"
)
DATE_PATTERN_ENGLISH_MDY = re.compile(
    r"\b((?:January|February|March|April|May|June|July|August"
    r"|September|October|November|December)\s+\d{1,2},?\s*\d{4})\b"
)
DATE_PATTERN_ENGLISH_ABBREV_DMY = re.compile(
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})\b"
)
DATE_PATTERN_GERMAN_DMY = re.compile(
    r"\b(\d{1,2}\.\s*(?:Januar|Februar|M[aä]rz|April|Mai|Juni|Juli|August"
    r"|September|Oktober|November|Dezember)\s+\d{4})\b"
)


def extract_date_via_anchors(  # noqa: C901
    clean_content: str,
    anchors: list["MetadataAnchor"],
    date_formats: "DateFormats | None" = None,
    date_validation: "DateValidation | None" = None,
    blocks: Sequence[BlockRow] | None = None,
) -> DateCandidate | None:
    """
    Apply publisher-specific byline_pattern anchors for date extraction.

    Each anchor carries its own regex ``pattern`` and strptime ``format``.
    The first successful match wins (anchors are ordered by priority in config).

    When *blocks* are supplied, matches that fall inside boilerplate blocks are
    skipped so that provenance is always anchored to meaningful content.
    """
    min_year = date_validation.min_year if date_validation else _FALLBACK_DATE_MIN_YEAR
    max_year = date_validation.max_year if date_validation else _FALLBACK_DATE_MAX_YEAR

    # Pre-compute non-boilerplate blocks for match validation
    nb_blocks: Sequence[BlockRow] | None = None
    if blocks is not None:
        nb_blocks = _get_nonboilerplate_blocks(blocks, char_limit=DATE_SEARCH_REGION)

    logger.debug(
        "Date anchor extraction: %d anchors, year bounds=[%d, %d], nb_blocks=%s",
        len(anchors),
        min_year,
        max_year,
        len(nb_blocks) if nb_blocks is not None else "n/a",
    )

    search_text = clean_content[:DATE_SEARCH_REGION]

    for anchor in anchors:
        if anchor.method != "byline_pattern" or not anchor.pattern:
            continue

        try:
            regex = re.compile(anchor.pattern, re.MULTILINE)
        except re.error:
            logger.warning("Invalid byline regex in config: %s", anchor.pattern)
            continue

        logger.debug("Trying date anchor: pattern=%r format=%r", anchor.pattern[:60], anchor.format)

        for match in regex.finditer(search_text):
            # Skip matches that fall inside boilerplate blocks
            if nb_blocks is not None and not _span_in_nonboilerplate_block(
                match.start(), match.end(), nb_blocks
            ):
                logger.debug(
                    "Date anchor: match at %d-%d falls in boilerplate, skipping",
                    match.start(),
                    match.end(),
                )
                continue

            raw_date = match.group(1) if match.lastindex else match.group(0)
            fmt = anchor.format
            if not fmt:
                continue

            publisher_locale = date_formats.locale if date_formats else None
            result = parse_date_string(raw_date, [fmt], publisher_locale)
            if result is None:
                logger.debug("Date anchor: could not parse %r with format %r", raw_date, fmt)
                continue

            parsed, fmt_used = result
            if parsed.year < min_year or parsed.year > max_year:
                logger.debug(
                    "Date anchor: %r parsed to year %d, outside bounds [%d, %d] — skipping",
                    raw_date,
                    parsed.year,
                    min_year,
                    max_year,
                )
                continue

            confidence = DATE_ANCHOR_BASE_CONFIDENCE
            if match.start() < DATE_EARLY_BONUS_THRESHOLD:
                confidence += DATE_EARLY_BONUS

            confidence = min(DATE_ANCHOR_MAX_CONFIDENCE, confidence)

            logger.debug(
                "Date anchor matched: %r -> %s (fmt=%r, span=%d-%d, conf=%.2f)",
                raw_date,
                parsed.strftime("%Y-%m-%d"),
                fmt_used,
                match.start(),
                match.end(),
                confidence,
            )

            return DateCandidate(
                parsed=parsed,
                raw=raw_date,
                format_used=fmt_used,
                source="content_span",
                confidence=confidence,
                span_start=match.start(),
                span_end=match.end(),
            )

    logger.debug("No date anchor matched")
    return None


def extract_date_from_content(
    clean_content: str,
    formats: Sequence[str] | None = None,
    publisher_locale: str | None = None,
    search_limit: int = DATE_CONTENT_SEARCH_LIMIT,
    date_validation: "DateValidation | None" = None,
    blocks: Sequence[BlockRow] | None = None,
) -> DateCandidate | None:
    """
    Extract publication date from document content using generic patterns.

    When *blocks* are supplied the search is confined to non-boilerplate block
    regions within the first ``search_limit`` characters, ensuring dates from
    navigation, footers, or other boilerplate are never selected.  Each match's
    span offsets are absolute positions in *clean_content* so provenance is
    naturally block-aligned.

    Falls back to scanning ``clean_content[:search_limit]`` when no blocks are
    available (backwards-compatible path).
    """
    min_year = date_validation.min_year if date_validation else _FALLBACK_DATE_MIN_YEAR
    max_year = date_validation.max_year if date_validation else _FALLBACK_DATE_MAX_YEAR

    patterns: list[tuple[re.Pattern[str], list[str]]] = [
        (DATE_PATTERN_ISO, ["%Y-%m-%d"]),
        (DATE_PATTERN_DMY_DOT, ["%d.%m.%Y"]),
        (DATE_PATTERN_DMY_SLASH, ["%d/%m/%Y", "%m/%d/%Y"]),
        (DATE_PATTERN_ENGLISH_DMY, ["%d %B %Y"]),
        (DATE_PATTERN_ENGLISH_MDY, ["%B %d, %Y", "%B %d %Y"]),
        (DATE_PATTERN_ENGLISH_ABBREV_DMY, ["%d %b %Y"]),
        (DATE_PATTERN_GERMAN_DMY, ["%d. %B %Y", "%d.%B %Y"]),
    ]

    best_candidate: DateCandidate | None = None
    candidates_found = 0

    # Build search regions: either block-aware or flat slice
    if blocks is not None:
        regions = _iter_block_regions(
            clean_content, blocks, char_limit=search_limit, exclude_boilerplate=True,
        )
        logger.debug(
            "Date content scan (block-aware): %d non-boilerplate regions within first %d chars",
            len(regions),
            search_limit,
        )
    else:
        # Backwards-compatible: single region covering the first search_limit chars
        slice_text = clean_content[:search_limit]
        regions = [(0, len(slice_text), slice_text)] if slice_text else []
        logger.debug(
            "Date content scan (flat): first %d chars",
            search_limit,
        )

    for region_start, _region_end, region_text in regions:
        for pattern, pattern_formats in patterns:
            for match in pattern.finditer(region_text):
                raw_date = match.group(1)
                result = parse_date_string(raw_date, pattern_formats, publisher_locale)
                if result:
                    parsed, fmt = result

                    if parsed.year < min_year or parsed.year > max_year:
                        logger.debug(
                            "Date content scan: %r -> year %d outside bounds [%d, %d]",
                            raw_date,
                            parsed.year,
                            min_year,
                            max_year,
                        )
                        continue

                    # Absolute offsets in clean_content
                    abs_start = region_start + match.start()
                    abs_end = region_start + match.end()

                    confidence = DATE_CONTENT_BASE_CONFIDENCE
                    if abs_start < DATE_EARLY_BONUS_THRESHOLD:
                        confidence += DATE_CONTENT_EARLY_BONUS

                    candidate = DateCandidate(
                        parsed=parsed,
                        raw=raw_date,
                        format_used=fmt,
                        source="content_span",
                        confidence=confidence,
                        span_start=abs_start,
                        span_end=abs_end,
                    )
                    candidates_found += 1

                    if best_candidate is None or candidate.confidence > best_candidate.confidence:
                        best_candidate = candidate

    if best_candidate:
        logger.debug(
            "Date from content scan: best=%r -> %s (conf=%.2f, %d candidates total)",
            best_candidate.raw,
            best_candidate.parsed.strftime("%Y-%m-%d"),
            best_candidate.confidence,
            candidates_found,
        )
    else:
        logger.debug("Date content scan: no valid dates found in first %d chars", search_limit)

    return best_candidate


def extract_date_from_source(
    scrape_record: ScrapeRecordRow,
) -> DateCandidate | None:
    """Extract publication date from scrape record's source_published_at field."""
    if scrape_record.source_published_at is None:
        logger.debug("No source_published_at in scrape record")
        return None

    logger.debug(
        "Date from source_db: %s (conf=%.2f)",
        scrape_record.source_published_at.isoformat(),
        DATE_SOURCE_DB_CONFIDENCE,
    )

    return DateCandidate(
        parsed=scrape_record.source_published_at,
        raw=scrape_record.source_published_at.isoformat(),
        format_used="source_db_timestamp",
        source="source_db_field(scrape_record.source_published_at)",
        confidence=DATE_SOURCE_DB_CONFIDENCE,
    )


def select_best_date(candidates: Sequence[DateCandidate]) -> DateCandidate | None:
    """Select the best date candidate by confidence."""
    if not candidates:
        return None
    best = max(candidates, key=lambda c: c.confidence)
    logger.debug(
        "Selected best date from %d candidates: source=%s conf=%.2f raw=%r -> %s",
        len(candidates),
        best.source,
        best.confidence,
        best.raw,
        best.parsed.strftime("%Y-%m-%d"),
    )
    return best



# Document-class detection

# Fallback keywords used only when config provides no taxonomy.document_types.
DOCUMENT_CLASS_KEYWORDS: dict[str, list[str]] = {
    "press_release": ["press release", "pressemitteilung", "media release"],
    "consultation": ["consultation", "konsultation", "public consultation"],
    "regulatory_decision": ["decision", "entscheidung", "beschluss", "ruling"],
    "report": ["report", "bericht", "annual report", "jahresbericht"],
    "market_data": ["market data", "marktdaten", "prices", "preise"],
    "news_article": ["article", "news", "nachricht", "artikel"],
    "technical_document": ["technical", "specification", "spezifikation"],
}


def detect_document_class(  # noqa: C901
    clean_content: str,
    blocks: Sequence[BlockRow],
    taxonomy: "Taxonomy | None" = None,
) -> DocumentClassCandidate | None:
    """
    Detect document class based on content and structural signals.

    This is a lightweight pre-classification using simple keyword matching.
    When a ``taxonomy`` is supplied, config-driven ``taxonomy.document_types``
    keywords and their per-type ``threshold`` are used.  Otherwise falls back
    to the built-in ``DOCUMENT_CLASS_KEYWORDS`` mapping with module-level
    default weights.

    **Block awareness:** When *blocks* are available the search text is
    constructed from non-boilerplate blocks within the first
    ``DOC_CLASS_SEARCH_REGION`` characters, ensuring navigation, footers, and
    other boilerplate cannot influence classification.  Keywords that appear
    inside heading blocks receive a structural boost
    (``DOC_CLASS_HEADING_KEYWORD_BOOST``).  Falls back to a raw content slice
    when no qualifying blocks exist.

    **Threshold gating:** Each document type is independently checked against
    its own threshold.  Among all types that *pass* their threshold the
    highest-scoring type wins.  This prevents a high-threshold type with the
    top raw score from shadowing a lower-scoring type that legitimately passes
    its own (lower) threshold.

    **Note:** ``taxonomy.classification_settings.keyword_weights`` are designed
    for the full multi-signal topic classification pipeline (keywords +
    mentions + structural + priors) and must NOT be applied here.  This
    function uses ``DOC_CLASS_KEYWORD_WEIGHT_FALLBACK`` for per-keyword
    scoring and each document type's own ``threshold`` for the acceptance
    gate.

    :param clean_content: The cleaned document text.
    :param blocks: Structural blocks from Stage 02.
    :param taxonomy: Optional taxonomy configuration from config.
    :return: DocumentClassCandidate or None if no class meets the threshold.
    """
    # Build search texts
    raw_slice_len = min(len(clean_content), DOC_CLASS_SEARCH_REGION)

    if blocks:
        search_text = _build_block_search_text(
            clean_content, blocks, char_limit=DOC_CLASS_SEARCH_REGION,
            exclude_boilerplate=True,
        ).lower()
        if not search_text:
            search_text = clean_content[:DOC_CLASS_SEARCH_REGION].lower()
            logger.info(
                "Doc-class: all blocks in first %d chars are boilerplate — "
                "falling back to raw slice (%d chars)",
                DOC_CLASS_SEARCH_REGION,
                raw_slice_len,
            )
        else:
            logger.debug(
                "Doc-class: block-filtered search text %d chars "
                "(raw slice would be %d chars, ratio=%.0f%%)",
                len(search_text),
                raw_slice_len,
                100.0 * len(search_text) / raw_slice_len if raw_slice_len else 0,
            )
    else:
        search_text = clean_content[:DOC_CLASS_SEARCH_REGION].lower()
        logger.debug(
            "Doc-class: no blocks provided; using raw slice (%d chars)",
            raw_slice_len,
        )

    # Heading-only search text for structural boost
    heading_text = ""
    if blocks:
        heading_blocks = [
            b for b in blocks
            if b.block_type == "HEADING"
            and b.boilerplate_flag is None
            and b.span_start < DOC_CLASS_SEARCH_REGION
        ]
        if heading_blocks:
            heading_parts = [
                clean_content[b.span_start : min(b.span_end, DOC_CLASS_SEARCH_REGION)]
                for b in sorted(heading_blocks, key=lambda b: b.span_start)
            ]
            heading_text = "\n".join(heading_parts).lower()
            logger.debug(
                "Doc-class: heading text %d chars from %d heading blocks",
                len(heading_text),
                len(heading_blocks),
            )

    # Scoring parameters
    keyword_weight = DOC_CLASS_KEYWORD_WEIGHT_FALLBACK
    heading_boost = DOC_CLASS_HEADING_KEYWORD_BOOST
    position_bonus = DOC_CLASS_POSITION_BONUS
    position_threshold = DOC_CLASS_POSITION_THRESHOLD
    max_confidence = DOC_CLASS_MAX_CONFIDENCE
    global_min_score = DOC_CLASS_MIN_SCORE

    # Build keyword map
    config_types: dict[str, tuple[dict[str, list[str]], float]] = {}
    fallback_flat_map: dict[str, list[str]] = {}

    if taxonomy and taxonomy.document_types:
        for doc_type_id, doc_type_def in taxonomy.document_types.items():
            lang_kws: dict[str, list[str]] = {}
            for lang, kw_list in doc_type_def.keywords.items():
                lang_kws[lang] = kw_list
            type_threshold = max(doc_type_def.threshold, global_min_score)
            config_types[doc_type_id] = (lang_kws, type_threshold)
        logger.debug(
            "Doc-class: %d config types (kw_weight=%.2f, heading_boost=%.2f, "
            "position_bonus=%.2f)",
            len(config_types),
            keyword_weight,
            heading_boost,
            position_bonus,
        )

    if not config_types:
        fallback_flat_map = DOCUMENT_CLASS_KEYWORDS
        logger.debug(
            "Doc-class: %d fallback classes (kw_weight=%.2f, threshold=%.2f)",
            len(fallback_flat_map),
            keyword_weight,
            global_min_score,
        )

    # Score every type and collect diagnostics
    @dataclass
    class _TypeScore:
        doc_class: str
        score: float
        threshold: float
        matched_keywords: list[str]
        heading_matches: list[str]

        @property
        def passes(self) -> bool:
            return self.score >= self.threshold

    all_scores: list[_TypeScore] = []

    def _score_keywords(
        doc_class: str,
        keywords: list[str],
        type_threshold: float,
    ) -> _TypeScore:
        score = 0.0
        matched: list[str] = []
        heading_matches: list[str] = []

        for keyword in keywords:
            kw_lower = keyword.lower()
            if kw_lower in search_text:
                score += keyword_weight
                matched.append(keyword)
                # Position bonus: early appearance in search text
                if search_text.find(kw_lower) < position_threshold:
                    score += position_bonus
                # Heading structural boost
                if heading_text and kw_lower in heading_text:
                    score += heading_boost
                    heading_matches.append(keyword)

        return _TypeScore(
            doc_class=doc_class,
            score=score,
            threshold=type_threshold,
            matched_keywords=matched,
            heading_matches=heading_matches,
        )

    if config_types:
        for doc_class, (lang_keywords, type_threshold) in config_types.items():
            # Flatten all language keywords for scoring
            all_kws: list[str] = []
            for lang, keywords in lang_keywords.items():  # noqa: B007
                all_kws.extend(keywords)
            ts = _score_keywords(doc_class, all_kws, type_threshold)
            all_scores.append(ts)
    else:
        for doc_class, keywords in fallback_flat_map.items():
            ts = _score_keywords(doc_class, keywords, global_min_score)
            all_scores.append(ts)

    # Diagnostic logging: every scored type
    all_scores.sort(key=lambda s: s.score, reverse=True)

    for ts in all_scores:
        if ts.score > 0:
            logger.debug(
                "Doc-class score: %-25s score=%.2f threshold=%.2f %s "
                "matched=%s heading=%s",
                ts.doc_class,
                ts.score,
                ts.threshold,
                "PASS" if ts.passes else "FAIL",
                ts.matched_keywords,
                ts.heading_matches or "[]",
            )

    # Log near-misses at INFO level (scored > 0 but below threshold)
    near_misses = [
        ts for ts in all_scores
        if ts.score > 0 and not ts.passes
    ]
    if near_misses:
        miss_parts = [
            f"{ts.doc_class}(score={ts.score:.2f}<thr={ts.threshold:.2f}, "
            f"kw={ts.matched_keywords})"
            for ts in near_misses
        ]
        logger.info(
            "Doc-class near-misses: %s",
            "; ".join(miss_parts),
        )

    # Select winner: highest score AMONG types that pass their threshold
    passing = [ts for ts in all_scores if ts.passes]

    if not passing:
        zero_score = all(ts.score == 0 for ts in all_scores)
        logger.info(
            "Doc-class: None — %s (search_text=%d chars, %d types evaluated)",
            "no keywords matched at all" if zero_score
            else f"best was {all_scores[0].doc_class}={all_scores[0].score:.2f} "
                 f"< threshold {all_scores[0].threshold:.2f}",
            len(search_text),
            len(all_scores),
        )
        return None

    winner = passing[0]  # already sorted by score desc
    confidence = min(max_confidence, winner.score)

    logger.debug(
        "Doc-class detected: %r (score=%.2f, threshold=%.2f, conf=%.2f, "
        "matched=%s, heading=%s, passing_types=%d)",
        winner.doc_class,
        winner.score,
        winner.threshold,
        confidence,
        winner.matched_keywords,
        winner.heading_matches or "[]",
        len(passing),
    )

    return DocumentClassCandidate(
        document_class=winner.doc_class,
        confidence=confidence,
    )


# Orchestrator – combines all extraction strategies
def extract_metadata(  # noqa: C901
    doc_version: DocumentVersionRow,
    blocks: Sequence[BlockRow],
    scrape_record: ScrapeRecordRow,
    publisher_config: "Publisher | None" = None,
    config: "Config | None" = None,
) -> MetadataResult:
    """
    Extract metadata from a document.

    Pure function that combines all extraction methods and selects best candidates.

    When a ``publisher_config`` is supplied the publisher-specific
    ``metadata_anchors`` are tried *first*; generic extraction runs as
    fallback.

    When ``config`` is supplied, global validation settings (e.g. date year
    bounds) and taxonomy keywords (for document-class detection) are used
    instead of hardcoded defaults.

    :param doc_version: Document version to extract metadata from.
    :param blocks: Structural blocks from Stage 02.
    :param scrape_record: Source scrape record.
    :param publisher_config: Optional publisher-specific configuration.
    :param config: Optional full configuration for global settings.
    :return: MetadataResult with extracted values.
    """
    clean_content = doc_version.clean_content
    extraction_log_parts: list[str] = []

    logger.info(
        "Extracting metadata for doc_version=%s (publisher=%s, content_length=%d, blocks=%d)",
        doc_version.doc_version_id[:16],
        scrape_record.publisher_id,
        len(clean_content),
        len(blocks),
    )

    # -- resolve publisher helpers ----------------------------------------
    title_anchors: list["MetadataAnchor"] = []
    date_anchors: list["MetadataAnchor"] = []
    date_formats_cfg: "DateFormats | None" = None
    publisher_locale: str | None = None

    if publisher_config is not None:
        if publisher_config.metadata_anchors:
            title_anchors = publisher_config.metadata_anchors.get("title", [])
            date_anchors = publisher_config.metadata_anchors.get("date", [])
        date_formats_cfg = publisher_config.date_formats
        if date_formats_cfg:
            publisher_locale = date_formats_cfg.locale
        logger.debug(
            "Publisher config: %s (tier=%s, title_anchors=%d, date_anchors=%d, locale=%s)",
            publisher_config.full_name,
            publisher_config.processing_tier,
            len(title_anchors),
            len(date_anchors),
            publisher_locale,
        )
    else:
        logger.debug("No publisher config available")

    # -- resolve global helpers -------------------------------------------
    date_validation: "DateValidation | None" = None
    taxonomy: "Taxonomy | None" = None
    if config is not None:
        date_validation = config.global_settings.validation.date_validation
        taxonomy = config.taxonomy
        logger.debug(
            "Global config: date_validation=[%d, %d], taxonomy_doc_types=%d",
            date_validation.min_year,
            date_validation.max_year,
            len(taxonomy.document_types) if taxonomy else 0,
        )
    else:
        logger.debug(
            "No global config — using fallback date bounds [%d, %d]",
            _FALLBACK_DATE_MIN_YEAR,
            _FALLBACK_DATE_MAX_YEAR,
        )

    # TITLE
    title_candidates: list[TitleCandidate] = []

    # 1. Publisher-specific anchor extraction (highest priority)
    if title_anchors:
        anchor_title = extract_title_via_anchor(clean_content, blocks, title_anchors)
        if anchor_title:
            title_candidates.append(anchor_title)
            extraction_log_parts.append(
                f"title_from_anchor({title_anchors[0].method}): "
                f"conf={anchor_title.confidence:.2f}"
            )

    # 2. Generic heading extraction
    content_title = extract_title_from_content(clean_content, blocks)
    if content_title:
        # Avoid adding a duplicate when the anchor already returned the same span
        if not any(
            c.span_start == content_title.span_start and c.span_end == content_title.span_end
            for c in title_candidates
        ):
            title_candidates.append(content_title)
            extraction_log_parts.append(
                f"title_from_content: conf={content_title.confidence:.2f}"
            )

    # 3. Scrape-record fallback
    source_title = extract_title_from_source(scrape_record)
    if source_title:
        title_candidates.append(source_title)
        extraction_log_parts.append(
            f"title_from_source: conf={source_title.confidence:.2f}"
        )

    best_title = select_best_title(title_candidates)

    logger.info(
        "Title result: %d candidates -> selected=%r (source=%s, conf=%s)",
        len(title_candidates),
        best_title.text[:60] if best_title else None,
        best_title.source if best_title else "none",
        f"{best_title.confidence:.2f}" if best_title else "n/a",
    )

    # DATE
    date_candidates: list[DateCandidate] = []

    # 1. Publisher-specific byline-pattern extraction (highest priority)
    if date_anchors:
        anchor_date = extract_date_via_anchors(
            clean_content, date_anchors, date_formats_cfg,
            date_validation=date_validation,
            blocks=blocks,
        )
        if anchor_date:
            date_candidates.append(anchor_date)
            extraction_log_parts.append(
                f"date_from_anchor: {anchor_date.raw} conf={anchor_date.confidence:.2f}"
            )

    # 2. Generic content scan – build format list from config + defaults
    scan_formats: list[str] = list(DATE_FORMATS_DEFAULT)
    if date_formats_cfg:
        # Prepend publisher-specific formats so they are tried first
        publisher_fmts = [date_formats_cfg.primary] + list(date_formats_cfg.secondary)
        seen: set[str] = set()
        deduped: list[str] = []
        for f in publisher_fmts + scan_formats:
            if f not in seen:
                deduped.append(f)
                seen.add(f)
        scan_formats = deduped

    content_date = extract_date_from_content(
        clean_content,
        formats=scan_formats,
        publisher_locale=publisher_locale,
        date_validation=date_validation,
        blocks=blocks,
    )
    if content_date:
        date_candidates.append(content_date)
        extraction_log_parts.append(f"date_from_content: {content_date.raw}")

    # 3. Scrape-record fallback
    source_date = extract_date_from_source(scrape_record)
    if source_date:
        date_candidates.append(source_date)
        extraction_log_parts.append(f"date_from_source: {source_date.raw}")

    best_date = select_best_date(date_candidates)

    logger.info(
        "Date result: %d candidates -> selected=%s (source=%s, conf=%s)",
        len(date_candidates),
        best_date.parsed.strftime("%Y-%m-%d") if best_date else None,
        best_date.source if best_date else "none",
        f"{best_date.confidence:.2f}" if best_date else "n/a",
    )

    # DOCUMENT CLASS
    doc_class = detect_document_class(clean_content, blocks, taxonomy=taxonomy)
    if doc_class:
        extraction_log_parts.append(
            f"doc_class: {doc_class.document_class} (conf={doc_class.confidence:.2f})"
        )
    else:
        extraction_log_parts.append("doc_class: None")

    logger.info(
        "Doc-class result: %s (conf=%s)",
        doc_class.document_class if doc_class else None,
        f"{doc_class.confidence:.2f}" if doc_class else "n/a",
    )

    return MetadataResult(
        title=best_title.text if best_title else None,
        title_source=best_title.source if best_title else "unknown",
        title_confidence=best_title.confidence if best_title else None,
        title_span_start=best_title.span_start if best_title and best_title.source == "content_span" else None,
        title_span_end=best_title.span_end if best_title and best_title.source == "content_span" else None,
        published_at=best_date.parsed if best_date else None,
        published_at_raw=best_date.raw if best_date else None,
        published_at_format=best_date.format_used if best_date else None,
        published_at_source=best_date.source if best_date else "unknown",
        published_at_confidence=best_date.confidence if best_date else None,
        published_at_span_start=best_date.span_start if best_date and best_date.source == "content_span" else None,
        published_at_span_end=best_date.span_end if best_date and best_date.source == "content_span" else None,
        detected_document_class=doc_class.document_class if doc_class else None,
        document_class_confidence=doc_class.confidence if doc_class else None,
        extraction_log="; ".join(extraction_log_parts) if extraction_log_parts else None,
    )


# Per-document processing
def process_document(
    db: Stage03DatabaseInterface,
    doc_version_id: str,
    run_id: str,
    config_hash: str,
    config: "Config | None" = None,
) -> Literal["ok", "failed", "blocked", "skipped"]:
    """
    Process a single document for metadata extraction.

    Handles prerequisite checking, extraction, evidence creation, and status updates.

    :param db: Database interface.
    :param doc_version_id: Document version to process.
    :param run_id: Current pipeline run ID.
    :param config_hash: Config version hash.
    :param config: Optional configuration for publisher-specific settings.
    :return: Final status for this document.
    """
    logger.debug("Processing document: %s", doc_version_id[:16])

    prereq_status = db.check_prerequisite_status(doc_version_id)
    if prereq_status is None or prereq_status.status != "ok":
        blocking_status = prereq_status.status if prereq_status else "missing"
        error_msg = f"prerequisite_not_ok:{PREREQUISITE_STAGE}:{blocking_status}"
        logger.debug(
            "Blocked %s: prerequisite %s status=%s",
            doc_version_id[:16],
            PREREQUISITE_STAGE,
            blocking_status,
        )
        db.upsert_doc_stage_status(
            doc_version_id=doc_version_id,
            stage=STAGE_NAME,
            run_id=run_id,
            config_hash=config_hash,
            status="blocked",
            error_message=error_msg,
        )
        return "blocked"

    try:
        with db.transaction():
            doc_version = db.get_document_version(doc_version_id)
            if doc_version is None:
                raise DBError(f"Document version not found: {doc_version_id}")

            blocks = db.get_blocks_by_doc_version_id(doc_version_id)
            scrape_record = db.get_scrape_record_for_doc_version(doc_version_id)
            if scrape_record is None:
                raise DBError(f"Scrape record not found for: {doc_version_id}")

            logger.debug(
                "Loaded doc data: version=%s, blocks=%d, publisher=%s, content_length=%d",
                doc_version_id[:16],
                len(blocks),
                scrape_record.publisher_id,
                len(doc_version.clean_content),
            )

            publisher_config = None
            if config is not None:
                publisher_name = scrape_record.publisher_id
                publisher_config = config.get_publisher(publisher_name)
                if publisher_config is None:
                    logger.warning(
                        "Publisher %r not found in config for doc %s",
                        publisher_name,
                        doc_version_id[:16],
                    )

            result = extract_metadata(
                doc_version=doc_version,
                blocks=blocks,
                scrape_record=scrape_record,
                publisher_config=publisher_config,
                config=config,
            )

            title_evidence_id: str | None = None
            if result.title_source == "content_span" and result.title_span_start is not None:
                evidence = db.get_or_create_evidence_span(
                    doc_version_id=doc_version_id,
                    span_start=result.title_span_start,
                    span_end=result.title_span_end,  # type: ignore
                    run_id=run_id,
                    purpose="title",
                    clean_content=doc_version.clean_content,
                )
                title_evidence_id = evidence.evidence_id
                logger.debug("Created title evidence span: %s", title_evidence_id[:16])

            published_at_evidence_id: str | None = None
            if result.published_at_source == "content_span" and result.published_at_span_start is not None:
                evidence = db.get_or_create_evidence_span(
                    doc_version_id=doc_version_id,
                    span_start=result.published_at_span_start,
                    span_end=result.published_at_span_end,  # type: ignore
                    run_id=run_id,
                    purpose="published_at",
                    clean_content=doc_version.clean_content,
                )
                published_at_evidence_id = evidence.evidence_id
                logger.debug("Created published_at evidence span: %s", published_at_evidence_id[:16])

            metadata_row = DocMetadataRow(
                doc_version_id=doc_version_id,
                title=result.title,
                title_span_start=result.title_span_start,
                title_span_end=result.title_span_end,
                title_source=result.title_source,
                title_confidence=result.title_confidence,
                title_evidence_id=title_evidence_id,
                published_at=result.published_at,
                published_at_raw=result.published_at_raw,
                published_at_format=result.published_at_format,
                published_at_span_start=result.published_at_span_start,
                published_at_span_end=result.published_at_span_end,
                published_at_source=result.published_at_source,
                published_at_confidence=result.published_at_confidence,
                published_at_evidence_id=published_at_evidence_id,
                detected_document_class=result.detected_document_class,
                document_class_confidence=result.document_class_confidence,
                metadata_extraction_log=result.extraction_log,
                created_in_run_id=run_id,
            )

            db.insert_doc_metadata(metadata_row)

            db.upsert_doc_stage_status(
                doc_version_id=doc_version_id,
                stage=STAGE_NAME,
                run_id=run_id,
                config_hash=config_hash,
                status="ok",
            )

        logger.info(
            "Processed %s -> title=%r date=%s class=%s (log: %s)",
            doc_version_id[:16],
            result.title[:50] if result.title else None,
            result.published_at.strftime("%Y-%m-%d") if result.published_at else None,
            result.detected_document_class,
            result.extraction_log,
        )
        return "ok"

    except Exception as e:
        logger.error("Failed to process %s: %s", doc_version_id[:16], str(e))

        try:
            db.upsert_doc_stage_status(
                doc_version_id=doc_version_id,
                stage=STAGE_NAME,
                run_id=run_id,
                config_hash=config_hash,
                status="failed",
                error_message=str(e)[:ERROR_MESSAGE_MAX_LENGTH],
            )
        except Exception as status_err:
            logger.error("Failed to write status for %s: %s", doc_version_id[:16], status_err)

        return "failed"


def run_stage(
    db: Stage03DatabaseInterface,
    run_id: str,
    config_hash: str,
    config: "Config | None" = None,
) -> tuple[int, int, int, int]:
    """
    Run Stage 03 over all eligible documents.

    :param db: Database interface.
    :param run_id: Pipeline run ID.
    :param config_hash: Config version hash.
    :param config: Optional configuration.
    :return: Tuple of (ok_count, failed_count, blocked_count, skipped_count).
    """
    iteration_set = db.get_iteration_set()
    logger.info("Stage 03 iteration set contains %d documents", len(iteration_set))

    ok_count = 0
    failed_count = 0
    blocked_count = 0
    skipped_count = 0

    for doc_version_id in iteration_set:
        status = process_document(
            db=db,
            doc_version_id=doc_version_id,
            run_id=run_id,
            config_hash=config_hash,
            config=config,
        )

        if status == "ok":
            ok_count += 1
        elif status == "failed":
            failed_count += 1
        elif status == "blocked":
            blocked_count += 1
        elif status == "skipped":
            skipped_count += 1

    logger.info(
        "Stage 03 complete: ok=%d, failed=%d, blocked=%d, skipped=%d",
        ok_count,
        failed_count,
        blocked_count,
        skipped_count,
    )

    return ok_count, failed_count, blocked_count, skipped_count


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stage 03: Metadata Extraction")
    parser.add_argument(
        "--run-id",
        type=str,
        default="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        help="Pipeline run ID (required; reused for resumption)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("../../../config/"),
        help="Configuration directory (default: ../../../config/)",
    )
    parser.add_argument(
        "--source-db",
        type=Path,
        default=Path("../../../database/preprocessed_posts.db"),
        help="Source database path (default: ../../../database/preprocewssed_posts.db)",
    )
    parser.add_argument(
        "--working-db",
        type=Path,
        default=Path("../../../database/processed_posts.db"),
        help="Working database path (default: ../../../database/processed_posts.db)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("../../../output/processed/logs/"),
        help="Log directory (default: ../../../output/processed/logs/)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main_stage_03_metadata() -> int:
    """
    Set main entry point for Stage 03 Metadata.

    :return: Exit code (0=success, 1=fatal error).
    """
    args = parse_args()

    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting Stage 03: Metadata Extraction")
    logger.info("Run ID: %s", args.run_id)
    logger.info("Working DB: %s", args.working_db)
    logger.info("Source DB: %s", args.source_db)
    logger.info("Config dir: %s", args.config_dir)
    logger.info("Log dir: %s", args.log_dir)

    # Ensure log directory exists
    args.log_dir.mkdir(parents=True, exist_ok=True)

    config = None
    config_hash = ""
    config_path = args.config_dir / "config.yaml"
    if config_path.exists():
        try:
            config = load_config(config_path)
            config_hash = get_config_version(config)
            logger.info("Loaded config version: %s", config_hash)
            logger.info(
                "Config date validation: min_year=%d, max_year=%d",
                config.global_settings.validation.date_validation.min_year,
                config.global_settings.validation.date_validation.max_year,
            )
            logger.info(
                "Config taxonomy: %d topics, %d document_types",
                len(config.taxonomy.topics),
                len(config.taxonomy.document_types),
            )
            logger.info(
                "Config publishers: %s",
                ", ".join(config.publishers.keys()),
            )
        except Exception as e:
            logger.warning("Failed to load config from %s: %s", config_path, e)
    else:
        logger.warning("Config file not found: %s", config_path)

    try:
        with Stage03DatabaseInterface(
            working_db_path=args.working_db,
            source_db_path=args.source_db,
        ) as db:
            pipeline_run = db.get_pipeline_run(args.run_id)
            if pipeline_run is None:
                logger.error("Pipeline run not found: %s", args.run_id)
                return 1

            if pipeline_run.status != "running":
                logger.error(
                    "Pipeline run %s has status '%s', expected 'running'",
                    args.run_id,
                    pipeline_run.status,
                )
                return 1

            if not config_hash:
                config_hash = pipeline_run.config_version
                logger.info("Using config_hash from pipeline run: %s", config_hash)

            ok_count, failed_count, blocked_count, skipped_count = run_stage(
                db=db,
                run_id=args.run_id,
                config_hash=config_hash,
                config=config,
            )

            attempted_count = ok_count + failed_count

            if attempted_count > 0 and ok_count == 0 and skipped_count == 0:
                logger.error(
                    "Systemic failure: all %d attempted documents failed",
                    attempted_count,
                )
                return 1

    except Exception as e:
        logger.exception("Fatal error in Stage 03: %s", e)
        return 1

    logger.info("Stage 03 completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main_stage_03_metadata())