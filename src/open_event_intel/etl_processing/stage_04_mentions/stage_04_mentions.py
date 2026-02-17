"""
Stage 04: Mention Extraction.

Scans each document's immutable document_version.clean_content and produces
span-anchored mentions (who/what is being talked about, and where exactly in
the text), then optionally links those mentions to the curated entity_registry.
Generates human-review proposals for new/unknown entities.

**Responsibility**:
  * scans each document's immutable document_version.clean_content and produces span-anchored mentions (who/what is being talked about, and where exactly in the text), then optionally links those mentions to the curated entity_registry. It also generates human-review proposals for new/unknown entities.
  * Build a gazetteer from entity_registry; Run mention extraction over block-scoped text; Span reconciliation and dedup; Disambiguation + linking (mention_link); Registry proposals

"""
import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

from pydantic import BaseModel

from open_event_intel.etl_processing.config_interface import (
    Config,
    DisambiguationRule,
    Entity,
    get_config_version,
    load_config,
)
from open_event_intel.etl_processing.database_interface import (
    BlockRow,
    ChunkRow,
    DBConstraintError,
    DBError,
    DocumentVersionRow,
    EntityRegistryRow,
    MentionLinkRow,
    MentionRow,
    RegistryUpdateProposalRow,
    compute_sha256_id,
)
from open_event_intel.etl_processing.stage_04_mentions.database_stage_04_mentions import PREREQUISITE_STAGE, STAGE_NAME, Stage04DatabaseInterface
from open_event_intel.logger import get_logger

logger = get_logger(__name__)

# Default context window size (characters around a mention)
DEFAULT_CONTEXT_WINDOW_CHARS = 150

# ---- Heuristic NER constants ----

# Organizational suffixes (German + European corporate forms)
_ORG_SUFFIXES = (
    "GmbH", "AG", "SE", "S.A.", "S.p.A.", "B.V.", "N.V.", "e.V.",
    "KG", "OHG", "gGmbH", "mbH", "Stiftung", "Verband",
    "Association", "Foundation", "Institute", "Group", "Corp",
)
# Compiled pattern: multi-word capitalised phrase ending in an org suffix
_ORG_SUFFIX_RE = re.compile(
    r"\b([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+(?:\s+[A-Za-zÄÖÜäöüß\-]+){0,5}\s+"
    + "|".join(re.escape(s) for s in _ORG_SUFFIXES)
    + r")\b",
    re.UNICODE,
)

# Capitalised multi-word phrase (2-5 words, each starting with an uppercase letter)
# Allows common German/English function words in between.
_FUNC_WORDS = {"und", "für", "der", "des", "von", "de", "for", "of", "and", "the", "in", "du"}
_CAP_PHRASE_RE = re.compile(
    r"\b("
    r"[A-ZÄÖÜ][a-zäöüß]{2,}"           # first word (capitalised, ≥3 chars)
    r"(?:\s+(?:[a-zäöüß]+\s+)?"        # optional function word
    r"[A-ZÄÖÜ][a-zäöüß]{2,}){1,4}"     # 1-4 more capitalised words
    r")\b",
    re.UNICODE,
)

# ALL-CAPS acronym (2-6 uppercase letters, optionally with digits or hyphens)
_ACRONYM_RE = re.compile(
    r"\b([A-Z][A-Z0-9\-]{1,5})\b",
)

# Common false-positive acronyms / stop-words to suppress
_ACRONYM_STOPLIST = frozenset({
    # common words that happen to be all-caps
    "EU", "US", "UK", "UN", "IT", "OR", "IF", "AN", "OF",
    # HTML / formatting artefacts
    "BR", "HR", "TR", "TD", "TH", "UL", "LI", "DIV", "IMG",
    # very short / ambiguous
    "AM", "PM", "VS", "NO", "OK", "II", "III", "IV",
    # German common
    "DER", "DIE", "DAS", "UND", "MIT", "VON", "FUR", "AUS",
    "ZUR", "ZUM",
})


class GazetteerEntry(BaseModel):
    """Entry in the entity gazetteer for fast lookup."""

    entity_id: str
    entity_type: str
    canonical_name: str
    pattern: str
    is_abbreviation: bool = False
    is_alias: bool = False
    is_compound_form: bool = False


@dataclass
class RawMention:
    """Intermediate representation of an extracted mention before dedup."""

    mention_type: str
    surface_form: str
    span_start: int
    span_end: int
    confidence: float
    extraction_method: str
    normalized_value: str | None = None
    context_window_start: int | None = None
    context_window_end: int | None = None
    metadata: dict | None = None


@dataclass
class MentionExtractionResult:
    """Result of mention extraction for a single document."""

    mentions: list[MentionRow] = field(default_factory=list)
    links: list[MentionLinkRow] = field(default_factory=list)
    proposals: list[RegistryUpdateProposalRow] = field(default_factory=list)


class Gazetteer:
    """
    Fast lookup structure for entity matching.

    Builds compiled regex patterns from entity_registry entries for efficient
    mention extraction.  Patterns are compiled with flexible whitespace so that
    match offsets correspond directly to the *original* (non-normalised) text,
    avoiding any offset-drift caused by whitespace collapsing.
    """

    def __init__(self, entities: list[EntityRegistryRow], config_entities: list[Entity]) -> None:
        """Initialize a Gazetteer instance."""
        self._entries: list[GazetteerEntry] = []
        self._patterns_by_type: dict[str, list[tuple[re.Pattern, GazetteerEntry]]] = {}
        self._entity_map: dict[str, EntityRegistryRow] = {e.entity_id: e for e in entities}
        # Collect all known surface forms (lowercased) for filtering heuristic candidates
        self._known_surface_forms: set[str] = set()
        self._build_gazetteer(entities, config_entities)
        logger.info(
            "[GAZETTEER] Built: %d entries across %d entity types, %d compiled patterns, "
            "%d known surface forms",
            len(self._entries),
            len(self._patterns_by_type),
            sum(len(v) for v in self._patterns_by_type.values()),
            len(self._known_surface_forms),
        )

    @staticmethod
    def _make_flexible_pattern(surface: str) -> str:
        r"""
        Build a regex pattern string from a surface following form.

        - escapes all regex metacharacters
        - replaces literal spaces with ``\\s+`` so it matches across
          varying whitespace in the source text

        This ensures match offsets are in the *original* text coordinate
        space rather than in a normalised copy, eliminating the class of
        offset-drift bugs caused by whitespace collapsing.
        """
        surface_clean = unicodedata.normalize("NFC", surface).strip()
        tokens = surface_clean.split()
        escaped_tokens = [re.escape(t) for t in tokens]
        return r"\s+".join(escaped_tokens)

    def _build_gazetteer(  # noqa: C901
        self, entities: list[EntityRegistryRow], config_entities: list[Entity]
    ) -> None:
        """Build gazetteer entries from entity registry and config."""
        config_entity_map = {e.entity_id: e for e in config_entities}

        for entity in entities:
            entity_type = entity.entity_type

            # Add canonical name
            self._add_entry(
                entity.entity_id,
                entity_type,
                entity.canonical_name,
                entity.canonical_name,
            )

            # Add aliases from DB
            if entity.aliases:
                for alias in entity.aliases:
                    self._add_entry(
                        entity.entity_id, entity_type, entity.canonical_name, alias, is_alias=True
                    )

            # Get additional patterns from config entity if available
            config_ent = config_entity_map.get(entity.entity_id)
            if config_ent:
                # Abbreviations from config
                for abbrev in config_ent.abbreviations:
                    self._add_entry(
                        entity.entity_id,
                        entity_type,
                        entity.canonical_name,
                        abbrev,
                        is_abbreviation=True,
                    )
                # Compound forms from config
                for compound in config_ent.compound_forms:
                    self._add_entry(
                        entity.entity_id,
                        entity_type,
                        entity.canonical_name,
                        compound,
                        is_compound_form=True,
                    )
                # German name variants
                for variant in config_ent.name_variants_de:
                    self._add_entry(
                        entity.entity_id, entity_type, entity.canonical_name, variant, is_alias=True
                    )
                # English name variants
                for variant in config_ent.name_variants_en:
                    self._add_entry(
                        entity.entity_id, entity_type, entity.canonical_name, variant, is_alias=True
                    )

        # Compile patterns grouped by entity type
        compiled_count = 0
        failed_count = 0
        for entry in self._entries:
            # Register known surface form
            self._known_surface_forms.add(entry.pattern.lower().strip())

            if entry.entity_type not in self._patterns_by_type:
                self._patterns_by_type[entry.entity_type] = []

            try:
                flexible = self._make_flexible_pattern(entry.pattern)
                pattern = re.compile(
                    r"\b" + flexible + r"\b",
                    re.IGNORECASE | re.UNICODE,
                )
                self._patterns_by_type[entry.entity_type].append((pattern, entry))
                compiled_count += 1
            except re.error as exc:
                logger.warning(
                    "Failed to compile pattern for entity %s (surface=%r): %s",
                    entry.entity_id, entry.pattern, exc,
                )
                failed_count += 1

        if failed_count:
            logger.warning(
                "[GAZETTEER] %d/%d patterns failed to compile",
                failed_count, compiled_count + failed_count,
            )

    def _add_entry(
        self,
        entity_id: str,
        entity_type: str,
        canonical_name: str,
        pattern: str,
        is_abbreviation: bool = False,
        is_alias: bool = False,
        is_compound_form: bool = False,
    ) -> None:
        """Add a gazetteer entry."""
        self._entries.append(
            GazetteerEntry(
                entity_id=entity_id,
                entity_type=entity_type,
                canonical_name=canonical_name,
                pattern=pattern,
                is_abbreviation=is_abbreviation,
                is_alias=is_alias,
                is_compound_form=is_compound_form,
            )
        )

    def find_matches(
        self, text: str, entity_types: list[str] | None = None
    ) -> Iterator[tuple[GazetteerEntry, int, int]]:
        """
        Find all entity matches in text.

        Searches the *original* text directly (no whitespace collapsing)
        so that returned (start, end) offsets are in the caller's coordinate
        space.

        :param text: Text to search (raw, not normalised).
        :param entity_types: Optional filter for entity types.
        :yields: Tuples of (entry, start, end) for each match.
        """
        types_to_search = entity_types or list(self._patterns_by_type.keys())

        for entity_type in types_to_search:
            patterns = self._patterns_by_type.get(entity_type, [])
            for pattern, entry in patterns:
                for match in pattern.finditer(text):
                    yield entry, match.start(), match.end()

    def is_known_surface_form(self, surface: str) -> bool:
        """Check whether *surface* (case-insensitive) is already in the gazetteer, i.e. it matches a registered entity."""
        return surface.lower().strip() in self._known_surface_forms

    def get_entity(self, entity_id: str) -> EntityRegistryRow | None:
        """Get entity by ID."""
        return self._entity_map.get(entity_id)


class MentionExtractor:
    """
    Extracts mentions from document text using gazetteer and pattern matching.

    Handles entity mentions, dates, quantities, legal references, geographic
    mentions, and **heuristic unknown-entity candidates** for proposal
    generation.
    """

    def __init__(
        self,
        gazetteer: Gazetteer,
        config: Config,
        disambiguation_rules: list[DisambiguationRule],
    ) -> None:
        self._gazetteer = gazetteer
        self._config = config
        self._disambiguation_rules = disambiguation_rules
        self._pattern_components = config.extraction.pattern_components
        self._mention_patterns = config.extraction.mention_patterns

    def extract_mentions(
        self, doc_version: DocumentVersionRow, blocks: list[BlockRow], chunks: list[ChunkRow]
    ) -> list[RawMention]:
        """
        Extract all mentions from a document by scanning block-scoped text.

        :param doc_version: Document version containing clean_content.
        :param blocks: Parsed blocks for the document.
        :param chunks: Chunks for the document (for chunk_id assignment).
        :return: List of raw mentions before dedup.
        """
        mentions: list[RawMention] = []
        clean_content = doc_version.clean_content
        content_length = len(clean_content)

        logger.debug(
            "[INPUT] extract_mentions: doc_version_id=%s, content_length=%d, "
            "blocks=%d, chunks=%d",
            doc_version.doc_version_id[:16], content_length, len(blocks), len(chunks),
        )

        # Separate content blocks from boilerplate
        content_blocks = [b for b in blocks if b.boilerplate_flag is None]
        boilerplate_blocks = [b for b in blocks if b.boilerplate_flag is not None]
        logger.debug(
            "[INPUT] Block breakdown: %d content blocks, %d boilerplate blocks",
            len(content_blocks), len(boilerplate_blocks),
        )

        # Extract entity mentions from gazetteer (block-scoped)
        entity_mentions = self._extract_entity_mentions(clean_content, content_blocks)
        logger.debug(
            "[PROCESSING] Entity mention extraction: %d raw hits from %d content blocks",
            len(entity_mentions), len(content_blocks),
        )
        mentions.extend(entity_mentions)

        # Extract pattern-based mentions (block-scoped)
        pattern_mentions = self._extract_pattern_mentions(clean_content, content_blocks)
        logger.debug(
            "[PROCESSING] Pattern-based extraction: %d raw hits (dates, quantities, "
            "legal refs, geo)",
            len(pattern_mentions),
        )
        mentions.extend(pattern_mentions)

        # Build set of spans already covered by known-entity matches so that
        # heuristic candidates don't duplicate them.
        known_spans: list[tuple[int, int]] = [
            (m.span_start, m.span_end) for m in mentions
        ]

        # Heuristic NER: detect potential unknown entities not in gazetteer
        heuristic_mentions = self._extract_unknown_entity_candidates(
            clean_content, content_blocks, known_spans,
        )
        logger.debug(
            "[PROCESSING] Heuristic unknown-entity candidates: %d",
            len(heuristic_mentions),
        )
        mentions.extend(heuristic_mentions)

        logger.debug(
            "[PROCESSING] Total raw mentions before dedup: %d", len(mentions),
        )

        return mentions

    def _extract_entity_mentions(
        self, clean_content: str, blocks: list[BlockRow]
    ) -> list[RawMention]:
        """Extract entity mentions by running gazetteer over each block's text."""
        mentions: list[RawMention] = []
        content_length = len(clean_content)

        for block in blocks:
            block_text = clean_content[block.span_start : block.span_end]
            block_match_count = 0

            for entry, rel_start, rel_end in self._gazetteer.find_matches(block_text):
                abs_start = block.span_start + rel_start
                abs_end = block.span_start + rel_end

                # Boundary check
                if abs_start < 0 or abs_end > content_length:
                    logger.warning(
                        "Span out of bounds for entity %s: abs=[%d,%d), "
                        "content_length=%d — skipping",
                        entry.entity_id, abs_start, abs_end, content_length,
                    )
                    continue

                surface_form = clean_content[abs_start:abs_end]

                # Calculate confidence based on match type
                confidence = 0.9
                if entry.is_abbreviation:
                    confidence = 0.7
                elif entry.is_compound_form:
                    confidence = 0.8
                elif entry.is_alias:
                    confidence = 0.85

                ctx_start = max(0, abs_start - DEFAULT_CONTEXT_WINDOW_CHARS)
                ctx_end = min(content_length, abs_end + DEFAULT_CONTEXT_WINDOW_CHARS)

                mentions.append(
                    RawMention(
                        mention_type=entry.entity_type,
                        surface_form=surface_form,
                        span_start=abs_start,
                        span_end=abs_end,
                        confidence=confidence,
                        extraction_method="gazetteer",
                        normalized_value=entry.canonical_name,
                        context_window_start=ctx_start,
                        context_window_end=ctx_end,
                        metadata={"entity_id": entry.entity_id},
                    )
                )
                block_match_count += 1

            if block_match_count > 0:
                logger.debug(
                    "[PROCESSING] Block %s (type=%s, span=[%d,%d)): %d entity matches",
                    block.block_id[:12], block.block_type,
                    block.span_start, block.span_end, block_match_count,
                )

        return mentions

    def _extract_pattern_mentions(
        self, clean_content: str, blocks: list[BlockRow]
    ) -> list[RawMention]:
        """Extract mentions using configured patterns (block-scoped)."""
        mentions: list[RawMention] = []

        date_mentions = self._extract_dates(clean_content, blocks)
        logger.debug("[PROCESSING]   Dates: %d", len(date_mentions))
        mentions.extend(date_mentions)

        qty_mentions = self._extract_quantities(clean_content, blocks)
        logger.debug("[PROCESSING]   Quantities: %d", len(qty_mentions))
        mentions.extend(qty_mentions)

        legal_mentions = self._extract_legal_refs(clean_content, blocks)
        logger.debug("[PROCESSING]   Legal refs: %d", len(legal_mentions))
        mentions.extend(legal_mentions)

        geo_mentions = self._extract_geo_mentions(clean_content, blocks)
        logger.debug("[PROCESSING]   Geo mentions: %d", len(geo_mentions))
        mentions.extend(geo_mentions)

        return mentions

    # -----------------------------------------------------------------
    # Heuristic unknown-entity candidate detection
    # -----------------------------------------------------------------

    def _extract_unknown_entity_candidates(  # noqa: C901
        self,
        clean_content: str,
        blocks: list[BlockRow],
        known_spans: list[tuple[int, int]],
    ) -> list[RawMention]:
        """
        Detect potential entity mentions **not** already in the gazetteer.

        Uses three complementary heuristics, in priority order:

        1. **Org-suffix detection** – phrases ending in corporate/institutional
           suffixes (GmbH, AG, SE, e.V., Stiftung, …).
        2. **Capitalised multi-word phrases** – two or more consecutive
           capitalised words (the typical pattern for organisation, project,
           and programme names in German and English).
        3. **Acronym detection** – ALL-CAPS tokens of 2-6 characters that are
           not in a stop-list and not already known to the gazetteer.

        Every candidate is checked against ``self._gazetteer.is_known_surface_form``
        and against ``known_spans`` (already-extracted mentions) to avoid
        duplicating work that the gazetteer already handles.

        Candidates receive ``extraction_method = "heuristic_ner"`` and carry
        no ``entity_id`` so that ``process_document`` routes them into the
        proposal pipeline.
        """
        mentions: list[RawMention] = []
        content_length = len(clean_content)

        for block in blocks:
            block_text = clean_content[block.span_start : block.span_end]

            # ---- 1. Org-suffix phrases ------------------------------------
            for match in _ORG_SUFFIX_RE.finditer(block_text):
                raw = self._qualify_candidate(
                    clean_content, block, match, known_spans, content_length,
                    confidence=0.55,
                    inferred_type="ORG",
                )
                if raw:
                    mentions.append(raw)

            # ---- 2. Capitalised multi-word phrases -------------------------
            for match in _CAP_PHRASE_RE.finditer(block_text):
                raw = self._qualify_candidate(
                    clean_content, block, match, known_spans, content_length,
                    confidence=0.40,
                    inferred_type="ORG",
                )
                if raw:
                    mentions.append(raw)

            # ---- 3. ALL-CAPS acronyms --------------------------------------
            for match in _ACRONYM_RE.finditer(block_text):
                surface = match.group(0)
                if surface in _ACRONYM_STOPLIST:
                    continue
                if len(surface) < 2:
                    continue
                raw = self._qualify_candidate(
                    clean_content, block, match, known_spans, content_length,
                    confidence=0.35,
                    inferred_type="ORG",
                )
                if raw:
                    mentions.append(raw)

        if mentions:
            logger.debug(
                "[HEURISTIC_NER] %d unknown-entity candidates found "
                "(before dedup)",
                len(mentions),
            )

        return mentions

    def _qualify_candidate(
        self,
        clean_content: str,
        block: BlockRow,
        match: re.Match,
        known_spans: list[tuple[int, int]],
        content_length: int,
        *,
        confidence: float,
        inferred_type: str,
    ) -> RawMention | None:
        """Return a ``RawMention`` if the regex *match* is a genuine novel candidate, or ``None`` if it should be suppressed."""
        abs_start = block.span_start + match.start()
        abs_end = block.span_start + match.end()
        surface = clean_content[abs_start:abs_end].strip()

        # Reject very short candidates
        if len(surface) < 3:
            return None

        # Already known to gazetteer → not a proposal candidate
        if self._gazetteer.is_known_surface_form(surface):
            return None

        # Overlaps with an already-extracted mention → skip
        for ks, ke in known_spans:
            if abs_start < ke and abs_end > ks:
                return None

        ctx_start = max(0, abs_start - DEFAULT_CONTEXT_WINDOW_CHARS)
        ctx_end = min(content_length, abs_end + DEFAULT_CONTEXT_WINDOW_CHARS)

        return RawMention(
            mention_type=inferred_type,
            surface_form=surface,
            span_start=abs_start,
            span_end=abs_end,
            confidence=confidence,
            extraction_method="heuristic_ner",
            normalized_value=None,
            context_window_start=ctx_start,
            context_window_end=ctx_end,
            metadata={"inferred_type": inferred_type},
        )

    # -----------------------------------------------------------------
    # Date extraction
    # -----------------------------------------------------------------

    def _extract_dates(
        self, clean_content: str, blocks: list[BlockRow]
    ) -> list[RawMention]:
        """Extract date mentions using configured patterns (block-scoped)."""
        mentions: list[RawMention] = []
        patterns = self._pattern_components
        content_length = len(clean_content)

        date_patterns = [
            (patterns.german_date_dmy, "DEADLINE"),
            (patterns.german_date_numeric, "DEADLINE"),
            (patterns.english_date_mdy, "DEADLINE"),
            (patterns.english_date_dmy, "DEADLINE"),
            (patterns.iso_date, "DEADLINE"),
        ]

        for block in blocks:
            block_text = clean_content[block.span_start : block.span_end]
            for pattern_str, mention_type in date_patterns:
                try:
                    pattern = re.compile(pattern_str, re.IGNORECASE | re.UNICODE)
                    for match in pattern.finditer(block_text):
                        abs_start = block.span_start + match.start()
                        abs_end = block.span_start + match.end()
                        surface_form = clean_content[abs_start:abs_end]

                        ctx_start = max(0, abs_start - DEFAULT_CONTEXT_WINDOW_CHARS)
                        ctx_end = min(content_length, abs_end + DEFAULT_CONTEXT_WINDOW_CHARS)

                        mentions.append(
                            RawMention(
                                mention_type=mention_type,
                                surface_form=surface_form,
                                span_start=abs_start,
                                span_end=abs_end,
                                confidence=0.8,
                                extraction_method="regex",
                                normalized_value=surface_form,
                                context_window_start=ctx_start,
                                context_window_end=ctx_end,
                            )
                        )
                except re.error:
                    continue

        return mentions

    # -----------------------------------------------------------------
    # Quantity extraction
    # -----------------------------------------------------------------

    def _extract_quantities(
        self, clean_content: str, blocks: list[BlockRow]
    ) -> list[RawMention]:
        """Extract quantity mentions (power, energy values) — block-scoped."""
        mentions: list[RawMention] = []
        patterns = self._pattern_components
        content_length = len(clean_content)

        quantity_patterns = [
            (f"({patterns.number_de}|{patterns.number_en})\\s*{patterns.power_unit}", "QUANTITY"),
            (f"({patterns.number_de}|{patterns.number_en})\\s*{patterns.energy_unit}", "QUANTITY"),
        ]

        for block in blocks:
            block_text = clean_content[block.span_start : block.span_end]
            for pattern_str, mention_type in quantity_patterns:
                try:
                    pattern = re.compile(pattern_str, re.IGNORECASE | re.UNICODE)
                    for match in pattern.finditer(block_text):
                        abs_start = block.span_start + match.start()
                        abs_end = block.span_start + match.end()
                        surface_form = clean_content[abs_start:abs_end]

                        ctx_start = max(0, abs_start - DEFAULT_CONTEXT_WINDOW_CHARS)
                        ctx_end = min(content_length, abs_end + DEFAULT_CONTEXT_WINDOW_CHARS)

                        mentions.append(
                            RawMention(
                                mention_type=mention_type,
                                surface_form=surface_form,
                                span_start=abs_start,
                                span_end=abs_end,
                                confidence=0.85,
                                extraction_method="regex",
                                normalized_value=surface_form.strip(),
                                context_window_start=ctx_start,
                                context_window_end=ctx_end,
                            )
                        )
                except re.error:
                    continue

        return mentions

    # -----------------------------------------------------------------
    # Legal reference extraction
    # -----------------------------------------------------------------

    def _extract_legal_refs(
        self, clean_content: str, blocks: list[BlockRow]
    ) -> list[RawMention]:
        """Extract legal reference mentions — block-scoped."""
        mentions: list[RawMention] = []
        content_length = len(clean_content)

        legal_ref_pattern = self._mention_patterns.get("LEGAL_REF")
        if not legal_ref_pattern:
            return mentions

        known_laws = legal_ref_pattern.known_german_laws or []
        for law in known_laws:
            abbrev = law.get("abbrev", "")
            if not abbrev:
                continue

            pattern = re.compile(
                r"\b" + re.escape(abbrev) + r"(?:\s+§\s*\d+(?:\s*(?:Abs\.|Absatz)\s*\d+)?)?",
                re.IGNORECASE | re.UNICODE,
            )

            for block in blocks:
                block_text = clean_content[block.span_start : block.span_end]

                for match in pattern.finditer(block_text):
                    abs_start = block.span_start + match.start()
                    abs_end = block.span_start + match.end()
                    surface_form = clean_content[abs_start:abs_end]

                    ctx_start = max(0, abs_start - DEFAULT_CONTEXT_WINDOW_CHARS)
                    ctx_end = min(content_length, abs_end + DEFAULT_CONTEXT_WINDOW_CHARS)

                    mentions.append(
                        RawMention(
                            mention_type="LEGAL_REF",
                            surface_form=surface_form,
                            span_start=abs_start,
                            span_end=abs_end,
                            confidence=0.9,
                            extraction_method="pattern",
                            normalized_value=abbrev,
                            context_window_start=ctx_start,
                            context_window_end=ctx_end,
                            metadata={"law_name": law.get("name", "")},
                        )
                    )

        return mentions

    # -----------------------------------------------------------------
    # Geographic mention extraction
    # -----------------------------------------------------------------

    def _extract_geo_mentions(
        self, clean_content: str, blocks: list[BlockRow]
    ) -> list[RawMention]:
        """Extract geographic mentions — block-scoped."""
        mentions: list[RawMention] = []
        content_length = len(clean_content)

        geo_country_pattern = self._mention_patterns.get("GEO_COUNTRY")
        if geo_country_pattern and geo_country_pattern.gazetteer:
            for entry in geo_country_pattern.gazetteer:
                name = entry.get("name", "")
                if not name:
                    continue

                pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE | re.UNICODE)

                for block in blocks:
                    block_text = clean_content[block.span_start : block.span_end]

                    for match in pattern.finditer(block_text):
                        abs_start = block.span_start + match.start()
                        abs_end = block.span_start + match.end()
                        surface_form = clean_content[abs_start:abs_end]

                        ctx_start = max(0, abs_start - DEFAULT_CONTEXT_WINDOW_CHARS)
                        ctx_end = min(content_length, abs_end + DEFAULT_CONTEXT_WINDOW_CHARS)

                        mentions.append(
                            RawMention(
                                mention_type="GEO_COUNTRY",
                                surface_form=surface_form,
                                span_start=abs_start,
                                span_end=abs_end,
                                confidence=0.85,
                                extraction_method="pattern",
                                normalized_value=entry.get("code", name),
                                context_window_start=ctx_start,
                                context_window_end=ctx_end,
                                metadata={"country_code": entry.get("code")},
                            )
                        )

        return mentions


# =====================================================================
#  Span reconciliation helpers
# =====================================================================

def deduplicate_mentions(mentions: list[RawMention]) -> list[RawMention]:
    """
    Deduplicate overlapping mentions, preferring higher confidence.

    Converts block-local extraction results into a clean set of document-level
    spans by collapsing overlaps and duplicates that arise from adjacent or
    nested blocks.

    :param mentions: List of raw mentions to deduplicate.
    :return: Deduplicated list of mentions.
    """
    if not mentions:
        return []

    before_count = len(mentions)

    # Sort: span_start asc, span length desc (longer preferred), confidence desc
    sorted_mentions = sorted(
        mentions,
        key=lambda m: (m.span_start, -(m.span_end - m.span_start), -m.confidence),
    )

    result: list[RawMention] = []
    for mention in sorted_mentions:
        overlaps = False
        for existing in result:
            if (
                mention.span_start < existing.span_end
                and mention.span_end > existing.span_start
            ):
                if mention.confidence > existing.confidence + 0.1:
                    logger.debug(
                        "[DEDUP] Replacing mention %r (conf=%.2f) with %r "
                        "(conf=%.2f) at span [%d,%d)",
                        existing.surface_form, existing.confidence,
                        mention.surface_form, mention.confidence,
                        mention.span_start, mention.span_end,
                    )
                    result.remove(existing)
                    result.append(mention)
                overlaps = True
                break

        if not overlaps:
            result.append(mention)

    removed = before_count - len(result)
    if removed > 0:
        logger.debug(
            "[DEDUP] Deduplicated %d -> %d mentions (%d overlaps removed)",
            before_count, len(result), removed,
        )

    return result


def assign_chunk_ids(
    mentions: list[RawMention], chunks: list[ChunkRow]
) -> list[tuple[RawMention, list[str]]]:
    """
    Assign chunk IDs to mentions based on span overlap.

    :param mentions: List of mentions.
    :param chunks: List of chunks for the document.
    :return: List of (mention, chunk_ids) tuples.
    """
    result: list[tuple[RawMention, list[str]]] = []

    for mention in mentions:
        chunk_ids: list[str] = []
        for chunk in chunks:
            if (
                mention.span_start < chunk.span_end
                and mention.span_end > chunk.span_start
            ):
                chunk_ids.append(chunk.chunk_id)
        result.append((mention, chunk_ids))

    return result


# =====================================================================
#  Document-level processing
# =====================================================================

def process_document(
    db: Stage04DatabaseInterface,
    doc_version: DocumentVersionRow,
    blocks: list[BlockRow],
    chunks: list[ChunkRow],
    extractor: MentionExtractor,
    gazetteer: Gazetteer,
    run_id: str,
) -> MentionExtractionResult:
    """
    Process a single document for mention extraction.

    Proposal logic
    ==============
    A mention becomes a **registry proposal** when it satisfies ALL of:

    1. It has **no** ``entity_id`` in its metadata (i.e. it wasn't resolved
       to a known entity by the gazetteer).
    2. Its ``extraction_method`` is one that can plausibly produce entity
       candidates: ``"gazetteer"`` (theoretically – see note below),
       ``"heuristic_ner"``, or ``"pattern"`` with certain mention types.

    In practice, gazetteer-sourced mentions always carry an ``entity_id``
    (that's how the gazetteer works), so the proposal path was structurally
    unreachable for entity mentions in the original code.  The addition of
    ``heuristic_ner`` extraction (capitalised phrases, org suffixes,
    acronyms not in the gazetteer) provides the missing source of unknown-
    entity candidates that feed the proposal pipeline.
    """
    result = MentionExtractionResult()
    doc_version_id = doc_version.doc_version_id
    clean_content = doc_version.clean_content

    logger.info(
        "[PROCESS_DOC] Starting mention extraction for doc_version_id=%s "
        "(content_length=%d, blocks=%d, chunks=%d)",
        doc_version_id[:16], len(clean_content), len(blocks), len(chunks),
    )

    # Extract raw mentions (block-scoped)
    raw_mentions = extractor.extract_mentions(doc_version, blocks, chunks)

    # Deduplicate — collapse overlaps/duplicates across adjacent/nested blocks
    deduped_mentions = deduplicate_mentions(raw_mentions)

    # Assign chunk IDs
    mentions_with_chunks = assign_chunk_ids(deduped_mentions, chunks)

    # Track unknown surface forms for proposals
    unknown_surface_forms: dict[str, list[RawMention]] = {}

    # Log type distribution
    type_counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}
    for raw_mention, _ in mentions_with_chunks:
        type_counts[raw_mention.mention_type] = type_counts.get(raw_mention.mention_type, 0) + 1
        method_counts[raw_mention.extraction_method] = method_counts.get(raw_mention.extraction_method, 0) + 1

    if type_counts:
        logger.debug(
            "[PROCESS_DOC] Mention type distribution: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items())),
        )
    if method_counts:
        logger.debug(
            "[PROCESS_DOC] Extraction method distribution: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(method_counts.items())),
        )

    for raw_mention, chunk_ids in mentions_with_chunks:
        mention_id = compute_sha256_id(
            doc_version_id,
            raw_mention.span_start,
            raw_mention.span_end,
            raw_mention.mention_type,
            raw_mention.normalized_value or "",
        )

        # Validate surface form matches clean_content slice
        expected_surface = clean_content[raw_mention.span_start : raw_mention.span_end]
        if expected_surface != raw_mention.surface_form:
            logger.warning(
                "[VALIDATION] Surface form mismatch for mention %s: "
                "expected %r, got %r — correcting",
                mention_id[:16], expected_surface, raw_mention.surface_form,
            )
            raw_mention.surface_form = expected_surface

        mention_row = MentionRow(
            mention_id=mention_id,
            doc_version_id=doc_version_id,
            chunk_ids=chunk_ids if chunk_ids else None,
            mention_type=raw_mention.mention_type,
            surface_form=raw_mention.surface_form,
            normalized_value=raw_mention.normalized_value,
            span_start=raw_mention.span_start,
            span_end=raw_mention.span_end,
            confidence=raw_mention.confidence,
            extraction_method=raw_mention.extraction_method,
            context_window_start=raw_mention.context_window_start,
            context_window_end=raw_mention.context_window_end,
            metadata=raw_mention.metadata,
            created_in_run_id=run_id,
        )
        result.mentions.append(mention_row)

        # ---- Linking vs. proposal routing ----
        entity_id = (raw_mention.metadata or {}).get("entity_id")

        if entity_id:
            # Known entity — create link
            link_id = compute_sha256_id(mention_id, entity_id)
            link_row = MentionLinkRow(
                link_id=link_id,
                mention_id=mention_id,
                entity_id=entity_id,
                link_confidence=raw_mention.confidence,
                link_method=raw_mention.extraction_method,
                created_in_run_id=run_id,
            )
            result.links.append(link_row)

        elif raw_mention.extraction_method == "heuristic_ner":
            # Unknown-entity candidate from heuristic detection → proposal
            surface_key = raw_mention.surface_form.lower().strip()
            if surface_key not in unknown_surface_forms:
                unknown_surface_forms[surface_key] = []
            unknown_surface_forms[surface_key].append(raw_mention)

        # NOTE: extraction_method "regex" and "pattern" do not generate
        # proposals — they represent known patterns (dates, quantities,
        # legal references, geo) and are not unknown-entity candidates.

    # Create proposals for unknown entity mentions
    for surface_key, raw_mentions_list in unknown_surface_forms.items():
        if len(raw_mentions_list) >= 1:
            first_mention = raw_mentions_list[0]
            proposal_id = compute_sha256_id("proposal", surface_key, doc_version_id)
            inferred_type = (first_mention.metadata or {}).get(
                "inferred_type", first_mention.mention_type,
            )
            proposal_row = RegistryUpdateProposalRow(
                proposal_id=proposal_id,
                surface_form=first_mention.surface_form,
                proposal_type="new_entity",
                inferred_type=inferred_type,
                evidence_doc_ids=[doc_version_id],
                occurrence_count=len(raw_mentions_list),
                status="pending",
                created_in_run_id=run_id,
            )
            result.proposals.append(proposal_row)

    # --- OUTPUT SUMMARY ---
    logger.info(
        "[OUTPUT] doc_version_id=%s: %d mentions, %d links, %d proposals",
        doc_version_id[:16],
        len(result.mentions),
        len(result.links),
        len(result.proposals),
    )

    for i, m in enumerate(result.mentions[:5]):
        logger.debug(
            "[OUTPUT]   mention[%d]: type=%s, surface=%r, span=[%d,%d), "
            "conf=%.2f, method=%s",
            i, m.mention_type, m.surface_form, m.span_start, m.span_end,
            m.confidence, m.extraction_method,
        )
    if len(result.mentions) > 5:
        logger.debug("[OUTPUT]   ... and %d more mentions", len(result.mentions) - 5)

    for p in result.proposals[:5]:
        logger.debug(
            "[OUTPUT]   proposal: surface=%r, inferred_type=%s, count=%d",
            p.surface_form, p.inferred_type, p.occurrence_count,
        )
    if len(result.proposals) > 5:
        logger.debug("[OUTPUT]   ... and %d more proposals", len(result.proposals) - 5)

    return result


# =====================================================================
#  Stage runner
# =====================================================================

def run_stage(
    db: Stage04DatabaseInterface,
    config: Config,
    run_id: str,
    config_hash: str,
) -> tuple[int, int, int, int]:
    """
    Run Stage 04 mention extraction.

    :param db: Database interface.
    :param config: Configuration.
    :param run_id: Pipeline run ID.
    :param config_hash: Config version hash.
    :return: Tuple of (ok_count, failed_count, blocked_count, skipped_count).
    """
    ok_count = 0
    failed_count = 0
    blocked_count = 0
    skipped_count = 0
    attempted_count = 0
    total_proposals = 0

    # Build gazetteer from entity registry and config
    entities = db.list_entity_registry()
    logger.info("[INIT] Loaded %d entities from entity_registry", len(entities))
    gazetteer = Gazetteer(entities, config.entities)

    # Create extractor
    extractor = MentionExtractor(gazetteer, config, config.disambiguation_rules)
    logger.info(
        "[INIT] MentionExtractor ready (disambiguation_rules=%d, "
        "mention_patterns=%d, config_entities=%d)",
        len(config.disambiguation_rules),
        len(config.extraction.mention_patterns),
        len(config.entities),
    )

    # Get iteration set
    iteration_set = db.get_stage04_iteration_set()
    logger.info("Stage 04 iteration set: %d documents", len(iteration_set))

    for doc_version_id in iteration_set:
        prereq_status = db.get_prerequisite_status(doc_version_id)

        if prereq_status is None or prereq_status.status != "ok":
            blocking_status = prereq_status.status if prereq_status else "missing"
            error_msg = f"prerequisite_not_ok:{PREREQUISITE_STAGE}:{blocking_status}"

            logger.debug(
                "[BLOCKED] doc_version_id=%s: prerequisite %s status=%s",
                doc_version_id[:16], PREREQUISITE_STAGE, blocking_status,
            )

            existing = db.get_doc_stage_status(doc_version_id, STAGE_NAME)
            if existing and existing.status == "blocked" and existing.error_message == error_msg:
                blocked_count += 1
                continue

            with db.transaction():
                db.upsert_doc_stage_status(
                    doc_version_id=doc_version_id,
                    stage=STAGE_NAME,
                    run_id=run_id,
                    config_hash=config_hash,
                    status="blocked",
                    error_message=error_msg,
                )
            blocked_count += 1
            continue

        attempted_count += 1

        try:
            with db.transaction():
                doc_version = db.get_document_version(doc_version_id)
                if doc_version is None:
                    raise DBError(f"Document version not found: {doc_version_id}")

                blocks = db.get_blocks_by_doc_version_id(doc_version_id)
                chunks = db.get_chunks_by_doc_version_id(doc_version_id)

                logger.debug(
                    "[STAGE] Processing doc_version_id=%s: "
                    "content_length=%d, blocks=%d, chunks=%d",
                    doc_version_id[:16],
                    len(doc_version.clean_content),
                    len(blocks),
                    len(chunks),
                )

                result = process_document(
                    db, doc_version, blocks, chunks, extractor, gazetteer, run_id
                )

                db.insert_mentions(result.mentions)
                db.insert_mention_links(result.links)

                for proposal in result.proposals:
                    existing = db.get_existing_proposal_by_surface_form(proposal.surface_form)
                    if existing:
                        db.update_proposal_occurrence(
                            existing.proposal_id,
                            doc_version_id,
                            existing.occurrence_count + proposal.occurrence_count,
                        )
                    else:
                        db.insert_registry_update_proposal(proposal)

                total_proposals += len(result.proposals)

                db.upsert_doc_stage_status(
                    doc_version_id=doc_version_id,
                    stage=STAGE_NAME,
                    run_id=run_id,
                    config_hash=config_hash,
                    status="ok",
                    details=(
                        f"mentions={len(result.mentions)},"
                        f"links={len(result.links)},"
                        f"proposals={len(result.proposals)}"
                    ),
                )
                ok_count += 1
                logger.debug(
                    "Processed %s: %d mentions, %d links, %d proposals",
                    doc_version_id[:16],
                    len(result.mentions),
                    len(result.links),
                    len(result.proposals),
                )

        except DBConstraintError as e:
            logger.warning("Constraint error for %s: %s", doc_version_id[:16], e)
            with db.transaction():
                db.upsert_doc_stage_status(
                    doc_version_id=doc_version_id,
                    stage=STAGE_NAME,
                    run_id=run_id,
                    config_hash=config_hash,
                    status="failed",
                    error_message=f"constraint_error:{str(e)[:200]}",
                )
            failed_count += 1

        except Exception as e:
            logger.error("Failed to process %s: %s", doc_version_id[:16], e, exc_info=True)
            with db.transaction():
                db.upsert_doc_stage_status(
                    doc_version_id=doc_version_id,
                    stage=STAGE_NAME,
                    run_id=run_id,
                    config_hash=config_hash,
                    status="failed",
                    error_message=str(e)[:500],
                )
            failed_count += 1

    logger.info(
        "[SUMMARY] Stage 04 run complete: attempted=%d, ok=%d, failed=%d, "
        "blocked=%d, skipped=%d, total_proposals=%d",
        attempted_count, ok_count, failed_count, blocked_count,
        skipped_count, total_proposals,
    )
    return ok_count, failed_count, blocked_count, skipped_count


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stage 04: Mention Extraction")
    parser.add_argument(
        "--run-id", type=str, default="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", help="Pipeline run ID (required)"
    )
    parser.add_argument(
        "--config-dir", type=Path, default=Path("../../../config/")
    )
    parser.add_argument(
        "--source-db", type=Path, default=Path("../../../database/preprocessed_posts.db")
    )
    parser.add_argument(
        "--working-db", type=Path, default=Path("../../../database/processed_posts.db")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("../../../output/processed/")
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("../../../output/processed/logs/")
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main_stage_04_mentions() -> int:
    """
    Set main entry point for Stage 04 Mentions.

    :return: Exit code (0 = success, 1 = fatal error).
    """
    args = parse_args()

    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting %s with run_id=%s", STAGE_NAME, args.run_id)

    config_path = args.config_dir / "config.yaml"
    try:
        config = load_config(config_path)
        config_hash = get_config_version(config)
        logger.info(
            "[INIT] Config loaded: config_hash=%s, entities=%d, "
            "disambiguation_rules=%d, mention_patterns=%d",
            config_hash,
            len(config.entities),
            len(config.disambiguation_rules),
            len(config.extraction.mention_patterns),
        )
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        return 1

    db = Stage04DatabaseInterface(args.working_db)
    try:
        db.open()

        run = db.get_pipeline_run(args.run_id)
        if run is None:
            logger.error("Pipeline run not found: %s", args.run_id)
            return 1
        if run.status != "running":
            logger.error("Pipeline run is not running: %s", run.status)
            return 1

        ok_count, failed_count, blocked_count, skipped_count = run_stage(
            db, config, args.run_id, config_hash
        )

        logger.info(
            "Stage 04 completed: ok=%d, failed=%d, blocked=%d, skipped=%d",
            ok_count, failed_count, blocked_count, skipped_count,
        )

        attempted = ok_count + failed_count
        if attempted > 0 and failed_count == attempted:
            logger.error(
                "Systemic failure: all attempted documents failed. "
                "Stage-level issue requires investigation."
            )
            return 1

        return 0

    except Exception as e:
        logger.error("Fatal error in %s: %s", STAGE_NAME, e, exc_info=True)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main_stage_04_mentions())