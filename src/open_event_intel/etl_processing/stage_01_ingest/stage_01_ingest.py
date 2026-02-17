"""
Stage 01: Document Ingestion.

Ingests publications from the source database into the working database.
Applies content cleaning, URL normalization, and quality scoring.

**Reads:** Source DB publisher tables; ``config.yaml`` seed sections (``publishers``, ``entities``, ``alerts``).
**Writes:** ``scrape_record``, ``document``, ``document_version``, ``doc_stage_status(stage_01_ingest)``,
            ``run_stage_status(stage_01_ingest)``.
**Also writes (fresh DB only; single transaction at start of stage):** config-derived seed tables:
``entity_registry``, ``alert_rule``, ``watchlist``.
**Updates:** ``pipeline_run`` counters only (does not create the row).
**Notes:** Produces immutable ``clean_content``. Applies content-quality skip.
**Seed rule (normative; minimal):**
- If the working DB has no completed runs (fresh DB), Stage 01 MUST seed ``entity_registry``, ``alert_rule``,
  and ``watchlist`` from ``config.yaml`` once before ingesting documents.
- If >=1 completed run exists, Stage 01 MUST NOT modify seed tables (validate-only; config drift is
  handled by Stage 0).
"""

import argparse
import hashlib
import html
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse, urlunparse

from pydantic import BaseModel, ValidationError

from open_event_intel.etl_processing.config_interface import (
    VALID_PUBLISHER_NAMES,
    Config,
    Entity,
    LanguageDetection,
    PIIMasking,
    Publisher,
    QualityThresholdsGlobal,
    get_config_version,
    load_config,
)
from open_event_intel.etl_processing.config_interface import (
    AlertRule as ConfigAlertRule,
)
from open_event_intel.etl_processing.config_interface import (
    Watchlist as ConfigWatchlist,
)
from open_event_intel.etl_processing.database_interface import (
    AlertRuleRow,
    DBConstraintError,
    DBError,
    DocumentRow,
    DocumentVersionRow,
    EntityRegistryRow,
    ScrapeRecordRow,
    SourcePublicationRow,
    WatchlistRow,
    compute_sha256_id,
)
from open_event_intel.etl_processing.stage_01_ingest.database_stage_01_ingest import STAGE_NAME, Stage01DatabaseInterface
from open_event_intel.logger import get_logger

logger = get_logger(__name__)

CLEANING_SPEC_VERSION = "clean_v1"

# Mapping from publisher config key to source DB table name.
# Validated against VALID_PUBLISHER_NAMES at module load time.
PUBLISHER_TABLE_MAP: dict[str, str] = {
    "SMARD": "smard",
    "EEX": "eex",
    "ENTSOE": "entsoe",
    "ACER": "acer",
    "EC": "ec",
    "BNETZA": "bnetza",
    "TRANSNETBW": "transnetbw",
    "TENNET": "tennet",
    "FIFTY_HERTZ": "fifty_hertz",
    "AMPRION": "amprion",
    "ICIS": "icis",
    "AGORA": "agora",
    "ENERGY_WIRE": "energy_wire",
}

_map_keys = set(PUBLISHER_TABLE_MAP.keys())
_valid_keys = set(VALID_PUBLISHER_NAMES)
if _map_keys != _valid_keys:
    raise RuntimeError(
        f"PUBLISHER_TABLE_MAP keys out of sync with VALID_PUBLISHER_NAMES. "
        f"Missing from map: {_valid_keys - _map_keys}, "
        f"Extra in map: {_map_keys - _valid_keys}"
    )

# Quality score component weights
QUALITY_WEIGHT_LENGTH = 0.3
QUALITY_WEIGHT_CONTENT = 0.4
QUALITY_WEIGHT_BOILERPLATE = 0.3
# Word count at which length component saturates to 1.0
QUALITY_LENGTH_REFERENCE_WORDS = 100

# RFC 3986 §2.3 unreserved characters
_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)

# Default fallback confidence when language detection config is unavailable.
# Matches the typical config value but only used as a last-resort fallback.
_FALLBACK_LANGUAGE_CONFIDENCE = 0.80


class CleaningResult(BaseModel):
    """Immutable result of content cleaning.

    Carries both the cleaned text and all metadata needed to populate
    ``document_version`` and quality-gate decisions.
    """

    clean_content: str
    content_hash_clean: str
    encoding_repairs: list[str]
    pii_mask_log: dict[str, int] | None
    content_length_raw: int
    content_length_clean: int
    boilerplate_ratio: float
    content_quality_score: float
    primary_language: str | None
    secondary_languages: list[str] | None
    language_detection_confidence: float | None


class IngestStats(BaseModel):
    """Aggregated statistics from an ingestion run."""

    processed: int = 0
    skipped: int = 0
    failed: int = 0
    skipped_reasons: dict[str, int] = {}


def normalize_url(url_raw: str, publisher: Publisher | None) -> str:  # noqa: C901
    """
    Normalize a URL according to the spec in §1.4.5.

    :param url_raw: Raw URL string from source data.
    :param publisher: Publisher config with URL normalization rules, or None.
    :return: Normalized URL string.

    Steps applied: parse, lowercase scheme/host, strip default ports,
    decode unreserved percent-encoding, remove fragment, sort query params,
    remove trailing slash, normalize path, apply publisher-specific rules.
    """
    try:
        parsed = urlparse(url_raw)
    except Exception:
        logger.warning(f"Malformed URL, returning as-is: {url_raw!r}")
        return url_raw

    scheme = parsed.scheme.lower()
    netloc = parsed.hostname or ""
    netloc = netloc.lower()

    port = parsed.port
    if port == 80 and scheme == "http":
        port = None
    if port == 443 and scheme == "https":
        port = None
    if port:
        netloc = f"{netloc}:{port}"

    path = _decode_unreserved(parsed.path)
    path = _normalize_path(path)
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    query_params = parse_qs(parsed.query, keep_blank_values=True)
    if publisher and publisher.url_normalization:
        norm = publisher.url_normalization
        if norm.canonical_host:
            netloc = norm.canonical_host.lower()
        if norm.strip_params:
            for param in norm.strip_params:
                query_params.pop(param, None)
        if norm.preserve_params:
            query_params = {
                k: v for k, v in query_params.items() if k in norm.preserve_params
            }

    sorted_params = sorted(query_params.items())
    query_parts = []
    for key, values in sorted_params:
        for val in sorted(values):
            query_parts.append(f"{key}={val}")
    query = "&".join(query_parts)

    normalized = urlunparse((scheme, netloc, path, "", query, ""))
    return normalized


def _decode_unreserved(s: str) -> str:
    """Decode percent-encoded unreserved characters (RFC 3986 §2.3).

    Only unreserved characters (A-Z a-z 0-9 - . _ ~) and '/' are kept decoded.
    All other characters that were percent-encoded are re-encoded.
    """
    decoded = unquote(s)
    result: list[str] = []
    for ch in decoded:
        if ch in _UNRESERVED or ch == "/":
            result.append(ch)
        else:
            result.append(quote(ch, safe=""))
    return "".join(result)


def _normalize_path(path: str) -> str:
    """Normalize path segments (remove ., resolve ..)."""
    if not path:
        return "/"
    segments = path.split("/")
    output: list[str] = []
    for seg in segments:
        if seg == ".":
            continue
        elif seg == "..":
            if output and output[-1] != "":
                output.pop()
        else:
            output.append(seg)
    result = "/".join(output)
    if not result.startswith("/"):
        result = "/" + result
    return result if result else "/"


def _strip_boilerplate(
    text: str,
    publisher: Publisher | None,
) -> str:
    """Strip publisher-specific boilerplate lines from text.

    Uses ``publisher.boilerplate_patterns`` (footer, navigation, contact,
    interactive) and ``publisher.boilerplate_mode`` to decide strictness.

    ``aggressive``: any partial, case-insensitive substring match removes the line.
    ``moderate``: the boilerplate token must appear as a distinct phrase
    (bounded by line start/end or non-alpha characters) to remove the line.
    """
    if not publisher or not publisher.boilerplate_patterns:
        return text

    bp = publisher.boilerplate_patterns
    all_patterns: list[str] = []
    for source in (bp.footer, bp.navigation, bp.contact, bp.interactive):
        all_patterns.extend(source)

    if not all_patterns:
        return text

    aggressive = publisher.boilerplate_mode == "aggressive"

    lines = text.split("\n")
    cleaned_lines: list[str] = []
    removed_count = 0
    for line in lines:
        stripped = line.strip()
        is_boilerplate = False
        for pattern in all_patterns:
            if aggressive:
                if pattern.lower() in stripped.lower():
                    is_boilerplate = True
                    break
            else:
                if re.search(
                    r"(?<![a-zA-Z])" + re.escape(pattern) + r"(?![a-zA-Z])",
                    stripped,
                    re.IGNORECASE,
                ):
                    is_boilerplate = True
                    break
        if is_boilerplate:
            removed_count += 1
        else:
            cleaned_lines.append(line)

    if removed_count:
        logger.debug(
            f"Boilerplate: removed {removed_count} lines "
            f"(mode={publisher.boilerplate_mode})"
        )

    return "\n".join(cleaned_lines)


def clean_content(  # noqa: C901
    raw_text: str,
    source_language: str | None,
    publisher: Publisher | None,
    pii_masking_enabled: bool,
    pii_config: PIIMasking | None = None,
    language_detection_config: LanguageDetection | None = None,
    quality_thresholds: QualityThresholdsGlobal | None = None,
) -> CleaningResult:
    r"""
    Produce clean content per §1.3 Content Cleaning Specification.

    Steps:

    1. Decode bytes using detected encoding; apply repairs
    2. Convert to Unicode NFC normalization
    3. Strip HTML tags (if present after markdown conversion)
    4. Strip publisher-specific boilerplate lines
    5. Collapse whitespace runs to single space (preserve paragraph breaks as ``\n\n``)
    6. Remove zero-width characters and control characters except ``\n``, ``\t``
    7. Apply PII masking if enabled (using config patterns when available)

    :param raw_text: Raw text content from source.
    :param source_language: Language code from source metadata, if available.
    :param publisher: Publisher config for boilerplate/URL rules.
    :param pii_masking_enabled: Whether PII masking is active.
    :param pii_config: Full PII masking configuration.
    :param language_detection_config: Language detection settings from
        ``config.global_settings.language_detection``.
    :param quality_thresholds: Quality threshold settings from
        ``config.global_settings.quality_thresholds``.
    :return: A :class:`CleaningResult` with cleaned text and metadata.
    """
    content_length_raw = len(raw_text)
    encoding_repairs: list[str] = []

    # Step 1: encoding
    text = raw_text
    try:
        if isinstance(raw_text, bytes):
            text = raw_text.decode("utf-8")
            encoding_repairs.append("decoded_utf8")
    except UnicodeDecodeError:
        try:
            text = raw_text.decode("latin-1")  # type: ignore[union-attr]
            encoding_repairs.append("fallback_latin1")
        except Exception:
            text = str(raw_text)
            encoding_repairs.append("forced_str")

    # Step 2: NFC
    text = unicodedata.normalize("NFC", text)

    # Step 3: HTML
    text = _strip_html_tags(text)

    pre_clean_len = len(text)

    # Step 4: boilerplate
    text = _strip_boilerplate(text, publisher)
    boilerplate_chars_removed = pre_clean_len - len(text)

    # Step 5: whitespace
    text = _collapse_whitespace(text)

    # Step 6: control chars
    text = _remove_control_chars(text)

    boilerplate_ratio = (
        boilerplate_chars_removed / max(pre_clean_len, 1) if pre_clean_len > 0 else 0.0
    )

    # Step 7: PII
    pii_mask_log: dict[str, int] | None = None
    if pii_masking_enabled:
        text, pii_mask_log = _apply_pii_masking(text, pii_config)

    content_length_clean = len(text)
    content_hash_clean = hashlib.sha256(text.encode("utf-8")).hexdigest().lower()

    primary_language, lang_confidence = _detect_language(
        text, source_language, publisher, language_detection_config
    )

    quality_score = _compute_quality_score(text, boilerplate_ratio, quality_thresholds)

    return CleaningResult(
        clean_content=text,
        content_hash_clean=content_hash_clean,
        encoding_repairs=encoding_repairs if encoding_repairs else [],
        pii_mask_log=pii_mask_log,
        content_length_raw=content_length_raw,
        content_length_clean=content_length_clean,
        boilerplate_ratio=round(boilerplate_ratio, 4),
        content_quality_score=quality_score,
        primary_language=primary_language,
        secondary_languages=None,
        language_detection_confidence=lang_confidence,
    )


def _detect_language(
    text: str,
    source_language: str | None,
    publisher: Publisher | None,
    lang_config: LanguageDetection | None,
) -> tuple[str | None, float | None]:
    """Determine the primary language of *text* using config-driven indicators.

    Falls back to ``source_language`` (from the source DB row) or the
    publisher's ``language_default`` when indicator-based detection is
    inconclusive, as controlled by ``fallback_to_publisher_default``.

    :return: ``(language_code, confidence)``
    """
    confidence_threshold = _FALLBACK_LANGUAGE_CONFIDENCE
    indicators: dict[str, list[str]] = {}
    fallback_to_publisher = True

    if lang_config is not None:
        confidence_threshold = lang_config.confidence_threshold
        indicators = lang_config.indicators
        fallback_to_publisher = lang_config.fallback_to_publisher_default
    else:
        logger.warning(
            "Language detection config unavailable; using fallback "
            f"confidence_threshold={_FALLBACK_LANGUAGE_CONFIDENCE}"
        )

    # Attempt indicator-based detection
    if indicators and text:
        words_lower = set(text.lower().split())
        scores: dict[str, int] = {}
        for lang, keywords in indicators.items():
            scores[lang] = sum(1 for kw in keywords if kw in words_lower)
        total_hits = sum(scores.values())
        if total_hits > 0:
            best_lang = max(scores, key=lambda k: scores[k])
            detected_confidence = round(scores[best_lang] / max(total_hits, 1), 3)
            if detected_confidence >= confidence_threshold:
                return best_lang, detected_confidence

    # Fall back to source metadata
    if source_language:
        return source_language, confidence_threshold

    # Fall back to publisher default
    if fallback_to_publisher and publisher:
        return publisher.language_default, confidence_threshold * 0.9

    return None, None


def _strip_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def _collapse_whitespace(text: str) -> str:
    r"""Collapse whitespace runs, preserving paragraph breaks as ``\n\n``."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = text.split("\n")
    lines = [line.strip() for line in lines]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_control_chars(text: str) -> str:
    r"""Remove zero-width and control characters except ``\n``, ``\t``."""
    allowed = {"\n", "\t"}
    result = []
    for ch in text:
        if ch in allowed:
            result.append(ch)
        elif unicodedata.category(ch) in ("Cc", "Cf"):
            continue
        else:
            result.append(ch)
    return "".join(result)


def _apply_pii_masking(  # noqa: C901
    text: str,
    pii_config: PIIMasking | None = None,
) -> tuple[str, dict[str, int]]:
    """Apply PII masking using config-defined patterns.

    Uses mask_format templates, whitelist domains and patterns,
    context exclusion zones, and consistent counters from config.
    If PII config provides no patterns for a category, that category
    is skipped with a warning rather than using hardcoded fallbacks.

    :return: ``(masked_text, mask_log_dict)``
    """
    log: dict[str, int] = {}

    # Resolve mask format templates from config
    email_template = "[EMAIL_{n}]"
    phone_template = "[PHONE_{n}]"

    if pii_config and pii_config.settings:
        mf = pii_config.settings.mask_format
        email_template = mf.email
        phone_template = mf.phone

    # Build patterns from config (no hardcoded fallbacks)
    email_patterns: list[str] = []
    phone_patterns: list[str] = []
    email_whitelist_domains: list[str] = []
    email_whitelist_patterns: list[str] = []

    if pii_config and pii_config.patterns:
        email_cfg = pii_config.patterns.get("email")
        if email_cfg and email_cfg.enabled:
            email_patterns = [p.pattern for p in email_cfg.patterns]
            if email_cfg.whitelist:
                email_whitelist_domains = email_cfg.whitelist.get("domains", [])
                email_whitelist_patterns = email_cfg.whitelist.get("patterns", [])

        phone_cfg = pii_config.patterns.get("phone")
        if phone_cfg and phone_cfg.enabled:
            phone_patterns = [p.pattern for p in phone_cfg.patterns]

    if not email_patterns:
        logger.debug("PII masking: no email patterns configured, skipping email masking")
    if not phone_patterns:
        logger.debug("PII masking: no phone patterns configured, skipping phone masking")

    # Context exclusion zones (press contact, impressum)
    exclusion_zones: list[tuple[int, int]] = []
    if pii_config and pii_config.context_exclusions:
        for section in pii_config.context_exclusions.sections:
            for indicator in section.indicators:
                idx = text.find(indicator)
                if idx != -1:
                    zone_end = idx
                    lines_counted = 0
                    for i in range(idx, len(text)):
                        if text[i] == "\n":
                            lines_counted += 1
                        if lines_counted >= section.scope_lines:
                            zone_end = i
                            break
                    else:
                        zone_end = len(text)
                    exclusion_zones.append((idx, zone_end))
                    logger.debug(
                        f"PII exclusion zone: '{section.name}' at "
                        f"chars {idx}..{zone_end}"
                    )

    def _in_exclusion_zone(match_start: int) -> bool:
        return any(start <= match_start < end for start, end in exclusion_zones)

    # Apply email masking
    email_counter = 0
    all_email_matches: list[str] = []
    for pattern in email_patterns:
        for m in re.finditer(pattern, text):
            addr = m.group(0)
            if addr not in all_email_matches and not _in_exclusion_zone(m.start()):
                all_email_matches.append(addr)

    for addr in all_email_matches:
        is_whitelisted = False
        for domain in email_whitelist_domains:
            if addr.lower().endswith(f"@{domain.lower()}") or domain.lower() in addr.lower():
                is_whitelisted = True
                break
        if not is_whitelisted:
            for wl_pattern in email_whitelist_patterns:
                if wl_pattern.lower() in addr.lower():
                    is_whitelisted = True
                    break
        if not is_whitelisted:
            email_counter += 1
            placeholder = email_template.replace("{n}", str(email_counter))
            text = text.replace(addr, placeholder, 1)
    if email_counter:
        log["email"] = email_counter

    # Apply phone masking via iterative re-scan.
    # Each replacement changes string offsets, so we re-scan from scratch
    # after every substitution until no new matches are found.
    phone_counter = 0
    found_match = True
    while found_match:
        found_match = False
        for pattern in phone_patterns:
            m = re.search(pattern, text)
            if m and not _in_exclusion_zone(m.start()):
                matched_text = m.group(0)
                # Guard against matching already-placed placeholders
                if phone_template.split("{")[0] in matched_text:
                    continue
                phone_counter += 1
                placeholder = phone_template.replace("{n}", str(phone_counter))
                text = text[: m.start()] + placeholder + text[m.end() :]
                found_match = True
                break  # restart scan after modification

    if phone_counter:
        log["phone"] = phone_counter

    if log:
        logger.debug(f"PII masking applied: {log}")

    return text, log if log else {}


def _compute_quality_score(
    text: str,
    boilerplate_ratio: float,
    quality_thresholds: QualityThresholdsGlobal | None = None,
) -> float:
    """Compute content quality score (0..1).

    Uses module-level weight constants. When *quality_thresholds* is provided,
    its ``min_content_ratio`` is used as the floor beneath which the content
    component is penalised.

    :param text: Cleaned text to score.
    :param boilerplate_ratio: Fraction of content removed as boilerplate.
    :param quality_thresholds: Typed quality thresholds from config.
    :return: Quality score in [0.0, 1.0].
    """
    if not text:
        return 0.0

    word_count = len(text.split())
    length_score = min(word_count / QUALITY_LENGTH_REFERENCE_WORDS, 1.0)

    alpha_count = sum(1 for c in text if c.isalpha())
    content_ratio = alpha_count / max(len(text), 1)

    min_content_ratio = (
        quality_thresholds.min_content_ratio
        if quality_thresholds is not None
        else 0.15
    )
    if content_ratio < min_content_ratio:
        content_ratio *= 0.5  # halve contribution when below floor

    quality = (
        QUALITY_WEIGHT_LENGTH * length_score
        + QUALITY_WEIGHT_CONTENT * content_ratio
        + QUALITY_WEIGHT_BOILERPLATE * (1 - boilerplate_ratio)
    )
    return round(min(max(quality, 0.0), 1.0), 3)


def entity_to_registry_row(entity: Entity) -> EntityRegistryRow:
    """Convert config :class:`Entity` to :class:`EntityRegistryRow`."""
    disambiguation_hints = None
    if entity.disambiguation_hints:
        disambiguation_hints = entity.disambiguation_hints.model_dump(exclude_none=True)

    return EntityRegistryRow(
        entity_id=entity.entity_id,
        entity_type=entity.entity_type,
        canonical_name=entity.canonical_name,
        aliases=entity.aliases if entity.aliases else None,
        name_variants_de=", ".join(entity.name_variants_de) if entity.name_variants_de else None,
        name_variants_en=", ".join(entity.name_variants_en) if entity.name_variants_en else None,
        abbreviations=", ".join(entity.abbreviations) if entity.abbreviations else None,
        compound_forms=", ".join(entity.compound_forms) if entity.compound_forms else None,
        valid_from=None,
        valid_to=None,
        source_authority="config",
        disambiguation_hints=disambiguation_hints,
        parent_entity_id=entity.parent_entity_id,
    )


def alert_rule_to_row(rule: ConfigAlertRule) -> AlertRuleRow:
    """Convert config :class:`AlertRule` to :class:`AlertRuleRow`."""
    conditions = rule.triggers.model_dump(exclude_none=True) if rule.triggers else {}
    return AlertRuleRow(
        rule_id=rule.id,
        name=rule.name,
        conditions_json=conditions,
        severity=rule.urgency,
        suppression_window_hours=rule.suppression_window_hours,
        active=1 if rule.enabled else 0,
    )


def watchlist_to_row(watchlist_id: str, watchlist: ConfigWatchlist) -> WatchlistRow:
    """Convert config :class:`Watchlist` to :class:`WatchlistRow`.

    Maps both ``entities`` and ``publishers`` into ``entity_values``
    (combined list), and preserves ``event_types`` and ``topics``.
    """
    combined_values = list(watchlist.entities)
    entity_type = "entity"
    if watchlist.publishers:
        combined_values.extend(watchlist.publishers)
        entity_type = "mixed" if watchlist.entities else "publisher"

    return WatchlistRow(
        watchlist_id=watchlist_id,
        name=watchlist.name,
        entity_type=entity_type,
        entity_values=combined_values,
        track_events=",".join(watchlist.event_types) if watchlist.event_types else None,
        track_topics=",".join(watchlist.topics) if watchlist.topics else None,
        alert_severity=watchlist.urgency_override or "info",
        active=1 if watchlist.enabled else 0,
    )


def seed_tables(db: Stage01DatabaseInterface, config: Config) -> None:
    """Seed entity_registry, alert_rule, and watchlist from config.

    Only called on fresh DB (no completed runs). Must run inside a
    transaction provided by the caller.
    """
    logger.info("Seeding entity_registry, alert_rule, and watchlist from config")

    entity_count = 0
    for entity in config.entities:
        row = entity_to_registry_row(entity)
        try:
            db.insert_entity_registry(row)
            entity_count += 1
            logger.debug(
                f"  Seeded entity: {entity.entity_id} "
                f"({entity.canonical_name}, type={entity.entity_type})"
            )
        except DBConstraintError as e:
            logger.warning(f"Entity {entity.entity_id} already exists: {e}")

    logger.info(f"Seeded {entity_count} entities into entity_registry")

    rule_count = 0
    for rule in config.alerts.rules:
        row = alert_rule_to_row(rule)
        try:
            db.insert_alert_rule(row)
            rule_count += 1
            logger.debug(
                f"  Seeded alert rule: {rule.id} "
                f"(urgency={rule.urgency}, enabled={rule.enabled})"
            )
        except DBConstraintError as e:
            logger.warning(f"Alert rule {rule.id} already exists: {e}")

    logger.info(f"Seeded {rule_count} alert rules into alert_rule")

    watchlist_count = 0
    for wl_id, watchlist in config.alerts.watchlists.items():
        row = watchlist_to_row(wl_id, watchlist)
        try:
            db.insert_watchlist(row)
            watchlist_count += 1
            logger.debug(
                f"  Seeded watchlist: {wl_id} "
                f"(entities={len(watchlist.entities)}, "
                f"publishers={len(watchlist.publishers)}, "
                f"enabled={watchlist.enabled})"
            )
        except DBConstraintError as e:
            logger.warning(f"Watchlist {wl_id} already exists: {e}")

    logger.info(f"Seeded {watchlist_count} watchlists")


def ingest_publication(  # noqa: C901
    db: Stage01DatabaseInterface,
    pub: SourcePublicationRow,
    publisher_id: str,
    publisher: Publisher | None,
    run_id: str,
    config_hash: str,
    pii_masking_enabled: bool,
    pii_config: PIIMasking | None,
    quality_threshold: float,
    min_meaningful_text_length: int,
    language_detection_config: LanguageDetection | None,
    quality_thresholds_config: QualityThresholdsGlobal | None,
) -> tuple[str, str]:
    """Ingest a single publication into the working database.

    :return: ``(status, error_message)`` where status is one of
        ``"ok"``, ``"skipped"``, or ``"failed"``.
    """
    scrape_id = compute_sha256_id(publisher_id, pub.id, "page")

    if db.scrape_record_exists(scrape_id):
        logger.debug(f"Scrape record {scrape_id[:12]}... already exists, skipping")
        return ("skipped", "already_ingested")

    url_normalized = normalize_url(pub.url, publisher)
    logger.debug(
        f"URL normalisation: {pub.url!r} -> {url_normalized!r}"
    )

    cleaning_result = clean_content(
        pub.content,
        pub.language,
        publisher,
        pii_masking_enabled,
        pii_config,
        language_detection_config=language_detection_config,
        quality_thresholds=quality_thresholds_config,
    )

    # Quality gate: min meaningful text length
    if cleaning_result.content_length_clean < min_meaningful_text_length:
        logger.debug(
            f"Skipped {pub.id}: content_length_clean={cleaning_result.content_length_clean} "
            f"< min_meaningful_text_length={min_meaningful_text_length}"
        )
        return (
            "skipped",
            f"below_min_text_length:{cleaning_result.content_length_clean}",
        )

    # Quality gate: quality score
    if cleaning_result.content_quality_score < quality_threshold:
        logger.debug(
            f"Skipped {pub.id}: quality_score={cleaning_result.content_quality_score} "
            f"< threshold={quality_threshold}"
        )
        return (
            "skipped",
            f"quality_below_threshold:{cleaning_result.content_quality_score}",
        )

    # Log cleaning summary
    logger.debug(
        f"Cleaned {pub.id}: "
        f"raw={cleaning_result.content_length_raw} -> "
        f"clean={cleaning_result.content_length_clean}, "
        f"boilerplate={cleaning_result.boilerplate_ratio:.2%}, "
        f"quality={cleaning_result.content_quality_score:.3f}, "
        f"lang={cleaning_result.primary_language} "
        f"(conf={cleaning_result.language_detection_confidence}), "
        f"encoding_repairs={cleaning_result.encoding_repairs}, "
        f"pii={cleaning_result.pii_mask_log}"
    )

    content_hash_raw = hashlib.sha256(pub.content.encode("utf-8")).hexdigest().lower()
    raw_content = pub.content.encode("utf-8")

    document_id = compute_sha256_id(
        publisher_id, url_normalized, pub.published_on.isoformat()
    )
    doc_version_id = compute_sha256_id(scrape_id)

    normalization_spec = {
        "nfc": True,
        "html_stripped": True,
        "boilerplate_stripped": bool(publisher and publisher.boilerplate_patterns),
        "boilerplate_mode": publisher.boilerplate_mode if publisher else None,
        "whitespace_collapsed": True,
        "control_chars_removed": True,
        "pii_masking_enabled": pii_masking_enabled,
    }

    with db.transaction():
        scrape_row = ScrapeRecordRow(
            scrape_id=scrape_id,
            publisher_id=publisher_id,
            source_id=pub.id,
            url_raw=pub.url,
            url_normalized=url_normalized,
            scraped_at=pub.added_on,
            source_published_at=pub.published_on,
            source_title=pub.title,
            source_language=pub.language,
            raw_content=raw_content,
            raw_encoding_detected="utf-8",
            scrape_kind="page",
            ingest_run_id=run_id,
            processing_status="pending",
        )
        db.insert_scrape_record(scrape_row)

        doc_row = DocumentRow(
            document_id=document_id,
            publisher_id=publisher_id,
            url_normalized=url_normalized,
            source_published_at=pub.published_on,
            url_raw_first_seen=pub.url,
            document_class=None,
            is_attachment=0,
        )
        doc_row = db.get_or_create_document(doc_row)
        document_id = doc_row.document_id

        doc_version_row = DocumentVersionRow(
            doc_version_id=doc_version_id,
            document_id=document_id,
            scrape_id=scrape_id,
            content_hash_raw=content_hash_raw,
            encoding_repairs_applied=cleaning_result.encoding_repairs or None,
            cleaning_spec_version=CLEANING_SPEC_VERSION,
            normalization_spec=normalization_spec,
            pii_masking_enabled=1 if pii_masking_enabled else 0,
            scrape_kind="page",
            pii_mask_log=cleaning_result.pii_mask_log,
            content_hash_clean=cleaning_result.content_hash_clean,
            clean_content=cleaning_result.clean_content,
            span_indexing="unicode_codepoint",
            content_length_raw=cleaning_result.content_length_raw,
            content_length_clean=cleaning_result.content_length_clean,
            boilerplate_ratio=cleaning_result.boilerplate_ratio,
            content_quality_score=cleaning_result.content_quality_score,
            primary_language=cleaning_result.primary_language,
            secondary_languages=cleaning_result.secondary_languages,
            language_detection_confidence=cleaning_result.language_detection_confidence,
            created_in_run_id=run_id,
        )
        db.insert_document_version(doc_version_row)

        db.upsert_doc_stage_status(
            doc_version_id=doc_version_id,
            stage=STAGE_NAME,
            run_id=run_id,
            config_hash=config_hash,
            status="ok",
        )

    return ("ok", "")


def run_ingest(
    db: Stage01DatabaseInterface,
    config: Config,
    run_id: str,
    config_hash: str,
) -> IngestStats:
    """Run the ingest process for all configured publishers.

    :return: Aggregated ingest statistics.
    """
    stats = IngestStats()

    # Resolve config-driven parameters up-front (typed, not dict)
    pii_settings = config.pii_masking.settings
    pii_masking_enabled = pii_settings.enabled if pii_settings else False
    pii_config = config.pii_masking if pii_masking_enabled else None

    quality_thresholds = config.global_settings.quality_thresholds
    quality_threshold = quality_thresholds.skip_below_quality_score
    min_meaningful_text_length = quality_thresholds.min_meaningful_text_length

    language_detection_config = config.global_settings.language_detection

    logger.info(
        f"Ingest config: pii_masking={pii_masking_enabled}, "
        f"quality_threshold={quality_threshold}, "
        f"min_meaningful_text_length={min_meaningful_text_length}, "
        f"language_confidence_threshold={language_detection_config.confidence_threshold}, "
        f"min_content_ratio={quality_thresholds.min_content_ratio}, "
        f"cleaning_spec={CLEANING_SPEC_VERSION}"
    )

    available_tables = set(db.get_source_table_names())
    logger.info(f"Source DB contains tables: {sorted(available_tables)}")

    # Determine which publishers will be processed
    active_publishers = [
        pid for pid in VALID_PUBLISHER_NAMES
        if PUBLISHER_TABLE_MAP.get(pid) in available_tables
    ]
    logger.info(
        f"Publishers to process: {len(active_publishers)}/{len(VALID_PUBLISHER_NAMES)} "
        f"({', '.join(active_publishers)})"
    )

    for publisher_id in VALID_PUBLISHER_NAMES:
        table_name = PUBLISHER_TABLE_MAP[publisher_id]

        if table_name not in available_tables:
            logger.debug(f"Table '{table_name}' not in source DB, skipping publisher {publisher_id}")
            continue

        publisher = config.get_publisher(publisher_id)
        if publisher:
            logger.info(
                f"Processing publisher {publisher_id} "
                f"(tier={publisher.processing_tier}, "
                f"lang={publisher.language_default}, "
                f"boilerplate_mode={publisher.boilerplate_mode}) "
                f"from table '{table_name}'"
            )
        else:
            logger.warning(
                f"Publisher {publisher_id} not found in config; "
                f"proceeding without publisher-specific rules"
            )

        try:
            publications = db.read_source_publications(table_name, sort_date=True)
        except Exception as e:
            logger.error(f"Failed to read publications from {table_name}: {e}")
            continue

        logger.info(f"Found {len(publications)} publications in '{table_name}'")

        pub_processed = 0
        pub_skipped = 0
        pub_failed = 0

        for pub in publications:
            try:
                status, error_msg = ingest_publication(
                    db=db,
                    pub=pub,
                    publisher_id=publisher_id,
                    publisher=publisher,
                    run_id=run_id,
                    config_hash=config_hash,
                    pii_masking_enabled=pii_masking_enabled,
                    pii_config=pii_config,
                    quality_threshold=quality_threshold,
                    min_meaningful_text_length=min_meaningful_text_length,
                    language_detection_config=language_detection_config,
                    quality_thresholds_config=quality_thresholds,
                )

                if status == "ok":
                    stats.processed += 1
                    pub_processed += 1
                elif status == "skipped":
                    stats.skipped += 1
                    pub_skipped += 1
                    reason_key = error_msg.split(":")[0] if ":" in error_msg else error_msg
                    stats.skipped_reasons[reason_key] = (
                        stats.skipped_reasons.get(reason_key, 0) + 1
                    )
                    if error_msg != "already_ingested":
                        logger.debug(f"Skipped {pub.id}: {error_msg}")
                else:
                    stats.failed += 1
                    pub_failed += 1
                    logger.warning(f"Failed {pub.id}: {error_msg}")

            except DBConstraintError as e:
                logger.warning(f"Constraint error for {pub.id}: {e}")
                stats.skipped += 1
                pub_skipped += 1
            except ValidationError as e:
                logger.error(f"Validation error for {pub.id}: {e}")
                stats.failed += 1
                pub_failed += 1
            except Exception as e:
                logger.error(f"Unexpected error for {pub.id}: {e}")
                stats.failed += 1
                pub_failed += 1

        logger.info(
            f"Publisher {publisher_id} complete: "
            f"processed={pub_processed}, skipped={pub_skipped}, failed={pub_failed}"
        )

    return stats


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stage 01: Document Ingestion")
    parser.add_argument(
        "--run-id",
        type=str,
        default="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        help="Pipeline run ID (required; created by Stage 00)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("../../../config/"),
        help="Directory containing config.yaml (default: ../../../config/)",
    )
    parser.add_argument(
        "--source-db",
        type=Path,
        default=Path("../../../database/preprocessed_posts.db"),
        help="Path to the source (preprocessed) database",
    )
    parser.add_argument(
        "--working-db",
        type=Path,
        default=Path("../../../database/processed_posts.db"),
        help="Path to the working (processed) database",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("../../../output/processed/logs/"),
        help="Directory for log output",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:  # noqa: C901
    """Run main entry point for Stage 01 Ingest."""
    args = parse_args()

    config_path = args.config_dir / "config.yaml"
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1

    config_hash = get_config_version(config)
    logger.info(f"Config loaded successfully, version hash: {config_hash}")

    if not args.source_db.exists():
        logger.error(f"Source database not found: {args.source_db}")
        return 1

    run_id = args.run_id
    logger.info(f"Starting {STAGE_NAME} for run {run_id[:16]}...")

    try:
        with Stage01DatabaseInterface(args.working_db, args.source_db) as db:
            # Validate pipeline run
            pipeline_run = db.get_pipeline_run(run_id)
            if pipeline_run is None:
                logger.error(
                    f"Pipeline run {run_id[:16]}... not found. Stage 00 must run first."
                )
                return 1

            if pipeline_run.status != "running":
                logger.error(
                    f"Pipeline run {run_id[:16]}... is not running "
                    f"(status={pipeline_run.status})"
                )
                return 1

            logger.info(
                f"Pipeline run validated: started_at={pipeline_run.started_at}, "
                f"config_version={pipeline_run.config_version}"
            )

            # Record run stage start
            db.upsert_run_stage_status(
                run_id=run_id,
                stage=STAGE_NAME,
                config_hash=config_hash,
                status="pending",
            )

            # Seed tables if fresh DB
            is_fresh = not db.has_completed_runs()
            if is_fresh:
                logger.info("Fresh database detected, seeding tables from config")
                with db.transaction():
                    seed_tables(db, config)
            else:
                entity_count = db.count_entity_registry()
                rule_count = db.count_alert_rules()
                watchlist_count = db.count_watchlists()
                logger.info(
                    f"Existing DB: {entity_count} entities, {rule_count} alert rules, "
                    f"{watchlist_count} watchlists"
                )

            # Run ingest
            stats = run_ingest(db, config, run_id, config_hash)

            # Update pipeline counters
            db.update_pipeline_run_counters(
                run_id=run_id,
                doc_count_processed=stats.processed,
                doc_count_skipped=stats.skipped,
                doc_count_failed=stats.failed,
            )

            # Record run stage completion
            stage_status = "ok" if stats.failed == 0 or stats.processed > 0 else "failed"
            stage_error = None
            if stats.processed == 0 and stats.failed > 0:
                stage_error = "All attempted documents failed - systemic error"

            db.upsert_run_stage_status(
                run_id=run_id,
                stage=STAGE_NAME,
                config_hash=config_hash,
                status=stage_status,
                error_message=stage_error,
            )

            # Final summary
            logger.info(
                f"{STAGE_NAME} complete: "
                f"processed={stats.processed}, "
                f"skipped={stats.skipped}, "
                f"failed={stats.failed}"
            )
            if stats.skipped_reasons:
                logger.info(f"Skip reasons: {dict(stats.skipped_reasons)}")

            if stats.processed == 0 and stats.failed > 0:
                logger.error("All attempted documents failed - systemic error")
                return 1

    except DBError as e:
        logger.error(f"Database error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())