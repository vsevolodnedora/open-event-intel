"""
Stage 01: Document Ingestion.

Ingests publications from the source database into the working database.
Applies content cleaning, URL normalization, and quality scoring.

On fresh DB (no completed runs), seeds entity_registry, alert_rule, and watchlist
from config.yaml. On subsequent runs, validates seed tables without modification.

Writes:
    scrape_record, document, document_version, doc_stage_status(stage_01_ingest)
    entity_registry, alert_rule, watchlist (fresh DB only)
"""
import argparse
import hashlib
import html
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse, urlunparse

from config_interface import (
    VALID_PUBLISHER_NAMES,
    Config,
    Entity,
    PIIMasking,
    PIIPatternConfig,
    Publisher,
    get_config_version,
    load_config,
)
from config_interface import (
    AlertRule as ConfigAlertRule,
)
from config_interface import (
    Watchlist as ConfigWatchlist,
)
from database_interface import (
    AlertRuleRow,
    DatabaseInterface,
    DBConstraintError,
    DBError,
    DocStageStatusRow,
    DocumentRow,
    DocumentVersionRow,
    EntityRegistryRow,
    PipelineRunRow,
    ScrapeRecordRow,
    SourcePublicationRow,
    WatchlistRow,
    compute_sha256_id,
)
from pydantic import ValidationError

from open_event_intel.logger import get_logger

logger = get_logger(__name__)

STAGE_NAME = "stage_01_ingest"
CLEANING_SPEC_VERSION = "clean_v1"

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

# RFC 3986 §2.3 unreserved characters
_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)


@dataclass(frozen=True)
class CleaningResult:
    """Result of content cleaning."""

    clean_content: str
    content_hash_clean: str
    encoding_repairs: list[str]
    pii_mask_log: dict | None
    content_length_raw: int
    content_length_clean: int
    boilerplate_ratio: float
    content_quality_score: float
    primary_language: str | None
    secondary_languages: list[str] | None
    language_detection_confidence: float | None


class Stage01DatabaseInterface(DatabaseInterface):
    """Database interface for Stage 01 Ingest."""

    READS = {
        "pipeline_run",
        "scrape_record",
        "document",
        "document_version",
        "doc_stage_status",
        "entity_registry",
        "alert_rule",
        "watchlist",
    }
    WRITES = {
        "pipeline_run",
        "scrape_record",
        "document",
        "document_version",
        "doc_stage_status",
        "entity_registry",
        "alert_rule",
        "watchlist",
    }

    def __init__(self, working_db_path: Path, source_db_path: Path) -> None:
        super().__init__(working_db_path, source_db_path, STAGE_NAME)

    def has_completed_runs(self) -> bool:
        """Check if any completed pipeline runs exist."""
        self._check_read_access("pipeline_run")
        row = self._fetchone("SELECT COUNT(*) as cnt FROM pipeline_run WHERE status = 'completed'")
        return row is not None and row["cnt"] > 0

    def scrape_record_exists(self, scrape_id: str) -> bool:
        """Check if a scrape record already exists."""
        self._check_read_access("scrape_record")
        row = self._fetchone("SELECT 1 FROM scrape_record WHERE scrape_id = ?", (scrape_id,))
        return row is not None

    def count_entity_registry(self) -> int:
        """Count entries in entity_registry."""
        self._check_read_access("entity_registry")
        row = self._fetchone("SELECT COUNT(*) as cnt FROM entity_registry")
        return row["cnt"] if row else 0

    def count_alert_rules(self) -> int:
        """Count entries in alert_rule."""
        self._check_read_access("alert_rule")
        row = self._fetchone("SELECT COUNT(*) as cnt FROM alert_rule")
        return row["cnt"] if row else 0

    def count_watchlists(self) -> int:
        """Count entries in watchlist."""
        self._check_read_access("watchlist")
        row = self._fetchone("SELECT COUNT(*) as cnt FROM watchlist")
        return row["cnt"] if row else 0


def normalize_url(url_raw: str, publisher: Publisher | None) -> str:  # noqa: C901
    """
    Normalize a URL according to the spec in §1.4.5.

    Steps:
        1. Parse URL; abort if malformed
        2. Lowercase scheme and host
        3. Remove default ports (80 for http, 443 for https)
        4. Decode percent-encoded unreserved characters (RFC 3986 §2.3)
        5. Remove fragment (#...)
        6. Sort query parameters alphabetically by key
        7. Remove trailing slash from path (unless path is /)
        8. Normalize path segments (remove ., resolve ..)
        9. Preserve scheme (do NOT fold http↔https)
        10. Apply publisher-specific rules from config
    """
    try:
        parsed = urlparse(url_raw)
    except Exception:
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
            query_params = {k: v for k, v in query_params.items() if k in norm.preserve_params}

    sorted_params = sorted(query_params.items())
    query_parts = []
    for key, values in sorted_params:
        for val in sorted(values):
            query_parts.append(f"{key}={val}")
    query = "&".join(query_parts)

    normalized = urlunparse((scheme, netloc, path, "", query, ""))
    return normalized


def _decode_unreserved(s: str) -> str:
    """
    Decode percent-encoded unreserved characters (RFC 3986 §2.3).

    Only unreserved characters (A-Z a-z 0-9 - . _ ~) and '/' are kept decoded.
    All other characters that were percent-encoded are re-encoded to preserve
    URL semantics.
    """
    decoded = unquote(s)
    result: list[str] = []
    for ch in decoded:
        if ch in _UNRESERVED or ch == "/":
            result.append(ch)
        else:
            # Re-encode reserved / non-unreserved characters
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


def _strip_boilerplate(text: str, publisher: Publisher | None) -> str:
    """
    Strip publisher-specific boilerplate lines from text.

    Uses patterns from publisher.boilerplate_patterns config (footer, navigation,
    contact, interactive) to identify and remove boilerplate lines.
    """
    if not publisher or not publisher.boilerplate_patterns:
        return text

    bp = publisher.boilerplate_patterns
    all_patterns: list[str] = []
    for source in (bp.footer, bp.navigation, bp.contact, bp.interactive):
        all_patterns.extend(source)

    if not all_patterns:
        return text

    lines = text.split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        is_boilerplate = False
        for pattern in all_patterns:
            if pattern.lower() in stripped.lower():
                is_boilerplate = True
                break
        if not is_boilerplate:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def clean_content(
    raw_text: str,
    source_language: str | None,
    publisher: Publisher | None,
    pii_masking_enabled: bool,
    pii_config: PIIMasking | None = None,
) -> CleaningResult:
    r"""
    Extract clean raw content per §1.3 Content Cleaning Specification.

    Steps:

        1. Decode bytes using detected encoding; apply repairs
        2. Convert to Unicode NFC normalization
        3. Strip HTML tags (if present after markdown conversion)
        4. Strip publisher-specific boilerplate lines
        5. Collapse whitespace runs to single space (preserve paragraph breaks as \\n\\n)
        6. Remove zero-width characters and control characters except \\n, \\t
        7. Apply PII masking if enabled (using config patterns when available)
    """
    content_length_raw = len(raw_text)
    encoding_repairs: list[str] = []

    text = raw_text
    try:
        if isinstance(raw_text, bytes):
            text = raw_text.decode("utf-8")
            encoding_repairs.append("decoded_utf8")
    except UnicodeDecodeError:
        try:
            text = raw_text.decode("latin-1")  # type: ignore
            encoding_repairs.append("fallback_latin1")
        except Exception:
            text = str(raw_text)
            encoding_repairs.append("forced_str")

    text = unicodedata.normalize("NFC", text)

    text = _strip_html_tags(text)

    # Length after HTML stripping but before boilerplate/whitespace cleaning
    pre_clean_len = len(text)

    # Apply publisher-specific boilerplate removal from config
    text = _strip_boilerplate(text, publisher)
    boilerplate_chars_removed = pre_clean_len - len(text)

    text = _collapse_whitespace(text)
    text = _remove_control_chars(text)

    # Boilerplate ratio: proportion of content identified as boilerplate
    boilerplate_ratio = boilerplate_chars_removed / max(pre_clean_len, 1) if pre_clean_len > 0 else 0.0

    pii_mask_log: dict | None = None
    if pii_masking_enabled:
        text, pii_mask_log = _apply_pii_masking(text, pii_config)

    content_length_clean = len(text)
    content_hash_clean = hashlib.sha256(text.encode("utf-8")).hexdigest().lower()

    primary_language = source_language
    language_detection_confidence = 0.8 if source_language else None
    quality_score = _compute_quality_score(text, boilerplate_ratio)

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
        language_detection_confidence=language_detection_confidence,
    )


def _strip_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def _collapse_whitespace(text: str) -> str:
    r"""Collapse whitespace runs, preserving paragraph breaks as \\n\\n."""
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
    r"""Remove zero-width and control characters except \\n, \\t."""
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

def _apply_pii_masking(
    text: str,
    pii_config: PIIMasking | None = None,
) -> tuple[str, dict]: # noqa: C901  # noqa: C901
    """Applies PII masking using config-defined patterns when available,"""
    log: dict[str, int] = {}

    # Build patterns from config if available, otherwise use defaults
    email_patterns: list[str] = []
    phone_patterns: list[str] = []
    email_whitelist_domains: list[str] = []

    if pii_config and pii_config.patterns:
        email_cfg = pii_config.patterns.get("email")
        if email_cfg and email_cfg.enabled:
            email_patterns = [p.pattern for p in email_cfg.patterns]
            if email_cfg.whitelist and "domains" in email_cfg.whitelist:
                email_whitelist_domains = email_cfg.whitelist["domains"]

        phone_cfg = pii_config.patterns.get("phone")
        if phone_cfg and phone_cfg.enabled:
            phone_patterns = [p.pattern for p in phone_cfg.patterns]

    # Fallback to hardcoded patterns if config didn't provide any
    if not email_patterns:
        email_patterns = [r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"]
    if not phone_patterns:
        phone_patterns = [
            r"\+49\s*[\d\s/-]{8,15}",            # German international
            r"0[1-9][0-9]{1,4}[\s/-]?[0-9]{4,10}",  # German national
            r"\+[1-9][0-9]{0,2}[\s.-]?[0-9\s.-]{6,14}",  # International
        ]

    # Apply email masking
    for pattern in email_patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Filter out whitelisted domains
            non_whitelisted = [
                m for m in matches
                if not any(m.lower().endswith(f"@{d.lower()}") or d.lower() in m.lower()
                           for d in email_whitelist_domains)
            ]
            if non_whitelisted:
                for addr in non_whitelisted:
                    text = text.replace(addr, "[EMAIL]")
                log["email"] = log.get("email", 0) + len(non_whitelisted)

    # Apply phone masking
    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        if matches:
            text = re.sub(pattern, "[PHONE]", text)
            log["phone"] = log.get("phone", 0) + len(matches)

    return text, log if log else {}


def _compute_quality_score(text: str, boilerplate_ratio: float) -> float:
    """Compute content quality score (0..1)."""
    if not text:
        return 0.0

    word_count = len(text.split())
    length_score = min(word_count / 100, 1.0)

    alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    content_ratio = alpha_ratio

    quality = (length_score * 0.3 + content_ratio * 0.4 + (1 - boilerplate_ratio) * 0.3)
    return round(min(max(quality, 0.0), 1.0), 3)


def entity_to_registry_row(entity: Entity) -> EntityRegistryRow:
    """Convert config Entity to EntityRegistryRow."""
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
    """Convert config AlertRule to AlertRuleRow."""
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
    """Convert config Watchlist to WatchlistRow."""
    return WatchlistRow(
        watchlist_id=watchlist_id,
        name=watchlist.name,
        entity_type="mixed",
        entity_values=watchlist.entities,
        track_events=",".join(watchlist.event_types) if watchlist.event_types else None,
        track_topics=",".join(watchlist.topics) if watchlist.topics else None,
        alert_severity=watchlist.urgency_override or "info",
        active=1 if watchlist.enabled else 0,
    )


def seed_tables(db: Stage01DatabaseInterface, config: Config) -> None:
    """
    Seed entity_registry, alert_rule, and watchlist from config.

    Only called on fresh DB (no completed runs).
    """
    logger.info("Seeding entity_registry, alert_rule, and watchlist from config")

    entity_count = 0
    for entity in config.entities:
        row = entity_to_registry_row(entity)
        try:
            db.insert_entity_registry(row)
            entity_count += 1
        except DBConstraintError as e:
            logger.warning(f"Entity {entity.entity_id} already exists: {e}")

    logger.info(f"Seeded {entity_count} entities into entity_registry")

    rule_count = 0
    for rule in config.alerts.rules:
        row = alert_rule_to_row(rule)
        try:
            db.insert_alert_rule(row)
            rule_count += 1
        except DBConstraintError as e:
            logger.warning(f"Alert rule {rule.id} already exists: {e}")

    logger.info(f"Seeded {rule_count} alert rules into alert_rule")

    watchlist_count = 0
    for wl_id, watchlist in config.alerts.watchlists.items():
        row = watchlist_to_row(wl_id, watchlist)
        try:
            db.insert_watchlist(row)
            watchlist_count += 1
        except DBConstraintError as e:
            logger.warning(f"Watchlist {wl_id} already exists: {e}")

    logger.info(f"Seeded {watchlist_count} watchlists")


def ingest_publication(
    db: Stage01DatabaseInterface,
    pub: SourcePublicationRow,
    publisher_id: str,
    publisher: Publisher | None,
    run_id: str,
    config_hash: str,
    pii_masking_enabled: bool,
    pii_config: PIIMasking | None,
    quality_threshold: float,
) -> tuple[str, str]:
    """
    Ingest a single publication into the working database.

    :return: (status, error_message or empty string)
    """
    scrape_id = compute_sha256_id(publisher_id, pub.id, "page")

    if db.scrape_record_exists(scrape_id):
        logger.debug(f"Scrape record {scrape_id} already exists, skipping")
        return ("skipped", "already_ingested")

    url_normalized = normalize_url(pub.url, publisher)

    cleaning_result = clean_content(
        pub.content,
        pub.language,
        publisher,
        pii_masking_enabled,
        pii_config,
    )

    if cleaning_result.content_quality_score < quality_threshold:
        return ("skipped", f"quality_below_threshold:{cleaning_result.content_quality_score}")

    content_hash_raw = hashlib.sha256(pub.content.encode("utf-8")).hexdigest().lower()
    raw_content = pub.content.encode("utf-8")

    document_id = compute_sha256_id(
        publisher_id, url_normalized, pub.published_on.isoformat()
    )
    doc_version_id = compute_sha256_id(scrape_id)

    normalization_spec = {
        "nfc": True,
        "html_stripped": True,
        "boilerplate_stripped": True,
        "whitespace_collapsed": True,
        "control_chars_removed": True,
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


@dataclass
class IngestStats:
    """Statistics from ingestion."""

    processed: int = 0
    skipped: int = 0
    failed: int = 0


def run_ingest(
    db: Stage01DatabaseInterface,
    config: Config,
    run_id: str,
    config_hash: str,
) -> IngestStats:
    """
    Run the ingest process for all publishers.

    :return: Ingest statistics.
    """
    stats = IngestStats()

    pii_settings = config.pii_masking.settings
    pii_masking_enabled = pii_settings.enabled if pii_settings else False
    pii_config = config.pii_masking if pii_masking_enabled else None

    quality_threshold = config.global_settings.quality_thresholds.skip_below_quality_score

    available_tables = set(db.get_source_table_names())
    logger.info(f"Source DB contains tables: {sorted(available_tables)}")

    for publisher_id in VALID_PUBLISHER_NAMES:
        table_name = PUBLISHER_TABLE_MAP.get(publisher_id)
        if not table_name:
            logger.warning(f"No table mapping for publisher {publisher_id}")
            continue

        if table_name not in available_tables:
            logger.debug(f"Table {table_name} not in source DB, skipping")
            continue

        publisher = config.get_publisher(publisher_id)
        logger.info(f"Processing publisher {publisher_id} from table {table_name}")

        try:
            publications = db.read_source_publications(table_name, sort_date=True)
        except Exception as e:
            logger.error(f"Failed to read publications from {table_name}: {e}")
            continue

        logger.info(f"Found {len(publications)} publications in {table_name}")

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
                )

                if status == "ok":
                    stats.processed += 1
                elif status == "skipped":
                    stats.skipped += 1
                    if error_msg != "already_ingested":
                        logger.debug(f"Skipped {pub.id}: {error_msg}")
                else:
                    stats.failed += 1
                    logger.warning(f"Failed {pub.id}: {error_msg}")

            except DBConstraintError as e:
                logger.warning(f"Constraint error for {pub.id}: {e}")
                stats.skipped += 1
            except ValidationError as e:
                logger.error(f"Validation error for {pub.id}: {e}")
                stats.failed += 1
            except Exception as e:
                logger.error(f"Unexpected error for {pub.id}: {e}")
                stats.failed += 1

    return stats


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stage 01: Document Ingestion")
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
    )
    parser.add_argument(
        "--source-db",
        type=Path,
        default=Path("../../../database/preprocessed_posts.db"),
    )
    parser.add_argument(
        "--working-db",
        type=Path,
        default=Path("../../../database/processed_posts.db"),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("../../../output/processed/logs/"),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int: # noqa: C901
    """Set main entry point for Stage 01 Ingest."""
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
    logger.info(f"Config version: {config_hash}")

    if not args.source_db.exists():
        logger.error(f"Source database not found: {args.source_db}")
        return 1

    run_id = args.run_id
    logger.info(f"Starting Stage 01 Ingest for run {run_id}")

    try:
        with Stage01DatabaseInterface(args.working_db, args.source_db) as db:
            pipeline_run = db.get_pipeline_run(run_id)
            if pipeline_run is None:
                logger.error(f"Pipeline run {run_id} not found. Stage 00 must run first.")
                return 1

            if pipeline_run.status != "running":
                logger.error(f"Pipeline run {run_id} is not running (status={pipeline_run.status})")
                return 1

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
                    f"Existing DB with {entity_count} entities, {rule_count} rules, "
                    f"{watchlist_count} watchlists"
                )

            stats = run_ingest(db, config, run_id, config_hash)

            db.update_pipeline_run_counters(
                run_id=run_id,
                doc_count_processed=stats.processed,
                doc_count_skipped=stats.skipped,
                doc_count_failed=stats.failed,
            )

            logger.info(
                f"Ingest complete: processed={stats.processed}, "
                f"skipped={stats.skipped}, failed={stats.failed}"
            )

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