"""
Stage 08: Event Extraction.

Converts document-level understanding (mentions, novelty, tables) into structured
events with revisions, provenance-locked evidence, and metric observations.

Logical pipeline per document:
1. Candidate generation via trigger keyword matching against config event_types
2. Slot filling from mentions, mention_links, and table_extracts
3. Canonical-key computation for event identity / cross-document merge
4. Event + event_revision + evidence creation (append-only)
5. Metric observation extraction from QUANTITY mentions and table cells
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from config_interface import Config, EventSlot, EventType, Extraction, get_config_version, load_config
from database_interface import (
    DatabaseInterface,
    DocStageStatusRow,
    DocumentVersionRow,
    EntityRegistryRow,
    EventCandidateRow,
    EventEntityLinkRow,
    EventRevisionEvidenceRow,
    EventRevisionRow,
    EventRow,
    EvidenceSpanRow,
    MentionLinkRow,
    MentionRow,
    MetricObservationRow,
    NoveltyLabelRow,
    TableExtractRow,
    _serialize_json,
    compute_sha256_id,
)
from pydantic import BaseModel, ConfigDict, Field

from open_event_intel.logger import get_logger

logger = get_logger(__name__)

STAGE_NAME = "stage_08_events"
PREREQUISITE_STAGES = ("stage_07_novelty",)

# Module-level defaults (used when config values are absent).
# Every value here is documented; nothing is silently embedded in logic.

# Confidence weights for aggregate event confidence calculation.
# aggregate = fill_ratio * FILL_RATIO_WEIGHT + avg_mention_confidence * CONFIDENCE_WEIGHT
FILL_RATIO_WEIGHT: float = 0.6
CONFIDENCE_WEIGHT: float = 0.4

# Default confidence assigned to extraction-method slots (heading/title heuristic)
EXTRACTION_METHOD_DEFAULT_CONFIDENCE: float = 0.7

# Maximum characters inspected for heading/title extraction methods
EXTRACTION_METHOD_TEXT_WINDOW: int = 500

# Maximum characters used for extracted title/heading values
EXTRACTION_METHOD_MAX_VALUE_LENGTH: int = 200

# Fallback evidence span size (characters from document start) when no slot spans exist
FALLBACK_EVIDENCE_SPAN_CHARS: int = 200

# Default parse_quality assigned to metric observations that have provenance evidence
METRIC_OBS_DEFAULT_PARSE_QUALITY: float = 0.8

# Default confidence for aggregate when no slot confidences are available
AGGREGATE_CONFIDENCE_FALLBACK: float = 0.5


class FilledSlot(BaseModel):
    """A slot filled during event extraction."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: Any
    mention_id: str | None = None
    span_start: int | None = None
    span_end: int | None = None
    confidence: float = 0.0


class ExtractedCandidate(BaseModel):
    """An event candidate extracted from a document."""

    model_config = ConfigDict(extra="forbid")

    event_type: str
    filled_slots: list[FilledSlot] = Field(default_factory=list)
    trigger_mention_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    extraction_tier: int = 1


class ExtractedMetricObs(BaseModel):
    """A metric observation extracted from a document."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str
    value_raw: str
    unit_raw: str | None = None
    value_norm: float | None = None
    unit_norm: str | None = None
    span_start: int | None = None
    span_end: int | None = None
    mention_id: str | None = None
    table_id: str | None = None
    table_row_index: int | None = None
    table_col_index: int | None = None
    period_start: str | None = None
    period_end: str | None = None
    period_granularity: str | None = None
    geography: str | None = None


class Stage08DatabaseInterface(DatabaseInterface):
    """
    Database interface for Stage 08 — Event Extraction.

    Adds SQL methods for event/revision/metric CRUD that are missing
    from the base ``DatabaseInterface``.  These are designed for future
    migration into the base class.
    """

    READS = {
        "pipeline_run",
        "document_version",
        "document",
        "mention",
        "mention_link",
        "novelty_label",
        "table_extract",
        "entity_registry",
        "doc_stage_status",
        "run_stage_status",
        "event",
        "event_revision",
    }
    WRITES = {
        "event",
        "event_revision",
        "event_revision_evidence",
        "event_entity_link",
        "metric_observation",
        "event_candidate",
        "evidence_span",
        "doc_stage_status",
    }

    def __init__(self, working_db_path: Path, source_db_path: Path | None = None) -> None:
        """Initialize."""
        super().__init__(working_db_path, source_db_path, STAGE_NAME)

    def get_iteration_set(self, run_id: str) -> list[tuple[str, str, str]]:
        """
        Return ``(publisher_id, url_normalized, doc_version_id)`` for Stage 08.

        Includes documents that:
        - have no ``doc_stage_status`` row for stage_08_events, OR
        - have status ``'failed'``, OR
        - have status ``'blocked'`` AND all prerequisites are now ``'ok'``.

        Ordered deterministically by ``(publisher_id, url_normalized, doc_version_id)``.
        """
        self._check_read_access("doc_stage_status")
        self._check_read_access("document_version")
        self._check_read_access("document")
        rows = self._fetchall(
            """
            SELECT d.publisher_id, d.url_normalized, dv.doc_version_id
            FROM document_version dv
            JOIN document d ON d.document_id = dv.document_id
            WHERE
                -- no status row yet
                NOT EXISTS (
                    SELECT 1 FROM doc_stage_status dss
                    WHERE dss.doc_version_id = dv.doc_version_id
                      AND dss.stage = 'stage_08_events'
                      AND dss.status IN ('ok', 'skipped')
                )
                AND (
                    -- new doc: no status row
                    NOT EXISTS (
                        SELECT 1 FROM doc_stage_status dss2
                        WHERE dss2.doc_version_id = dv.doc_version_id
                          AND dss2.stage = 'stage_08_events'
                    )
                    -- or failed
                    OR EXISTS (
                        SELECT 1 FROM doc_stage_status dss3
                        WHERE dss3.doc_version_id = dv.doc_version_id
                          AND dss3.stage = 'stage_08_events'
                          AND dss3.status = 'failed'
                    )
                    -- or blocked with prereqs now ok
                    OR (
                        EXISTS (
                            SELECT 1 FROM doc_stage_status dss4
                            WHERE dss4.doc_version_id = dv.doc_version_id
                              AND dss4.stage = 'stage_08_events'
                              AND dss4.status = 'blocked'
                        )
                        AND EXISTS (
                            SELECT 1 FROM doc_stage_status prereq
                            WHERE prereq.doc_version_id = dv.doc_version_id
                              AND prereq.stage = 'stage_07_novelty'
                              AND prereq.status = 'ok'
                        )
                    )
                )
            ORDER BY d.publisher_id, d.url_normalized, dv.doc_version_id
            """,
        )
        return [(r["publisher_id"], r["url_normalized"], r["doc_version_id"]) for r in rows]

    def check_prerequisites_ok(self, doc_version_id: str) -> tuple[bool, str | None]:
        """
        Check whether all prerequisite stages have status ``'ok'``.

        :return: ``(True, None)`` if all ok, else ``(False, error_message)``.
        """
        for prereq in PREREQUISITE_STAGES:
            status_row = self.get_doc_stage_status(doc_version_id, prereq)
            if status_row is None or status_row.status != "ok":
                blocking_status = status_row.status if status_row else "missing"
                return False, f"prerequisite_not_ok:{prereq}:{blocking_status}"
        return True, None

    def get_mention_links_by_doc(self, doc_version_id: str) -> list[MentionLinkRow]:
        """Get all mention_link rows for a document."""
        self._check_read_access("mention_link")
        rows = self._fetchall(
            """SELECT ml.* FROM mention_link ml
               JOIN mention m ON m.mention_id = ml.mention_id
               WHERE m.doc_version_id = ?""",
            (doc_version_id,),
        )
        return [MentionLinkRow.model_validate(dict(r)) for r in rows]

    def get_novelty_label(self, doc_version_id: str) -> NoveltyLabelRow | None:
        """Get the novelty label for a document."""
        self._check_read_access("novelty_label")
        row = self._fetchone(
            "SELECT * FROM novelty_label WHERE doc_version_id = ?",
            (doc_version_id,),
        )
        return NoveltyLabelRow.model_validate(dict(row)) if row else None

    def get_event_by_canonical_key(self, event_type: str, canonical_key: str) -> EventRow | None:
        """Retrieve an event by its type and canonical key."""
        self._check_read_access("event")
        row = self._fetchone(
            "SELECT * FROM event WHERE event_type = ? AND canonical_key = ?",
            (event_type, canonical_key),
        )
        return EventRow.model_validate(dict(row)) if row else None

    def get_event(self, event_id: str) -> EventRow | None:
        """Retrieve an event by ID."""
        self._check_read_access("event")
        row = self._fetchone("SELECT * FROM event WHERE event_id = ?", (event_id,))
        return EventRow.model_validate(dict(row)) if row else None

    def get_max_revision_no(self, event_id: str) -> int:
        """Return the current maximum ``revision_no`` for an event, or 0."""
        self._check_read_access("event_revision")
        row = self._fetchone(
            "SELECT COALESCE(MAX(revision_no), 0) AS max_rev FROM event_revision WHERE event_id = ?",
            (event_id,),
        )
        return int(row["max_rev"]) if row else 0

    def get_current_revision(self, event_id: str) -> EventRevisionRow | None:
        """Get the current revision for an event (via ``event.current_revision_id``)."""
        self._check_read_access("event")
        self._check_read_access("event_revision")
        ev = self.get_event(event_id)
        if ev is None or ev.current_revision_id is None:
            return None
        row = self._fetchone(
            "SELECT * FROM event_revision WHERE revision_id = ?",
            (ev.current_revision_id,),
        )
        return EventRevisionRow.model_validate(dict(row)) if row else None

    def insert_event(self, row: EventRow) -> None:
        """Insert a new event."""
        self._check_write_access("event")
        logger.debug(
            "INSERT event: event_id=%s, event_type=%s, canonical_key=%s",
            row.event_id[:12], row.event_type, row.canonical_key
        )
        self._execute(
            """INSERT INTO event (event_id, event_type, canonical_key,
                current_revision_id, created_in_run_id)
            VALUES (?, ?, ?, ?, ?)""",
            (
                row.event_id,
                row.event_type,
                row.canonical_key,
                row.current_revision_id,
                row.created_in_run_id,
            ),
        )

    def update_event_current_revision(self, event_id: str, revision_id: str) -> None:
        """
        Update ``event.current_revision_id``.

        Per §4, this is the only allowed UPDATE on the ``event`` table.
        """
        self._check_write_access("event")
        logger.debug(
            "UPDATE event current_revision: event_id=%s, revision_id=%s",
            event_id[:12], revision_id[:12]
        )
        self._execute(
            "UPDATE event SET current_revision_id = ? WHERE event_id = ?",
            (revision_id, event_id),
        )

    def insert_event_revision(self, row: EventRevisionRow) -> None:
        """Insert an event revision (append-only)."""
        self._check_write_access("event_revision")
        logger.debug(
            "INSERT event_revision: revision_id=%s, event_id=%s, revision_no=%d, "
            "doc_count=%d, confidence=%.3f",
            row.revision_id[:12], row.event_id[:12], row.revision_no,
            len(row.doc_version_ids), row.confidence
        )
        self._execute(
            """INSERT INTO event_revision
            (revision_id, event_id, revision_no, slots_json, doc_version_ids,
             confidence, extraction_method, extraction_tier,
             supersedes_revision_id, created_in_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row.revision_id,
                row.event_id,
                row.revision_no,
                _serialize_json(row.slots_json),
                _serialize_json(row.doc_version_ids),
                row.confidence,
                row.extraction_method,
                row.extraction_tier,
                row.supersedes_revision_id,
                row.created_in_run_id,
            ),
        )

    def insert_event_revision_evidence(self, row: EventRevisionEvidenceRow) -> None:
        """Insert evidence for an event revision."""
        self._check_write_access("event_revision_evidence")
        logger.debug(
            "INSERT event_revision_evidence: revision_id=%s, evidence_id=%s, purpose=%s",
            row.revision_id[:12], row.evidence_id[:12], row.purpose
        )
        self._execute(
            "INSERT INTO event_revision_evidence (revision_id, evidence_id, purpose) VALUES (?, ?, ?)",
            (row.revision_id, row.evidence_id, row.purpose),
        )

    def insert_event_entity_link(self, row: EventEntityLinkRow) -> None:
        """Insert an event–entity link."""
        self._check_write_access("event_entity_link")
        logger.debug(
            "INSERT event_entity_link: revision_id=%s, entity_id=%s, role=%s",
            row.revision_id[:12], row.entity_id[:12], row.role
        )
        self._execute(
            """INSERT INTO event_entity_link
            (revision_id, entity_id, role, confidence, created_in_run_id)
            VALUES (?, ?, ?, ?, ?)""",
            (row.revision_id, row.entity_id, row.role, row.confidence, row.created_in_run_id),
        )

    def insert_metric_observation(self, row: MetricObservationRow) -> None:
        """Insert a metric observation (idempotent — ignores natural-key duplicates)."""
        self._check_write_access("metric_observation")
        logger.debug(
            "INSERT metric_observation: metric_id=%s, metric_name=%s, value=%s",
            row.metric_id[:12], row.metric_name, row.value_raw
        )
        self._execute(
            """INSERT OR IGNORE INTO metric_observation
            (metric_id, doc_version_id, table_id, metric_name,
             value_raw, unit_raw, value_norm, unit_norm,
             period_start, period_end, period_granularity, geography,
             table_row_index, table_col_index, evidence_id,
             parse_quality, created_in_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row.metric_id,
                row.doc_version_id,
                row.table_id,
                row.metric_name,
                row.value_raw,
                row.unit_raw,
                row.value_norm,
                row.unit_norm,
                row.period_start.isoformat() if row.period_start else None,
                row.period_end.isoformat() if row.period_end else None,
                row.period_granularity,
                row.geography,
                row.table_row_index,
                row.table_col_index,
                row.evidence_id,
                row.parse_quality,
                row.created_in_run_id,
            ),
        )

    def insert_event_candidate(self, row: EventCandidateRow) -> None:
        """Insert an event candidate (internal/debugging)."""
        self._check_write_access("event_candidate")
        logger.debug(
            "INSERT event_candidate: candidate_id=%s, event_type=%s, status=%s",
            row.candidate_id[:12], row.event_type, row.status
        )
        self._execute(
            """INSERT INTO event_candidate
            (candidate_id, doc_version_id, event_type, partial_slots,
             confidence, status, rejection_reason, extraction_tier, created_in_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row.candidate_id,
                row.doc_version_id,
                row.event_type,
                _serialize_json(row.partial_slots),
                row.confidence,
                row.status,
                row.rejection_reason,
                row.extraction_tier,
                row.created_in_run_id,
            ),
        )



# Config helpers — build indexes / maps from the typed config models
def _build_trigger_index(
    extraction: Extraction,
) -> dict[str, list[str]]:
    """
    Build keyword → [event_type] index from config trigger_keywords.

    Returns lowercase keyword to list of event types that use it.
    """
    index: dict[str, list[str]] = {}
    for etype_name, etype_def in extraction.event_types.items():
        if etype_def.trigger_keywords is None:
            continue
        for _lang, kw_value in etype_def.trigger_keywords.items():
            keywords: list[str] = []
            if isinstance(kw_value, list):
                keywords = kw_value
            elif isinstance(kw_value, dict):
                for sub_kws in kw_value.values():
                    if isinstance(sub_kws, list):
                        keywords.extend(sub_kws)
            for kw in keywords:
                key = kw.lower()
                index.setdefault(key, [])
                if etype_name not in index[key]:
                    index[key].append(etype_name)
    return index


def _build_unit_category_map(extraction: Extraction) -> dict[str, str]:
    """
    Build unit → category mapping from config ``QUANTITY.unit_categories``.

    Falls back to a minimal hardcoded map only if the config section is absent.
    """
    quantity_pattern = extraction.mention_patterns.get("QUANTITY")
    if quantity_pattern and quantity_pattern.unit_categories:
        unit_map: dict[str, str] = {}
        for category_name, cat_info in quantity_pattern.unit_categories.items():
            if isinstance(cat_info, dict):
                patterns = cat_info.get("patterns", [])
                for p in patterns:
                    unit_map[p] = category_name
                # Also map the canonical unit
                canonical = cat_info.get("canonical")
                if canonical:
                    unit_map[canonical] = category_name
        if unit_map:
            logger.info(
                "Built unit→category map from config QUANTITY.unit_categories: %d mappings across %d categories",
                len(unit_map), len(extraction.mention_patterns["QUANTITY"].unit_categories),
            )
            return unit_map

    # Fallback — log a warning so it's never silent
    logger.warning(
        "QUANTITY.unit_categories not found in config; using built-in fallback unit→category map"
    )
    return {
        "MW": "power", "GW": "power", "kW": "power", "TW": "power",
        "MWh": "energy", "GWh": "energy", "TWh": "energy", "kWh": "energy",
        "EUR": "currency", "USD": "currency", "€": "currency",
        "km": "length", "m": "length",
        "%": "percentage",
    }


def _get_pipeline_thresholds(extraction: Extraction) -> dict[str, float]:
    """Extract pipeline thresholds from config with logging."""
    thresholds = dict(extraction.pipeline.thresholds)
    logger.info(
        "Pipeline thresholds from config: %s",
        {k: f"{v:.2f}" for k, v in thresholds.items()},
    )
    return thresholds


def _get_context_windows(extraction: Extraction) -> dict[str, int]:
    """Extract context window sizes from config with logging."""
    windows = dict(extraction.pipeline.context_windows)
    logger.info(
        "Context windows from config: %s",
        windows,
    )
    return windows


# Trigger detection
def detect_triggered_event_types(
    clean_content: str,
    trigger_index: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Detect which event types are triggered by keywords in the document.

    :return: ``{event_type: [matched_keywords]}``.
    """
    content_lower = clean_content.lower()
    results: dict[str, list[str]] = {}
    for kw, etypes in trigger_index.items():
        if kw in content_lower:
            for et in etypes:
                results.setdefault(et, [])
                if kw not in results[et]:
                    results[et].append(kw)
    return results


# Slot filling
def _slot_value_from_mentions(  # noqa: C901
    slot: EventSlot,
    mentions: list[MentionRow],
    mention_links: list[MentionLinkRow],
    entities: dict[str, EntityRegistryRow],
    clean_content: str,
) -> FilledSlot | None:
    """
    Attempt to fill a single slot from available mentions.

    Matches by selector criteria (entity_types, mention_types) and
    optional proximity / enum constraints.
    """
    selector = slot.selector or {}
    entity_types: list[str] = selector.get("entity_types", [])
    mention_types: list[str] = selector.get("mention_types", [])
    enum_values: list[str] = slot.enum_values or []
    unit_filter: list[str] = slot.unit_filter or []

    link_map: dict[str, list[MentionLinkRow]] = {}
    for ml in mention_links:
        link_map.setdefault(ml.mention_id, []).append(ml)

    best: FilledSlot | None = None
    best_conf = -1.0

    for m in mentions:
        matched = False
        value: Any = None

        if mention_types and m.mention_type in mention_types:
            matched = True
            value = m.normalized_value or m.surface_form

        if entity_types:
            for ml in link_map.get(m.mention_id, []):
                if ml.entity_id and ml.entity_id in entities:
                    ent = entities[ml.entity_id]
                    if ent.entity_type in entity_types:
                        matched = True
                        value = ent.canonical_name
                        break

        if not matched:
            continue

        if unit_filter and m.mention_type == "QUANTITY":
            meta = m.metadata or {}
            unit = meta.get("unit", "")
            if unit not in unit_filter:
                continue

        if enum_values:
            if value and str(value) not in enum_values:
                continue

        conf = m.confidence
        if conf > best_conf:
            best_conf = conf
            best = FilledSlot(
                name=slot.name,
                value=value,
                mention_id=m.mention_id,
                span_start=m.span_start,
                span_end=m.span_end,
                confidence=conf,
            )
    return best


def _slot_value_from_extraction_method(
    slot: EventSlot,
    clean_content: str,
    doc_version_id: str,
) -> FilledSlot | None:
    """Fill a slot using its ``extraction_method`` (e.g. heading_or_first_sentence)."""
    method = slot.extraction_method
    if not method:
        return None
    if method in ("heading_or_first_sentence", "title_or_first_paragraph"):
        text = clean_content[:EXTRACTION_METHOD_TEXT_WINDOW]
        lines = text.split("\n")
        first_line = ""
        for line in lines:
            stripped = line.strip()
            if stripped:
                first_line = stripped
                break
        if first_line:
            start = clean_content.index(first_line)
            end = start + len(first_line)
            return FilledSlot(
                name=slot.name,
                value=first_line[:EXTRACTION_METHOD_MAX_VALUE_LENGTH],
                span_start=start,
                span_end=end,
                confidence=EXTRACTION_METHOD_DEFAULT_CONFIDENCE,
            )
    return None


def fill_event_slots(
    event_type_def: EventType,
    mentions: list[MentionRow],
    mention_links: list[MentionLinkRow],
    entities: dict[str, EntityRegistryRow],
    clean_content: str,
    doc_version_id: str,
) -> tuple[list[FilledSlot], float]:
    """
    Fill required + optional slots for an event type.

    :return: ``(filled_slots, aggregate_confidence)``.
    """
    all_filled: list[FilledSlot] = []
    required_filled = 0

    for slot in event_type_def.required_slots:
        filled = _slot_value_from_mentions(
            slot, mentions, mention_links, entities, clean_content,
        )
        if filled is None:
            filled = _slot_value_from_extraction_method(slot, clean_content, doc_version_id)
        if filled is not None:
            all_filled.append(filled)
            required_filled += 1

    for slot in event_type_def.optional_slots:
        filled = _slot_value_from_mentions(
            slot, mentions, mention_links, entities, clean_content,
        )
        if filled is not None:
            all_filled.append(filled)

    total_required = len(event_type_def.required_slots)
    if total_required == 0:
        fill_ratio = 1.0
    else:
        fill_ratio = required_filled / total_required

    confidences = [s.confidence for s in all_filled if s.confidence > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else AGGREGATE_CONFIDENCE_FALLBACK

    aggregate = fill_ratio * FILL_RATIO_WEIGHT + avg_conf * CONFIDENCE_WEIGHT

    logger.debug(
        "Aggregate confidence: fill_ratio=%.3f (weight=%.2f) * avg_conf=%.3f (weight=%.2f) = %.3f "
        "(required_filled=%d/%d, optional_filled=%d)",
        fill_ratio, FILL_RATIO_WEIGHT, avg_conf, CONFIDENCE_WEIGHT, aggregate,
        required_filled, total_required, len(all_filled) - required_filled,
    )

    return all_filled, aggregate


# Canonical key computation
def compute_canonical_key(
    event_type: str,
    template: str | None,
    slots: dict[str, Any],
) -> str:
    """
    Compute a deterministic canonical key for event identity.

    Falls back to hashing the event_type + sorted slots if no template.
    """
    if template:
        try:
            key = template.format(**slots)
        except KeyError:
            available = ":".join(f"{k}={v}" for k, v in sorted(slots.items()))
            key = f"{event_type}:{available}"
    else:
        available = ":".join(f"{k}={v}" for k, v in sorted(slots.items()))
        key = f"{event_type}:{available}"
    return key


# Temporal / geographic context helpers
def _infer_period_granularity(date_str: str) -> str | None:
    """
    Infer period granularity from a date string format.

    Returns one of: ``'year'``, ``'quarter'``, ``'month'``, ``'day'``, or ``None``.
    """
    if not date_str:
        return None
    # Quarter patterns: Q1 2025, 2025-Q2, etc.
    if re.search(r"Q[1-4]", date_str, re.IGNORECASE):
        return "quarter"
    # Full ISO date: 2025-01-15 or similar
    if re.match(r"^\d{4}-\d{2}-\d{2}", date_str):
        return "day"
    # Year-month: 2025-01
    if re.match(r"^\d{4}-\d{2}$", date_str):
        return "month"
    # Bare year: 2025
    if re.match(r"^\d{4}$", date_str):
        return "year"
    # German date formats: dd.mm.yyyy
    if re.match(r"^\d{1,2}\.\d{1,2}\.\d{4}$", date_str):
        return "day"
    # Month name + year patterns
    if re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}", date_str, re.IGNORECASE):
        return "month"
    if re.search(r"(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{4}", date_str, re.IGNORECASE):
        return "month"
    return None


def _normalize_date_to_iso(date_str: str) -> str | None:
    """
    Best-effort normalization of a date string to ISO format (YYYY-MM-DD or YYYY-MM or YYYY).

    Returns ``None`` if parsing fails.
    """
    if not date_str:
        return None
    stripped = date_str.strip()

    # Already ISO
    if re.match(r"^\d{4}-\d{2}-\d{2}$", stripped):
        return stripped
    if re.match(r"^\d{4}-\d{2}$", stripped):
        return stripped
    if re.match(r"^\d{4}$", stripped):
        return stripped

    # German numeric: dd.mm.yyyy
    m = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$", stripped)
    if m:
        return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"

    # Quarter: Q1 2025 or 2025-Q1
    m = re.search(r"Q([1-4])\s*(\d{4})", stripped, re.IGNORECASE)
    if m:
        q = int(m.group(1))
        year = m.group(2)
        month = (q - 1) * 3 + 1
        return f"{year}-{month:02d}-01"
    m = re.search(r"(\d{4})\s*-?\s*Q([1-4])", stripped, re.IGNORECASE)
    if m:
        year = m.group(1)
        q = int(m.group(2))
        month = (q - 1) * 3 + 1
        return f"{year}-{month:02d}-01"

    # Try common datetime formats
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y"):
        try:
            dt = datetime.strptime(stripped, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return None


def _find_nearest_temporal_context(  # noqa: C901
    target_start: int | None,
    target_end: int | None,
    mentions: list[MentionRow],
    proximity_chars: int,
) -> tuple[str | None, str | None, str | None]:
    """
    Find the nearest temporal mention (DEADLINE, EFFECTIVE_DATE) to a target span.

    :param proximity_chars: Maximum character distance to consider (from config
        ``extraction.pipeline.context_windows.event_slot_proximity``).
    :return: ``(period_start_iso, period_end_iso, period_granularity)``
    """
    if target_start is None or target_end is None:
        return None, None, None

    target_mid = (target_start + target_end) / 2
    best_distance = float("inf")
    best_date_raw: str | None = None

    for m in mentions:
        if m.mention_type not in ("DEADLINE", "EFFECTIVE_DATE"):
            continue
        if m.span_start is None or m.span_end is None:
            continue
        mention_mid = (m.span_start + m.span_end) / 2
        distance = abs(mention_mid - target_mid)
        if distance > proximity_chars:
            continue
        if distance < best_distance:
            best_distance = distance
            best_date_raw = m.normalized_value or m.surface_form

    if best_date_raw is None:
        return None, None, None

    iso_date = _normalize_date_to_iso(best_date_raw)
    granularity = _infer_period_granularity(best_date_raw)

    # For a single date, set it as period_start; period_end is left None
    # unless we can infer a range from granularity.
    period_start = iso_date
    period_end: str | None = None
    if iso_date and granularity == "year" and re.match(r"^\d{4}$", iso_date):
        period_start = f"{iso_date}-01-01"
        period_end = f"{iso_date}-12-31"
    elif iso_date and granularity == "month" and re.match(r"^\d{4}-\d{2}$", iso_date):
        period_start = f"{iso_date}-01"
        # Approximate month end
        try:
            from calendar import monthrange
            y, mo = int(iso_date[:4]), int(iso_date[5:7])
            _, last_day = monthrange(y, mo)
            period_end = f"{iso_date}-{last_day:02d}"
        except Exception:
            period_end = None

    return period_start, period_end, granularity


def _find_nearest_geography(
    target_start: int | None,
    target_end: int | None,
    mentions: list[MentionRow],
    proximity_chars: int,
) -> str | None:
    """
    Find the nearest geographic mention (GEO_COUNTRY, GEO_REGION, BIDDING_ZONE) to a target span.

    :param proximity_chars: Maximum character distance (from config
        ``extraction.pipeline.context_windows.event_slot_proximity``).
    """
    if target_start is None or target_end is None:
        return None

    target_mid = (target_start + target_end) / 2
    best_distance = float("inf")
    best_geo: str | None = None

    for m in mentions:
        if m.mention_type not in ("GEO_COUNTRY", "GEO_REGION", "BIDDING_ZONE"):
            continue
        if m.span_start is None or m.span_end is None:
            continue
        mention_mid = (m.span_start + m.span_end) / 2
        distance = abs(mention_mid - target_mid)
        if distance > proximity_chars:
            continue
        if distance < best_distance:
            best_distance = distance
            best_geo = m.normalized_value or m.surface_form

    return best_geo


# Quantity validation
def _validate_quantity_value(
    value_norm: float | None,
    unit_raw: str | None,
    validation_ranges: dict[str, Any],
    unit_category_map: dict[str, str],
) -> bool:
    """
    Validate a numeric value against config quantity ranges.

    :param unit_category_map: Built from config via ``_build_unit_category_map()``.
    :return: ``True`` if valid or no applicable range found.
    """
    if value_norm is None:
        return True
    if not unit_raw:
        return True

    category = unit_category_map.get(unit_raw)
    if category is None:
        return True

    vrange = validation_ranges.get(category)
    if vrange is None:
        return True

    # vrange can be a dict with min/max or a QuantityValidationRange-like object
    if isinstance(vrange, dict):
        range_min = vrange.get("min", float("-inf"))
        range_max = vrange.get("max", float("inf"))
    else:
        range_min = getattr(vrange, "min", float("-inf"))
        range_max = getattr(vrange, "max", float("inf"))

    return range_min <= value_norm <= range_max


# Table parsing helpers
def _parse_table_numeric(cell_value: str) -> tuple[float | None, str | None]:
    """
    Attempt to parse a table cell as a numeric value with optional unit.

    :return: ``(value_norm, unit_raw)``
    """
    if not cell_value or not cell_value.strip():
        return None, None

    stripped = cell_value.strip()

    # Try to extract unit from the end
    unit_match = re.search(
        r"(MW|GW|kW|MWh|GWh|TWh|kWh|EUR|USD|€|km|m|%|Mrd\.?|Mio\.?)$",
        stripped,
    )
    unit_raw: str | None = None
    numeric_part = stripped
    if unit_match:
        unit_raw = unit_match.group(1)
        numeric_part = stripped[: unit_match.start()].strip()

    # Clean and parse numeric
    try:
        clean_val = re.sub(r"[^\d.\-eE]", "", numeric_part.replace(",", "."))
        value_norm = float(clean_val) if clean_val else None
    except (ValueError, TypeError):
        value_norm = None

    return value_norm, unit_raw


def _parse_raw_table_text(raw_table_text: str) -> tuple[list[str], list[list[str]]]:
    """Parse ``raw_table_text`` into headers and rows.

    Supports pipe-delimited markdown tables and tab-separated tables.
    Returns ``(headers, rows)`` where rows does NOT include the header row.
    """
    if not raw_table_text or not raw_table_text.strip():
        return [], []

    lines = [line.strip() for line in raw_table_text.strip().splitlines() if line.strip()]
    if not lines:
        return [], []

    # Detect format: pipe-delimited markdown or tab-separated
    if "|" in lines[0]:
        # Markdown-style table: split by pipe, strip outer empties
        def parse_pipe_row(line: str) -> list[str]:
            cells = [c.strip() for c in line.split("|")]
            # Remove leading/trailing empty cells from outer pipes
            if cells and cells[0] == "":
                cells = cells[1:]
            if cells and cells[-1] == "":
                cells = cells[:-1]
            return cells

        headers = parse_pipe_row(lines[0])
        rows: list[list[str]] = []
        for line in lines[1:]:
            # Skip separator rows (e.g., |---|---|)
            if re.match(r"^[\s|:\-]+$", line):
                continue
            rows.append(parse_pipe_row(line))
        return headers, rows
    elif "\t" in lines[0]:
        # Tab-separated
        headers = [c.strip() for c in lines[0].split("\t")]
        rows = [[c.strip() for c in line.split("\t")] for line in lines[1:]]
        return headers, rows
    else:
        # Fallback: treat each line as a single-cell row (no useful table structure)
        logger.debug("raw_table_text has no recognizable table structure (no pipes or tabs)")
        return [], []


# Metric observation extraction
def extract_metric_observations(  # noqa: C901
    mentions: list[MentionRow],
    table_extracts: list[TableExtractRow],
    doc_version_id: str,
    clean_content: str,
    validation_ranges: dict[str, Any] | None = None,
    unit_category_map: dict[str, str] | None = None,
    mention_surfacing_threshold: float = 0.70,
    slot_proximity_chars: int = 500,
) -> list[ExtractedMetricObs]:
    """
    Extract metric observations from QUANTITY mentions and table extracts.

    :param mention_surfacing_threshold: Minimum confidence for QUANTITY mentions
        (from config ``extraction.pipeline.thresholds.mention_surfacing``).
    :param slot_proximity_chars: Proximity window for temporal/geo context
        (from config ``extraction.pipeline.context_windows.event_slot_proximity``).
    :param unit_category_map: Built from config via ``_build_unit_category_map()``.

    Enriches each observation with temporal context (``period_start``,
    ``period_granularity``) from nearby DATE/DEADLINE/EFFECTIVE_DATE mentions,
    and geographic context from nearby GEO mentions.
    """
    results: list[ExtractedMetricObs] = []
    if validation_ranges is None:
        validation_ranges = {}
    if unit_category_map is None:
        unit_category_map = {}

    logger.debug(
        "Extracting metrics: %d mentions, %d table_extracts, threshold=%.2f, proximity=%d",
        len(mentions), len(table_extracts), mention_surfacing_threshold, slot_proximity_chars,
    )

    # --- 1. Metrics from QUANTITY mentions -----------------------------------
    quantity_count = 0
    skipped_low_conf = 0
    skipped_validation = 0

    for m in mentions:
        if m.mention_type != "QUANTITY":
            continue
        quantity_count += 1

        if m.confidence < mention_surfacing_threshold:
            skipped_low_conf += 1
            continue

        meta = m.metadata or {}
        unit_raw = meta.get("unit")
        value_str = m.normalized_value or m.surface_form

        try:
            clean_val = re.sub(r"[^\d.\-eE]", "", value_str.replace(",", "."))
            value_norm = float(clean_val) if clean_val else None
        except (ValueError, TypeError):
            value_norm = None

        # Validate against config ranges
        if not _validate_quantity_value(value_norm, unit_raw, validation_ranges, unit_category_map):
            logger.debug(
                "Skipping QUANTITY mention %s: value %.3f %s outside validation range",
                m.mention_id[:12] if m.mention_id else "?",
                value_norm if value_norm is not None else 0.0,
                unit_raw or "",
            )
            skipped_validation += 1
            continue

        # Temporal context from nearby date mentions
        period_start, period_end, period_granularity = _find_nearest_temporal_context(
            m.span_start, m.span_end, mentions,
            proximity_chars=slot_proximity_chars,
        )

        # Geographic context from nearby geo mentions
        geography = _find_nearest_geography(
            m.span_start, m.span_end, mentions,
            proximity_chars=slot_proximity_chars,
        )

        results.append(
            ExtractedMetricObs(
                metric_name=meta.get("category", "unknown"),
                value_raw=m.surface_form,
                unit_raw=unit_raw,
                value_norm=value_norm,
                unit_norm=unit_raw,
                span_start=m.span_start,
                span_end=m.span_end,
                mention_id=m.mention_id,
                period_start=period_start,
                period_end=period_end,
                period_granularity=period_granularity,
                geography=geography,
            )
        )

    logger.debug(
        "QUANTITY mentions: total=%d, surfaced=%d, skipped_low_conf=%d, skipped_validation=%d",
        quantity_count, len(results), skipped_low_conf, skipped_validation,
    )

    # --- 2. Metrics from table extracts --------------------------------------
    table_metric_count = 0

    for te in table_extracts:
        # TableExtractRow has headers_json and raw_table_text — NOT table_data
        headers: list[str] = []
        rows: list[list[str]] = []

        if te.headers_json and te.raw_table_text:
            # Use raw_table_text for full row data, headers_json for verified headers
            parsed_headers, parsed_rows = _parse_raw_table_text(te.raw_table_text)
            # Prefer headers_json from the DB (already validated by earlier stage)
            headers = [str(h) for h in te.headers_json] if te.headers_json else parsed_headers
            rows = parsed_rows
        elif te.raw_table_text:
            headers, rows = _parse_raw_table_text(te.raw_table_text)
        elif te.headers_json:
            # Headers only, no row data — skip
            logger.debug(
                "Table %s has headers_json but no raw_table_text — skipping metric extraction",
                te.table_id[:12],
            )
            continue
        else:
            continue

        if not headers or not rows:
            logger.debug(
                "Table %s produced no parseable rows (headers=%d, rows=%d)",
                te.table_id[:12], len(headers), len(rows),
            )
            continue

        logger.debug(
            "Processing table %s for metrics: %d headers, %d rows",
            te.table_id[:12], len(headers), len(rows),
        )

        # Detect temporal header columns (years, dates)
        header_temporal: dict[int, tuple[str | None, str | None, str | None]] = {}
        for col_idx, h in enumerate(headers):
            h_stripped = str(h).strip()
            iso = _normalize_date_to_iso(h_stripped)
            if iso:
                gran = _infer_period_granularity(h_stripped)
                p_start = iso
                p_end: str | None = None
                if gran == "year" and re.match(r"^\d{4}$", h_stripped):
                    p_start = f"{h_stripped}-01-01"
                    p_end = f"{h_stripped}-12-31"
                header_temporal[col_idx] = (p_start, p_end, gran)

        # Geographic context: use nearby GEO mentions within proximity of the
        # table's block span (approximated via raw_table_text length)
        table_geo: str | None = None
        # We don't have context_before on TableExtractRow; instead find geo
        # mentions near the start of the document as a rough approximation.
        table_geo = _find_nearest_geography(
            0, len(te.raw_table_text) if te.raw_table_text else 0,
            mentions, proximity_chars=slot_proximity_chars * 2,
        )

        for row_idx, row in enumerate(rows):
            # First column is typically the metric label
            metric_label = str(row[0]).strip() if row else ""
            if not metric_label:
                continue

            for col_idx in range(1, len(row)):
                cell_value = str(row[col_idx]).strip() if col_idx < len(row) else ""
                if not cell_value:
                    continue

                value_norm, unit_raw = _parse_table_numeric(cell_value)
                if value_norm is None:
                    continue

                # Validate against config ranges
                if not _validate_quantity_value(value_norm, unit_raw, validation_ranges, unit_category_map):
                    continue

                # Get temporal info from header if available
                t_start, t_end, t_gran = header_temporal.get(col_idx, (None, None, None))

                col_header = str(headers[col_idx]).strip() if col_idx < len(headers) else ""
                metric_name = f"{metric_label}"
                if col_header and col_idx not in header_temporal:
                    metric_name = f"{metric_label}:{col_header}"

                results.append(
                    ExtractedMetricObs(
                        metric_name=metric_name,
                        value_raw=cell_value,
                        unit_raw=unit_raw,
                        value_norm=value_norm,
                        unit_norm=unit_raw,
                        table_id=te.table_id,
                        table_row_index=row_idx,
                        table_col_index=col_idx,
                        period_start=t_start,
                        period_end=t_end,
                        period_granularity=t_gran,
                        geography=table_geo,
                    )
                )
                table_metric_count += 1

    logger.debug(
        "Table metric extraction: %d observations from %d table extracts",
        table_metric_count, len(table_extracts),
    )

    return results


# Metric deduplication
def metric_natural_key(
    doc_version_id: str, obs: ExtractedMetricObs,
) -> tuple[str, str, str, str, str, str, int, int]:
    """
    Compute the natural-key tuple that mirrors ``idx_metric_obs_natural_key``.

    The DB index uses ``IFNULL`` to coalesce NULLs into deterministic
    sentinel values.  This function replicates that logic in Python so
    that metric identity in Stage 08 is consistent with the schema's
    uniqueness constraint.
    """
    return (
        doc_version_id,
        obs.table_id or "",
        obs.metric_name,
        obs.period_start or "",
        obs.period_end or "",
        obs.geography or "",
        obs.table_row_index if obs.table_row_index is not None else -1,
        obs.table_col_index if obs.table_col_index is not None else -1,
    )


def deduplicate_metric_observations(
    observations: list[ExtractedMetricObs],
    doc_version_id: str,
) -> list[ExtractedMetricObs]:
    """
    Collapse observations that share the same DB natural key.

    Selection policy (deterministic tie-break):
      1. Prefer observations with a successfully parsed ``value_norm``.
      2. Prefer higher mention confidence (proxied by earlier span
         position for now — upstream sorts by confidence then position).
      3. Prefer the observation with the earliest span (stable ordering).
    """
    chosen: dict[tuple, ExtractedMetricObs] = {}
    for obs in observations:
        key = metric_natural_key(doc_version_id, obs)
        prev = chosen.get(key)
        if prev is None:
            chosen[key] = obs
            continue
        # Prefer parsed numeric value
        obs_has_norm = obs.value_norm is not None
        prev_has_norm = prev.value_norm is not None
        if obs_has_norm and not prev_has_norm:
            chosen[key] = obs
        elif obs_has_norm == prev_has_norm:
            # Tie-break: earliest span (deterministic)
            obs_start = obs.span_start if obs.span_start is not None else float("inf")
            prev_start = prev.span_start if prev.span_start is not None else float("inf")
            if obs_start < prev_start:
                chosen[key] = obs
    return list(chosen.values())


# Per-document processing
def process_single_document(  # noqa: C901
    db: Stage08DatabaseInterface,
    doc_version_id: str,
    run_id: str,
    config_hash: str,
    extraction: Extraction,
    trigger_index: dict[str, list[str]],
    entity_map: dict[str, EntityRegistryRow],
    quantity_validation_ranges: dict[str, Any] | None = None,
    unit_category_map: dict[str, str] | None = None,
    mention_surfacing_threshold: float = 0.70,
    slot_proximity_chars: int = 500,
) -> None:
    """
    Process a single document for event extraction.

    Runs inside the caller's transaction context. Raises on error
    so the caller can ROLLBACK.

    :param mention_surfacing_threshold: From config ``extraction.pipeline.thresholds.mention_surfacing``.
    :param slot_proximity_chars: From config ``extraction.pipeline.context_windows.event_slot_proximity``.
    :param unit_category_map: Built from config via ``_build_unit_category_map()``.
    """
    logger.info("Processing document: doc_version_id=%s", doc_version_id[:12])

    dv = db.get_document_version(doc_version_id)
    if dv is None:
        raise ValueError(f"document_version not found: {doc_version_id}")

    clean_content = dv.clean_content
    logger.debug("Document content length: %d characters", len(clean_content))

    mentions = db.get_mentions_by_doc_version_id(doc_version_id)
    logger.debug("Loaded %d mentions for document", len(mentions))

    # Log mention type distribution for inspection
    mention_type_counts: dict[str, int] = {}
    for m in mentions:
        mention_type_counts[m.mention_type] = mention_type_counts.get(m.mention_type, 0) + 1
    if mention_type_counts:
        logger.debug("Mention types: %s", mention_type_counts)

    mention_links = db.get_mention_links_by_doc(doc_version_id)
    logger.debug("Loaded %d mention links for document", len(mention_links))

    table_extracts = db.get_table_extracts_by_doc_version_id(doc_version_id)
    logger.debug("Loaded %d table extracts for document", len(table_extracts))

    triggered = detect_triggered_event_types(clean_content, trigger_index)
    logger.debug("Triggered %d event types: %s", len(triggered), list(triggered.keys()))

    if not triggered:
        logger.debug("No event triggers for doc %s — marking ok (no events)", doc_version_id[:12])
        db.upsert_doc_stage_status(
            doc_version_id, STAGE_NAME, run_id, config_hash, "ok",
            details=json.dumps({"events_promoted": 0, "metrics_extracted": 0, "metrics_deduped": 0}),
        )
        return

    promoted_count = 0

    for event_type_name, matched_kws in triggered.items():
        logger.info(
            "Processing event type '%s' for doc %s (triggered by: %s)",
            event_type_name, doc_version_id[:12], ', '.join(matched_kws)
        )

        event_type_def = extraction.event_types.get(event_type_name)
        if event_type_def is None:
            logger.warning("Event type '%s' not found in config", event_type_name)
            continue

        filled_slots, confidence = fill_event_slots(
            event_type_def, mentions, mention_links, entity_map, clean_content, doc_version_id,
        )
        logger.debug(
            "Filled %d slots with confidence %.3f for event type '%s' (threshold=%.3f)",
            len(filled_slots), confidence, event_type_name, event_type_def.confidence_threshold,
        )

        # Log filled slots details
        for i, fs in enumerate(filled_slots):
            logger.debug(
                "  Slot %d: name=%s, value=%s, span=(%s-%s), mention_id=%s, conf=%.3f",
                i, fs.name, str(fs.value)[:50], fs.span_start, fs.span_end,
                fs.mention_id[:12] if fs.mention_id else "None", fs.confidence,
            )

        slots_dict = {s.name: s.value for s in filled_slots}
        candidate_id = compute_sha256_id(doc_version_id, event_type_name, json.dumps(slots_dict, sort_keys=True))

        if confidence < event_type_def.confidence_threshold:
            logger.debug(
                "Rejecting candidate (confidence %.3f < threshold %.3f)",
                confidence, event_type_def.confidence_threshold
            )
            db.insert_event_candidate(
                EventCandidateRow(
                    candidate_id=candidate_id,
                    doc_version_id=doc_version_id,
                    event_type=event_type_name,
                    partial_slots=slots_dict,
                    confidence=confidence,
                    status="rejected",
                    rejection_reason=f"below_threshold:{confidence:.3f}<{event_type_def.confidence_threshold}",
                    extraction_tier=1,
                    created_in_run_id=run_id,
                )
            )
            continue

        logger.info(
            "Promoting candidate to event (confidence %.3f >= threshold %.3f)",
            confidence, event_type_def.confidence_threshold
        )

        db.insert_event_candidate(
            EventCandidateRow(
                candidate_id=candidate_id,
                doc_version_id=doc_version_id,
                event_type=event_type_name,
                partial_slots=slots_dict,
                confidence=confidence,
                status="promoted",
                extraction_tier=1,
                created_in_run_id=run_id,
            )
        )

        canonical_key = compute_canonical_key(
            event_type_name, event_type_def.canonical_key_template, slots_dict,
        )
        logger.debug("Computed canonical_key: %s", canonical_key)

        event_id = compute_sha256_id(event_type_name, canonical_key)
        logger.debug("Computed event_id: %s", event_id[:12])

        existing_event = db.get_event_by_canonical_key(event_type_name, canonical_key)
        if existing_event:
            logger.debug(
                "Found existing event: event_id=%s (will create new revision)",
                existing_event.event_id[:12]
            )

        # Collect evidence spans from filled slots.
        # We track (EvidenceSpanRow, join_purpose) tuples because the stored
        # ev_span.purpose may differ from Stage 8's intended role (e.g. an
        # existing span created by Stage 2 will have purpose="chunk").
        evidence_links: list[tuple[EvidenceSpanRow, str]] = []
        logger.debug("Collecting evidence spans from %d filled slots", len(filled_slots))

        for i, fs in enumerate(filled_slots):
            if fs.span_start is not None and fs.span_end is not None:
                join_purpose = f"event_slot:{fs.name}"
                logger.debug(
                    "  Creating evidence span %d: slot=%s, span=(%d-%d)",
                    i, fs.name, fs.span_start, fs.span_end
                )
                ev_span = db.get_or_create_evidence_span(
                    doc_version_id, fs.span_start, fs.span_end,
                    run_id, purpose=join_purpose, clean_content=clean_content,
                )
                if ev_span.purpose != join_purpose:
                    logger.debug(
                        "    → evidence_id=%s, stored_purpose=%s (using join_purpose=%s)",
                        ev_span.evidence_id[:12], ev_span.purpose, join_purpose
                    )
                else:
                    logger.debug(
                        "    → evidence_id=%s, purpose=%s",
                        ev_span.evidence_id[:12], ev_span.purpose
                    )
                evidence_links.append((ev_span, join_purpose))
            else:
                logger.debug("  Skipping slot %d (no span): name=%s", i, fs.name)

        # Fallback: create evidence from document start if no evidence collected
        if not evidence_links:
            logger.debug(
                "No evidence spans collected, creating fallback context (%d chars)",
                FALLBACK_EVIDENCE_SPAN_CHARS,
            )
            first_line_end = min(FALLBACK_EVIDENCE_SPAN_CHARS, len(clean_content))
            if first_line_end > 0:
                ev_span = db.get_or_create_evidence_span(
                    doc_version_id, 0, first_line_end,
                    run_id, purpose="event_fallback_context", clean_content=clean_content,
                )
                logger.debug(
                    "  Created fallback evidence: evidence_id=%s, span=(0-%d)",
                    ev_span.evidence_id[:12], first_line_end
                )
                evidence_links.append((ev_span, "event_fallback_context"))

        if not evidence_links:
            logger.warning(
                "Cannot create evidence for event in doc %s — skipping",
                doc_version_id[:12]
            )
            continue

        logger.info("Collected %d evidence spans for event", len(evidence_links))

        # Deduplicate by (evidence_id, join_purpose) — the DB unique key
        # components within a single revision. First occurrence wins.
        seen_keys: set[tuple[str, str]] = set()
        deduped_evidence: list[tuple[str, str]] = []
        for ev_span, join_purpose in evidence_links:
            key = (ev_span.evidence_id, join_purpose)
            if key in seen_keys:
                logger.warning(
                    "Dropping duplicate evidence link: evidence_id=%s, purpose=%s",
                    ev_span.evidence_id[:12], join_purpose,
                )
                continue
            seen_keys.add(key)
            deduped_evidence.append(key)

        # Create or update event
        if existing_event is None:
            logger.info("Creating new event: event_id=%s", event_id[:12])
            db.insert_event(
                EventRow(
                    event_id=event_id,
                    event_type=event_type_name,
                    canonical_key=canonical_key,
                    current_revision_id=None,
                    created_in_run_id=run_id,
                )
            )
            prev_rev_no = 0
            supersedes = None
        else:
            event_id = existing_event.event_id
            prev_rev_no = db.get_max_revision_no(event_id)
            supersedes = existing_event.current_revision_id
            logger.debug(
                "Using existing event_id=%s, prev_rev_no=%d, supersedes=%s",
                event_id[:12], prev_rev_no,
                supersedes[:12] if supersedes else "None"
            )

        new_rev_no = prev_rev_no + 1
        logger.info("Creating revision %d for event %s", new_rev_no, event_id[:12])

        # Determine document IDs for revision
        if existing_event and existing_event.current_revision_id:
            prev_rev = db.get_current_revision(event_id)
            if prev_rev:
                doc_ids = list(set(prev_rev.doc_version_ids + [doc_version_id]))
                logger.debug(
                    "Merging doc_version_ids: previous=%d, total=%d",
                    len(prev_rev.doc_version_ids), len(doc_ids)
                )
            else:
                doc_ids = [doc_version_id]
        else:
            doc_ids = [doc_version_id]

        revision_id = compute_sha256_id(event_id, str(new_rev_no))
        logger.debug("Computed revision_id: %s", revision_id[:12])

        # Insert event revision
        db.insert_event_revision(
            EventRevisionRow(
                revision_id=revision_id,
                event_id=event_id,
                revision_no=new_rev_no,
                slots_json=slots_dict,
                doc_version_ids=doc_ids,
                confidence=confidence,
                extraction_method="rule_based_tier1",
                extraction_tier=1,
                supersedes_revision_id=supersedes,
                created_in_run_id=run_id,
            )
        )

        # Insert evidence for revision
        logger.info(
            "Inserting %d evidence entries for revision %s (from %d raw links)",
            len(deduped_evidence), revision_id[:12], len(evidence_links)
        )

        for i, (evidence_id_val, join_purpose) in enumerate(deduped_evidence):
            logger.debug(
                "  Inserting evidence %d/%d: revision_id=%s, evidence_id=%s, purpose=%s",
                i + 1, len(deduped_evidence),
                revision_id[:12], evidence_id_val[:12], join_purpose
            )
            try:
                db.insert_event_revision_evidence(
                    EventRevisionEvidenceRow(
                        revision_id=revision_id,
                        evidence_id=evidence_id_val,
                        purpose=join_purpose,
                    )
                )
            except Exception as e:
                logger.error(
                    "Failed to insert evidence %d/%d: %s",
                    i + 1, len(deduped_evidence), str(e)
                )
                logger.error(
                    "  Failed parameters: revision_id=%s, evidence_id=%s, purpose=%s",
                    revision_id[:12], evidence_id_val[:12], join_purpose
                )
                raise

        # Update event's current revision pointer
        db.update_event_current_revision(event_id, revision_id)

        # Link entities to event revision
        entity_link_count = 0
        for fs in filled_slots:
            if fs.mention_id is None:
                continue
            for ml in mention_links:
                if ml.mention_id == fs.mention_id and ml.entity_id:
                    db.insert_event_entity_link(
                        EventEntityLinkRow(
                            revision_id=revision_id,
                            entity_id=ml.entity_id,
                            role=fs.name,
                            confidence=fs.confidence,
                            created_in_run_id=run_id,
                        )
                    )
                    entity_link_count += 1
                    break

        logger.debug("Created %d entity links for revision", entity_link_count)
        promoted_count += 1
        logger.info(
            "Successfully promoted event: type=%s, revision=%d, evidence_count=%d, entity_links=%d",
            event_type_name, new_rev_no, len(deduped_evidence), entity_link_count,
        )

    # --- Metric observations ------------------------------------------------
    logger.debug("Extracting metric observations")
    metric_obs_raw = extract_metric_observations(
        mentions, table_extracts, doc_version_id, clean_content,
        validation_ranges=quantity_validation_ranges or {},
        unit_category_map=unit_category_map or {},
        mention_surfacing_threshold=mention_surfacing_threshold,
        slot_proximity_chars=slot_proximity_chars,
    )
    logger.debug("Extracted %d raw metric observations", len(metric_obs_raw))

    metric_obs = deduplicate_metric_observations(metric_obs_raw, doc_version_id)
    metrics_deduped = len(metric_obs_raw) - len(metric_obs)
    logger.debug(
        "Deduplicated metrics: kept %d, removed %d duplicates",
        len(metric_obs), metrics_deduped
    )

    for i, obs in enumerate(metric_obs):
        evidence_id: str | None = None
        if obs.span_start is not None and obs.span_end is not None:
            ev_span = db.get_or_create_evidence_span(
                doc_version_id, obs.span_start, obs.span_end,
                run_id, purpose="metric_observation", clean_content=clean_content,
            )
            evidence_id = ev_span.evidence_id

        # metric_id derived from the natural key — aligned with DB uniqueness
        nk = metric_natural_key(doc_version_id, obs)
        metric_id = compute_sha256_id(*[str(c) for c in nk])

        logger.debug(
            "Inserting metric %d/%d: name=%s, value=%s, unit=%s, evidence=%s",
            i + 1, len(metric_obs), obs.metric_name, obs.value_raw,
            obs.unit_raw or "none", "yes" if evidence_id else "no",
        )

        db.insert_metric_observation(
            MetricObservationRow(
                metric_id=metric_id,
                doc_version_id=doc_version_id,
                table_id=obs.table_id,
                metric_name=obs.metric_name,
                value_raw=obs.value_raw,
                unit_raw=obs.unit_raw,
                value_norm=obs.value_norm,
                unit_norm=obs.unit_norm,
                period_start=obs.period_start,
                period_end=obs.period_end,
                period_granularity=obs.period_granularity,
                geography=obs.geography,
                evidence_id=evidence_id,
                table_row_index=obs.table_row_index,
                table_col_index=obs.table_col_index,
                parse_quality=METRIC_OBS_DEFAULT_PARSE_QUALITY if evidence_id else None,
                created_in_run_id=run_id,
            )
        )

    logger.info(
        "Document processing complete: promoted_events=%d, metrics=%d (deduped=%d)",
        promoted_count, len(metric_obs), metrics_deduped
    )

    db.upsert_doc_stage_status(
        doc_version_id, STAGE_NAME, run_id, config_hash, "ok",
        details=json.dumps({
            "events_promoted": promoted_count,
            "metrics_extracted": len(metric_obs),
            "metrics_deduped": metrics_deduped,
        }),
    )

# Stage runner
def run_stage(
    db: Stage08DatabaseInterface,
    run_id: str,
    config: Config,
    config_hash: str,
) -> int:
    """
    Execute Stage 08 over the full iteration set.

    :return: Exit code (0 = success, 1 = fatal).
    """
    extraction = config.extraction

    # --- Build indexes from config ---
    trigger_index = _build_trigger_index(extraction)
    logger.info("Built trigger index with %d keywords", len(trigger_index))

    unit_category_map = _build_unit_category_map(extraction)

    pipeline_thresholds = _get_pipeline_thresholds(extraction)
    context_windows = _get_context_windows(extraction)

    mention_surfacing_threshold = pipeline_thresholds.get("mention_surfacing", 0.70)
    slot_proximity_chars = context_windows.get("event_slot_proximity", 500)

    logger.info(
        "Effective settings: mention_surfacing_threshold=%.2f, slot_proximity_chars=%d",
        mention_surfacing_threshold, slot_proximity_chars,
    )

    entity_rows = db.list_entity_registry()
    entity_map = {e.entity_id: e for e in entity_rows}
    logger.info("Loaded %d entities from registry", len(entity_map))

    # Extract quantity validation ranges from global config
    quantity_validation_ranges: dict[str, Any] = {}
    try:
        qv = config.global_settings.validation.quantity_validation
        # Convert QuantityValidationRange objects to dicts for the helper
        quantity_validation_ranges = {
            k: {"min": v.min, "max": v.max} if hasattr(v, "min") else v
            for k, v in qv.items()
        }
        logger.info(
            "Loaded quantity validation ranges for %d categories: %s",
            len(quantity_validation_ranges), list(quantity_validation_ranges.keys()),
        )
    except (AttributeError, KeyError) as e:
        logger.warning("Could not load quantity validation ranges from config: %s", e)

    # --- Log module-level defaults for transparency ---
    logger.info(
        "Module-level defaults: fill_ratio_weight=%.2f, confidence_weight=%.2f, "
        "extraction_method_conf=%.2f, extraction_method_window=%d, "
        "extraction_method_max_len=%d, fallback_evidence_chars=%d, "
        "metric_parse_quality=%.2f",
        FILL_RATIO_WEIGHT, CONFIDENCE_WEIGHT,
        EXTRACTION_METHOD_DEFAULT_CONFIDENCE, EXTRACTION_METHOD_TEXT_WINDOW,
        EXTRACTION_METHOD_MAX_VALUE_LENGTH, FALLBACK_EVIDENCE_SPAN_CHARS,
        METRIC_OBS_DEFAULT_PARSE_QUALITY,
    )

    # --- Log event type thresholds from config ---
    for etype_name, etype_def in extraction.event_types.items():
        logger.info(
            "Event type '%s': confidence_threshold=%.2f, priority=%d, "
            "required_slots=%d, optional_slots=%d",
            etype_name, etype_def.confidence_threshold, etype_def.priority,
            len(etype_def.required_slots), len(etype_def.optional_slots),
        )

    iteration_set = db.get_iteration_set(run_id)
    logger.info("Stage 08 iteration set: %d documents", len(iteration_set))

    if not iteration_set:
        logger.info("No documents to process — stage complete")
        return 0

    ok_count = 0
    failed_count = 0
    blocked_count = 0
    skipped_count = 0
    attempted_count = 0

    for doc_idx, (publisher_id, url_normalized, doc_version_id) in enumerate(iteration_set):
        logger.info(
            "Processing iteration item %d/%d: publisher=%s, url=%s, doc=%s",
            doc_idx + 1, len(iteration_set),
            publisher_id, url_normalized[:80], doc_version_id[:12],
        )

        prereqs_ok, err_msg = db.check_prerequisites_ok(doc_version_id)

        if not prereqs_ok:
            logger.debug("Prerequisites not met: %s", err_msg)
            existing = db.get_doc_stage_status(doc_version_id, STAGE_NAME)
            if existing and existing.status == "blocked" and existing.error_message == err_msg:
                blocked_count += 1
                continue
            with db.transaction():
                db.upsert_doc_stage_status(
                    doc_version_id, STAGE_NAME, run_id, config_hash,
                    "blocked", error_message=err_msg,
                )
            blocked_count += 1
            continue

        attempted_count += 1
        try:
            with db.transaction():
                process_single_document(
                    db, doc_version_id, run_id, config_hash,
                    extraction, trigger_index, entity_map,
                    quantity_validation_ranges=quantity_validation_ranges,
                    unit_category_map=unit_category_map,
                    mention_surfacing_threshold=mention_surfacing_threshold,
                    slot_proximity_chars=slot_proximity_chars,
                )
            ok_count += 1
            logger.info("Successfully processed doc %s", doc_version_id[:12])
        except Exception:
            logger.exception("Failed processing doc %s", doc_version_id[:12])
            try:
                with db.transaction():
                    db.upsert_doc_stage_status(
                        doc_version_id, STAGE_NAME, run_id, config_hash,
                        "failed", error_message=f"exception:{doc_version_id}",
                    )
            except Exception:
                logger.exception("Failed to write failure status for %s", doc_version_id[:12])
            failed_count += 1

    logger.info(
        "Stage 08 complete: ok=%d failed=%d blocked=%d skipped=%d attempted=%d total=%d",
        ok_count, failed_count, blocked_count, skipped_count, attempted_count, len(iteration_set),
    )

    if attempted_count > 0 and failed_count == attempted_count and ok_count == 0:
        logger.error("Systemic failure: all %d attempted documents failed", attempted_count)
        return 1

    return 0


# CLI entry point
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Stage 08: Event Extraction")
    parser.add_argument("--run-id", type=str, default="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", help="Pipeline run ID")
    parser.add_argument("--config-dir", type=Path, default=Path("../../../config/"))
    parser.add_argument("--source-db", type=Path, default=Path("../../../database/preprocessed_posts.db"))
    parser.add_argument("--working-db", type=Path, default=Path("../../../database/processed_posts.db"))
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose (DEBUG) logging")
    return parser.parse_args()


def main_stage_08_events() -> int:
    """
    Set main entry point for Stage 08.

    :return: 0 on success, 1 on fatal error.
    """
    args = parse_args()

    logger.info("Stage 08 starting: working_db=%s, config_dir=%s", args.working_db, args.config_dir)

    config_path = args.config_dir / "config.yaml"
    config = load_config(config_path)
    config_hash = get_config_version(config)
    logger.info("Loaded config: hash=%s", config_hash)

    db = Stage08DatabaseInterface(working_db_path=args.working_db)
    try:
        db.open()
        logger.info("Database opened: %s", args.working_db)

        run_row = db.get_pipeline_run(args.run_id)
        if run_row is None:
            logger.error("Pipeline run %s not found", args.run_id)
            return 1
        if run_row.status != "running":
            logger.error("Pipeline run %s is not running (status=%s)", args.run_id, run_row.status)
            return 1
        logger.info("Pipeline run verified: run_id=%s, status=%s", args.run_id[:12], run_row.status)

        return run_stage(db, args.run_id, config, config_hash)
    except Exception:
        logger.exception("Fatal error in Stage 08")
        return 1
    finally:
        db.close()
        logger.info("Database closed")


if __name__ == "__main__":
    sys.exit(main_stage_08_events())