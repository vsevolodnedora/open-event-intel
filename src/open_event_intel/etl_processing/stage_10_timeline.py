"""
Stage 10 — Entity Timeline Materialization.

Run-scoped stage that turns eligible events and entity mentions into
queryable, evidence-linked ``entity_timeline_item`` rows grouped by entity.

**Reads:** event, event_revision, event_revision_evidence, event_entity_link,
    mention, mention_link, novelty_label, entity_registry, document,
    document_version, evidence_span, pipeline_run, run_stage_status,
    doc_stage_status.
**Writes (run-scoped):** entity_timeline_item, entity_timeline_item_evidence,
    evidence_span, run_stage_status.
**Prerequisite:** ``run_stage_status(stage_09_outputs) = 'ok'``.
**Deletion order (FK-safe):**
    ``entity_timeline_item_evidence`` → ``entity_timeline_item``.
"""
import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar, Sequence

from config_interface import Config, get_config_version, load_config
from database_interface import (
    DatabaseInterface,
    DBError,
    EntityRegistryRow,
    EntityTimelineItemEvidenceRow,
    EntityTimelineItemRow,
    EventEntityLinkRow,
    EventRevisionEvidenceRow,
    EventRevisionRow,
    EventRow,
    EvidenceSpanRow,
    MentionLinkRow,
    MentionRow,
    RunStageStatusRow,
    compute_sha256_id,
)
from pydantic import BaseModel, ConfigDict

from open_event_intel.logger import get_logger

logger = get_logger(__name__)

# Module-level constants — values that are structural to the pipeline and not
# expected to change via config.  Collected here so they are never silently
# buried inside function bodies.

STAGE_NAME = "stage_10_timeline"
PREREQUISITE_RUN_STAGE = "stage_09_outputs"
PREREQUISITE_DOC_STAGE = "stage_08_events"

# Item type identifiers written to entity_timeline_item.item_type
ITEM_TYPE_EVENT_REVISION = "event_revision"
ITEM_TYPE_MENTION = "mention"

# Evidence purpose strings written to entity_timeline_item_evidence.purpose
EVIDENCE_PURPOSE_SUMMARY_QUOTE = "summary_quote"
EVIDENCE_PURPOSE_TIMELINE_MENTION = "timeline_mention"

# Preferred purposes when selecting evidence from event revisions (ordered)
PREFERRED_EVIDENCE_PURPOSES: tuple[str, ...] = (
    "summary",
    "justification",
    "slot:effective_date",
)

# Slot keys inspected for deriving event time
EVENT_TIME_SLOT_KEYS: tuple[tuple[str, str], ...] = (
    ("effective_date", "slot_effective_date"),
    ("deadline", "slot_deadline"),
)
EVENT_TIME_FALLBACK_SOURCE = "doc_published_at"

# Slot keys used for building event summaries
EVENT_SUMMARY_SLOT_KEYS: tuple[str, ...] = ("title", "summary")


class _EligibleEvent(BaseModel):
    """An eligible event together with its current revision."""

    model_config = ConfigDict(frozen=True)

    event: EventRow
    revision: EventRevisionRow


class _LinkedMention(BaseModel):
    """A mention linked to an entity from an eligible document."""

    model_config = ConfigDict(frozen=True)

    mention: MentionRow
    link: MentionLinkRow
    entity_id: str


class Stage10DatabaseInterface(DatabaseInterface):
    """
    Database adapter for Stage 10 — timeline materialization.

    .. note::
       Query methods defined here should eventually migrate into
       ``database_interface.py`` once the interface stabilises.
    """

    READS: ClassVar[set[str]] = {
        "pipeline_run",
        "run_stage_status",
        "doc_stage_status",
        "event",
        "event_revision",
        "event_revision_evidence",
        "event_entity_link",
        "mention",
        "mention_link",
        "novelty_label",
        "entity_registry",
        "document",
        "document_version",
        "evidence_span",
    }
    WRITES: ClassVar[set[str]] = {
        "entity_timeline_item",
        "entity_timeline_item_evidence",
        "evidence_span",
        "run_stage_status",
    }

    def __init__(
        self,
        working_db_path: Path,
        source_db_path: Path | None = None,
    ) -> None:
        """Initialize a Stage 10 timeline materialization."""
        super().__init__(working_db_path, source_db_path, STAGE_NAME)

    def get_eligible_doc_version_ids(self) -> frozenset[str]:
        """
        Return doc_version_ids where the prerequisite doc stage is 'ok'.

        Per §6.3.3 if the prerequisite doc stage is 'ok' all transitive
        per-doc prerequisites are necessarily 'ok' as well.
        """
        self._check_read_access("doc_stage_status")
        rows = self._fetchall(
            "SELECT doc_version_id FROM doc_stage_status "
            "WHERE stage = ? AND status = 'ok'",
            (PREREQUISITE_DOC_STAGE,),
        )
        return frozenset(r["doc_version_id"] for r in rows)

    def get_eligible_events(
        self, eligible_doc_ids: frozenset[str]
    ) -> list[_EligibleEvent]:
        """
        Return events whose current revision's docs are all eligible.

        Enforces:
        * ``event.current_revision_id IS NOT NULL``
        * Every ``doc_version_id`` in ``event_revision.doc_version_ids``
          is a member of *eligible_doc_ids*.

        Results are sorted by ``(event_type, canonical_key)`` for
        deterministic iteration.
        """
        self._check_read_access("event")
        self._check_read_access("event_revision")
        rows = self._fetchall(
            """
            SELECT
                e.event_id, e.event_type, e.canonical_key,
                e.current_revision_id, e.created_in_run_id,
                e.created_at AS e_created_at,
                er.revision_id, er.event_id AS er_event_id,
                er.revision_no, er.slots_json,
                er.doc_version_ids, er.confidence,
                er.extraction_method, er.extraction_tier,
                er.supersedes_revision_id, er.created_in_run_id AS er_created_in_run_id,
                er.created_at AS er_created_at
            FROM event e
            JOIN event_revision er ON er.revision_id = e.current_revision_id
            WHERE e.current_revision_id IS NOT NULL
            ORDER BY e.event_type, e.canonical_key
            """
        )
        result: list[_EligibleEvent] = []
        skipped_ineligible_docs = 0
        for r in rows:
            ev_row = EventRow(
                event_id=r["event_id"],
                event_type=r["event_type"],
                canonical_key=r["canonical_key"],
                current_revision_id=r["current_revision_id"],
                created_in_run_id=r["created_in_run_id"],
                created_at=r["e_created_at"],
            )
            rev_row = EventRevisionRow(
                revision_id=r["revision_id"],
                event_id=r["er_event_id"],
                revision_no=r["revision_no"],
                slots_json=r["slots_json"],
                doc_version_ids=r["doc_version_ids"],
                confidence=r["confidence"],
                extraction_method=r["extraction_method"],
                extraction_tier=r["extraction_tier"],
                supersedes_revision_id=r["supersedes_revision_id"],
                created_in_run_id=r["er_created_in_run_id"],
                created_at=r["er_created_at"],
            )
            if all(dvid in eligible_doc_ids for dvid in rev_row.doc_version_ids):
                result.append(_EligibleEvent(event=ev_row, revision=rev_row))
            else:
                skipped_ineligible_docs += 1
                logger.debug(
                    "Skipping event=%s: not all doc_version_ids eligible "
                    "(revision=%s, docs=%s)",
                    ev_row.event_id,
                    rev_row.revision_id,
                    rev_row.doc_version_ids,
                )

        if skipped_ineligible_docs:
            logger.info(
                "Skipped %d events due to ineligible doc_version_ids",
                skipped_ineligible_docs,
            )
        return result

    def get_event_entity_links(
        self, revision_id: str
    ) -> list[EventEntityLinkRow]:
        """Return entity links for a specific revision, ordered by entity_id."""
        self._check_read_access("event_entity_link")
        rows = self._fetchall(
            "SELECT * FROM event_entity_link "
            "WHERE revision_id = ? ORDER BY entity_id, role",
            (revision_id,),
        )
        return [EventEntityLinkRow.model_validate(dict(r)) for r in rows]

    def get_event_revision_evidence(
        self, revision_id: str
    ) -> list[EventRevisionEvidenceRow]:
        """Return evidence rows for an event revision."""
        self._check_read_access("event_revision_evidence")
        rows = self._fetchall(
            "SELECT * FROM event_revision_evidence "
            "WHERE revision_id = ? ORDER BY evidence_id",
            (revision_id,),
        )
        return [EventRevisionEvidenceRow.model_validate(dict(r)) for r in rows]

    def get_linked_mentions_for_docs(
        self, eligible_doc_ids: frozenset[str]
    ) -> list[_LinkedMention]:
        """
        Return mentions linked to entities from eligible documents.

        Only includes links where ``mention_link.entity_id IS NOT NULL``.
        Results are sorted by ``(entity_id, doc_version_id, span_start)``
        for deterministic output.
        """
        if not eligible_doc_ids:
            return []
        self._check_read_access("mention")
        self._check_read_access("mention_link")
        rows = self._fetchall(
            """
            SELECT
                m.mention_id, m.doc_version_id, m.chunk_ids,
                m.mention_type, m.surface_form, m.normalized_value,
                m.span_start, m.span_end, m.confidence,
                m.extraction_method, m.context_window_start,
                m.context_window_end, m.rejection_reason,
                m.metadata AS m_metadata,
                m.created_in_run_id AS m_created_in_run_id,
                m.created_at AS m_created_at,
                ml.link_id, ml.mention_id AS ml_mention_id,
                ml.entity_id, ml.link_confidence,
                ml.link_method,
                ml.created_in_run_id AS ml_created_in_run_id,
                ml.created_at AS ml_created_at
            FROM mention m
            JOIN mention_link ml ON ml.mention_id = m.mention_id
            WHERE ml.entity_id IS NOT NULL
            ORDER BY ml.entity_id, m.doc_version_id, m.span_start
            """
        )
        result: list[_LinkedMention] = []
        skipped_ineligible = 0
        for r in rows:
            if r["doc_version_id"] not in eligible_doc_ids:
                skipped_ineligible += 1
                continue
            mention = MentionRow(
                mention_id=r["mention_id"],
                doc_version_id=r["doc_version_id"],
                chunk_ids=r["chunk_ids"],
                mention_type=r["mention_type"],
                surface_form=r["surface_form"],
                normalized_value=r["normalized_value"],
                span_start=r["span_start"],
                span_end=r["span_end"],
                confidence=r["confidence"],
                extraction_method=r["extraction_method"],
                context_window_start=r["context_window_start"],
                context_window_end=r["context_window_end"],
                rejection_reason=r["rejection_reason"],
                metadata=r["m_metadata"],
                created_in_run_id=r["m_created_in_run_id"],
                created_at=r["m_created_at"],
            )
            link = MentionLinkRow(
                link_id=r["link_id"],
                mention_id=r["ml_mention_id"],
                entity_id=r["entity_id"],
                link_confidence=r["link_confidence"],
                link_method=r["link_method"],
                created_in_run_id=r["ml_created_in_run_id"],
                created_at=r["ml_created_at"],
            )
            result.append(
                _LinkedMention(
                    mention=mention, link=link, entity_id=r["entity_id"]
                )
            )

        if skipped_ineligible:
            logger.debug(
                "Filtered out %d mention links from ineligible documents",
                skipped_ineligible,
            )
        return result

    def get_document_published_at(self, doc_version_id: str) -> datetime | None:
        """Return ``document.source_published_at`` for a doc_version_id."""
        self._check_read_access("document_version")
        self._check_read_access("document")
        row = self._fetchone(
            """
            SELECT d.source_published_at
            FROM document_version dv
            JOIN document d ON d.document_id = dv.document_id
            WHERE dv.doc_version_id = ?
            """,
            (doc_version_id,),
        )
        if row is None:
            return None
        raw = row["source_published_at"]
        if isinstance(raw, str):
            return datetime.fromisoformat(raw)
        return raw

    def get_entity_registry_map(self) -> dict[str, EntityRegistryRow]:
        """Return a mapping of entity_id → EntityRegistryRow."""
        entities = self.list_entity_registry()
        return {e.entity_id: e for e in entities}

    def get_evidence_span(self, evidence_id: str) -> EvidenceSpanRow | None:
        """Retrieve an evidence span by ID."""
        self._check_read_access("evidence_span")
        row = self._fetchone(
            "SELECT * FROM evidence_span WHERE evidence_id = ?", (evidence_id,)
        )
        return EvidenceSpanRow.model_validate(dict(row)) if row else None


# Date validation helper
def validate_event_time(
    event_time: datetime | None,
    time_source: str | None,
    config: Config,
) -> tuple[datetime | None, str | None, str | None]:
    """
    Validate an extracted event time against config date validation rules.

    :param event_time: The candidate timestamp.
    :param time_source: How the time was derived.
    :param config: Loaded pipeline config for validation bounds.
    :return: ``(event_time, time_source, warning)`` — *warning* is ``None``
        if the time passed validation, otherwise a human-readable reason.
    """
    if event_time is None:
        return None, None, None

    dv = config.global_settings.validation.date_validation
    year = event_time.year

    if year < dv.min_year:
        reason = (
            f"event_time year {year} < min_year {dv.min_year} "
            f"(source={time_source})"
        )
        return None, None, reason

    if year > dv.max_year:
        reason = (
            f"event_time year {year} > max_year {dv.max_year} "
            f"(source={time_source})"
        )
        return None, None, reason

    now = datetime.now()
    max_future = now + timedelta(
        days=dv.reject_future_effective_dates_beyond_years * 365
    )
    if event_time > max_future and time_source in (
        "slot_effective_date",
        "slot_deadline",
    ):
        reason = (
            f"event_time {event_time.isoformat()} is more than "
            f"{dv.reject_future_effective_dates_beyond_years} years in the "
            f"future (source={time_source})"
        )
        return None, None, reason

    return event_time, time_source, None


# Core helpers
def extract_event_time(
    slots_json: dict,
    doc_version_ids: list[str],
    db: Stage10DatabaseInterface,
) -> tuple[datetime | None, str | None]:
    """
    Derive the best event timestamp from slot data or document metadata.

    :param slots_json: Parsed ``event_revision.slots_json``.
    :param doc_version_ids: Documents contributing to the revision.
    :param db: Database handle for fallback document time lookups.
    :return: ``(event_time, time_source)`` or ``(None, None)``.
    """
    for key, source_label in EVENT_TIME_SLOT_KEYS:
        raw = slots_json.get(key)
        if raw is not None:
            try:
                parsed = datetime.fromisoformat(str(raw))
                logger.debug(
                    "Extracted event_time from slot '%s': %s",
                    key,
                    parsed.isoformat(),
                )
                return parsed, source_label
            except (ValueError, TypeError):
                logger.debug(
                    "Could not parse slot '%s' value '%s' as datetime",
                    key,
                    raw,
                )

    for dvid in sorted(doc_version_ids):
        pub = db.get_document_published_at(dvid)
        if pub is not None:
            logger.debug(
                "Falling back to doc_published_at for event_time: "
                "doc_version_id=%s, published_at=%s",
                dvid,
                pub.isoformat(),
            )
            return pub, EVENT_TIME_FALLBACK_SOURCE

    logger.debug(
        "No event_time derivable from slots or documents (docs=%s)",
        doc_version_ids,
    )
    return None, None


def compute_timeline_item_id(
    run_id: str, entity_id: str, item_type: str, ref_id: str
) -> str:
    """
    Deterministic item_id for a timeline entry.

    :param run_id: Pipeline run identifier.
    :param entity_id: Entity the item belongs to.
    :param item_type: ``event_revision`` or ``mention``.
    :param ref_id: ``revision_id`` or ``mention_id``.
    :return: 64-char lowercase hex SHA-256 digest.
    """
    return compute_sha256_id(run_id, entity_id, item_type, ref_id)


def build_event_revision_items(  # noqa: C901
    eligible_events: Sequence[_EligibleEvent],
    run_id: str,
    db: Stage10DatabaseInterface,
    entity_map: dict[str, EntityRegistryRow],
    config: Config,
) -> list[tuple[EntityTimelineItemRow, list[EntityTimelineItemEvidenceRow]]]:
    """
    Create timeline items from eligible event→entity links.

    For each eligible event, expands linked entities via
    ``event_entity_link`` and produces one timeline item per
    ``(entity, event)`` pair.

    :return: List of ``(item, [evidence_rows])`` pairs.
    """
    items: list[tuple[EntityTimelineItemRow, list[EntityTimelineItemEvidenceRow]]] = []
    seen: set[str] = set()
    skipped_no_links = 0
    skipped_not_in_registry = 0
    skipped_no_evidence = 0
    skipped_date_validation = 0
    entity_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()

    for ee in eligible_events:
        ev = ee.event
        rev = ee.revision
        entity_links = db.get_event_entity_links(rev.revision_id)
        if not entity_links:
            skipped_no_links += 1
            logger.debug(
                "Event %s (revision=%s): no entity links, skipping",
                ev.event_id,
                rev.revision_id,
            )
            continue

        rev_evidence = db.get_event_revision_evidence(rev.revision_id)
        raw_event_time, raw_time_source = extract_event_time(
            rev.slots_json, rev.doc_version_ids, db
        )

        # Validate event time against config date rules
        event_time, time_source, date_warning = validate_event_time(
            raw_event_time, raw_time_source, config
        )
        if date_warning:
            skipped_date_validation += 1
            logger.warning(
                "Date validation failed for event=%s: %s — "
                "clearing event_time",
                ev.event_id,
                date_warning,
            )

        summary = _build_event_summary(ev, rev)

        for el in entity_links:
            if el.entity_id not in entity_map:
                skipped_not_in_registry += 1
                logger.warning(
                    "Skipping entity_id=%s (not in registry) for event=%s",
                    el.entity_id,
                    ev.event_id,
                )
                continue

            item_id = compute_timeline_item_id(
                run_id, el.entity_id, ITEM_TYPE_EVENT_REVISION, rev.revision_id
            )
            if item_id in seen:
                logger.debug(
                    "Duplicate item_id=%s (entity=%s, event=%s), skipping",
                    item_id,
                    el.entity_id,
                    ev.event_id,
                )
                continue
            seen.add(item_id)

            context = {
                "event_type": ev.event_type,
                "canonical_key": ev.canonical_key,
                "role": el.role,
                "confidence": el.confidence,
                "revision_no": rev.revision_no,
            }

            item = EntityTimelineItemRow(
                item_id=item_id,
                run_id=run_id,
                entity_id=el.entity_id,
                item_type=ITEM_TYPE_EVENT_REVISION,
                category=ev.event_type,
                ref_revision_id=rev.revision_id,
                ref_mention_id=None,
                ref_doc_version_id=None,
                event_time=event_time,
                time_source=time_source,
                summary_text=summary,
                context_json=context,
            )

            evidence_rows = _pick_evidence_for_revision(
                item_id, rev_evidence
            )
            if not evidence_rows:
                skipped_no_evidence += 1
                logger.warning(
                    "No evidence available for timeline item %s "
                    "(event=%s, entity=%s); skipping.",
                    item_id,
                    ev.event_id,
                    el.entity_id,
                )
                continue

            items.append((item, evidence_rows))
            entity_counter[el.entity_id] += 1
            category_counter[ev.event_type] += 1

            logger.debug(
                "Built event-revision item: item_id=%s, entity=%s, "
                "event_type=%s, event_time=%s",
                item_id,
                el.entity_id,
                ev.event_type,
                event_time.isoformat() if event_time else "N/A",
            )

    # Summary logging
    if skipped_no_links:
        logger.info(
            "Event-revision items: skipped %d events with no entity links",
            skipped_no_links,
        )
    if skipped_not_in_registry:
        logger.info(
            "Event-revision items: skipped %d entity links not in registry",
            skipped_not_in_registry,
        )
    if skipped_no_evidence:
        logger.info(
            "Event-revision items: skipped %d items with no evidence",
            skipped_no_evidence,
        )
    if skipped_date_validation:
        logger.info(
            "Event-revision items: cleared event_time for %d items "
            "(date validation failure)",
            skipped_date_validation,
        )
    if category_counter:
        logger.info(
            "Event-revision items by category: %s",
            dict(category_counter.most_common()),
        )
    if entity_counter:
        logger.info(
            "Event-revision items span %d distinct entities (top-5: %s)",
            len(entity_counter),
            dict(entity_counter.most_common(5)),
        )

    return items


def build_mention_items(
    linked_mentions: Sequence[_LinkedMention],
    run_id: str,
    db: Stage10DatabaseInterface,
    entity_map: dict[str, EntityRegistryRow],
) -> list[tuple[EntityTimelineItemRow, list[EntityTimelineItemEvidenceRow]]]:
    """
    Create timeline items from entity mentions in eligible documents.

    Each linked mention produces one timeline item referencing the
    mention directly.

    :return: List of ``(item, [evidence_rows])`` pairs.
    """
    items: list[tuple[EntityTimelineItemRow, list[EntityTimelineItemEvidenceRow]]] = []
    seen: set[str] = set()
    skipped_not_in_registry = 0
    skipped_duplicate = 0
    entity_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()

    for lm in linked_mentions:
        m = lm.mention
        entity_id = lm.entity_id

        if entity_id not in entity_map:
            skipped_not_in_registry += 1
            logger.debug(
                "Mention %s: entity_id=%s not in registry, skipping",
                m.mention_id,
                entity_id,
            )
            continue

        item_id = compute_timeline_item_id(
            run_id, entity_id, ITEM_TYPE_MENTION, m.mention_id
        )
        if item_id in seen:
            skipped_duplicate += 1
            continue
        seen.add(item_id)

        pub_time = db.get_document_published_at(m.doc_version_id)
        time_source = EVENT_TIME_FALLBACK_SOURCE if pub_time else None

        summary = (
            f"Mentioned as '{m.surface_form}' "
            f"({m.mention_type})"
        )
        category = f"{m.mention_type}"

        context = {
            "surface_form": m.surface_form,
            "link_confidence": lm.link.link_confidence,
            "link_method": lm.link.link_method,
        }

        evidence_id = compute_sha256_id(
            m.doc_version_id, m.span_start, m.span_end
        )

        item = EntityTimelineItemRow(
            item_id=item_id,
            run_id=run_id,
            entity_id=entity_id,
            item_type=ITEM_TYPE_MENTION,
            category=category,
            ref_revision_id=None,
            ref_mention_id=m.mention_id,
            ref_doc_version_id=None,
            event_time=pub_time,
            time_source=time_source,
            summary_text=summary,
            context_json=context,
        )

        ev_row = EntityTimelineItemEvidenceRow(
            item_id=item_id,
            evidence_id=evidence_id,
            purpose=EVIDENCE_PURPOSE_SUMMARY_QUOTE,
        )
        items.append((item, [ev_row]))
        entity_counter[entity_id] += 1
        category_counter[category] += 1

        logger.debug(
            "Built mention item: item_id=%s, entity=%s, "
            "mention_type=%s, doc=%s",
            item_id,
            entity_id,
            m.mention_type,
            m.doc_version_id,
        )

    # Summary logging
    if skipped_not_in_registry:
        logger.info(
            "Mention items: skipped %d mentions with entity not in registry",
            skipped_not_in_registry,
        )
    if skipped_duplicate:
        logger.info(
            "Mention items: skipped %d duplicate item_ids",
            skipped_duplicate,
        )
    if category_counter:
        logger.info(
            "Mention items by category: %s",
            dict(category_counter.most_common()),
        )
    if entity_counter:
        logger.info(
            "Mention items span %d distinct entities (top-5: %s)",
            len(entity_counter),
            dict(entity_counter.most_common(5)),
        )

    return items


def _build_event_summary(ev: EventRow, rev: EventRevisionRow) -> str:
    """Produce a short human-readable summary for an event timeline item."""
    slots = rev.slots_json
    title = None
    for key in EVENT_SUMMARY_SLOT_KEYS:
        title = slots.get(key)
        if title:
            break
    if title:
        return f"[{ev.event_type}] {title}"
    return f"[{ev.event_type}] {ev.canonical_key}"


def _pick_evidence_for_revision(
    item_id: str,
    rev_evidence: Sequence[EventRevisionEvidenceRow],
) -> list[EntityTimelineItemEvidenceRow]:
    """
    Map event-revision evidence to timeline-item evidence rows.

    Picks the first evidence span with a summary-like purpose, falling
    back to the first available span.
    """
    if not rev_evidence:
        return []

    chosen: EventRevisionEvidenceRow | None = None
    for er in rev_evidence:
        if er.purpose in PREFERRED_EVIDENCE_PURPOSES:
            chosen = er
            logger.debug(
                "Selected preferred evidence (purpose=%s) for item=%s",
                er.purpose,
                item_id,
            )
            break
    if chosen is None:
        chosen = rev_evidence[0]
        logger.debug(
            "Fell back to first evidence (purpose=%s) for item=%s",
            chosen.purpose,
            item_id,
        )

    return [
        EntityTimelineItemEvidenceRow(
            item_id=item_id,
            evidence_id=chosen.evidence_id,
            purpose=EVIDENCE_PURPOSE_SUMMARY_QUOTE,
        )
    ]


def ensure_mention_evidence_spans(
    items: Sequence[tuple[EntityTimelineItemRow, list[EntityTimelineItemEvidenceRow]]],
    db: Stage10DatabaseInterface,
    run_id: str,
) -> None:
    """
    Get-or-create evidence spans for mention-sourced timeline items.

    Event-revision items reuse existing evidence_span rows already
    created by Stage 8.  Mention items may reference spans that are not
    yet in ``evidence_span``; this helper ensures they exist.
    """
    created_count = 0
    reused_count = 0

    for item, ev_rows in items:
        if item.item_type != ITEM_TYPE_MENTION or item.ref_mention_id is None:
            continue
        for ev in ev_rows:
            existing = db.get_evidence_span(ev.evidence_id)
            if existing is not None:
                reused_count += 1
                continue
            # Re-derive span coordinates from mention; the evidence_id
            # encodes (doc_version_id, span_start, span_end).
            row = db._fetchone(
                "SELECT doc_version_id, span_start, span_end "
                "FROM mention WHERE mention_id = ?",
                (item.ref_mention_id,),
            )
            if row is None:
                raise DBError(
                    f"Mention {item.ref_mention_id} not found for "
                    f"timeline item {item.item_id}"
                )
            db.get_or_create_evidence_span(
                doc_version_id=row["doc_version_id"],
                span_start=row["span_start"],
                span_end=row["span_end"],
                run_id=run_id,
                purpose=EVIDENCE_PURPOSE_TIMELINE_MENTION,
            )
            created_count += 1
            logger.debug(
                "Created evidence span for mention=%s (doc=%s, %d:%d)",
                item.ref_mention_id,
                row["doc_version_id"],
                row["span_start"],
                row["span_end"],
            )

    logger.info(
        "Evidence spans for mention items: %d created, %d reused",
        created_count,
        reused_count,
    )


def run_stage_10(
    run_id: str,
    config: Config,
    config_hash: str,
    db: Stage10DatabaseInterface,
) -> RunStageStatusRow:
    """
    Execute the Stage 10 pipeline step.

    :param run_id: Current pipeline run identifier.
    :param config: Loaded and validated pipeline configuration.
    :param config_hash: Config version hash for audit trail.
    :param db: Open database adapter.
    :return: The final ``RunStageStatusRow`` written for this stage.
    :raises DBError: On unrecoverable DB-level errors.
    """
    logger.info(
        "Stage 10 starting: computing entity timelines for run=%s "
        "(config_hash=%s)",
        run_id,
        config_hash,
    )

    # Log relevant config values used by this stage
    dv = config.global_settings.validation.date_validation
    logger.info(
        "Config — date validation: min_year=%d, max_year=%d, "
        "reject_future_effective_dates_beyond_years=%d, "
        "reject_past_deadlines_beyond_days=%d",
        dv.min_year,
        dv.max_year,
        dv.reject_future_effective_dates_beyond_years,
        dv.reject_past_deadlines_beyond_days,
    )

    # Check prerequisite
    prereq = db.get_run_stage_status(run_id, PREREQUISITE_RUN_STAGE)
    if prereq is None or prereq.status != "ok":
        msg = (
            f"Prerequisite {PREREQUISITE_RUN_STAGE} not satisfied "
            f"(status={prereq.status if prereq else 'missing'})"
        )
        logger.error(msg)
        return db.upsert_run_stage_status(
            run_id=run_id,
            stage=STAGE_NAME,
            config_hash=config_hash,
            status="failed",
            error_message=msg,
        )

    logger.info(
        "Prerequisite %s satisfied (status=%s)",
        PREREQUISITE_RUN_STAGE,
        prereq.status,
    )

    # --- INPUT PHASE ---
    eligible_docs = db.get_eligible_doc_version_ids()
    logger.info("INPUT — Eligible documents: %d", len(eligible_docs))

    eligible_events = db.get_eligible_events(eligible_docs)
    logger.info("INPUT — Eligible events (with current revision): %d", len(eligible_events))
    if eligible_events:
        event_type_dist = Counter(ee.event.event_type for ee in eligible_events)
        logger.info(
            "INPUT — Eligible events by type: %s",
            dict(event_type_dist.most_common()),
        )

    entity_map = db.get_entity_registry_map()
    logger.info("INPUT — Entity registry entries: %d", len(entity_map))

    linked_mentions = db.get_linked_mentions_for_docs(eligible_docs)
    logger.info("INPUT — Linked mentions in eligible docs: %d", len(linked_mentions))

    # --- PROCESSING PHASE ---
    logger.info("PROCESSING — Building event-revision timeline items...")
    event_items = build_event_revision_items(
        eligible_events, run_id, db, entity_map, config
    )
    logger.info(
        "PROCESSING — Event-revision timeline items produced: %d",
        len(event_items),
    )

    logger.info("PROCESSING — Building mention timeline items...")
    mention_items = build_mention_items(
        linked_mentions, run_id, db, entity_map
    )
    logger.info(
        "PROCESSING — Mention timeline items produced: %d",
        len(mention_items),
    )

    all_items = event_items + mention_items
    logger.info(
        "PROCESSING — Total timeline items to write: %d "
        "(event_revision=%d, mention=%d)",
        len(all_items),
        len(event_items),
        len(mention_items),
    )

    # --- OUTPUT PHASE ---
    logger.info("OUTPUT — Writing timeline items to database...")
    with db.transaction():
        deleted = db.delete_timeline_items_for_run(run_id)
        if deleted:
            logger.info(
                "OUTPUT — Deleted %d prior timeline items for run=%s",
                deleted,
                run_id,
            )

        logger.info("OUTPUT — Ensuring mention evidence spans...")
        ensure_mention_evidence_spans(mention_items, db, run_id)

        for idx, (item, evidence_rows) in enumerate(all_items):
            db.insert_timeline_item_with_evidence(item, evidence_rows)
            if (idx + 1) % 500 == 0:
                logger.info(
                    "OUTPUT — Written %d / %d timeline items...",
                    idx + 1,
                    len(all_items),
                )

        details = json.dumps({
            "event_revision_items": len(event_items),
            "mention_items": len(mention_items),
            "total_items": len(all_items),
            "eligible_events": len(eligible_events),
            "eligible_docs": len(eligible_docs),
            "entity_registry_size": len(entity_map),
            "linked_mentions_input": len(linked_mentions),
        })

        status_row = db.upsert_run_stage_status(
            run_id=run_id,
            stage=STAGE_NAME,
            config_hash=config_hash,
            status="ok",
            details=details,
        )

    logger.info(
        "OUTPUT — Stage 10 completed: %d items written for run=%s "
        "(event_revision=%d, mention=%d)",
        len(all_items),
        run_id,
        len(event_items),
        len(mention_items),
    )
    return status_row


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Stage 10: Entity Timeline")
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
        help="Directory containing config.yaml",
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
        help="Directory for stage log files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args()


def setup_file_logging(log_dir: Path, run_id: str) -> None:
    """
    Add a file handler to the root logger for persistent log capture.

    :param log_dir: Directory to write the log file into.
    :param run_id: Current run identifier (used in the filename).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{STAGE_NAME}_{run_id[:12]}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(fh)
    logger.info("Log file: %s", log_file)


def main_stage_10_timeline() -> int:
    """
    Entry point for Stage 10.

    :return: 0 on success, 1 on fatal error.
    """
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(
        "Stage 10 invoked with: run_id=%s, config_dir=%s, "
        "working_db=%s, source_db=%s, log_dir=%s, verbose=%s",
        args.run_id,
        args.config_dir,
        args.working_db,
        args.source_db,
        args.log_dir,
        args.verbose,
    )

    # Set up persistent file logging
    setup_file_logging(args.log_dir, args.run_id)

    # Load and validate config
    config_path = args.config_dir / "config.yaml"
    logger.info("Loading configuration from %s", config_path)
    try:
        config = load_config(config_path)
    except Exception:
        logger.exception("Failed to load configuration from %s", config_path)
        return 1
    config_hash = get_config_version(config)
    logger.info("Configuration loaded successfully (hash=%s)", config_hash)

    db = Stage10DatabaseInterface(
        working_db_path=args.working_db,
        source_db_path=args.source_db if args.source_db.exists() else None,
    )

    run = None
    try:
        db.open()
        logger.info("Database connections opened")

        run = db.get_pipeline_run(args.run_id)
        if run is None:
            logger.error("Pipeline run %s not found", args.run_id)
            return 1
        if run.status != "running":
            logger.error(
                "Pipeline run %s is not running (status=%s)",
                args.run_id,
                run.status,
            )
            return 1
        logger.info(
            "Pipeline run validated: run_id=%s, status=%s, config_version=%s",
            run.run_id,
            run.status,
            run.config_version,
        )

        result = run_stage_10(args.run_id, config, config_hash, db)
        if result.status != "ok":
            logger.error(
                "Stage 10 failed: %s", result.error_message or "unknown error"
            )
            return 1
        return 0
    except Exception:
        logger.exception("Stage 10 fatal error")
        try:
            db.upsert_run_stage_status(
                run_id=args.run_id,
                stage=STAGE_NAME,
                config_hash=config_hash if config_hash else (
                    run.config_version if run else "unknown"
                ),
                status="failed",
                error_message="Fatal exception; see logs.",
            )
        except Exception:
            logger.exception("Failed to record stage failure status")
        return 1
    finally:
        db.close()
        logger.info("Database connections closed")


if __name__ == "__main__":
    sys.exit(main_stage_10_timeline())