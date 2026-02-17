# Module-level constants — values that are structural to the pipeline and not
# expected to change via config.  Collected here so they are never silently
# buried inside function bodies.
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from open_event_intel.etl_processing.database_interface import (
    DatabaseInterface,
    EntityRegistryRow,
    EventEntityLinkRow,
    EventRevisionEvidenceRow,
    EventRevisionRow,
    EventRow,
    EvidenceSpanRow,
    MentionLinkRow,
    MentionRow,
)
from open_event_intel.logger import get_logger

STAGE_NAME = "stage_10_timeline"
PREREQUISITE_RUN_STAGE = "stage_09_outputs"
PREREQUISITE_DOC_STAGE = "stage_08_events"

logger = get_logger(__name__)


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

    def get_mention_by_id(self, mention_id: str) -> MentionRow | None:
        """Retrieve a mention row by its ``mention_id``."""
        self._check_read_access("mention")
        row = self._fetchone(
            "SELECT * FROM mention WHERE mention_id = ?", (mention_id,)
        )
        return MentionRow.model_validate(dict(row)) if row else None

    def get_evidence_span(self, evidence_id: str) -> EvidenceSpanRow | None:
        """Retrieve an evidence span by ID."""
        self._check_read_access("evidence_span")
        row = self._fetchone(
            "SELECT * FROM evidence_span WHERE evidence_id = ?", (evidence_id,)
        )
        return EvidenceSpanRow.model_validate(dict(row)) if row else None