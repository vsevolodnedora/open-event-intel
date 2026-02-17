from pathlib import Path

from open_event_intel.etl_processing.database_interface import (
    DatabaseInterface,
    EventCandidateRow,
    EventEntityLinkRow,
    EventRevisionEvidenceRow,
    EventRevisionRow,
    EventRow,
    MentionLinkRow,
    MetricObservationRow,
    NoveltyLabelRow,
    _serialize_json,
)
from open_event_intel.logger import get_logger

logger = get_logger(__name__)

STAGE_NAME = "stage_08_events"
PREREQUISITE_STAGES = ("stage_07_novelty",)

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

