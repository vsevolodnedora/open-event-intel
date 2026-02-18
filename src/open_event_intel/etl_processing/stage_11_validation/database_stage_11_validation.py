# Stage identity
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from open_event_intel.etl_processing.database_interface import DatabaseInterface

# Sampling defaults (override via argparse)
DEFAULT_EVIDENCE_ID_SAMPLE_SIZE = 200
DEFAULT_SPAN_CONFORMANCE_SAMPLE_SIZE = 100
DEFAULT_BAD_ID_LOG_LIMIT = 50

STAGE_NAME = "stage_11_validation"
PREREQUISITE_STAGE = "stage_10_timeline"

class Stage11DatabaseInterface(DatabaseInterface):
    """
    Database adapter for Stage 11 validation.

    Reads broadly across all pipeline tables; writes to
    ``validation_failure``, ``run_stage_status``, and ``pipeline_run``
    (to mark the run as completed).

    .. note::
       The SQL queries here are stage-specific and should eventually be
       migrated into ``DatabaseInterface`` proper.
    """

    READS: set[str] = {  # type: ignore[assignment]
        "pipeline_run",
        "document_version",
        "evidence_span",
        "doc_stage_status",
        "run_stage_status",
        "block",
        "chunk",
        "chunk_embedding",
        "doc_metadata",
        "table_extract",
        "mention",
        "mention_link",
        "registry_update_proposal",
        "facet_assignment",
        "facet_assignment_evidence",
        "novelty_label",
        "novelty_label_evidence",
        "document_fingerprint",
        "chunk_novelty",
        "chunk_novelty_score",
        "event",
        "event_revision",
        "event_revision_evidence",
        "event_entity_link",
        "metric_observation",
        "event_candidate",
        "metric_series",
        "metric_series_point",
        "alert",
        "alert_evidence",
        "digest_item",
        "digest_item_evidence",
        "entity_timeline_item",
        "entity_timeline_item_evidence",
        "embedding_index",
        "story_cluster",
        "story_cluster_member",
    }
    WRITES: set[str] = {"validation_failure", "run_stage_status", "pipeline_run"}  # type: ignore[assignment]

    def __init__(self, working_db_path: Path) -> None:
        """Initialize a Stage 11 validation."""
        super().__init__(working_db_path, source_db_path=None, stage_name=STAGE_NAME)

    # Deletion

    def delete_validation_failures_for_run_stage(self, run_id: str, stage: str) -> int:
        """Delete previous validation failures for *run_id* and *stage*."""
        self._check_write_access("validation_failure")
        cursor = self._execute(
            "DELETE FROM validation_failure WHERE run_id = ? AND stage = ?",
            (run_id, stage),
        )
        return cursor.rowcount

    # Evidence-linkage counts

    def count_alerts_without_evidence(self, run_id: str) -> int:
        """Count alerts that lack at least one ``alert_evidence`` row."""
        self._check_read_access("alert")
        self._check_read_access("alert_evidence")
        row = self._fetchone(
            """SELECT COUNT(*) AS cnt FROM alert a
               WHERE a.run_id = ?
                 AND NOT EXISTS (
                     SELECT 1 FROM alert_evidence ae WHERE ae.alert_id = a.alert_id
                 )""",
            (run_id,),
        )
        return row["cnt"] if row else 0

    def get_alert_ids_without_evidence(self, run_id: str, limit: int = DEFAULT_BAD_ID_LOG_LIMIT) -> list[str]:
        """Get alert IDs without evidence."""
        self._check_read_access("alert")
        self._check_read_access("alert_evidence")
        rows = self._fetchall(
            """SELECT a.alert_id FROM alert a
               WHERE a.run_id = ?
                 AND NOT EXISTS (
                     SELECT 1 FROM alert_evidence ae WHERE ae.alert_id = a.alert_id
                 ) LIMIT ?""",
            (run_id, limit),
        )
        return [r["alert_id"] for r in rows]

    def count_digest_items_without_evidence(self, run_id: str) -> int:
        """Count digest items without evidence."""
        self._check_read_access("digest_item")
        self._check_read_access("digest_item_evidence")
        row = self._fetchone(
            """SELECT COUNT(*) AS cnt FROM digest_item di
               WHERE di.run_id = ?
                 AND NOT EXISTS (
                     SELECT 1 FROM digest_item_evidence die WHERE die.item_id = di.item_id
                 )""",
            (run_id,),
        )
        return row["cnt"] if row else 0

    def get_digest_item_ids_without_evidence(self, run_id: str, limit: int = DEFAULT_BAD_ID_LOG_LIMIT) -> list[str]:
        """Get digest item IDs without evidence."""
        self._check_read_access("digest_item")
        self._check_read_access("digest_item_evidence")
        rows = self._fetchall(
            """SELECT di.item_id FROM digest_item di
               WHERE di.run_id = ?
                 AND NOT EXISTS (
                     SELECT 1 FROM digest_item_evidence die WHERE die.item_id = di.item_id
                 ) LIMIT ?""",
            (run_id, limit),
        )
        return [r["item_id"] for r in rows]

    def count_timeline_items_without_evidence(self, run_id: str) -> int:
        """Count timeline items without evidence."""
        self._check_read_access("entity_timeline_item")
        self._check_read_access("entity_timeline_item_evidence")
        row = self._fetchone(
            """SELECT COUNT(*) AS cnt FROM entity_timeline_item eti
               WHERE eti.run_id = ?
                 AND NOT EXISTS (
                     SELECT 1 FROM entity_timeline_item_evidence etie
                     WHERE etie.item_id = eti.item_id
                 )""",
            (run_id,),
        )
        return row["cnt"] if row else 0

    def get_timeline_item_ids_without_evidence(self, run_id: str, limit: int = DEFAULT_BAD_ID_LOG_LIMIT) -> list[str]:
        """Get timeline item IDs without evidence."""
        self._check_read_access("entity_timeline_item")
        self._check_read_access("entity_timeline_item_evidence")
        rows = self._fetchall(
            """SELECT eti.item_id FROM entity_timeline_item eti
               WHERE eti.run_id = ?
                 AND NOT EXISTS (
                     SELECT 1 FROM entity_timeline_item_evidence etie
                     WHERE etie.item_id = eti.item_id
                 ) LIMIT ?""",
            (run_id, limit),
        )
        return [r["item_id"] for r in rows]

    def count_metric_points_without_evidence(self, run_id: str) -> int:
        """Count metric points without evidence."""
        self._check_read_access("metric_series_point")
        row = self._fetchone(
            """SELECT COUNT(*) AS cnt FROM metric_series_point
               WHERE run_id = ? AND evidence_id IS NULL""",
            (run_id,),
        )
        return row["cnt"] if row else 0

    def count_event_revisions_without_evidence(self) -> int:
        """Count current-revision event_revisions lacking evidence rows."""
        self._check_read_access("event")
        self._check_read_access("event_revision")
        self._check_read_access("event_revision_evidence")
        row = self._fetchone(
            """SELECT COUNT(*) AS cnt FROM event e
               JOIN event_revision er ON er.revision_id = e.current_revision_id
               WHERE e.current_revision_id IS NOT NULL
                 AND NOT EXISTS (
                     SELECT 1 FROM event_revision_evidence ere
                     WHERE ere.revision_id = er.revision_id
                 )"""
        )
        return row["cnt"] if row else 0

    def count_facet_assignments_without_evidence(self) -> int:
        """Count facet_assignments lacking evidence rows."""
        self._check_read_access("facet_assignment")
        self._check_read_access("facet_assignment_evidence")
        row = self._fetchone(
            """SELECT COUNT(*) AS cnt FROM facet_assignment fa
               WHERE NOT EXISTS (
                   SELECT 1 FROM facet_assignment_evidence fae
                   WHERE fae.facet_id = fa.facet_id
               )"""
        )
        return row["cnt"] if row else 0

    def count_novelty_labels_without_evidence(self) -> int:
        """Count novelty_labels lacking evidence rows."""
        self._check_read_access("novelty_label")
        self._check_read_access("novelty_label_evidence")
        row = self._fetchone(
            """SELECT COUNT(*) AS cnt FROM novelty_label nl
               WHERE NOT EXISTS (
                   SELECT 1 FROM novelty_label_evidence nle
                   WHERE nle.doc_version_id = nl.doc_version_id
               )"""
        )
        return row["cnt"] if row else 0

    # Span / evidence sampling

    def sample_evidence_spans(self, limit: int = DEFAULT_EVIDENCE_ID_SAMPLE_SIZE) -> list[dict]:
        """Return a sample of evidence_span rows for determinism checks."""
        self._check_read_access("evidence_span")
        rows = self._fetchall(
            """SELECT evidence_id, doc_version_id, span_start, span_end
               FROM evidence_span ORDER BY evidence_id LIMIT ?""",
            (limit,),
        )
        return [dict(r) for r in rows]

    def verify_span_text_with_sqlite(
        self, doc_version_id: str, span_start: int, span_end: int
    ) -> dict | None:
        """Return both stored text and SQLite-sliced text for a span."""
        self._check_read_access("evidence_span")
        self._check_read_access("document_version")
        row = self._fetchone(
            """SELECT
                   es.text AS stored_text,
                   substr(dv.clean_content, ? + 1, ? - ?) AS sqlite_text
               FROM evidence_span es
               JOIN document_version dv ON dv.doc_version_id = es.doc_version_id
               WHERE es.doc_version_id = ?
                 AND es.span_start = ?
                 AND es.span_end = ?""",
            (span_start, span_end, span_start, doc_version_id, span_start, span_end),
        )
        return dict(row) if row else None

    def sample_chunk_spans(self, limit: int = DEFAULT_SPAN_CONFORMANCE_SAMPLE_SIZE) -> list[dict]:
        """Sample chunks and compare chunk_text with SQLite substr of clean_content."""
        self._check_read_access("chunk")
        self._check_read_access("document_version")
        rows = self._fetchall(
            """SELECT
                   c.chunk_id,
                   c.doc_version_id,
                   c.span_start,
                   c.span_end,
                   c.chunk_text,
                   substr(dv.clean_content, c.span_start + 1, c.span_end - c.span_start) AS sqlite_text
               FROM chunk c
               JOIN document_version dv ON dv.doc_version_id = c.doc_version_id
               ORDER BY c.chunk_id
               LIMIT ?""",
            (limit,),
        )
        return [dict(r) for r in rows]

    # Config coherence

    def get_eligible_doc_config_hashes(self) -> list[dict]:
        """Return distinct config_hash values for eligible docs (stage_08_events='ok')."""
        self._check_read_access("doc_stage_status")
        rows = self._fetchall(
            """SELECT DISTINCT dss.config_hash
               FROM doc_stage_status dss
               WHERE dss.stage = 'stage_08_events'
                 AND dss.status = 'ok'"""
        )
        return [dict(r) for r in rows]

    # Event eligibility

    def get_current_revision_doc_version_ids(self) -> list[dict]:
        """Return current-revision doc_version_ids and their event stage status."""
        self._check_read_access("event")
        self._check_read_access("event_revision")
        self._check_read_access("doc_stage_status")
        rows = self._fetchall(
            """SELECT e.event_id, er.revision_id, er.doc_version_ids
               FROM event e
               JOIN event_revision er ON er.revision_id = e.current_revision_id
               WHERE e.current_revision_id IS NOT NULL"""
        )
        return [dict(r) for r in rows]

    def get_doc_stage_status_for_ids(
        self, doc_version_ids: Sequence[str], stage: str
    ) -> dict[str, str]:
        """Return {doc_version_id: status} for the given IDs and stage."""
        self._check_read_access("doc_stage_status")
        if not doc_version_ids:
            return {}
        placeholders = ",".join("?" for _ in doc_version_ids)
        rows = self._fetchall(
            f"SELECT doc_version_id, status FROM doc_stage_status "  # noqa: S608
            f"WHERE stage = ? AND doc_version_id IN ({placeholders})",
            (stage, *doc_version_ids),
        )
        return {r["doc_version_id"]: r["status"] for r in rows}

    # Index file presence

    def get_embedding_index_rows(self, run_id: str) -> list[dict]:
        self._check_read_access("embedding_index")
        rows = self._fetchall(
            "SELECT index_id, index_path FROM embedding_index WHERE run_id = ?",
            (run_id,),
        )
        return [dict(r) for r in rows]

    #  Orphaned events

    def get_orphaned_events(self) -> list[dict]:
        """Events with NULL current_revision_id but existing revisions."""
        self._check_read_access("event")
        self._check_read_access("event_revision")
        rows = self._fetchall(
            """SELECT e.event_id FROM event e
               WHERE e.current_revision_id IS NULL
                 AND EXISTS (
                     SELECT 1 FROM event_revision er WHERE er.event_id = e.event_id
                 )"""
        )
        return [dict(r) for r in rows]

    # Run-state hygiene

    def get_other_running_runs(self, current_run_id: str) -> list[dict]:
        """Return all other running runs for a given current_run_id."""
        self._check_read_access("pipeline_run")
        rows = self._fetchall(
            "SELECT run_id FROM pipeline_run WHERE status = 'running' AND run_id != ?",
            (current_run_id,),
        )
        return [dict(r) for r in rows]

    def get_abandoned_runs(self) -> list[dict]:
        """Return all abandoned runs for a given current_run_id."""
        self._check_read_access("pipeline_run")
        rows = self._fetchall(
            """SELECT run_id, status, started_at, completed_at
               FROM pipeline_run WHERE status IN ('failed', 'aborted')
               ORDER BY started_at"""
        )
        return [dict(r) for r in rows]

    def count_non_completed_run_artifacts(
        self, current_run_id: str, table: str
    ) -> int:
        """Count rows whose ``created_in_run_id`` references a non-completed, non-current pipeline_run."""
        self._check_read_access(table)
        self._check_read_access("pipeline_run")
        row = self._fetchone(
            f"SELECT COUNT(*) AS cnt FROM \"{table}\" t "  # noqa: S608
            f"WHERE t.created_in_run_id != ? "
            f"  AND t.created_in_run_id NOT IN ("
            f"      SELECT run_id FROM pipeline_run WHERE status = 'completed'"
            f"  )",
            (current_run_id,),
        )
        return row["cnt"] if row else 0

    # Embedding model versions

    def get_distinct_embedding_model_versions(self) -> list[str]:
        """Return distinct embedding model versions."""
        self._check_read_access("chunk_embedding")
        rows = self._fetchall(
            "SELECT DISTINCT model_version FROM chunk_embedding"
        )
        return [r["model_version"] for r in rows]

    # Table row counts (for audit / export-readiness)

    def count_table_rows(self, table: str, run_id: str | None = None) -> int:
        """Return total row count for *table*, optionally filtered by run_id."""
        self._check_read_access(table)
        if run_id is not None:
            row = self._fetchone(
                f'SELECT COUNT(*) AS cnt FROM "{table}" WHERE run_id = ?',  # noqa: S608
                (run_id,),
            )
        else:
            row = self._fetchone(f'SELECT COUNT(*) AS cnt FROM "{table}"')  # noqa: S608
        return row["cnt"] if row else 0

    def count_table_rows_by_run(self, table: str, run_id: str, id_column: str = "created_in_run_id") -> int:
        """Count rows created in a specific run via *id_column*."""
        self._check_read_access(table)
        row = self._fetchone(
            f'SELECT COUNT(*) AS cnt FROM "{table}" WHERE "{id_column}" = ?',  # noqa: S608
            (run_id,),
        )
        return row["cnt"] if row else 0

    # Run-stage status for export readiness

    def get_run_stage_statuses_for_run(self, run_id: str) -> dict[str, str]:
        """Return {stage: status} for all run_stage_status rows for *run_id*."""
        self._check_read_access("run_stage_status")
        rows = self._fetchall(
            "SELECT stage, status FROM run_stage_status WHERE run_id = ?",
            (run_id,),
        )
        return {r["stage"]: r["status"] for r in rows}

    def count_ok_docs_for_stage(self, stage: str) -> int:
        """Count doc_stage_status rows with ``status='ok'`` for *stage*."""
        self._check_read_access("doc_stage_status")
        row = self._fetchone(
            "SELECT COUNT(*) AS cnt FROM doc_stage_status WHERE stage = ? AND status = 'ok'",
            (stage,),
        )
        return row["cnt"] if row else 0

    # Pipeline run completion

    def complete_pipeline_run(self, run_id: str) -> None:
        """
        Transition a pipeline run from ``'running'`` to ``'completed'``.

        Sets ``status='completed'`` and ``completed_at`` to the current UTC
        timestamp.  Only updates rows whose current status is ``'running'``
        to avoid accidentally overwriting a run that was concurrently failed
        or aborted.

        :param run_id: The pipeline run ID to mark as completed.
        :raises DBError: If no row was updated (run not found or not running).
        """
        self._check_write_access("pipeline_run")
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._execute(
            """UPDATE pipeline_run
               SET status = 'completed', completed_at = ?
               WHERE run_id = ? AND status = 'running'""",
            (now, run_id),
        )
        if cursor.rowcount == 0:
            from open_event_intel.etl_processing.database_interface import DBError
            raise DBError(
                f"Cannot complete pipeline run '{run_id}': "
                f"no running row found (already completed, failed, or missing)"
            )