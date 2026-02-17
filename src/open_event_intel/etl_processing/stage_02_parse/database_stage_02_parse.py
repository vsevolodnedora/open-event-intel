from pathlib import Path
from typing import ClassVar

from open_event_intel.etl_processing.database_interface import DatabaseInterface, DocStageStatusRow

STAGE_NAME = "stage_02_parse"
PREREQUISITE_STAGE = "stage_01_ingest"

class Stage02DatabaseInterface(DatabaseInterface):
    """Database interface for stage 02 parse operations."""

    READS: ClassVar[set[str]] = {
        "document_version",
        "document",
        "pipeline_run",
        "doc_stage_status",
        "block",
        "chunk",
        "evidence_span",
        "table_extract",
    }
    WRITES: ClassVar[set[str]] = {
        "block",
        "chunk",
        "evidence_span",
        "table_extract",
        "doc_stage_status",
    }

    def __init__(self, working_db_path: Path, source_db_path: Path | None = None) -> None:
        """Initialize Stage 02 Database Interface."""
        super().__init__(working_db_path, source_db_path, STAGE_NAME)

    def get_iteration_set(self, run_id: str) -> list[tuple[str, str, str]]:
        """
        Get documents requiring stage 02 processing.

        Returns documents in deterministic order: (publisher_id, url_normalized, doc_version_id)
        where:
        - No doc_stage_status row exists for stage_02_parse, OR
        - Status is 'failed', OR
        - Status is 'blocked' AND stage_01_ingest is now 'ok'
        """
        self._check_read_access("document_version")
        self._check_read_access("document")
        self._check_read_access("doc_stage_status")

        rows = self._fetchall(
            """
            SELECT d.publisher_id, d.url_normalized, dv.doc_version_id
            FROM document_version dv
            JOIN document d ON d.document_id = dv.document_id
            LEFT JOIN doc_stage_status dss_02
                ON dss_02.doc_version_id = dv.doc_version_id
                AND dss_02.stage = ?
            LEFT JOIN doc_stage_status dss_01
                ON dss_01.doc_version_id = dv.doc_version_id
                AND dss_01.stage = ?
            WHERE
                dss_02.status IS NULL
                OR dss_02.status = 'failed'
                OR (dss_02.status = 'blocked' AND dss_01.status = 'ok')
            ORDER BY d.publisher_id, d.url_normalized, dv.doc_version_id
            """,
            (STAGE_NAME, PREREQUISITE_STAGE),
        )
        return [(r["publisher_id"], r["url_normalized"], r["doc_version_id"]) for r in rows]

    def get_prerequisite_status(self, doc_version_id: str) -> DocStageStatusRow | None:
        """Get the prerequisite stage status for a document."""
        return self.get_doc_stage_status(doc_version_id, PREREQUISITE_STAGE)

    def get_table_extract_run_summary(self, run_id: str) -> dict[str, int] | None:
        """
        Get aggregate table_extract statistics for a pipeline run.

        :param run_id: Pipeline run ID to summarise.
        :returns: Dict with keys total, with_headers, with_period_gran,
            with_units; or None if no rows found.
        """
        self._check_read_access("table_extract")

        rows = self._fetchall(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN headers_json IS NOT NULL AND headers_json != '[]' THEN 1 ELSE 0 END) AS with_headers,
                SUM(CASE WHEN period_granularity IS NOT NULL THEN 1 ELSE 0 END) AS with_period_gran,
                SUM(CASE WHEN units_detected IS NOT NULL THEN 1 ELSE 0 END) AS with_units
            FROM table_extract
            WHERE created_in_run_id = ?
            """,
            (run_id,),
        )
        if not rows:
            return None
        r = rows[0]
        return {
            "total": r["total"],
            "with_headers": r["with_headers"],
            "with_period_gran": r["with_period_gran"],
            "with_units": r["with_units"],
        }