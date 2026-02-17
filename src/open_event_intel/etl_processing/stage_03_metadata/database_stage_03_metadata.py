from pathlib import Path

from open_event_intel.etl_processing.database_interface import DatabaseInterface, DocStageStatusRow, ScrapeRecordRow

STAGE_NAME = "stage_03_metadata"
PREREQUISITE_STAGE = "stage_02_parse"

class Stage03DatabaseInterface(DatabaseInterface):
    """Database interface for Stage 03 metadata extraction."""

    READS = {
        "document_version",
        "block",
        "scrape_record",
        "doc_stage_status",
        "pipeline_run",
        "document",
    }
    WRITES = {"doc_metadata", "evidence_span", "doc_stage_status"}

    def __init__(
        self,
        working_db_path: Path,
        source_db_path: Path | None = None,
    ) -> None:
        """Initialize Stage 03 database interface."""
        super().__init__(working_db_path, source_db_path, STAGE_NAME)

    def get_iteration_set(self) -> list[str]:
        """
        Get documents eligible for Stage 03 processing.

        Returns doc_version_ids where:
        - No status row exists for stage_03_metadata, OR
        - Status is 'failed', OR
        - Status is 'blocked' AND prerequisite is now 'ok'
        """
        self._check_read_access("doc_stage_status")
        self._check_read_access("document_version")

        rows = self._fetchall(
            """
            SELECT dv.doc_version_id
            FROM document_version dv
            JOIN document d ON dv.document_id = d.document_id
            LEFT JOIN doc_stage_status dss
                ON dv.doc_version_id = dss.doc_version_id AND dss.stage = ?
            LEFT JOIN doc_stage_status prereq
                ON dv.doc_version_id = prereq.doc_version_id AND prereq.stage = ?
            WHERE (
                dss.doc_version_id IS NULL
                OR dss.status = 'failed'
                OR (dss.status = 'blocked' AND prereq.status = 'ok')
            )
            ORDER BY d.publisher_id, d.url_normalized, dv.doc_version_id
            """,
            (STAGE_NAME, PREREQUISITE_STAGE),
        )
        return [row["doc_version_id"] for row in rows]

    def check_prerequisite_status(self, doc_version_id: str) -> DocStageStatusRow | None:
        """Check if prerequisite stage is complete for a document."""
        return self.get_doc_stage_status(doc_version_id, PREREQUISITE_STAGE)

    def get_scrape_record_for_doc_version(
        self, doc_version_id: str
    ) -> ScrapeRecordRow | None:
        """Get scrape record associated with a document version."""
        self._check_read_access("document_version")
        self._check_read_access("scrape_record")

        row = self._fetchone(
            """
            SELECT sr.*
            FROM scrape_record sr
            JOIN document_version dv ON sr.scrape_id = dv.scrape_id
            WHERE dv.doc_version_id = ?
            """,
            (doc_version_id,),
        )
        return ScrapeRecordRow.model_validate(dict(row)) if row else None
