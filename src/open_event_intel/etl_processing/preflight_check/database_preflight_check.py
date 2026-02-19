"""
Database interface for the pre-flight "should I run" check.

Read-only adapter that compares source-DB publication IDs against
ingested ``scrape_record`` rows in the working DB, and inspects
``doc_stage_status`` for retryable (failed/blocked) documents.

.. note::
    The SQL helper methods added here (``get_source_publication_ids``,
    ``get_ingested_scrape_ids_for_publisher``, ``get_stage_failure_counts``,
    ``get_retryable_doc_version_ids``, ``get_total_document_count``)
    should be migrated into ``DatabaseInterface`` once the preflight
    check is integrated into the main codebase.
"""
from pathlib import Path
from typing import ClassVar

from open_event_intel.etl_processing.database_interface import DatabaseInterface, DBError


class PreflightDatabaseInterface(DatabaseInterface):
    """Read-only database adapter for the pre-flight publication check.

    Reads ``pipeline_run``, ``scrape_record``, ``doc_stage_status``, and
    ``document_version``.  Writes nothing.
    """

    READS: ClassVar[set[str]] = {
        "pipeline_run",
        "scrape_record",
        "doc_stage_status",
        "document_version",
    }
    WRITES: ClassVar[set[str]] = set()

    def __init__(
        self,
        working_db_path: Path,
        source_db_path: Path,
    ) -> None:
        super().__init__(
            working_db_path=working_db_path,
            source_db_path=source_db_path,
            stage_name="preflight_should_run",
        )

    def has_completed_runs(self) -> bool:
        """Return ``True`` if the working DB contains at least one completed run."""
        self._check_read_access("pipeline_run")
        row = self._fetchone(
            "SELECT COUNT(*) AS cnt FROM pipeline_run WHERE status = 'completed'"
        )
        return row is not None and row["cnt"] > 0

    def get_ingested_scrape_ids_for_publisher(self, publisher_id: str) -> set[str]:
        """Return the set of ``scrape_id`` values already ingested for *publisher_id*.

        .. note::
            Candidate for promotion into ``DatabaseInterface``.
        """
        self._check_read_access("scrape_record")
        rows = self._fetchall(
            "SELECT scrape_id FROM scrape_record WHERE publisher_id = ?",
            (publisher_id,),
        )
        return {row["scrape_id"] for row in rows}

    def get_source_publication_ids(self, table_name: str) -> list[str]:
        """Return all ``ID`` values from a source-DB publisher table.

        Lightweight alternative to ``read_source_publications`` — avoids
        deserialising content blobs.

        :param table_name: Publisher table name (e.g. ``"entsoe"``).
        :raises DBError: If the source connection is not available.

        .. note::
            Candidate for promotion into ``DatabaseInterface``.
        """
        if self._source_conn is None:
            raise DBError("Source database not connected")
        cursor = self._source_conn.execute(
            f'SELECT ID FROM "{table_name}"'  # noqa: S608
        )
        return [str(row[0]) for row in cursor.fetchall()]

    def get_total_document_count(self) -> int:
        """Return the total number of ``document_version`` rows.

        .. note::
            Candidate for promotion into ``DatabaseInterface``.
        """
        self._check_read_access("document_version")
        row = self._fetchone("SELECT COUNT(*) AS cnt FROM document_version")
        return row["cnt"] if row else 0

    def get_stage_failure_counts(self) -> list[tuple[str, str, int]]:
        """Return ``(stage, status, count)`` for every failed or blocked stage.

        Rows are ordered by stage then status for deterministic output.

        .. note::
            Candidate for promotion into ``DatabaseInterface``.
        """
        self._check_read_access("doc_stage_status")
        rows = self._fetchall(
            """SELECT stage, status, COUNT(*) AS cnt
               FROM doc_stage_status
               WHERE status IN ('failed', 'blocked')
               GROUP BY stage, status
               ORDER BY stage, status"""
        )
        return [(row["stage"], row["status"], row["cnt"]) for row in rows]

    def get_retryable_doc_version_ids(self) -> tuple[set[str], set[str]]:
        """Return ``(failed_ids, blocked_only_ids)`` from ``doc_stage_status``.

        *failed_ids*: every ``doc_version_id`` with at least one ``'failed'``
        stage (root-cause failure).

        *blocked_only_ids*: every ``doc_version_id`` with at least one
        ``'blocked'`` stage but **no** ``'failed'`` stage (pure
        prerequisite-propagation).

        .. note::
            Candidate for promotion into ``DatabaseInterface``.
        """
        self._check_read_access("doc_stage_status")
        failed_rows = self._fetchall(
            "SELECT DISTINCT doc_version_id FROM doc_stage_status WHERE status = 'failed'"
        )
        failed_ids = {row["doc_version_id"] for row in failed_rows}

        blocked_rows = self._fetchall(
            "SELECT DISTINCT doc_version_id FROM doc_stage_status WHERE status = 'blocked'"
        )
        blocked_all = {row["doc_version_id"] for row in blocked_rows}
        blocked_only_ids = blocked_all - failed_ids

        return failed_ids, blocked_only_ids