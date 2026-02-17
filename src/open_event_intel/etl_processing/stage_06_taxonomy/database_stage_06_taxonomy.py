from pathlib import Path

from open_event_intel.etl_processing.database_interface import DatabaseInterface, FacetAssignmentEvidenceRow, FacetAssignmentRow, _serialize_json

STAGE_NAME = "stage_06_taxonomy"
PREREQUISITE_STAGES: tuple[str, ...] = ("stage_03_metadata", "stage_04_mentions")

class Stage06DatabaseInterface(DatabaseInterface):
    """
    Database adapter for Stage 06 (taxonomy classification).

    All SQL lives in :class:`DatabaseInterface`; this subclass only declares
    table ownership and adds the few missing CRUD helpers for
    ``facet_assignment`` / ``facet_assignment_evidence``.
    """

    READS = {
        "pipeline_run",
        "doc_stage_status",
        "document_version",
        "block",
        "mention",
        "document",
    }
    WRITES = {
        "doc_stage_status",
        "facet_assignment",
        "facet_assignment_evidence",
        "evidence_span",
    }

    def __init__(
        self,
        working_db_path: Path,
        source_db_path: Path | None = None,
    ) -> None:
        """Initialize a Stage06DatabaseInterface."""
        super().__init__(working_db_path, source_db_path, STAGE_NAME)

    def get_iteration_set(self) -> list[str]:
        """
        Return ``doc_version_id``s that need (re-)evaluation.

        Per §6.3.0 the iteration set comprises documents with:
        * no ``doc_stage_status`` row for this stage, OR
        * ``status = 'failed'``, OR
        * ``status = 'blocked'`` **and** all prerequisites are now ``ok``.
        """
        self._check_read_access("doc_stage_status")
        self._check_read_access("document_version")
        rows = self._fetchall(
            """
            SELECT dv.doc_version_id
            FROM document_version dv
            JOIN document d ON dv.document_id = d.document_id
            WHERE dv.doc_version_id NOT IN (
                SELECT doc_version_id FROM doc_stage_status
                WHERE stage = ? AND status IN ('ok', 'skipped')
            )
            ORDER BY d.publisher_id, d.url_normalized, dv.doc_version_id
            """,
            (STAGE_NAME,),
        )
        return [r["doc_version_id"] for r in rows]

    def check_prerequisites(self, doc_version_id: str) -> tuple[bool, str | None]:
        """
        Check whether all prerequisite stages are ``ok``.

        :returns: ``(True, None)`` if all ok, else ``(False, error_message)``.
        """
        for prereq in PREREQUISITE_STAGES:
            status_row = self.get_doc_stage_status(doc_version_id, prereq)
            if status_row is None or status_row.status != "ok":
                blocking_status = status_row.status if status_row else "missing"
                return False, f"prerequisite_not_ok:{prereq}:{blocking_status}"
        return True, None

    def insert_facet_assignment(self, row: FacetAssignmentRow) -> None:
        """Insert a facet assignment row."""
        self._check_write_access("facet_assignment")
        self._execute(
            """INSERT INTO facet_assignment
            (facet_id, doc_version_id, facet_type, facet_value, confidence,
             signals_json, created_in_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                row.facet_id,
                row.doc_version_id,
                row.facet_type,
                row.facet_value,
                row.confidence,
                _serialize_json(row.signals_json),
                row.created_in_run_id,
            ),
        )

    def insert_facet_assignment_evidence(
        self, row: FacetAssignmentEvidenceRow
    ) -> None:
        """Insert a facet-assignment ↔ evidence link."""
        self._check_write_access("facet_assignment_evidence")
        self._execute(
            """INSERT INTO facet_assignment_evidence
            (facet_id, evidence_id, purpose) VALUES (?, ?, ?)""",
            (row.facet_id, row.evidence_id, row.purpose),
        )

    def get_facet_assignments_for_doc(
        self, doc_version_id: str
    ) -> list[FacetAssignmentRow]:
        """Return all facet assignments for a document version."""
        self._check_read_access("facet_assignment")
        rows = self._fetchall(
            "SELECT * FROM facet_assignment WHERE doc_version_id = ?",
            (doc_version_id,),
        )
        return [FacetAssignmentRow.model_validate(dict(r)) for r in rows]

    def get_document_publisher_id(self, doc_version_id: str) -> str | None:
        """Look up the publisher_id for a given doc_version_id."""
        self._check_read_access("document_version")
        self._check_read_access("document")
        row = self._fetchone(
            """SELECT d.publisher_id
            FROM document_version dv
            JOIN document d ON dv.document_id = d.document_id
            WHERE dv.doc_version_id = ?""",
            (doc_version_id,),
        )
        return row["publisher_id"] if row else None

    def delete_facets_for_doc(self, doc_version_id: str) -> int:
        """
        Delete all facet assignments and their evidence links for a document.

        Used for idempotent re-processing: cleans up partial results from a
        previously failed run before re-classifying.

        :returns: Number of facet_assignment rows deleted.
        """
        self._check_write_access("facet_assignment_evidence")
        self._check_write_access("facet_assignment")
        self._execute(
            """DELETE FROM facet_assignment_evidence
            WHERE facet_id IN (
                SELECT facet_id FROM facet_assignment WHERE doc_version_id = ?
            )""",
            (doc_version_id,),
        )
        cursor = self._execute(
            "DELETE FROM facet_assignment WHERE doc_version_id = ?",
            (doc_version_id,),
        )
        return cursor.rowcount