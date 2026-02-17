import json
from pathlib import Path
from typing import Sequence

from open_event_intel.etl_processing.database_interface import DatabaseInterface, DocStageStatusRow, MentionLinkRow, RegistryUpdateProposalRow, _serialize_json

STAGE_NAME = "stage_04_mentions"
PREREQUISITE_STAGE = "stage_02_parse"

class Stage04DatabaseInterface(DatabaseInterface):
    """
    Database interface for Stage 04 Mentions.

    Reads: document_version, block, entity_registry, chunk, doc_stage_status, pipeline_run
    Writes: mention, mention_link, registry_update_proposal, doc_stage_status
    """

    READS = {
        "document_version",
        "document",
        "block",
        "chunk",
        "entity_registry",
        "doc_stage_status",
        "pipeline_run",
        "registry_update_proposal",
    }
    WRITES = {
        "mention",
        "mention_link",
        "registry_update_proposal",
        "doc_stage_status",
    }

    def __init__(self, working_db_path: Path, source_db_path: Path | None = None) -> None:
        """Initialize."""
        super().__init__(working_db_path, source_db_path, STAGE_NAME)

    def get_stage04_iteration_set(self) -> list[str]:
        """
        Get documents requiring processing for stage 04.

        Returns doc_version_ids where:
        - No doc_stage_status row exists for stage_04_mentions, OR
        - Status is 'failed', OR
        - Status is 'blocked' AND all prerequisites are now 'ok'

        Ordered by (publisher_id, url_normalized, doc_version_id) for determinism.
        """
        self._check_read_access("doc_stage_status")
        self._check_read_access("document_version")
        self._check_read_access("document")

        sql = """
            SELECT dv.doc_version_id
            FROM document_version dv
            JOIN document d ON d.document_id = dv.document_id
            LEFT JOIN doc_stage_status dss4 
                ON dss4.doc_version_id = dv.doc_version_id 
                AND dss4.stage = ?
            LEFT JOIN doc_stage_status dss2
                ON dss2.doc_version_id = dv.doc_version_id
                AND dss2.stage = ?
            WHERE
                dss4.status IS NULL
                OR dss4.status = 'failed'
                OR (dss4.status = 'blocked' AND dss2.status = 'ok')
            ORDER BY d.publisher_id, d.url_normalized, dv.doc_version_id
        """
        rows = self._fetchall(sql, (STAGE_NAME, PREREQUISITE_STAGE))
        return [row["doc_version_id"] for row in rows]

    def get_prerequisite_status(self, doc_version_id: str) -> DocStageStatusRow | None:
        """Get the prerequisite stage status for a document."""
        return self.get_doc_stage_status(doc_version_id, PREREQUISITE_STAGE)

    def insert_registry_update_proposal(self, row: RegistryUpdateProposalRow) -> None:
        """Insert a registry update proposal."""
        self._check_write_access("registry_update_proposal")
        self._execute(
            """INSERT INTO registry_update_proposal 
               (proposal_id, surface_form, proposal_type, target_entity_id, 
                inferred_type, evidence_doc_ids, occurrence_count, status, 
                review_notes, created_in_run_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row.proposal_id,
                row.surface_form,
                row.proposal_type,
                row.target_entity_id,
                row.inferred_type,
                _serialize_json(row.evidence_doc_ids),
                row.occurrence_count,
                row.status,
                row.review_notes,
                row.created_in_run_id,
            ),
        )

    def get_existing_proposal_by_surface_form(
        self, surface_form: str
    ) -> RegistryUpdateProposalRow | None:
        """Get existing proposal for a surface form if it exists."""
        self._check_read_access("registry_update_proposal")
        row = self._fetchone(
            "SELECT * FROM registry_update_proposal WHERE surface_form = ?",
            (surface_form,),
        )
        if row:
            return RegistryUpdateProposalRow.model_validate(dict(row))
        return None

    def update_proposal_occurrence(
        self, proposal_id: str, doc_version_id: str, new_count: int
    ) -> None:
        """Update proposal with additional occurrence."""
        self._check_write_access("registry_update_proposal")
        row = self._fetchone(
            "SELECT evidence_doc_ids FROM registry_update_proposal WHERE proposal_id = ?",
            (proposal_id,),
        )
        if row:
            existing_docs = json.loads(row["evidence_doc_ids"])
            if doc_version_id not in existing_docs:
                existing_docs.append(doc_version_id)
            self._execute(
                """UPDATE registry_update_proposal 
                   SET occurrence_count = ?, evidence_doc_ids = ?
                   WHERE proposal_id = ?""",
                (new_count, _serialize_json(existing_docs), proposal_id),
            )

    def insert_mention_links(self, rows: Sequence[MentionLinkRow]) -> None:
        """Insert multiple mention links."""
        for row in rows:
            self.insert_mention_link(row)