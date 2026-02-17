from pathlib import Path
from typing import ClassVar, Sequence

from open_event_intel.etl_processing.database_interface import ChunkEmbeddingRow, DatabaseInterface, DocStageStatusRow, EmbeddingIndexRow, _serialize_json

STAGE_NAME = "stage_05_embeddings"
STAGE_NAME_INDEX = "stage_05_embeddings_index"
PREREQUISITE_STAGE = "stage_02_parse"

class Stage05DatabaseInterface(DatabaseInterface):
    """
    Database adapter for Stage 05 (embeddings + ANN index).

    SQL methods here are candidates for future migration into
    :class:`DatabaseInterface`.
    """

    READS: ClassVar[set[str]] = {
        "pipeline_run",
        "doc_stage_status",
        "run_stage_status",
        "chunk",
        "chunk_embedding",
        "document_version",
        "document",
        "llm_cache",
    }
    WRITES: ClassVar[set[str]] = {
        "chunk_embedding",
        "embedding_index",
        "doc_stage_status",
        "run_stage_status",
        "pipeline_run",
        "llm_usage_log",
        "llm_cache",
    }

    def __init__(self, working_db_path: Path, source_db_path: Path | None = None) -> None:
        """Initialize Stage 05 (embeddings + ANN index)."""
        super().__init__(working_db_path, source_db_path, STAGE_NAME)

    def get_iteration_set(self) -> list[str]:
        r"""
        Return ``doc_version_id``\s needing (re-)evaluation for Stage 05.

        Includes documents with no status row, ``failed`` status, or ``blocked``
        status whose prerequisite is now ``'ok'``.

        Results ordered by ``(publisher_id, url_normalized, doc_version_id)``.
        """
        self._check_read_access("doc_stage_status")
        self._check_read_access("document_version")
        self._check_read_access("document")
        rows = self._fetchall(
            """
            SELECT dv.doc_version_id
            FROM document_version dv
            JOIN document d ON d.document_id = dv.document_id
            JOIN doc_stage_status prereq
              ON prereq.doc_version_id = dv.doc_version_id
             AND prereq.stage = ?
             AND prereq.status = 'ok'
            LEFT JOIN doc_stage_status cur
              ON cur.doc_version_id = dv.doc_version_id
             AND cur.stage = ?
            WHERE cur.status IS NULL
               OR cur.status = 'failed'
               OR (cur.status = 'blocked'
                   AND EXISTS (
                       SELECT 1 FROM doc_stage_status p2
                       WHERE p2.doc_version_id = dv.doc_version_id
                         AND p2.stage = ?
                         AND p2.status = 'ok'
                   ))
            ORDER BY d.publisher_id, d.url_normalized, dv.doc_version_id
            """,
            (PREREQUISITE_STAGE, STAGE_NAME, PREREQUISITE_STAGE),
        )
        return [r["doc_version_id"] for r in rows]

    def get_embedding_input_docs(self) -> list[str]:
        r"""All ``doc_version_id``\s with ``stage_02_parse='ok'`` (global)."""
        self._check_read_access("doc_stage_status")
        rows = self._fetchall(
            "SELECT doc_version_id FROM doc_stage_status "
            "WHERE stage = ? AND status = 'ok' ORDER BY doc_version_id",
            (PREREQUISITE_STAGE,),
        )
        return [r["doc_version_id"] for r in rows]

    def get_eligible_embedding_docs(self) -> list[str]:
        r"""All ``doc_version_id``\s with ``stage_05_embeddings='ok'``."""
        self._check_read_access("doc_stage_status")
        rows = self._fetchall(
            "SELECT doc_version_id FROM doc_stage_status "
            "WHERE stage = ? AND status = 'ok' ORDER BY doc_version_id",
            (STAGE_NAME,),
        )
        return [r["doc_version_id"] for r in rows]

    def check_prerequisite(self, doc_version_id: str) -> DocStageStatusRow | None:
        """Return the prerequisite status row for a document, or ``None``."""
        return self.get_doc_stage_status(doc_version_id, PREREQUISITE_STAGE)

    def insert_chunk_embedding(self, row: ChunkEmbeddingRow) -> None:
        """Insert a single chunk embedding row."""
        self._check_write_access("chunk_embedding")
        self._execute(
            """INSERT INTO chunk_embedding
               (chunk_id, embedding_vector, embedding_dim, model_version,
                language_used, created_in_run_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                row.chunk_id,
                row.embedding_vector,
                row.embedding_dim,
                row.model_version,
                row.language_used,
                row.created_in_run_id,
            ),
        )

    def insert_chunk_embeddings(self, rows: Sequence[ChunkEmbeddingRow]) -> None:
        """Insert multiple chunk embedding rows."""
        for row in rows:
            self.insert_chunk_embedding(row)

    def get_chunk_embedding(self, chunk_id: str) -> ChunkEmbeddingRow | None:
        """Retrieve a chunk embedding by chunk_id."""
        self._check_read_access("chunk_embedding")
        row = self._fetchone(
            "SELECT * FROM chunk_embedding WHERE chunk_id = ?", (chunk_id,)
        )
        return ChunkEmbeddingRow.model_validate(dict(row)) if row else None

    def get_all_eligible_embeddings(self) -> list[tuple[str, bytes, int]]:
        """
        Return ``(chunk_id, embedding_vector, embedding_dim)`` for chunks belonging to eligible documents.

        Ordered by ``chunk_id`` for determinism.
        """
        self._check_read_access("chunk_embedding")
        self._check_read_access("doc_stage_status")
        self._check_read_access("chunk")
        rows = self._fetchall(
            """
            SELECT ce.chunk_id, ce.embedding_vector, ce.embedding_dim
            FROM chunk_embedding ce
            JOIN chunk c ON c.chunk_id = ce.chunk_id
            JOIN doc_stage_status dss
              ON dss.doc_version_id = c.doc_version_id
             AND dss.stage = ?
             AND dss.status = 'ok'
            ORDER BY ce.chunk_id
            """,
            (STAGE_NAME,),
        )
        return [(r["chunk_id"], r["embedding_vector"], r["embedding_dim"]) for r in rows]

    def insert_embedding_index(self, row: EmbeddingIndexRow) -> None:
        """Insert an embedding index metadata row."""
        self._check_write_access("embedding_index")
        self._execute(
            """INSERT INTO embedding_index
               (index_id, run_id, model_version, embedding_dim, method,
                index_path, built_at, chunk_count, build_params_json, checksum)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row.index_id,
                row.run_id,
                row.model_version,
                row.embedding_dim,
                row.method,
                row.index_path,
                row.built_at.isoformat(),
                row.chunk_count,
                _serialize_json(row.build_params_json),
                row.checksum,
            ),
        )

    def delete_embedding_indexes_for_run(self, run_id: str) -> int:
        """Delete embedding index rows for a run (§6.4 run-scoped idempotency)."""
        self._check_write_access("embedding_index")
        cursor = self._execute(
            "DELETE FROM embedding_index WHERE run_id = ?", (run_id,)
        )
        return cursor.rowcount

    def count_doc_stage_statuses(self, stage: str) -> dict[str, int]:
        """Count documents grouped by status for a given stage."""
        self._check_read_access("doc_stage_status")
        rows = self._fetchall(
            "SELECT status, COUNT(*) as cnt FROM doc_stage_status "
            "WHERE stage = ? GROUP BY status",
            (stage,),
        )
        return {r["status"]: r["cnt"] for r in rows}

    def get_stage_doc_version_ids(self, stage: str) -> set[str]:
        """Return all doc_version_ids that have a status row for *stage*."""
        self._check_read_access("doc_stage_status")
        rows = self._fetchall(
            "SELECT doc_version_id FROM doc_stage_status WHERE stage = ?",
            (stage,),
        )
        return {r["doc_version_id"] for r in rows}