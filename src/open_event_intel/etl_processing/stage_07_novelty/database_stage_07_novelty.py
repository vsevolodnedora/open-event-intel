from collections import defaultdict
from pathlib import Path

import numpy as np

from open_event_intel.etl_processing.database_interface import (
    ChunkEmbeddingRow,
    ChunkNoveltyRow,
    ChunkNoveltyScoreRow,
    ChunkRow,
    DatabaseInterface,
    DocumentFingerprintRow,
    FacetAssignmentRow,
    MentionRow,
    NoveltyLabelEvidenceRow,
    NoveltyLabelRow,
    StoryClusterMemberRow,
    StoryClusterRow,
    _serialize_json,
)
from open_event_intel.logger import get_logger

STAGE_NAME = "stage_07_novelty"
CLUSTER_STAGE_NAME = "stage_07_story_cluster"
PREREQUISITE_STAGES = ("stage_05_embeddings", "stage_06_taxonomy")

logger = get_logger(__name__)

def _bytes_to_vector(data: bytes, dim: int) -> np.ndarray:
    """
    Deserialise a little-endian float32 blob to a numpy array.

    :param data: Raw bytes (length must equal ``dim * 4``).
    :param dim: Expected number of float32 elements.
    :raises ValueError: If byte length does not match expected dimension.
    """
    expected_bytes = dim * 4
    if len(data) != expected_bytes:
        raise ValueError(
            f"Embedding blob length {len(data)} does not match expected "
            f"{expected_bytes} bytes for dim={dim}"
        )
    return np.frombuffer(data, dtype="<f4").copy()

class Stage07DatabaseInterface(DatabaseInterface):
    """
    Database adapter for Stage 07.

    Extends the base class with SQL helpers for novelty-specific tables.
    All SQL lives here (or in the base class) per project convention.

    .. note::
       Insert/read methods below should be migrated into
       ``DatabaseInterface`` in a future consolidation pass.
    """

    READS = {
        "document_version",
        "document",
        "chunk",
        "chunk_embedding",
        "mention",
        "facet_assignment",
        "doc_stage_status",
        "run_stage_status",
        "pipeline_run",
        "novelty_label",
        "document_fingerprint",
        "chunk_novelty",
        "chunk_novelty_score",
        "novelty_label_evidence",
        "story_cluster",
        "story_cluster_member",
        "evidence_span",
    }
    WRITES = {
        "novelty_label",
        "novelty_label_evidence",
        "document_fingerprint",
        "chunk_novelty",
        "chunk_novelty_score",
        "evidence_span",
        "doc_stage_status",
        "run_stage_status",
        "story_cluster",
        "story_cluster_member",
    }

    def __init__(
        self,
        working_db_path: Path,
        source_db_path: Path | None = None,
    ) -> None:
        """Initialize a :class:`Stage07DatabaseInterface`."""
        super().__init__(working_db_path, source_db_path, STAGE_NAME)

    def get_iteration_set(self) -> list[str]:
        """
        Return ``doc_version_id`` values requiring (re)processing.

        Per §6.3.0 this is the union of:
        * documents with no status row for this stage,
        * documents with ``status='failed'``,
        * documents with ``status='blocked'`` whose prerequisites are now
          all ``ok``.

        Results are ordered by ``(publisher_id, url_normalized, doc_version_id)``
        per §6.4.
        """
        self._check_read_access("doc_stage_status")
        self._check_read_access("document_version")
        self._check_read_access("document")
        rows = self._fetchall(
            """
            SELECT dv.doc_version_id
            FROM document_version dv
            JOIN document d ON d.document_id = dv.document_id
            WHERE (
                -- no status row yet
                NOT EXISTS (
                    SELECT 1 FROM doc_stage_status dss
                    WHERE dss.doc_version_id = dv.doc_version_id
                      AND dss.stage = ?
                )
                OR
                -- failed: always retry
                EXISTS (
                    SELECT 1 FROM doc_stage_status dss
                    WHERE dss.doc_version_id = dv.doc_version_id
                      AND dss.stage = ?
                      AND dss.status = 'failed'
                )
                OR
                -- blocked but prereqs now all ok
                (
                    EXISTS (
                        SELECT 1 FROM doc_stage_status dss
                        WHERE dss.doc_version_id = dv.doc_version_id
                          AND dss.stage = ?
                          AND dss.status = 'blocked'
                    )
                    AND EXISTS (
                        SELECT 1 FROM doc_stage_status p5
                        WHERE p5.doc_version_id = dv.doc_version_id
                          AND p5.stage = 'stage_05_embeddings'
                          AND p5.status = 'ok'
                    )
                    AND EXISTS (
                        SELECT 1 FROM doc_stage_status p6
                        WHERE p6.doc_version_id = dv.doc_version_id
                          AND p6.stage = 'stage_06_taxonomy'
                          AND p6.status = 'ok'
                    )
                )
            )
            ORDER BY d.publisher_id, d.url_normalized, dv.doc_version_id
            """,
            (STAGE_NAME, STAGE_NAME, STAGE_NAME),
        )
        result = [r["doc_version_id"] for r in rows]
        logger.info("get_iteration_set: %d documents require (re)processing for %s", len(result), STAGE_NAME)
        return result

    def check_prerequisites(self, doc_version_id: str) -> tuple[bool, str | None]:
        """
        Check whether all prerequisite stages are ``ok``.

        :return: ``(all_ok, blocking_message | None)``
        """
        for prereq in PREREQUISITE_STAGES:
            status_row = self.get_doc_stage_status(doc_version_id, prereq)
            if status_row is None or status_row.status != "ok":
                actual = "missing" if status_row is None else status_row.status
                msg = f"prerequisite_not_ok:{prereq}:{actual}"
                return False, msg
        return True, None

    def get_chunks_by_doc(self, doc_version_id: str) -> list[ChunkRow]:
        """Retrieve chunks for a document ordered by span_start."""
        return self.get_chunks_by_doc_version_id(doc_version_id)

    def get_chunk_embeddings_by_doc(
        self, doc_version_id: str
    ) -> list[ChunkEmbeddingRow]:
        """Retrieve chunk embeddings for a document."""
        self._check_read_access("chunk_embedding")
        rows = self._fetchall(
            """
            SELECT ce.* FROM chunk_embedding ce
            JOIN chunk c ON c.chunk_id = ce.chunk_id
            WHERE c.doc_version_id = ?
            ORDER BY c.span_start
            """,
            (doc_version_id,),
        )
        return [ChunkEmbeddingRow.model_validate(dict(r)) for r in rows]

    def get_mentions_by_doc(self, doc_version_id: str) -> list[MentionRow]:
        """Retrieve mentions for a document."""
        return self.get_mentions_by_doc_version_id(doc_version_id)

    def get_facets_by_doc(self, doc_version_id: str) -> list[FacetAssignmentRow]:
        """Retrieve facet assignments for a document."""
        self._check_read_access("facet_assignment")
        rows = self._fetchall(
            "SELECT * FROM facet_assignment WHERE doc_version_id = ? ORDER BY facet_id",
            (doc_version_id,),
        )
        return [FacetAssignmentRow.model_validate(dict(r)) for r in rows]

    def get_all_doc_embeddings_centroids(
        self,
    ) -> list[tuple[str, np.ndarray]]:
        r"""
        Return ``(doc_version_id, centroid_vector)`` for every document whose ``stage_07_novelty`` is already ``ok`` *or* whose ``stage_05_embeddings`` is ``ok``.

        Only docs with at least one chunk embedding are included.
        """
        self._check_read_access("chunk_embedding")
        self._check_read_access("chunk")
        rows = self._fetchall(
            """
            SELECT c.doc_version_id, ce.embedding_vector, ce.embedding_dim
            FROM chunk_embedding ce
            JOIN chunk c ON c.chunk_id = ce.chunk_id
            ORDER BY c.doc_version_id, c.span_start
            """
        )
        if not rows:
            logger.warning("get_all_doc_embeddings_centroids: no chunk_embedding rows found")
            return []
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for r in rows:
            vec = _bytes_to_vector(r["embedding_vector"], r["embedding_dim"])
            grouped[r["doc_version_id"]].append(vec)
        result: list[tuple[str, np.ndarray]] = []
        for dvid in sorted(grouped):
            centroid = np.mean(grouped[dvid], axis=0)
            result.append((dvid, centroid))
        logger.info("get_all_doc_embeddings_centroids: %d total chunk embeddings → %d doc centroids "
                     "(avg %.1f chunks/doc)",
                     len(rows), len(result), len(rows) / max(len(result), 1))
        return result

    def get_novelty_ok_doc_ids(self) -> list[str]:
        """Return ``doc_version_id`` values with ``stage_07_novelty='ok'``."""
        self._check_read_access("doc_stage_status")
        rows = self._fetchall(
            """
            SELECT doc_version_id FROM doc_stage_status
            WHERE stage = ? AND status = 'ok'
            ORDER BY doc_version_id
            """,
            (STAGE_NAME,),
        )
        result = [r["doc_version_id"] for r in rows]
        logger.info("get_novelty_ok_doc_ids: %d documents have stage_07_novelty='ok'", len(result))
        return result

    def delete_prior_doc_results(self, doc_version_id: str) -> int:
        """
        Delete any partial results from a prior failed processing attempt.

        This prevents UNIQUE constraint violations when retrying a document
        whose previous run inserted some rows before failing.

        Called inside the per-doc transaction, before inserting new results.

        :return: Total number of rows deleted across all tables.
        """
        deleted = 0

        # chunk_novelty_score (references chunk, not doc directly — join through chunk)
        self._check_write_access("chunk_novelty_score")
        cursor = self._execute(
            """DELETE FROM chunk_novelty_score WHERE chunk_id IN (
                SELECT c.chunk_id FROM chunk c WHERE c.doc_version_id = ?
            )""",
            (doc_version_id,),
        )
        deleted += cursor.rowcount

        # chunk_novelty (references chunk)
        self._check_write_access("chunk_novelty")
        cursor = self._execute(
            """DELETE FROM chunk_novelty WHERE chunk_id IN (
                SELECT c.chunk_id FROM chunk c WHERE c.doc_version_id = ?
            )""",
            (doc_version_id,),
        )
        deleted += cursor.rowcount

        # novelty_label_evidence
        self._check_write_access("novelty_label_evidence")
        cursor = self._execute(
            "DELETE FROM novelty_label_evidence WHERE doc_version_id = ?",
            (doc_version_id,),
        )
        deleted += cursor.rowcount

        # novelty_label
        self._check_write_access("novelty_label")
        cursor = self._execute(
            "DELETE FROM novelty_label WHERE doc_version_id = ?",
            (doc_version_id,),
        )
        deleted += cursor.rowcount

        # document_fingerprint
        self._check_write_access("document_fingerprint")
        cursor = self._execute(
            "DELETE FROM document_fingerprint WHERE doc_version_id = ?",
            (doc_version_id,),
        )
        deleted += cursor.rowcount

        if deleted > 0:
            logger.debug("delete_prior_doc_results: doc=%s removed %d rows from prior attempt",
                         doc_version_id[:12], deleted)
        return deleted

    def insert_novelty_label(self, row: NoveltyLabelRow) -> None:
        """Insert a novelty label row."""
        self._check_write_access("novelty_label")
        self._execute(
            """INSERT INTO novelty_label
            (doc_version_id, label, neighbor_doc_version_ids,
             similarity_score, shared_mentions, linking_window_days,
             confidence, created_in_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row.doc_version_id,
                row.label,
                _serialize_json(row.neighbor_doc_version_ids),
                row.similarity_score,
                _serialize_json(row.shared_mentions),
                row.linking_window_days,
                row.confidence,
                row.created_in_run_id,
            ),
        )

    def insert_novelty_label_evidence(self, row: NoveltyLabelEvidenceRow) -> None:
        """Insert a novelty-label ↔ evidence join row."""
        self._check_write_access("novelty_label_evidence")
        self._execute(
            "INSERT INTO novelty_label_evidence (doc_version_id, evidence_id, purpose) VALUES (?, ?, ?)",
            (row.doc_version_id, row.evidence_id, row.purpose),
        )

    def insert_document_fingerprint(self, row: DocumentFingerprintRow) -> None:
        """Insert a document fingerprint."""
        self._check_write_access("document_fingerprint")
        self._execute(
            """INSERT INTO document_fingerprint
            (doc_version_id, minhash_signature, simhash_signature, created_in_run_id)
            VALUES (?, ?, ?, ?)""",
            (
                row.doc_version_id,
                row.minhash_signature,
                row.simhash_signature,
                row.created_in_run_id,
            ),
        )

    def insert_chunk_novelty(self, row: ChunkNoveltyRow) -> None:
        """Insert a chunk novelty row."""
        self._check_write_access("chunk_novelty")
        self._execute(
            """INSERT INTO chunk_novelty
            (chunk_id, novelty_label, source_chunk_ids, similarity_scores)
            VALUES (?, ?, ?, ?)""",
            (
                row.chunk_id,
                row.novelty_label,
                _serialize_json(row.source_chunk_ids),
                _serialize_json(row.similarity_scores),
            ),
        )

    def insert_chunk_novelty_score(self, row: ChunkNoveltyScoreRow) -> None:
        """Insert a chunk novelty score."""
        self._check_write_access("chunk_novelty_score")
        self._execute(
            """INSERT INTO chunk_novelty_score
            (chunk_id, novelty_score, method, created_in_run_id)
            VALUES (?, ?, ?, ?)""",
            (row.chunk_id, row.novelty_score, row.method, row.created_in_run_id),
        )

    def insert_story_cluster(self, row: StoryClusterRow) -> None:
        """Insert a story cluster."""
        self._check_write_access("story_cluster")
        self._execute(
            """INSERT INTO story_cluster
            (run_id, story_id, cluster_method, seed_doc_version_id, summary_text)
            VALUES (?, ?, ?, ?, ?)""",
            (
                row.run_id,
                row.story_id,
                row.cluster_method,
                row.seed_doc_version_id,
                row.summary_text,
            ),
        )

    def insert_story_cluster_member(self, row: StoryClusterMemberRow) -> None:
        """Insert a story cluster member."""
        self._check_write_access("story_cluster_member")
        self._execute(
            """INSERT INTO story_cluster_member
            (run_id, story_id, doc_version_id, score, role)
            VALUES (?, ?, ?, ?, ?)""",
            (row.run_id, row.story_id, row.doc_version_id, row.score, row.role),
        )

    def get_novelty_label_distribution(self) -> list[dict]:
        """
        Return novelty label counts for documents with ``stage_07_novelty='ok'``.

        :return: List of dicts with keys ``label`` and ``cnt``, ordered by
            count descending.
        """
        self._check_read_access("novelty_label")
        self._check_read_access("doc_stage_status")
        rows = self._fetchall(
            """SELECT nl.label, COUNT(*) as cnt
               FROM novelty_label nl
               JOIN doc_stage_status dss
                 ON dss.doc_version_id = nl.doc_version_id
                AND dss.stage = ? AND dss.status = 'ok'
               GROUP BY nl.label ORDER BY cnt DESC""",
            (STAGE_NAME,),
        )
        return [dict(r) for r in rows]

    def delete_story_clusters_for_run(self, run_id: str) -> int:
        """Delete story clusters for a run (FK-safe order)."""
        self._check_write_access("story_cluster_member")
        self._check_write_access("story_cluster")
        self._execute(
            "DELETE FROM story_cluster_member WHERE run_id = ?", (run_id,)
        )
        cursor = self._execute(
            "DELETE FROM story_cluster WHERE run_id = ?", (run_id,)
        )
        return cursor.rowcount