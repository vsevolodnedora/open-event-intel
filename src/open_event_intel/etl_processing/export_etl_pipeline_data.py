#!/usr/bin/env python3
"""
Export processed_posts.db → SQLite databases for Run Explorer.

Reads the working database via :class:`SourceDatabaseReader` (inherits from
:class:`DatabaseInterface`) and writes compact, frontend-friendly SQLite files
via :class:`ExportDatabaseWriter` so GitHub Pages can serve them to sql.js in
the browser.

Output layout::

    etl_data/sqlite/
        catalog.sqlite    — meta, docs, stage status, totals (always loaded)
        traces.sqlite     — per-doc trace headers + stage samples (lazy)
        runs/
            <run_id>.sqlite — run status, new counts, artifact counts, impact (lazy)

Usage::

    python export_etl_pipeline_data.py --db path/to/processed_posts.db --out docs/etl_data
"""
import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from open_event_intel.etl_processing.processed_posts_db_interface import DatabaseInterface
from open_event_intel.logger import get_logger

logger = get_logger(__name__)

EXPORT_VERSION = "2.0.0"
SAMPLE_LIMIT = 5
RELEASE_SCHEMA_PATH = Path(__file__).resolve().parent / "release_database_schema.sql"


# Pydantic models

class StageDefinition(BaseModel):
    """A single pipeline stage descriptor used for export ordering."""

    model_config = ConfigDict(frozen=True)

    stage_id: str
    scope: Literal["doc", "run"]
    label: str


class ExportConfig(BaseModel):
    """Validated CLI / caller configuration for the export."""

    model_config = ConfigDict(frozen=True)

    db_path: Path
    out_dir: Path
    max_runs: int | None = None
    skip_traces: bool = False
    completed_only: bool = False
    verbose: bool = False

    @field_validator("db_path")
    @classmethod
    def _db_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Database not found: {v}")
        return v


class RunRecord(BaseModel):
    """Pipeline run row enriched with the previous-completed-run link."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    config_version: str | None = None
    prev_completed_run_id: str | None = None


class DocRecord(BaseModel):
    """Document row projected for export (joined from several source tables)."""

    model_config = ConfigDict(frozen=True)

    doc_version_id: str
    document_id: str | None = None
    publisher_id: str | None = None
    title: str | None = None
    url_normalized: str | None = None
    source_published_at: str | None = None
    created_in_run_id: str | None = None
    content_quality_score: float | None = None
    primary_language: str | None = None


class PublisherRecord(BaseModel):
    """A distinct publisher id."""

    model_config = ConfigDict(frozen=True)

    publisher_id: str


class DocStageEntry(BaseModel):
    """Single doc-stage status row as read from the source DB."""

    stage: str
    status: str | None = None
    attempt: int | None = None
    last_run_id: str | None = None
    processed_at: str | None = None
    error_message: str | None = None
    details: str | None = None


class RunStageEntry(BaseModel):
    """Run-level stage status entry."""

    stage_id: str
    status: str | None = None
    attempt: int | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None
    details: str | None = None


class ArtifactCounts(BaseModel):
    """Aggregate artifact counts for a single run."""

    embedding_indexes: int = 0
    chunk_count_total: int = 0
    clusters: int = 0
    cluster_memberships: int = 0
    metric_series: int = 0
    metric_points: int = 0
    alerts: int = 0
    digest_items: int = 0
    timeline_items: int = 0
    validation_failures: int = 0


class CatalogResult(BaseModel):
    """Value object returned after writing catalog.sqlite."""

    export_version: str
    generated_at: str
    stages: list[StageDefinition]
    runs: list[RunRecord]
    publishers: list[PublisherRecord]
    docs: list[DocRecord]


# Stage definitions

DOC_STAGES: list[StageDefinition] = [
    StageDefinition(stage_id="stage_01_ingest",     scope="doc", label="01 Ingest"),
    StageDefinition(stage_id="stage_02_parse",      scope="doc", label="02 Parse"),
    StageDefinition(stage_id="stage_03_metadata",   scope="doc", label="03 Metadata"),
    StageDefinition(stage_id="stage_04_mentions",   scope="doc", label="04 Mentions"),
    StageDefinition(stage_id="stage_05_embeddings", scope="doc", label="05 Embeddings"),
    StageDefinition(stage_id="stage_06_taxonomy",   scope="doc", label="06 Taxonomy"),
    StageDefinition(stage_id="stage_07_novelty",    scope="doc", label="07 Novelty"),
    StageDefinition(stage_id="stage_08_events",     scope="doc", label="08 Events"),
]

RUN_STAGES: list[StageDefinition] = [
    StageDefinition(stage_id="stage_05_embeddings_index", scope="run", label="05 Index"),
    StageDefinition(stage_id="stage_07_story_cluster",    scope="run", label="07 Cluster"),
    StageDefinition(stage_id="stage_09_outputs",          scope="run", label="09 Outputs"),
    StageDefinition(stage_id="stage_10_timeline",         scope="run", label="10 Timeline"),
    StageDefinition(stage_id="stage_11_validation",       scope="run", label="11 Validation"),
]

ALL_STAGES: list[StageDefinition] = DOC_STAGES + RUN_STAGES
DOC_STAGE_IDS: list[str] = [s.stage_id for s in DOC_STAGES]
RUN_STAGE_IDS: list[str] = [s.stage_id for s in RUN_STAGES]

_ARTIFACT_STAGE_QUERIES: dict[str, list[tuple[str, str]]] = {
    "stage_05_embeddings_index": [
        ("embedding_index", "SELECT COUNT(*) FROM embedding_index WHERE run_id = ?"),
    ],
    "stage_07_story_cluster": [
        ("story_cluster", "SELECT COUNT(*) FROM story_cluster WHERE run_id = ?"),
        ("story_cluster_member", "SELECT COUNT(*) FROM story_cluster_member WHERE run_id = ?"),
    ],
    "stage_10_timeline": [
        ("entity_timeline_item", "SELECT COUNT(*) FROM entity_timeline_item WHERE run_id = ?"),
    ],
    "stage_11_validation": [
        ("validation_failure", "SELECT COUNT(*) FROM validation_failure WHERE run_id = ?"),
    ],
}


# Utilities

def _j(obj: Any) -> str:
    """Serialize *obj* to compact JSON text."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, default=str)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def _rows_to_dicts(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [_row_to_dict(r) for r in cursor.fetchall()]


# Source database reader

class SourceDatabaseReader(DatabaseInterface):
    """
    Read-only access to ``processed_posts.db`` for export.

    Inherits connection management, stage-isolation checks, and
    transaction helpers from :class:`DatabaseInterface`.  The connection
    is opened in read-only mode because the export never mutates the
    working database.
    """

    READS: ClassVar[set[str]] = {
        "pipeline_run", "document", "document_version", "scrape_record",
        "doc_metadata", "doc_stage_status", "run_stage_status",
        "block", "chunk", "table_extract", "chunk_embedding", "embedding_index",
        "mention", "mention_link", "entity_registry",
        "facet_assignment", "novelty_label",
        "metric_observation", "event_candidate", "event_revision", "event",
        "story_cluster", "story_cluster_member",
        "metric_series", "metric_series_point",
        "alert", "digest_item",
        "entity_timeline_item", "validation_failure",
    }
    WRITES: ClassVar[set[str]] = set()

    def __init__(self, db_path: Path) -> None:
        super().__init__(
            working_db_path=db_path,
            source_db_path=None,
            stage_name="export_reader",
        )

    def open(self) -> None:
        """Open a **read-only** connection to the working database."""
        uri = f"file:{self._working_db_path}?mode=ro"
        self._working_conn = sqlite3.connect(uri, uri=True)
        self._working_conn.row_factory = sqlite3.Row
        self._working_conn.execute("PRAGMA foreign_keys = ON")
        self._validate_schema()

    # Runs

    def fetch_runs(self) -> list[RunRecord]:
        """Return all pipeline runs ordered newest-first, with prev-completed link."""
        rows = self._fetchall("""
            SELECT run_id, status, started_at, completed_at, config_version
            FROM pipeline_run
            ORDER BY CASE WHEN completed_at IS NOT NULL
                          THEN completed_at ELSE started_at END DESC
        """)

        completed_ids: list[str] = [
            r["run_id"] for r in rows if r["status"] == "completed"
        ]
        completed_index = {rid: i for i, rid in enumerate(completed_ids)}

        result: list[RunRecord] = []
        for row in rows:
            d = _row_to_dict(row)
            prev: str | None = None
            if d["run_id"] in completed_index:
                idx = completed_index[d["run_id"]]
                if idx + 1 < len(completed_ids):
                    prev = completed_ids[idx + 1]
            result.append(RunRecord(
                run_id=d["run_id"],
                status=d["status"],
                started_at=d["started_at"],
                completed_at=d["completed_at"],
                config_version=d["config_version"],
                prev_completed_run_id=prev,
            ))
        return result

    # Publishers

    def fetch_publishers(self) -> list[PublisherRecord]:
        """Return distinct publisher IDs in sorted order."""
        rows = self._fetchall(
            "SELECT DISTINCT publisher_id FROM document ORDER BY publisher_id"
        )
        return [PublisherRecord(publisher_id=r["publisher_id"]) for r in rows]

    # Documents

    def fetch_documents(self) -> list[DocRecord]:
        """Return documents projected for the catalog export."""
        rows = self._fetchall("""
            SELECT dv.doc_version_id, dv.document_id, d.publisher_id,
                   COALESCE(dm.title, sr.source_title) AS title,
                   d.url_normalized, d.source_published_at,
                   dv.created_in_run_id, dv.content_quality_score, dv.primary_language
            FROM document_version dv
            JOIN document d ON d.document_id = dv.document_id
            JOIN scrape_record sr ON sr.scrape_id = dv.scrape_id
            LEFT JOIN doc_metadata dm ON dm.doc_version_id = dv.doc_version_id
            ORDER BY d.publisher_id, d.source_published_at DESC
        """)
        return [DocRecord.model_validate(_row_to_dict(r)) for r in rows]

    # Doc stage status

    def fetch_doc_stage_status(
        self, dvid: str, stage_id: str
    ) -> DocStageEntry | None:
        """Return the doc-stage status row, or *None* if absent."""
        row = self._fetchone("""
            SELECT stage, status, attempt, run_id AS last_run_id, processed_at,
                   error_message, details
            FROM doc_stage_status WHERE doc_version_id = ? AND stage = ?
        """, (dvid, stage_id))
        return DocStageEntry.model_validate(_row_to_dict(row)) if row else None

    # Doc totals (one entry per doc-stage)

    def fetch_doc_totals(self, dvid: str) -> list[dict[str, Any] | None]:
        """
        Return per-stage aggregate totals for one document.

        Index positions correspond to :data:`DOC_STAGES`.
        """
        totals: list[dict[str, Any] | None] = []

        row = self._fetchone(
            "SELECT content_length_raw, content_length_clean "
            "FROM document_version WHERE doc_version_id = ?",
            (dvid,),
        )
        totals.append({
            "content_length_raw": row["content_length_raw"] if row else None,
            "content_length_clean": row["content_length_clean"] if row else None,
        })

        row = self._fetchone("""
            SELECT
                (SELECT COUNT(*) FROM block WHERE doc_version_id = ?) AS blocks,
                (SELECT COUNT(*) FROM chunk WHERE doc_version_id = ?) AS chunks,
                (SELECT COUNT(*) FROM table_extract WHERE doc_version_id = ?) AS tables
        """, (dvid, dvid, dvid))
        totals.append({"blocks": row["blocks"], "chunks": row["chunks"], "tables": row["tables"]})

        row = self._fetchone(
            "SELECT COUNT(*) AS n FROM doc_metadata WHERE doc_version_id = ?", (dvid,))
        totals.append({"metadata_rows": row["n"]})

        row = self._fetchone("""
            SELECT
                (SELECT COUNT(*) FROM mention WHERE doc_version_id = ?) AS mentions,
                (SELECT COUNT(*) FROM mention_link ml
                 JOIN mention m ON m.mention_id = ml.mention_id
                 WHERE m.doc_version_id = ?) AS mention_links
        """, (dvid, dvid))
        totals.append({"mentions": row["mentions"], "mention_links": row["mention_links"]})

        row = self._fetchone("""
            SELECT COUNT(*) AS n FROM chunk_embedding ce
            JOIN chunk c ON c.chunk_id = ce.chunk_id WHERE c.doc_version_id = ?
        """, (dvid,))
        totals.append({"embeddings": row["n"]})

        row = self._fetchone(
            "SELECT COUNT(*) AS n FROM facet_assignment WHERE doc_version_id = ?", (dvid,))
        totals.append({"facet_assignments": row["n"]})

        row = self._fetchone(
            "SELECT label FROM novelty_label WHERE doc_version_id = ?", (dvid,))
        totals.append({"novelty_label": row["label"] if row else None})

        row = self._fetchone("""
            SELECT
                (SELECT COUNT(*) FROM metric_observation WHERE doc_version_id = ?) AS metric_observations,
                (SELECT COUNT(*) FROM event_candidate WHERE doc_version_id = ?) AS event_candidates,
                (SELECT COUNT(*) FROM event_revision er
                 WHERE json_valid(er.doc_version_ids)
                 AND EXISTS (SELECT 1 FROM json_each(er.doc_version_ids) j WHERE j.value = ?)
                ) AS event_revisions_contributed
        """, (dvid, dvid, dvid))
        totals.append({
            "metric_observations": row["metric_observations"],
            "event_candidates": row["event_candidates"],
            "event_revisions_contributed": row["event_revisions_contributed"],
        })

        return totals

    # Trace header + samples

    def fetch_trace_header(self, dvid: str) -> dict[str, Any] | None:
        """Return the trace header dict for one document, or *None*."""
        row = self._fetchone("""
            SELECT dv.doc_version_id, dv.document_id, d.publisher_id,
                   COALESCE(dm.title, sr.source_title) AS title,
                   d.url_normalized, d.source_published_at,
                   dv.content_quality_score, dv.primary_language
            FROM document_version dv
            JOIN document d ON d.document_id = dv.document_id
            JOIN scrape_record sr ON sr.scrape_id = dv.scrape_id
            LEFT JOIN doc_metadata dm ON dm.doc_version_id = dv.doc_version_id
            WHERE dv.doc_version_id = ?
        """, (dvid,))
        return _row_to_dict(row) if row else None

    def _sample_rows(
        self, sql: str, params: tuple[Any, ...], limit: int = SAMPLE_LIMIT
    ) -> list[dict[str, Any]]:
        rows = self._fetchall(f"{sql} LIMIT ?", (*params, limit))
        return [_row_to_dict(r) for r in rows]

    def fetch_trace_samples(
        self, dvid: str
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Return stage→table→rows sample map for one document."""
        m: dict[str, dict[str, list[dict[str, Any]]]] = {}

        m["stage_01_ingest"] = {
            "document_version": self._sample_rows(
                """SELECT doc_version_id, document_id, content_hash_clean,
                          content_length_raw, content_length_clean, boilerplate_ratio,
                          content_quality_score, primary_language, cleaning_spec_version,
                          created_in_run_id
                   FROM document_version WHERE doc_version_id = ?""", (dvid,)),
            "scrape_record": self._sample_rows(
                """SELECT sr.scrape_id, sr.publisher_id, sr.source_id, sr.url_raw,
                          sr.scraped_at, sr.source_published_at, sr.source_title,
                          sr.source_language, sr.scrape_kind, sr.processing_status
                   FROM scrape_record sr
                   JOIN document_version dv ON dv.scrape_id = sr.scrape_id
                   WHERE dv.doc_version_id = ?""", (dvid,)),
        }

        m["stage_02_parse"] = {
            "blocks": self._sample_rows(
                "SELECT block_id, block_type, block_level, span_start, span_end "
                "FROM block WHERE doc_version_id = ? ORDER BY span_start", (dvid,)),
            "chunks": self._sample_rows(
                "SELECT chunk_id, chunk_type, span_start, span_end, token_count_approx, heading_context "
                "FROM chunk WHERE doc_version_id = ? ORDER BY span_start", (dvid,)),
            "tables": self._sample_rows(
                "SELECT table_id, row_count, col_count, table_class, parse_method "
                "FROM table_extract WHERE doc_version_id = ? ORDER BY table_id", (dvid,)),
        }

        m["stage_03_metadata"] = {
            "metadata": self._sample_rows(
                "SELECT title, published_at, detected_document_class, title_source, published_at_source "
                "FROM doc_metadata WHERE doc_version_id = ?", (dvid,)),
        }

        m["stage_04_mentions"] = {
            "mentions": self._sample_rows(
                "SELECT mention_id, mention_type, surface_form, normalized_value, confidence, extraction_method "
                "FROM mention WHERE doc_version_id = ? ORDER BY span_start", (dvid,)),
            "mention_links": self._sample_rows(
                """SELECT ml.link_id, ml.entity_id, er.canonical_name, ml.link_confidence
                   FROM mention_link ml
                   JOIN mention m ON m.mention_id = ml.mention_id
                   LEFT JOIN entity_registry er ON er.entity_id = ml.entity_id
                   WHERE m.doc_version_id = ? ORDER BY ml.link_confidence DESC""", (dvid,)),
        }

        m["stage_05_embeddings"] = {
            "chunk_embeddings": self._sample_rows(
                """SELECT ce.chunk_id, ce.embedding_dim, ce.model_version,
                          ce.language_used, ce.created_in_run_id
                   FROM chunk_embedding ce
                   JOIN chunk c ON c.chunk_id = ce.chunk_id
                   WHERE c.doc_version_id = ? ORDER BY ce.chunk_id""", (dvid,)),
        }

        m["stage_06_taxonomy"] = {
            "facets": self._sample_rows(
                "SELECT facet_id, facet_type, facet_value, confidence "
                "FROM facet_assignment WHERE doc_version_id = ? ORDER BY confidence DESC", (dvid,)),
        }

        m["stage_07_novelty"] = {
            "novelty": self._sample_rows(
                "SELECT label, similarity_score, confidence "
                "FROM novelty_label WHERE doc_version_id = ?", (dvid,)),
        }

        m["stage_08_events"] = {
            "metric_observations": self._sample_rows(
                "SELECT metric_id, metric_name, value_raw, unit_raw, value_norm, period_start "
                "FROM metric_observation WHERE doc_version_id = ? ORDER BY metric_name", (dvid,)),
            "event_candidates": self._sample_rows(
                "SELECT candidate_id, event_type, confidence, status, rejection_reason "
                "FROM event_candidate WHERE doc_version_id = ? ORDER BY confidence DESC", (dvid,)),
            "event_revisions": self._sample_rows(
                """SELECT er.revision_id, er.event_id, e.canonical_key, er.revision_no, er.confidence
                   FROM event_revision er
                   JOIN event e ON e.event_id = er.event_id
                   WHERE json_valid(er.doc_version_ids)
                   AND EXISTS (SELECT 1 FROM json_each(er.doc_version_ids) j WHERE j.value = ?)
                   ORDER BY er.created_at DESC""", (dvid,)),
        }

        return m

    # Run-level stage status

    def fetch_run_stage_statuses(
        self, run_id: str
    ) -> dict[str, RunStageEntry]:
        """Return a ``{stage_id: RunStageEntry}`` mapping for *run_id*."""
        rows = self._fetchall("""
            SELECT stage, status, attempt, started_at, completed_at,
                   error_message, details
            FROM run_stage_status WHERE run_id = ?
        """, (run_id,))
        result: dict[str, RunStageEntry] = {}
        for r in rows:
            d = _row_to_dict(r)
            sid = d.pop("stage")
            d["stage_id"] = sid
            result[sid] = RunStageEntry.model_validate(d)
        return result

    def synthesize_artifact_status(
        self, stage_id: str, run_id: str
    ) -> RunStageEntry | None:
        """
        Build a synthetic status from artifact table counts.

        Returns *None* when the stage has no artifact queries or
        all counts are zero.
        """
        queries = _ARTIFACT_STAGE_QUERIES.get(stage_id)
        if not queries:
            return None
        total = 0
        table_counts: dict[str, int] = {}
        for table_name, sql in queries:
            row = self._fetchone(sql, (run_id,))
            c = row[0] if row else 0
            table_counts[table_name] = c
            total += c
        if total == 0:
            return None
        return RunStageEntry(
            stage_id=stage_id,
            status="ok",
            attempt=1,
            started_at=None,
            completed_at=None,
            error_message=None,
            details=json.dumps({"synthesized_from_artifacts": table_counts}),
        )

    # Artifact counts

    def fetch_artifact_counts(self, run_id: str) -> ArtifactCounts:
        """Return aggregate artifact counts for *run_id*."""
        def _scalar(sql: str) -> int:
            row = self._fetchone(sql, (run_id,))
            return row[0] if row else 0

        return ArtifactCounts(
            embedding_indexes=_scalar(
                "SELECT COUNT(*) FROM embedding_index WHERE run_id = ?"),
            chunk_count_total=_scalar(
                "SELECT COALESCE(SUM(chunk_count), 0) FROM embedding_index WHERE run_id = ?"),
            clusters=_scalar(
                "SELECT COUNT(*) FROM story_cluster WHERE run_id = ?"),
            cluster_memberships=_scalar(
                "SELECT COUNT(*) FROM story_cluster_member WHERE run_id = ?"),
            metric_series=_scalar(
                "SELECT COUNT(*) FROM metric_series WHERE run_id = ?"),
            metric_points=_scalar(
                "SELECT COUNT(*) FROM metric_series_point WHERE run_id = ?"),
            alerts=_scalar(
                "SELECT COUNT(*) FROM alert WHERE run_id = ?"),
            digest_items=_scalar(
                "SELECT COUNT(*) FROM digest_item WHERE run_id = ?"),
            timeline_items=_scalar(
                "SELECT COUNT(*) FROM entity_timeline_item WHERE run_id = ?"),
            validation_failures=_scalar(
                "SELECT COUNT(*) FROM validation_failure WHERE run_id = ?"),
        )

    # New counts per doc per run

    def fetch_new_counts_for_doc(
        self, dvid: str, run_id: str
    ) -> list[dict[str, Any] | None]:
        """
        Return per-stage new-in-run counts for one document.

        Index positions correspond to :data:`DOC_STAGES`.
        """
        counts: list[dict[str, Any] | None] = []

        row = self._fetchone(
            "SELECT COUNT(*) AS n FROM document_version "
            "WHERE doc_version_id = ? AND created_in_run_id = ?",
            (dvid, run_id))
        counts.append({"new_doc": row["n"]})

        row = self._fetchone("""
            SELECT
                (SELECT COUNT(*) FROM block WHERE doc_version_id = ? AND created_in_run_id = ?) AS blocks,
                (SELECT COUNT(*) FROM chunk WHERE doc_version_id = ? AND created_in_run_id = ?) AS chunks,
                (SELECT COUNT(*) FROM table_extract WHERE doc_version_id = ? AND created_in_run_id = ?) AS tables
        """, (dvid, run_id, dvid, run_id, dvid, run_id))
        counts.append({"blocks": row["blocks"], "chunks": row["chunks"], "tables": row["tables"]})

        row = self._fetchone(
            "SELECT COUNT(*) AS n FROM doc_metadata "
            "WHERE doc_version_id = ? AND created_in_run_id = ?",
            (dvid, run_id))
        counts.append({"metadata_rows": row["n"]})

        row = self._fetchone("""
            SELECT
                (SELECT COUNT(*) FROM mention WHERE doc_version_id = ? AND created_in_run_id = ?) AS mentions,
                (SELECT COUNT(*) FROM mention_link ml
                 JOIN mention m ON m.mention_id = ml.mention_id
                 WHERE m.doc_version_id = ? AND ml.created_in_run_id = ?) AS mention_links
        """, (dvid, run_id, dvid, run_id))
        counts.append({"mentions": row["mentions"], "mention_links": row["mention_links"]})

        row = self._fetchone("""
            SELECT COUNT(*) AS n FROM chunk_embedding ce
            JOIN chunk c ON c.chunk_id = ce.chunk_id
            WHERE c.doc_version_id = ? AND ce.created_in_run_id = ?
        """, (dvid, run_id))
        counts.append({"embeddings": row["n"]})

        row = self._fetchone(
            "SELECT COUNT(*) AS n FROM facet_assignment "
            "WHERE doc_version_id = ? AND created_in_run_id = ?",
            (dvid, run_id))
        counts.append({"facet_assignments": row["n"]})

        row = self._fetchone(
            "SELECT label FROM novelty_label "
            "WHERE doc_version_id = ? AND created_in_run_id = ?",
            (dvid, run_id))
        counts.append({"novelty_label": row["label"] if row else None})

        row = self._fetchone("""
            SELECT
                (SELECT COUNT(*) FROM metric_observation
                 WHERE doc_version_id = ? AND created_in_run_id = ?) AS metric_observations,
                (SELECT COUNT(*) FROM event_candidate
                 WHERE doc_version_id = ? AND created_in_run_id = ?) AS event_candidates,
                (SELECT COUNT(*) FROM event_revision er
                 WHERE er.created_in_run_id = ?
                 AND json_valid(er.doc_version_ids)
                 AND EXISTS (SELECT 1 FROM json_each(er.doc_version_ids) j WHERE j.value = ?)
                ) AS event_revisions_contributed
        """, (dvid, run_id, dvid, run_id, run_id, dvid))
        counts.append({
            "metric_observations": row["metric_observations"],
            "event_candidates": row["event_candidates"],
            "event_revisions_contributed": row["event_revisions_contributed"],
        })

        return counts

    # Impact data

    def fetch_impact_data(
        self, dvid: str, run_id: str
    ) -> dict[str, list[dict[str, Any]]]:
        """Return impact sample rows keyed by category for one document."""
        impact: dict[str, list[dict[str, Any]]] = {}

        impact["clusters"] = self._sample_rows(
            """SELECT sc.story_id, sc.summary_text, scm.score, scm.role
               FROM story_cluster_member scm
               JOIN story_cluster sc ON sc.run_id = scm.run_id AND sc.story_id = scm.story_id
               WHERE scm.run_id = ? AND scm.doc_version_id = ?""",
            (run_id, dvid))

        impact["metric_points"] = self._sample_rows(
            """SELECT msp.series_id, ms.metric_name, msp.period_start, msp.value_norm
               FROM metric_series_point msp
               JOIN metric_series ms ON ms.run_id = msp.run_id AND ms.series_id = msp.series_id
               WHERE msp.run_id = ? AND msp.source_doc_version_id = ?""",
            (run_id, dvid))

        impact["alerts"] = self._sample_rows(
            """SELECT alert_id, triggered_at, trigger_payload_json
               FROM alert WHERE run_id = ?
               AND json_valid(doc_version_ids)
               AND EXISTS (SELECT 1 FROM json_each(doc_version_ids) j WHERE j.value = ?)""",
            (run_id, dvid))

        impact["digest_items"] = self._sample_rows(
            """SELECT item_id, digest_date, section, item_type
               FROM digest_item WHERE run_id = ?
               AND json_valid(doc_version_ids)
               AND EXISTS (SELECT 1 FROM json_each(doc_version_ids) j WHERE j.value = ?)""",
            (run_id, dvid))

        impact["timeline_items"] = self._sample_rows(
            """SELECT item_id, entity_id, item_type, event_time, summary_text
               FROM entity_timeline_item WHERE run_id = ? AND ref_doc_version_id = ?""",
            (run_id, dvid))

        impact["validation_failures"] = self._sample_rows(
            """SELECT failure_id, stage, check_name, severity, details
               FROM validation_failure WHERE run_id = ? AND doc_version_id = ?""",
            (run_id, dvid))

        return impact


# Target (output) database writer

class ExportDatabaseWriter:
    """
    Writes compact SQLite files consumed by the Run Explorer frontend.

    Each public method creates one output file (catalog, traces, or a
    per-run database).  The caller supplies pre-fetched data read from
    :class:`SourceDatabaseReader`; this class is responsible only for
    schema creation, row insertion, and file finalisation.
    """

    def __init__(self, schema_path: Path = RELEASE_SCHEMA_PATH) -> None:
        self._schema_sql = schema_path.read_text(encoding="utf-8")

    @staticmethod
    def _create_db(path: Path) -> sqlite3.Connection:
        """Create a fresh output database at *path*."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    @staticmethod
    def _finalize(conn: sqlite3.Connection, path: Path) -> None:
        """VACUUM, switch to DELETE journal, close, and log size."""
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.execute("VACUUM")
        conn.close()
        size_kb = path.stat().st_size / 1024
        logger.info("Wrote %s (%.1f KB)", path, size_kb)

    def _init_catalog_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(self._schema_sql)

    @staticmethod
    def _init_traces_schema(conn: sqlite3.Connection) -> None:
        conn.executescript("""
            CREATE TABLE trace_header (
                doc_version_id TEXT PRIMARY KEY,
                header_json    TEXT NOT NULL
            );
            CREATE TABLE trace_sample (
                doc_version_id TEXT NOT NULL,
                stage_id       TEXT NOT NULL,
                table_name     TEXT NOT NULL,
                rows_json      TEXT NOT NULL,
                PRIMARY KEY (doc_version_id, stage_id, table_name)
            );
        """)

    @staticmethod
    def _init_run_schema(conn: sqlite3.Connection) -> None:
        conn.executescript("""
            CREATE TABLE run_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE run_stage_status (
                stage_index INTEGER PRIMARY KEY,
                status_json TEXT
            );
            CREATE TABLE new_counts (
                doc_index   INTEGER NOT NULL,
                stage_index INTEGER NOT NULL,
                counts_json TEXT,
                PRIMARY KEY (doc_index, stage_index)
            );
            CREATE TABLE run_artifact_counts (key TEXT PRIMARY KEY, value INTEGER);
            CREATE TABLE impact (
                doc_version_id TEXT NOT NULL,
                impact_key     TEXT NOT NULL,
                rows_json      TEXT NOT NULL,
                PRIMARY KEY (doc_version_id, impact_key)
            );
        """)

    # Catalog

    def write_catalog(
        self,
        out_path: Path,
        *,
        runs: list[RunRecord],
        publishers: list[PublisherRecord],
        docs: list[DocRecord],
        reader: SourceDatabaseReader,
    ) -> CatalogResult:
        """
        Write ``catalog.sqlite`` and return a :class:`CatalogResult`.

        :param out_path: File path for the output database.
        :param runs: Pre-fetched run records.
        :param publishers: Pre-fetched publisher records.
        :param docs: Pre-fetched document records.
        :param reader: Source reader used to fetch per-doc stage status and totals.
        """
        db = self._create_db(out_path)
        self._init_catalog_schema(db)

        generated_at = datetime.now(timezone.utc).isoformat()
        db.execute("INSERT INTO meta VALUES (?, ?)", ("export_version", EXPORT_VERSION))
        db.execute("INSERT INTO meta VALUES (?, ?)", ("generated_at", generated_at))

        for i, stage in enumerate(ALL_STAGES):
            db.execute(
                "INSERT INTO stage VALUES (?, ?, ?, ?)",
                (i, stage.stage_id, stage.scope, stage.label),
            )

        for run in runs:
            db.execute(
                "INSERT INTO run VALUES (?, ?, ?, ?, ?, ?)",
                (run.run_id, run.status, run.started_at, run.completed_at,
                 run.config_version, run.prev_completed_run_id),
            )

        for pub in publishers:
            db.execute("INSERT INTO publisher VALUES (?)", (pub.publisher_id,))

        for doc_index, doc in enumerate(docs):
            db.execute(
                "INSERT INTO doc VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (doc_index, doc.doc_version_id, doc.document_id, doc.publisher_id,
                 doc.title, doc.url_normalized, doc.source_published_at,
                 doc.created_in_run_id, doc.content_quality_score, doc.primary_language),
            )

            for si, stage_def in enumerate(DOC_STAGES):
                entry = reader.fetch_doc_stage_status(doc.doc_version_id, stage_def.stage_id)
                if entry is not None:
                    db.execute(
                        "INSERT INTO doc_stage_status VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (doc_index, si, entry.status, entry.attempt,
                         entry.last_run_id, entry.processed_at,
                         entry.error_message, entry.details),
                    )

            totals = reader.fetch_doc_totals(doc.doc_version_id)
            for si, counts in enumerate(totals):
                if counts:
                    db.execute(
                        "INSERT INTO doc_totals VALUES (?, ?, ?)",
                        (doc_index, si, _j(counts)),
                    )

        db.commit()

        logger.info(
            "  Runs: %d, Publishers: %d, Docs: %d",
            len(runs), len(publishers), len(docs),
        )
        self._finalize(db, out_path)

        return CatalogResult(
            export_version=EXPORT_VERSION,
            generated_at=generated_at,
            stages=list(ALL_STAGES),
            runs=runs,
            publishers=publishers,
            docs=docs,
        )

    # Traces

    def write_traces(
        self,
        out_path: Path,
        docs: list[DocRecord],
        reader: SourceDatabaseReader,
    ) -> None:
        """
        Write ``traces.sqlite`` with per-doc trace headers and stage samples.

        :param out_path: File path for the output database.
        :param docs: Document records whose traces to export.
        :param reader: Source reader for fetching trace data.
        """
        db = self._create_db(out_path)
        self._init_traces_schema(db)

        for i, doc in enumerate(docs):
            dvid = doc.doc_version_id
            header = reader.fetch_trace_header(dvid)
            if header is None:
                continue
            db.execute(
                "INSERT INTO trace_header VALUES (?, ?)", (dvid, _j(header))
            )

            samples_map = reader.fetch_trace_samples(dvid)
            for stage_id, tables in samples_map.items():
                for table_name, rows in tables.items():
                    if rows:
                        db.execute(
                            "INSERT INTO trace_sample VALUES (?, ?, ?, ?)",
                            (dvid, stage_id, table_name, _j(rows)),
                        )

            if (i + 1) % 100 == 0:
                logger.info("  traces: %d / %d", i + 1, len(docs))

        db.commit()
        self._finalize(db, out_path)

    # Per-run database

    def write_run_db(
        self,
        out_path: Path,
        run_id: str,
        docs: list[DocRecord],
        reader: SourceDatabaseReader,
        skip_traces: bool = False,
    ) -> None:
        """
        Write a per-run SQLite database.

        :param out_path: File path for the output database.
        :param run_id: Pipeline run to export.
        :param docs: All documents (index-aligned with catalog).
        :param reader: Source reader for run data.
        :param skip_traces: When *True*, skip impact data export.
        """
        db = self._create_db(out_path)
        self._init_run_schema(db)

        db.execute("INSERT INTO run_meta VALUES (?, ?)", ("run_id", run_id))

        run_stage_map = reader.fetch_run_stage_statuses(run_id)
        for ri, stage_def in enumerate(RUN_STAGES):
            sid = stage_def.stage_id
            entry = run_stage_map.get(sid)
            if entry is None and sid in _ARTIFACT_STAGE_QUERIES:
                entry = reader.synthesize_artifact_status(sid, run_id)
            if entry is not None:
                db.execute(
                    "INSERT INTO run_stage_status VALUES (?, ?)",
                    (ri, _j(entry.model_dump())),
                )

        artifact_counts = reader.fetch_artifact_counts(run_id)
        for key, value in artifact_counts.model_dump().items():
            db.execute("INSERT INTO run_artifact_counts VALUES (?, ?)", (key, value))

        docs_with_new = 0
        for doc_index, doc in enumerate(docs):
            stage_counts = reader.fetch_new_counts_for_doc(doc.doc_version_id, run_id)
            has_data = False
            for si, counts in enumerate(stage_counts):
                if counts:
                    if any(v not in (None, 0, "") for v in counts.values()):
                        has_data = True
                    db.execute(
                        "INSERT INTO new_counts VALUES (?, ?, ?)",
                        (doc_index, si, _j(counts)),
                    )
            if has_data:
                docs_with_new += 1

        logger.info("  %d/%d docs with new-in-run data", docs_with_new, len(docs))

        if not skip_traces:
            impact_count = 0
            for doc in docs:
                impact = reader.fetch_impact_data(doc.doc_version_id, run_id)
                for key, rows in impact.items():
                    if rows:
                        db.execute(
                            "INSERT INTO impact VALUES (?, ?, ?)",
                            (doc.doc_version_id, key, _j(rows)),
                        )
                        impact_count += 1
            logger.info("  %d impact entries written", impact_count)

        db.commit()
        self._finalize(db, out_path)


# CLI

def _parse_args(argv: Sequence[str] | None = None) -> ExportConfig:
    """Parse CLI arguments and return a validated :class:`ExportConfig`."""
    parser = argparse.ArgumentParser(
        description="Export processed_posts.db → SQLite databases for Run Explorer",
    )
    parser.add_argument(
        "--db",
        default=Path("../../../database/processed_posts.db"),
        type=Path,
        help="Path to processed_posts.db",
    )
    parser.add_argument(
        "--out",
        default=Path("../../../docs/etl_data/"),
        type=Path,
        help="Output directory",
    )
    parser.add_argument(
        "--runs", type=int, default=None,
        help="Max runs to export (default: all non-aborted)",
    )
    parser.add_argument("--skip-traces", action="store_true", help="Skip trace + impact export")
    parser.add_argument("--completed-only", action="store_true", help="Export only completed runs")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    return ExportConfig(
        db_path=args.db,
        out_dir=args.out,
        max_runs=args.runs,
        skip_traces=args.skip_traces,
        completed_only=args.completed_only,
        verbose=args.verbose,
    )


def run_export(config: ExportConfig) -> int:
    """
    Execute the full export pipeline.

    :param config: Validated export configuration.
    :return: Exit code (0 on success).
    """
    sqlite_dir = config.out_dir / "sqlite"
    reader = SourceDatabaseReader(config.db_path)
    writer = ExportDatabaseWriter()

    reader.open()
    try:
        logger.info("Exporting catalog.sqlite...")
        runs = reader.fetch_runs()
        publishers = reader.fetch_publishers()
        docs = reader.fetch_documents()

        catalog = writer.write_catalog(
            sqlite_dir / "catalog.sqlite",
            runs=runs,
            publishers=publishers,
            docs=docs,
            reader=reader,
        )
        logger.info("Exported %d documents", len(catalog.docs))

        if not config.skip_traces:
            logger.info("Exporting traces.sqlite (%d docs)...", len(docs))
            writer.write_traces(
                sqlite_dir / "traces.sqlite",
                docs=docs,
                reader=reader,
            )

        exportable_statuses = (
            {"completed"} if config.completed_only else {"completed", "running"}
        )
        exportable_runs = [
            r for r in catalog.runs if r.status in exportable_statuses
        ]
        if config.max_runs is not None:
            exportable_runs = exportable_runs[: config.max_runs]

        if not exportable_runs:
            logger.warning("No exportable runs found.")
        else:
            logger.info("Exporting %d run(s)...", len(exportable_runs))

        for run_info in exportable_runs:
            logger.info(
                "━━ Exporting run %s (status=%s) ━━",
                run_info.run_id, run_info.status,
            )
            writer.write_run_db(
                sqlite_dir / "runs" / f"{run_info.run_id}.sqlite",
                run_id=run_info.run_id,
                docs=docs,
                reader=reader,
                skip_traces=config.skip_traces,
            )

        logger.info("Export complete → %s", sqlite_dir)
        return 0
    finally:
        reader.close()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    config = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if config.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return run_export(config)


if __name__ == "__main__":
    sys.exit(main())