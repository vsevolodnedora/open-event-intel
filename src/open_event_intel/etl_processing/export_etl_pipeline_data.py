#!/usr/bin/env python3
"""
Export processed_posts.db → SQLite databases for Run Explorer.

Reads the working database and writes compact, frontend-friendly SQLite files
so GitHub Pages can serve them to sql.js in the browser.

Output layout::

    etl_data/sqlite/
        catalog.sqlite    — meta, docs, stage status, totals (always loaded)
        traces.sqlite     — per-doc trace headers + stage samples (lazy)
        runs/
            <run_id>.sqlite — run status, new counts, artifact counts, impact (lazy)

Usage:
    python export_etl_pipeline_data.py --db path/to/processed_posts.db --out docs/etl_data
"""
import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from open_event_intel.logger import get_logger

EXPORT_VERSION = "2.0.0"

DOC_STAGES: list[dict[str, str]] = [
    {"stage_id": "stage_01_ingest",     "scope": "doc", "label": "01 Ingest"},
    {"stage_id": "stage_02_parse",      "scope": "doc", "label": "02 Parse"},
    {"stage_id": "stage_03_metadata",   "scope": "doc", "label": "03 Metadata"},
    {"stage_id": "stage_04_mentions",   "scope": "doc", "label": "04 Mentions"},
    {"stage_id": "stage_05_embeddings", "scope": "doc", "label": "05 Embeddings"},
    {"stage_id": "stage_06_taxonomy",   "scope": "doc", "label": "06 Taxonomy"},
    {"stage_id": "stage_07_novelty",    "scope": "doc", "label": "07 Novelty"},
    {"stage_id": "stage_08_events",     "scope": "doc", "label": "08 Events"},
]

RUN_STAGES: list[dict[str, str]] = [
    {"stage_id": "stage_05_embeddings_index", "scope": "run", "label": "05 Index"},
    {"stage_id": "stage_07_story_cluster",    "scope": "run", "label": "07 Cluster"},
    {"stage_id": "stage_09_outputs",          "scope": "run", "label": "09 Outputs"},
    {"stage_id": "stage_10_timeline",         "scope": "run", "label": "10 Timeline"},
    {"stage_id": "stage_11_validation",       "scope": "run", "label": "11 Validation"},
]

ALL_STAGES = DOC_STAGES + RUN_STAGES
DOC_STAGE_IDS = [s["stage_id"] for s in DOC_STAGES]
RUN_STAGE_IDS = [s["stage_id"] for s in RUN_STAGES]

SAMPLE_LIMIT = 5

_ARTIFACT_STAGE_QUERIES: dict[str, list[tuple[str, str]]] = {
    "stage_05_embeddings_index": [
        ("embedding_index",  "SELECT COUNT(*) FROM embedding_index WHERE run_id = ?"),
    ],
    "stage_07_story_cluster": [
        ("story_cluster",        "SELECT COUNT(*) FROM story_cluster WHERE run_id = ?"),
        ("story_cluster_member", "SELECT COUNT(*) FROM story_cluster_member WHERE run_id = ?"),
    ],
    "stage_10_timeline": [
        ("entity_timeline_item", "SELECT COUNT(*) FROM entity_timeline_item WHERE run_id = ?"),
    ],
    "stage_11_validation": [
        ("validation_failure",   "SELECT COUNT(*) FROM validation_failure WHERE run_id = ?"),
    ],
}

log = get_logger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────

def _connect(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def _rows_to_list(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [_row_to_dict(r) for r in cursor.fetchall()]


def _sample_rows(conn: sqlite3.Connection, sql: str, params: tuple, limit: int = SAMPLE_LIMIT) -> list[dict]:
    rows = conn.execute(f"{sql} LIMIT ?", (*params, limit)).fetchall()
    return [_row_to_dict(r) for r in rows]


def _create_output_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    out = sqlite3.connect(str(path))
    out.execute("PRAGMA journal_mode = WAL")
    out.execute("PRAGMA synchronous = NORMAL")
    return out


def _finalize_db(conn: sqlite3.Connection, path: Path) -> None:
    conn.execute("PRAGMA journal_mode = DELETE")
    conn.execute("VACUUM")
    conn.close()
    size_kb = path.stat().st_size / 1024
    log.info("Wrote %s (%.1f KB)", path, size_kb)


def _j(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, default=str)


# ── Catalog.sqlite ──────────────────────────────────────────────────────

def _init_catalog_schema(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE stage (
            sort_order INTEGER PRIMARY KEY,
            stage_id   TEXT NOT NULL UNIQUE,
            scope      TEXT NOT NULL,
            label      TEXT NOT NULL
        );
        CREATE TABLE run (
            run_id                TEXT PRIMARY KEY,
            status                TEXT NOT NULL,
            started_at            TEXT,
            completed_at          TEXT,
            config_version        TEXT,
            prev_completed_run_id TEXT
        );
        CREATE TABLE publisher (publisher_id TEXT PRIMARY KEY);
        CREATE TABLE doc (
            doc_index             INTEGER PRIMARY KEY,
            doc_version_id        TEXT NOT NULL UNIQUE,
            document_id           TEXT,
            publisher_id          TEXT,
            title                 TEXT,
            url_normalized        TEXT,
            source_published_at   TEXT,
            created_in_run_id     TEXT,
            content_quality_score REAL,
            primary_language      TEXT
        );
        CREATE TABLE doc_stage_status (
            doc_index     INTEGER NOT NULL,
            stage_index   INTEGER NOT NULL,
            status        TEXT,
            attempt       INTEGER,
            last_run_id   TEXT,
            processed_at  TEXT,
            error_message TEXT,
            details       TEXT,
            PRIMARY KEY (doc_index, stage_index)
        );
        CREATE TABLE doc_totals (
            doc_index   INTEGER NOT NULL,
            stage_index INTEGER NOT NULL,
            counts_json TEXT,
            PRIMARY KEY (doc_index, stage_index)
        );
    """)


def export_catalog(conn: sqlite3.Connection, out_path: Path) -> tuple[dict, list[dict]]:
    db = _create_output_db(out_path)
    _init_catalog_schema(db)

    generated_at = datetime.now(timezone.utc).isoformat()
    db.execute("INSERT INTO meta VALUES (?, ?)", ("export_version", EXPORT_VERSION))
    db.execute("INSERT INTO meta VALUES (?, ?)", ("generated_at", generated_at))

    # Stages
    for i, s in enumerate(ALL_STAGES):
        db.execute("INSERT INTO stage VALUES (?, ?, ?, ?)",
                   (i, s["stage_id"], s["scope"], s["label"]))

    # Runs
    runs_rows = conn.execute("""
        SELECT run_id, status, started_at, completed_at, config_version
        FROM pipeline_run
        ORDER BY CASE WHEN completed_at IS NOT NULL THEN completed_at ELSE started_at END DESC
    """).fetchall()

    runs = []
    completed_ids: list[str] = []
    for row in runs_rows:
        r = _row_to_dict(row)
        if r["status"] == "completed":
            completed_ids.append(r["run_id"])

    completed_index = {rid: i for i, rid in enumerate(completed_ids)}
    for row in runs_rows:
        r = _row_to_dict(row)
        prev = None
        if r["run_id"] in completed_index:
            idx = completed_index[r["run_id"]]
            if idx + 1 < len(completed_ids):
                prev = completed_ids[idx + 1]
        r["prev_completed_run_id"] = prev
        runs.append(r)
        db.execute("INSERT INTO run VALUES (?, ?, ?, ?, ?, ?)",
                   (r["run_id"], r["status"], r["started_at"], r["completed_at"],
                    r["config_version"], prev))

    # Publishers
    pub_rows = conn.execute("SELECT DISTINCT publisher_id FROM document ORDER BY publisher_id").fetchall()
    publishers = [{"publisher_id": r["publisher_id"]} for r in pub_rows]
    for p in publishers:
        db.execute("INSERT INTO publisher VALUES (?)", (p["publisher_id"],))

    # Docs + stage status + totals
    doc_rows = conn.execute("""
        SELECT dv.doc_version_id, dv.document_id, d.publisher_id,
               COALESCE(dm.title, sr.source_title) AS title,
               d.url_normalized, d.source_published_at,
               dv.created_in_run_id, dv.content_quality_score, dv.primary_language
        FROM document_version dv
        JOIN document d ON d.document_id = dv.document_id
        JOIN scrape_record sr ON sr.scrape_id = dv.scrape_id
        LEFT JOIN doc_metadata dm ON dm.doc_version_id = dv.doc_version_id
        ORDER BY d.publisher_id, d.source_published_at DESC
    """).fetchall()

    docs: list[dict] = []
    for doc_index, drow in enumerate(doc_rows):
        d = _row_to_dict(drow)
        dvid = d["doc_version_id"]
        docs.append(d)

        db.execute("INSERT INTO doc VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (doc_index, d["doc_version_id"], d["document_id"], d["publisher_id"],
                    d["title"], d["url_normalized"], d["source_published_at"],
                    d["created_in_run_id"], d["content_quality_score"], d["primary_language"]))

        # Stage status
        for si, stage_def in enumerate(DOC_STAGES):
            sid = stage_def["stage_id"]
            srow = conn.execute("""
                SELECT stage, status, attempt, run_id AS last_run_id, processed_at,
                       error_message, details
                FROM doc_stage_status WHERE doc_version_id = ? AND stage = ?
            """, (dvid, sid)).fetchone()
            if srow:
                entry = _row_to_dict(srow)
                db.execute("INSERT INTO doc_stage_status VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (doc_index, si, entry["status"], entry["attempt"],
                            entry["last_run_id"], entry["processed_at"],
                            entry["error_message"], entry["details"]))

        # Totals
        totals = _get_doc_totals_list(conn, dvid)
        for si, counts in enumerate(totals):
            if counts:
                db.execute("INSERT INTO doc_totals VALUES (?, ?, ?)",
                           (doc_index, si, _j(counts)))

    db.commit()

    meta = {
        "export_version": EXPORT_VERSION,
        "generated_at": generated_at,
        "stages": ALL_STAGES,
        "runs": runs,
        "publishers": publishers,
    }

    log.info("  Runs: %d, Publishers: %d, Docs: %d", len(runs), len(publishers), len(docs))
    _finalize_db(db, out_path)
    return meta, docs


def _get_doc_totals_list(conn: sqlite3.Connection, dvid: str) -> list[Optional[dict]]:
    totals: list[Optional[dict]] = []

    # Stage 01: content lengths
    row = conn.execute(
        "SELECT content_length_raw, content_length_clean FROM document_version WHERE doc_version_id = ?",
        (dvid,)).fetchone()
    totals.append({
        "content_length_raw": row["content_length_raw"] if row else None,
        "content_length_clean": row["content_length_clean"] if row else None,
    })

    # Stage 02: blocks, chunks, tables
    row = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM block WHERE doc_version_id = ?) AS blocks,
            (SELECT COUNT(*) FROM chunk WHERE doc_version_id = ?) AS chunks,
            (SELECT COUNT(*) FROM table_extract WHERE doc_version_id = ?) AS tables
    """, (dvid, dvid, dvid)).fetchone()
    totals.append({"blocks": row["blocks"], "chunks": row["chunks"], "tables": row["tables"]})

    # Stage 03
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM doc_metadata WHERE doc_version_id = ?", (dvid,)).fetchone()
    totals.append({"metadata_rows": row["n"]})

    # Stage 04
    row = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM mention WHERE doc_version_id = ?) AS mentions,
            (SELECT COUNT(*) FROM mention_link ml JOIN mention m ON m.mention_id = ml.mention_id
             WHERE m.doc_version_id = ?) AS mention_links
    """, (dvid, dvid)).fetchone()
    totals.append({"mentions": row["mentions"], "mention_links": row["mention_links"]})

    # Stage 05
    row = conn.execute("""
        SELECT COUNT(*) AS n FROM chunk_embedding ce
        JOIN chunk c ON c.chunk_id = ce.chunk_id WHERE c.doc_version_id = ?
    """, (dvid,)).fetchone()
    totals.append({"embeddings": row["n"]})

    # Stage 06
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM facet_assignment WHERE doc_version_id = ?", (dvid,)).fetchone()
    totals.append({"facet_assignments": row["n"]})

    # Stage 07
    row = conn.execute(
        "SELECT label FROM novelty_label WHERE doc_version_id = ?", (dvid,)).fetchone()
    totals.append({"novelty_label": row["label"] if row else None})

    # Stage 08
    row = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM metric_observation WHERE doc_version_id = ?) AS metric_observations,
            (SELECT COUNT(*) FROM event_candidate WHERE doc_version_id = ?) AS event_candidates,
            (SELECT COUNT(*) FROM event_revision er
             WHERE json_valid(er.doc_version_ids)
             AND EXISTS (SELECT 1 FROM json_each(er.doc_version_ids) j WHERE j.value = ?)
            ) AS event_revisions_contributed
    """, (dvid, dvid, dvid)).fetchone()
    totals.append({
        "metric_observations": row["metric_observations"],
        "event_candidates": row["event_candidates"],
        "event_revisions_contributed": row["event_revisions_contributed"],
    })

    return totals


# ── Traces.sqlite ───────────────────────────────────────────────────────

def _init_traces_schema(db: sqlite3.Connection) -> None:
    db.executescript("""
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


def export_traces(conn: sqlite3.Connection, docs: list[dict], out_path: Path) -> None:
    db = _create_output_db(out_path)
    _init_traces_schema(db)

    for i, doc in enumerate(docs):
        dvid = doc["doc_version_id"]
        _export_single_trace(conn, dvid, db)
        if (i + 1) % 100 == 0:
            log.info("  traces: %d / %d", i + 1, len(docs))

    db.commit()
    _finalize_db(db, out_path)


def _export_single_trace(conn: sqlite3.Connection, dvid: str, db: sqlite3.Connection) -> None:
    header_row = conn.execute("""
        SELECT dv.doc_version_id, dv.document_id, d.publisher_id,
               COALESCE(dm.title, sr.source_title) AS title,
               d.url_normalized, d.source_published_at,
               dv.content_quality_score, dv.primary_language
        FROM document_version dv
        JOIN document d ON d.document_id = dv.document_id
        JOIN scrape_record sr ON sr.scrape_id = dv.scrape_id
        LEFT JOIN doc_metadata dm ON dm.doc_version_id = dv.doc_version_id
        WHERE dv.doc_version_id = ?
    """, (dvid,)).fetchone()
    if not header_row:
        return
    header = _row_to_dict(header_row)
    db.execute("INSERT INTO trace_header VALUES (?, ?)", (dvid, _j(header)))

    # Stage samples
    samples_map: dict[str, dict[str, list[dict]]] = {}

    samples_map["stage_01_ingest"] = {
        "document_version": _sample_rows(conn,
            """SELECT doc_version_id, document_id, content_hash_clean,
                      content_length_raw, content_length_clean, boilerplate_ratio,
                      content_quality_score, primary_language, cleaning_spec_version,
                      created_in_run_id
               FROM document_version WHERE doc_version_id = ?""", (dvid,)),
        "scrape_record": _sample_rows(conn,
            """SELECT sr.scrape_id, sr.publisher_id, sr.source_id, sr.url_raw,
                      sr.scraped_at, sr.source_published_at, sr.source_title,
                      sr.source_language, sr.scrape_kind, sr.processing_status
               FROM scrape_record sr
               JOIN document_version dv ON dv.scrape_id = sr.scrape_id
               WHERE dv.doc_version_id = ?""", (dvid,)),
    }

    samples_map["stage_02_parse"] = {
        "blocks": _sample_rows(conn,
            "SELECT block_id, block_type, block_level, span_start, span_end FROM block WHERE doc_version_id = ? ORDER BY span_start",
            (dvid,)),
        "chunks": _sample_rows(conn,
            "SELECT chunk_id, chunk_type, span_start, span_end, token_count_approx, heading_context FROM chunk WHERE doc_version_id = ? ORDER BY span_start",
            (dvid,)),
        "tables": _sample_rows(conn,
            "SELECT table_id, row_count, col_count, table_class, parse_method FROM table_extract WHERE doc_version_id = ? ORDER BY table_id",
            (dvid,)),
    }

    samples_map["stage_03_metadata"] = {
        "metadata": _sample_rows(conn,
            "SELECT title, published_at, detected_document_class, title_source, published_at_source FROM doc_metadata WHERE doc_version_id = ?",
            (dvid,)),
    }

    samples_map["stage_04_mentions"] = {
        "mentions": _sample_rows(conn,
            "SELECT mention_id, mention_type, surface_form, normalized_value, confidence, extraction_method FROM mention WHERE doc_version_id = ? ORDER BY span_start",
            (dvid,)),
        "mention_links": _sample_rows(conn,
            """SELECT ml.link_id, ml.entity_id, er.canonical_name, ml.link_confidence
               FROM mention_link ml
               JOIN mention m ON m.mention_id = ml.mention_id
               LEFT JOIN entity_registry er ON er.entity_id = ml.entity_id
               WHERE m.doc_version_id = ? ORDER BY ml.link_confidence DESC""",
            (dvid,)),
    }

    samples_map["stage_05_embeddings"] = {
        "chunk_embeddings": _sample_rows(conn,
            """SELECT ce.chunk_id, ce.embedding_dim, ce.model_version,
                      ce.language_used, ce.created_in_run_id
               FROM chunk_embedding ce
               JOIN chunk c ON c.chunk_id = ce.chunk_id
               WHERE c.doc_version_id = ? ORDER BY ce.chunk_id""",
            (dvid,)),
    }

    samples_map["stage_06_taxonomy"] = {
        "facets": _sample_rows(conn,
            "SELECT facet_id, facet_type, facet_value, confidence FROM facet_assignment WHERE doc_version_id = ? ORDER BY confidence DESC",
            (dvid,)),
    }

    samples_map["stage_07_novelty"] = {
        "novelty": _sample_rows(conn,
            "SELECT label, similarity_score, confidence FROM novelty_label WHERE doc_version_id = ?",
            (dvid,)),
    }

    samples_map["stage_08_events"] = {
        "metric_observations": _sample_rows(conn,
            "SELECT metric_id, metric_name, value_raw, unit_raw, value_norm, period_start FROM metric_observation WHERE doc_version_id = ? ORDER BY metric_name",
            (dvid,)),
        "event_candidates": _sample_rows(conn,
            "SELECT candidate_id, event_type, confidence, status, rejection_reason FROM event_candidate WHERE doc_version_id = ? ORDER BY confidence DESC",
            (dvid,)),
        "event_revisions": _sample_rows(conn,
            """SELECT er.revision_id, er.event_id, e.canonical_key, er.revision_no, er.confidence
               FROM event_revision er
               JOIN event e ON e.event_id = er.event_id
               WHERE json_valid(er.doc_version_ids)
               AND EXISTS (SELECT 1 FROM json_each(er.doc_version_ids) j WHERE j.value = ?)
               ORDER BY er.created_at DESC""",
            (dvid,)),
    }

    for stage_id, tables in samples_map.items():
        for table_name, rows in tables.items():
            if rows:
                db.execute("INSERT INTO trace_sample VALUES (?, ?, ?, ?)",
                           (dvid, stage_id, table_name, _j(rows)))


# ── Run SQLite ──────────────────────────────────────────────────────────

def _init_run_schema(db: sqlite3.Connection) -> None:
    db.executescript("""
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


def _synthesize_artifact_status(conn: sqlite3.Connection, stage_id: str, run_id: str) -> Optional[dict]:
    queries = _ARTIFACT_STAGE_QUERIES.get(stage_id)
    if not queries:
        return None
    total = 0
    table_counts = {}
    for table_name, sql in queries:
        row = conn.execute(sql, (run_id,)).fetchone()
        c = row[0] if row else 0
        table_counts[table_name] = c
        total += c
    if total == 0:
        return None
    return {
        "stage_id": stage_id, "status": "ok", "attempt": 1,
        "started_at": None, "completed_at": None, "error_message": None,
        "details": json.dumps({"synthesized_from_artifacts": table_counts}),
    }


def _get_new_counts_for_doc(conn: sqlite3.Connection, dvid: str, run_id: str) -> list[Optional[dict]]:
    counts: list[Optional[dict]] = []

    row = conn.execute(
        "SELECT COUNT(*) AS n FROM document_version WHERE doc_version_id = ? AND created_in_run_id = ?",
        (dvid, run_id)).fetchone()
    counts.append({"new_doc": row["n"]})

    row = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM block WHERE doc_version_id = ? AND created_in_run_id = ?) AS blocks,
            (SELECT COUNT(*) FROM chunk WHERE doc_version_id = ? AND created_in_run_id = ?) AS chunks,
            (SELECT COUNT(*) FROM table_extract WHERE doc_version_id = ? AND created_in_run_id = ?) AS tables
    """, (dvid, run_id, dvid, run_id, dvid, run_id)).fetchone()
    counts.append({"blocks": row["blocks"], "chunks": row["chunks"], "tables": row["tables"]})

    row = conn.execute(
        "SELECT COUNT(*) AS n FROM doc_metadata WHERE doc_version_id = ? AND created_in_run_id = ?",
        (dvid, run_id)).fetchone()
    counts.append({"metadata_rows": row["n"]})

    row = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM mention WHERE doc_version_id = ? AND created_in_run_id = ?) AS mentions,
            (SELECT COUNT(*) FROM mention_link ml JOIN mention m ON m.mention_id = ml.mention_id
             WHERE m.doc_version_id = ? AND ml.created_in_run_id = ?) AS mention_links
    """, (dvid, run_id, dvid, run_id)).fetchone()
    counts.append({"mentions": row["mentions"], "mention_links": row["mention_links"]})

    row = conn.execute("""
        SELECT COUNT(*) AS n FROM chunk_embedding ce
        JOIN chunk c ON c.chunk_id = ce.chunk_id
        WHERE c.doc_version_id = ? AND ce.created_in_run_id = ?
    """, (dvid, run_id)).fetchone()
    counts.append({"embeddings": row["n"]})

    row = conn.execute(
        "SELECT COUNT(*) AS n FROM facet_assignment WHERE doc_version_id = ? AND created_in_run_id = ?",
        (dvid, run_id)).fetchone()
    counts.append({"facet_assignments": row["n"]})

    row = conn.execute(
        "SELECT label FROM novelty_label WHERE doc_version_id = ? AND created_in_run_id = ?",
        (dvid, run_id)).fetchone()
    counts.append({"novelty_label": row["label"] if row else None})

    row = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM metric_observation WHERE doc_version_id = ? AND created_in_run_id = ?) AS metric_observations,
            (SELECT COUNT(*) FROM event_candidate WHERE doc_version_id = ? AND created_in_run_id = ?) AS event_candidates,
            (SELECT COUNT(*) FROM event_revision er
             WHERE er.created_in_run_id = ?
             AND json_valid(er.doc_version_ids)
             AND EXISTS (SELECT 1 FROM json_each(er.doc_version_ids) j WHERE j.value = ?)
            ) AS event_revisions_contributed
    """, (dvid, run_id, dvid, run_id, run_id, dvid)).fetchone()
    counts.append({
        "metric_observations": row["metric_observations"],
        "event_candidates": row["event_candidates"],
        "event_revisions_contributed": row["event_revisions_contributed"],
    })

    return counts


def export_run_db(conn: sqlite3.Connection, run_id: str, docs: list[dict],
                  out_path: Path, skip_traces: bool = False) -> None:
    db = _create_output_db(out_path)
    _init_run_schema(db)

    db.execute("INSERT INTO run_meta VALUES (?, ?)", ("run_id", run_id))

    # Run stage status
    run_stage_rows = conn.execute("""
        SELECT stage, status, attempt, started_at, completed_at, error_message, details
        FROM run_stage_status WHERE run_id = ?
    """, (run_id,)).fetchall()

    run_stage_map: dict[str, dict] = {}
    for r in run_stage_rows:
        d = _row_to_dict(r)
        d["stage_id"] = d.pop("stage")
        run_stage_map[d["stage_id"]] = d

    for ri, stage_def in enumerate(RUN_STAGES):
        sid = stage_def["stage_id"]
        entry = run_stage_map.get(sid)
        if entry is None and sid in _ARTIFACT_STAGE_QUERIES:
            entry = _synthesize_artifact_status(conn, sid, run_id)
        if entry is not None:
            db.execute("INSERT INTO run_stage_status VALUES (?, ?)", (ri, _j(entry)))

    # Artifact counts
    def _scalar(sql: str, params: tuple = ()) -> int:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else 0

    artifact_counts = {
        "embedding_indexes": _scalar("SELECT COUNT(*) FROM embedding_index WHERE run_id = ?", (run_id,)),
        "chunk_count_total": _scalar("SELECT COALESCE(SUM(chunk_count), 0) FROM embedding_index WHERE run_id = ?", (run_id,)),
        "clusters": _scalar("SELECT COUNT(*) FROM story_cluster WHERE run_id = ?", (run_id,)),
        "cluster_memberships": _scalar("SELECT COUNT(*) FROM story_cluster_member WHERE run_id = ?", (run_id,)),
        "metric_series": _scalar("SELECT COUNT(*) FROM metric_series WHERE run_id = ?", (run_id,)),
        "metric_points": _scalar("SELECT COUNT(*) FROM metric_series_point WHERE run_id = ?", (run_id,)),
        "alerts": _scalar("SELECT COUNT(*) FROM alert WHERE run_id = ?", (run_id,)),
        "digest_items": _scalar("SELECT COUNT(*) FROM digest_item WHERE run_id = ?", (run_id,)),
        "timeline_items": _scalar("SELECT COUNT(*) FROM entity_timeline_item WHERE run_id = ?", (run_id,)),
        "validation_failures": _scalar("SELECT COUNT(*) FROM validation_failure WHERE run_id = ?", (run_id,)),
    }
    for k, v in artifact_counts.items():
        db.execute("INSERT INTO run_artifact_counts VALUES (?, ?)", (k, v))

    # New counts per doc
    docs_with_new = 0
    for doc_index, doc in enumerate(docs):
        dvid = doc["doc_version_id"]
        stage_counts = _get_new_counts_for_doc(conn, dvid, run_id)
        has_data = False
        for si, counts in enumerate(stage_counts):
            if counts:
                if any(v not in (None, 0, "") for v in counts.values()):
                    has_data = True
                db.execute("INSERT INTO new_counts VALUES (?, ?, ?)",
                           (doc_index, si, _j(counts)))
        if has_data:
            docs_with_new += 1

    log.info("  %d/%d docs with new-in-run data", docs_with_new, len(docs))

    # Impact data
    if not skip_traces:
        impact_count = 0
        for doc in docs:
            dvid = doc["doc_version_id"]
            impact = _get_impact_data(conn, dvid, run_id)
            for key, rows in impact.items():
                if rows:
                    db.execute("INSERT INTO impact VALUES (?, ?, ?)",
                               (dvid, key, _j(rows)))
                    impact_count += 1
        log.info("  %d impact entries written", impact_count)

    db.commit()
    _finalize_db(db, out_path)


def _get_impact_data(conn: sqlite3.Connection, dvid: str, run_id: str) -> dict[str, list[dict]]:
    impact: dict[str, list[dict]] = {}

    impact["clusters"] = _sample_rows(conn,
        """SELECT sc.story_id, sc.summary_text, scm.score, scm.role
           FROM story_cluster_member scm
           JOIN story_cluster sc ON sc.run_id = scm.run_id AND sc.story_id = scm.story_id
           WHERE scm.run_id = ? AND scm.doc_version_id = ?""",
        (run_id, dvid))

    impact["metric_points"] = _sample_rows(conn,
        """SELECT msp.series_id, ms.metric_name, msp.period_start, msp.value_norm
           FROM metric_series_point msp
           JOIN metric_series ms ON ms.run_id = msp.run_id AND ms.series_id = msp.series_id
           WHERE msp.run_id = ? AND msp.source_doc_version_id = ?""",
        (run_id, dvid))

    impact["alerts"] = _sample_rows(conn,
        """SELECT alert_id, triggered_at, trigger_payload_json
           FROM alert WHERE run_id = ?
           AND json_valid(doc_version_ids)
           AND EXISTS (SELECT 1 FROM json_each(doc_version_ids) j WHERE j.value = ?)""",
        (run_id, dvid))

    impact["digest_items"] = _sample_rows(conn,
        """SELECT item_id, digest_date, section, item_type
           FROM digest_item WHERE run_id = ?
           AND json_valid(doc_version_ids)
           AND EXISTS (SELECT 1 FROM json_each(doc_version_ids) j WHERE j.value = ?)""",
        (run_id, dvid))

    impact["timeline_items"] = _sample_rows(conn,
        """SELECT item_id, entity_id, item_type, event_time, summary_text
           FROM entity_timeline_item WHERE run_id = ? AND ref_doc_version_id = ?""",
        (run_id, dvid))

    impact["validation_failures"] = _sample_rows(conn,
        """SELECT failure_id, stage, check_name, severity, details
           FROM validation_failure WHERE run_id = ? AND doc_version_id = ?""",
        (run_id, dvid))

    return impact


# ── Main ────────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export processed_posts.db → SQLite databases for Run Explorer")
    parser.add_argument("--db", default=Path("../../../database/processed_posts.db"),
                        help="Path to processed_posts.db")
    parser.add_argument("--out", default=Path("../../../docs/etl_data/"),
                        help="Output directory")
    parser.add_argument("--runs", type=int, default=None,
                        help="Max runs to export (default: all non-aborted)")
    parser.add_argument("--skip-traces", action="store_true",
                        help="Skip trace + impact export")
    parser.add_argument("--completed-only", action="store_true",
                        help="Export only completed runs")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    out = Path(args.out)
    sqlite_dir = out / "sqlite"
    conn = _connect(args.db)

    try:
        log.info("Exporting catalog.sqlite...")
        meta, docs = export_catalog(conn, sqlite_dir / "catalog.sqlite")
        log.info("Exported %d documents", len(docs))

        if not args.skip_traces:
            log.info("Exporting traces.sqlite (%d docs)...", len(docs))
            export_traces(conn, docs, sqlite_dir / "traces.sqlite")

        exportable_statuses = {"completed"} if args.completed_only else {"completed", "running"}
        exportable_runs = [r for r in meta["runs"] if r["status"] in exportable_statuses]
        if args.runs is not None:
            exportable_runs = exportable_runs[:args.runs]

        if not exportable_runs:
            log.warning("No exportable runs found.")
        else:
            log.info("Exporting %d run(s)...", len(exportable_runs))

        for run_info in exportable_runs:
            rid = run_info["run_id"]
            log.info("━━ Exporting run %s (status=%s) ━━", rid, run_info["status"])
            export_run_db(conn, rid, docs, sqlite_dir / "runs" / f"{rid}.sqlite",
                          skip_traces=args.skip_traces)

        log.info("Export complete → %s", sqlite_dir)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
