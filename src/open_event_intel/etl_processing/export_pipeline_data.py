#!/usr/bin/env python3
"""
public_view_extraction.py — Export processed_posts.db → static JSON for Run Explorer.

Reads the working database and writes compact, frontend-friendly JSON into
docs/public_view/ so GitHub Pages can serve the Run Explorer directly.

Usage:
    python public_view_extraction.py --db path/to/processed_posts.db --out docs/public_view
"""
import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from open_event_intel.logger import get_logger

EXPORT_VERSION = "1.0.0"

# ── Stage definitions (ordered) ────────────────────────────────────────

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

log = get_logger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────

def _connect(db_path: str) -> sqlite3.Connection:
    """Open read-only SQLite connection with required pragmas."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _write_json(path: Path, data: Any) -> None:
    """Write compact JSON, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False, default=str)
    log.info("Wrote %s (%.1f KB)", path, path.stat().st_size / 1024)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def _rows_to_list(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [_row_to_dict(r) for r in cursor.fetchall()]


# ── Export: meta.json ───────────────────────────────────────────────────

def export_meta(conn: sqlite3.Connection, out: Path) -> dict:
    """Export meta.json with stages, runs, and publishers."""
    # Runs
    runs_rows = conn.execute("""
        SELECT run_id, status, started_at, completed_at, config_version
        FROM pipeline_run
        ORDER BY
            CASE WHEN completed_at IS NOT NULL THEN completed_at ELSE started_at END DESC
    """).fetchall()

    runs = []
    prev_completed_id: Optional[str] = None
    # Build runs list; attach prev_completed_run_id (walk in reverse chronological)
    completed_ids: list[str] = []
    for row in runs_rows:
        r = _row_to_dict(row)
        if r["status"] == "completed":
            completed_ids.append(r["run_id"])

    # Map each run to its predecessor completed run
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

    # Publishers (distinct from documents)
    pub_rows = conn.execute("""
        SELECT DISTINCT publisher_id FROM document ORDER BY publisher_id
    """).fetchall()
    publishers = [{"publisher_id": r["publisher_id"]} for r in pub_rows]

    meta = {
        "export_version": EXPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stages": ALL_STAGES,
        "runs": runs,
        "publishers": publishers,
    }
    _write_json(out / "meta.json", meta)

    # ── Diagnostic: run status summary ───────────────────────────────
    status_dist: dict[str, int] = {}
    for r in runs:
        status_dist[r["status"]] = status_dist.get(r["status"], 0) + 1
    log.info(
        "  Runs: %d total (%s). Publishers: %d (%s)",
        len(runs),
        ", ".join(f"{k}={v}" for k, v in sorted(status_dist.items())),
        len(publishers),
        ", ".join(p["publisher_id"] for p in publishers[:8])
        + ("…" if len(publishers) > 8 else ""),
    )

    return meta


# ── Export: docs.json ───────────────────────────────────────────────────

def _get_doc_totals(conn: sqlite3.Connection, dvid: str) -> list[Optional[dict]]:
    """Per-doc, stage-ordered total counts for tooltip display."""
    totals: list[Optional[dict]] = []

    # Stage 02: blocks, chunks, tables
    row = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM block WHERE doc_version_id = ?) AS blocks,
            (SELECT COUNT(*) FROM chunk WHERE doc_version_id = ?) AS chunks,
            (SELECT COUNT(*) FROM table_extract WHERE doc_version_id = ?) AS tables
    """, (dvid, dvid, dvid)).fetchone()
    totals.append({"blocks": row["blocks"], "chunks": row["chunks"], "tables": row["tables"]})

    # Stage 03: metadata (1 row or 0)
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM doc_metadata WHERE doc_version_id = ?", (dvid,)
    ).fetchone()
    totals.append({"metadata_rows": row["n"]})

    # Stage 04: mentions, mention_links
    row = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM mention WHERE doc_version_id = ?) AS mentions,
            (SELECT COUNT(*) FROM mention_link ml
             JOIN mention m ON m.mention_id = ml.mention_id
             WHERE m.doc_version_id = ?) AS mention_links
    """, (dvid, dvid)).fetchone()
    totals.append({"mentions": row["mentions"], "mention_links": row["mention_links"]})

    # Stage 05: embeddings
    row = conn.execute("""
        SELECT COUNT(*) AS n FROM chunk_embedding ce
        JOIN chunk c ON c.chunk_id = ce.chunk_id
        WHERE c.doc_version_id = ?
    """, (dvid,)).fetchone()
    totals.append({"embeddings": row["n"]})

    # Stage 06: facet_assignments
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM facet_assignment WHERE doc_version_id = ?", (dvid,)
    ).fetchone()
    totals.append({"facet_assignments": row["n"]})

    # Stage 07: novelty
    row = conn.execute(
        "SELECT label FROM novelty_label WHERE doc_version_id = ?", (dvid,)
    ).fetchone()
    totals.append({"novelty_label": row["label"] if row else None})

    # Stage 08: events
    row = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM metric_observation WHERE doc_version_id = ?) AS metric_observations,
            (SELECT COUNT(*) FROM event_candidate WHERE doc_version_id = ?) AS event_candidates,
            (SELECT COUNT(*) FROM event_revision er
             WHERE json_valid(er.doc_version_ids)
             AND EXISTS (
                SELECT 1 FROM json_each(er.doc_version_ids) j WHERE j.value = ?
             )) AS event_revisions_contributed
    """, (dvid, dvid, dvid)).fetchone()
    totals.append({
        "metric_observations": row["metric_observations"],
        "event_candidates": row["event_candidates"],
        "event_revisions_contributed": row["event_revisions_contributed"],
    })

    return totals


def export_docs(conn: sqlite3.Connection, out: Path) -> list[dict]:
    """Export docs.json with stage status and total counts per document."""
    doc_rows = conn.execute("""
        SELECT
            dv.doc_version_id,
            dv.document_id,
            d.publisher_id,
            COALESCE(dm.title, sr.source_title) AS title,
            d.url_normalized,
            d.source_published_at,
            dv.created_in_run_id,
            dv.content_quality_score,
            dv.primary_language
        FROM document_version dv
        JOIN document d ON d.document_id = dv.document_id
        JOIN scrape_record sr ON sr.scrape_id = dv.scrape_id
        LEFT JOIN doc_metadata dm ON dm.doc_version_id = dv.doc_version_id
        ORDER BY d.publisher_id, d.source_published_at DESC
    """).fetchall()

    docs: list[dict] = []
    stage_status_by_doc: list[list[Optional[dict]]] = []
    totals_by_doc: list[list[Optional[dict]]] = []

    for drow in doc_rows:
        d = _row_to_dict(drow)
        dvid = d["doc_version_id"]
        docs.append(d)

        # Stage status for doc stages (01–08)
        status_entries: list[Optional[dict]] = []
        for stage_def in DOC_STAGES:
            sid = stage_def["stage_id"]
            srow = conn.execute("""
                SELECT stage, status, attempt, run_id AS last_run_id, processed_at,
                       error_message, details
                FROM doc_stage_status
                WHERE doc_version_id = ? AND stage = ?
            """, (dvid, sid)).fetchone()
            if srow:
                entry = _row_to_dict(srow)
                entry["stage_id"] = entry.pop("stage")
                status_entries.append(entry)
            else:
                status_entries.append(None)
        stage_status_by_doc.append(status_entries)

        # Totals (stage 01 is ingest → content lengths are the "count")
        doc_totals: list[Optional[dict]] = []
        # Stage 01: content lengths from document_version
        len_row = conn.execute(
            "SELECT content_length_raw, content_length_clean FROM document_version WHERE doc_version_id = ?",
            (dvid,)
        ).fetchone()
        doc_totals.append({
            "content_length_raw": len_row["content_length_raw"] if len_row else None,
            "content_length_clean": len_row["content_length_clean"] if len_row else None,
        })
        # Stages 02–08
        doc_totals.extend(_get_doc_totals(conn, dvid))
        totals_by_doc.append(doc_totals)

    payload = {
        "docs": docs,
        "stage_status_by_doc": stage_status_by_doc,
        "totals_by_doc": totals_by_doc,
    }
    _write_json(out / "docs.json", payload)

    # ── Diagnostic: per-stage status distribution across all docs ─────
    log.info("── Doc-stage status distribution (all %d docs) ──", len(docs))
    for si, stage_def in enumerate(DOC_STAGES):
        sid = stage_def["stage_id"]
        dist: dict[str, int] = {}
        for status_row in stage_status_by_doc:
            entry = status_row[si] if si < len(status_row) else None
            s = entry["status"] if entry else "NOT_STARTED"
            dist[s] = dist.get(s, 0) + 1
        parts = "  ".join(f"{k}={v}" for k, v in sorted(dist.items()))
        log.info("  %-24s %s", sid, parts)

    return docs


# ── Export: runs/<run_id>.json ──────────────────────────────────────────

# Maps run-scoped stage IDs to artifact tables + count queries used to
# synthesize their status when no run_stage_status row exists.
# stage_05 and stage_07 produce artifacts directly and may not always
# write run_stage_status; stage_10 and stage_11 normally do write it,
# but the fallback lets the exporter detect partial runs where the
# stage produced output but crashed before recording its status.
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


def _diagnose_run_scoped_stages(  # noqa: C901
    conn: sqlite3.Connection,
    run_id: str,
    run_stage_map: dict[str, dict],
    artifact_counts: dict[str, int],
) -> None:
    """
    Log detailed diagnostics explaining why run-scoped stage cells may be empty.

    This is the primary troubleshooting output.  It compares:
      • what RUN_STAGES the frontend expects
      • what run_stage_status rows actually exist in the DB
      • whether artifact tables have data (even when no status row exists)
    """
    # 1) Pipeline run context
    run_row = conn.execute(
        "SELECT run_id, status, started_at, completed_at, config_version "
        "FROM pipeline_run WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if run_row:
        r = _row_to_dict(run_row)
        log.info(
            "  pipeline_run: status=%s  started=%s  completed=%s  config=%s",
            r["status"], r["started_at"], r["completed_at"],
            (r["config_version"] or "")[:16],
        )
    else:
        log.warning("  pipeline_run row NOT FOUND for run_id=%s", run_id)
        return

    # 2) All actual run_stage_status rows for this run
    all_rows = conn.execute(
        "SELECT stage, status, attempt, error_message "
        "FROM run_stage_status WHERE run_id = ? ORDER BY stage",
        (run_id,),
    ).fetchall()
    db_stage_ids = set()
    if all_rows:
        log.info("  run_stage_status rows in DB for this run (%d):", len(all_rows))
        for row in all_rows:
            d = _row_to_dict(row)
            db_stage_ids.add(d["stage"])
            err_snippet = ""
            if d["error_message"]:
                err_snippet = f"  error={d['error_message'][:80]}"
            log.info(
                "    %-30s status=%-7s attempt=%s%s",
                d["stage"], d["status"], d["attempt"], err_snippet,
            )
    else:
        log.warning(
            "  run_stage_status has ZERO rows for run_id=%s.  "
            "Stages 09/10/11 have not written status yet, or the run "
            "has not reached those stages.",
            run_id,
        )

    # 3) Per-expected-stage diagnosis
    log.info("  Expected run-scoped stages vs DB reality:")
    for stage_def in RUN_STAGES:
        sid = stage_def["stage_id"]
        matched = run_stage_map.get(sid)

        # Is this an artifact-only stage?
        is_artifact_only = sid in _ARTIFACT_STAGE_QUERIES

        if matched:
            log.info(
                "    %-24s ✓ run_stage_status.status=%s",
                sid, matched["status"],
            )
        elif is_artifact_only:
            # Check artifact tables
            queries = _ARTIFACT_STAGE_QUERIES[sid]
            table_counts = {}
            for table_name, sql in queries:
                row = conn.execute(sql, (run_id,)).fetchone()
                table_counts[table_name] = row[0] if row else 0
            has_artifacts = any(c > 0 for c in table_counts.values())
            counts_str = ", ".join(f"{t}={c}" for t, c in table_counts.items())
            if has_artifacts:
                log.info(
                    "    %-24s ⚠ NO run_stage_status row (artifact-only stage). "
                    "Artifacts found: %s → synthesising 'ok'",
                    sid, counts_str,
                )
            else:
                log.warning(
                    "    %-24s ✗ NO run_stage_status row AND no artifact data "
                    "(%s). Stage likely has not run yet.",
                    sid, counts_str,
                )
        else:
            # Should be in run_stage_status but isn't
            log.warning(
                "    %-24s ✗ MISSING from run_stage_status.  "
                "Stage has not run or used a different stage name.  "
                "DB has these stage names: %s",
                sid,
                sorted(db_stage_ids) if db_stage_ids else "(none)",
            )

    # 4) Artifact counts summary
    nonempty = {k: v for k, v in artifact_counts.items() if v > 0}
    all_zero = {k: v for k, v in artifact_counts.items() if v == 0}
    if nonempty:
        log.info("  Run artifact counts (non-zero): %s", nonempty)
    else:
        log.warning("  ALL run artifact counts are zero — no run-scoped output was produced.")
    if all_zero:
        log.debug("  Run artifact counts (zero): %s", sorted(all_zero.keys()))


def _synthesize_artifact_status(
    conn: sqlite3.Connection,
    stage_id: str,
    run_id: str,
) -> Optional[dict]:
    """
    For stages with artifact-table fallback, build a synthetic status entry from the artifact tables so the frontend can show a marker.

    Returns a dict shaped like a run_stage_status row, or None.
    """
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
        "stage_id": stage_id,
        "status": "ok",
        "attempt": 1,
        "started_at": None,
        "completed_at": None,
        "error_message": None,
        "details": json.dumps({"synthesized_from_artifacts": table_counts}),
    }


def _get_new_counts_for_doc(conn: sqlite3.Connection, dvid: str, run_id: str) -> list[Optional[dict]]:
    """Per-doc new-in-run counts (created_in_run_id == run_id)."""
    counts: list[Optional[dict]] = []

    # Stage 01: was document itself created in this run?
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM document_version WHERE doc_version_id = ? AND created_in_run_id = ?",
        (dvid, run_id)
    ).fetchone()
    counts.append({"new_doc": row["n"]})

    # Stage 02
    row = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM block WHERE doc_version_id = ? AND created_in_run_id = ?) AS blocks,
            (SELECT COUNT(*) FROM chunk WHERE doc_version_id = ? AND created_in_run_id = ?) AS chunks,
            (SELECT COUNT(*) FROM table_extract WHERE doc_version_id = ? AND created_in_run_id = ?) AS tables
    """, (dvid, run_id, dvid, run_id, dvid, run_id)).fetchone()
    counts.append({"blocks": row["blocks"], "chunks": row["chunks"], "tables": row["tables"]})

    # Stage 03
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM doc_metadata WHERE doc_version_id = ? AND created_in_run_id = ?",
        (dvid, run_id)
    ).fetchone()
    counts.append({"metadata_rows": row["n"]})

    # Stage 04
    row = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM mention WHERE doc_version_id = ? AND created_in_run_id = ?) AS mentions,
            (SELECT COUNT(*) FROM mention_link ml
             JOIN mention m ON m.mention_id = ml.mention_id
             WHERE m.doc_version_id = ? AND ml.created_in_run_id = ?) AS mention_links
    """, (dvid, run_id, dvid, run_id)).fetchone()
    counts.append({"mentions": row["mentions"], "mention_links": row["mention_links"]})

    # Stage 05
    row = conn.execute("""
        SELECT COUNT(*) AS n FROM chunk_embedding ce
        JOIN chunk c ON c.chunk_id = ce.chunk_id
        WHERE c.doc_version_id = ? AND ce.created_in_run_id = ?
    """, (dvid, run_id)).fetchone()
    counts.append({"embeddings": row["n"]})

    # Stage 06
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM facet_assignment WHERE doc_version_id = ? AND created_in_run_id = ?",
        (dvid, run_id)
    ).fetchone()
    counts.append({"facet_assignments": row["n"]})

    # Stage 07
    row = conn.execute(
        "SELECT label FROM novelty_label WHERE doc_version_id = ? AND created_in_run_id = ?",
        (dvid, run_id)
    ).fetchone()
    counts.append({"novelty_label": row["label"] if row else None})

    # Stage 08
    row = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM metric_observation WHERE doc_version_id = ? AND created_in_run_id = ?) AS metric_observations,
            (SELECT COUNT(*) FROM event_candidate WHERE doc_version_id = ? AND created_in_run_id = ?) AS event_candidates,
            (SELECT COUNT(*) FROM event_revision er
             WHERE er.created_in_run_id = ?
             AND json_valid(er.doc_version_ids)
             AND EXISTS (
                SELECT 1 FROM json_each(er.doc_version_ids) j WHERE j.value = ?
             )) AS event_revisions_contributed
    """, (dvid, run_id, dvid, run_id, run_id, dvid)).fetchone()
    counts.append({
        "metric_observations": row["metric_observations"],
        "event_candidates": row["event_candidates"],
        "event_revisions_contributed": row["event_revisions_contributed"],
    })

    return counts


def export_run(conn: sqlite3.Connection, run_id: str, docs: list[dict], out: Path) -> None:
    """
    Export runs/<run_id>.json with run stage status, new counts, and artifact counts.

    This function also runs detailed diagnostics (logged at INFO/WARNING)
    explaining why individual run-scoped stage cells may appear empty in
    the overview matrix.
    """
    log.info("── Run %s: collecting run-scoped stage status ──", run_id)

    # ── 1. Fetch actual run_stage_status rows ────────────────────────
    run_stage_rows = conn.execute("""
        SELECT stage, status, attempt, started_at, completed_at, error_message, details
        FROM run_stage_status
        WHERE run_id = ?
    """, (run_id,)).fetchall()

    run_stage_map: dict[str, dict] = {}
    for r in run_stage_rows:
        d = _row_to_dict(r)
        d["stage_id"] = d.pop("stage")
        run_stage_map[d["stage_id"]] = d

    log.info(
        "  run_stage_status returned %d rows; matched stage IDs: %s",
        len(run_stage_rows),
        sorted(run_stage_map.keys()) if run_stage_map else "(none)",
    )

    # ── 2. Build artifact counts early (needed for diagnostics) ──────
    def _scalar(sql: str, params: tuple = ()) -> int:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else 0

    run_artifact_counts = {
        "embedding_indexes": _scalar(
            "SELECT COUNT(*) FROM embedding_index WHERE run_id = ?", (run_id,)),
        "chunk_count_total": _scalar("""
            SELECT COALESCE(SUM(chunk_count), 0) FROM embedding_index WHERE run_id = ?
        """, (run_id,)),
        "clusters": _scalar(
            "SELECT COUNT(*) FROM story_cluster WHERE run_id = ?", (run_id,)),
        "cluster_memberships": _scalar(
            "SELECT COUNT(*) FROM story_cluster_member WHERE run_id = ?", (run_id,)),
        "metric_series": _scalar(
            "SELECT COUNT(*) FROM metric_series WHERE run_id = ?", (run_id,)),
        "metric_points": _scalar(
            "SELECT COUNT(*) FROM metric_series_point WHERE run_id = ?", (run_id,)),
        "alerts": _scalar(
            "SELECT COUNT(*) FROM alert WHERE run_id = ?", (run_id,)),
        "digest_items": _scalar(
            "SELECT COUNT(*) FROM digest_item WHERE run_id = ?", (run_id,)),
        "timeline_items": _scalar(
            "SELECT COUNT(*) FROM entity_timeline_item WHERE run_id = ?", (run_id,)),
        "validation_failures": _scalar(
            "SELECT COUNT(*) FROM validation_failure WHERE run_id = ?", (run_id,)),
    }

    # ── 3. Detailed diagnostics ──────────────────────────────────────
    _diagnose_run_scoped_stages(conn, run_id, run_stage_map, run_artifact_counts)

    # ── 4. Assemble run_stage_status array for JSON ──────────────────
    #   • Use the DB row if it exists
    #   • Otherwise, for artifact-only stages, synthesize from table counts
    #   • Otherwise, emit None (frontend shows "not started")
    run_stage_status: list[Optional[dict]] = []
    for stage_def in RUN_STAGES:
        sid = stage_def["stage_id"]
        entry = run_stage_map.get(sid)
        if entry is None and sid in _ARTIFACT_STAGE_QUERIES:
            entry = _synthesize_artifact_status(conn, sid, run_id)
            if entry:
                log.debug("  Synthesized status for %s from artifact tables", sid)
        if entry is None:
            log.debug("  %s → null (will display as 'not started')", sid)
        run_stage_status.append(entry)

    resolved_count = sum(1 for e in run_stage_status if e is not None)
    log.info(
        "  Final run_stage_status: %d/%d stages have a status entry "
        "(remaining %d will show as empty/not-started in UI)",
        resolved_count, len(RUN_STAGES), len(RUN_STAGES) - resolved_count,
    )

    # ── 5. Per-doc new counts (aligned to docs.json order) ───────────
    new_counts_by_doc = []
    docs_with_new_data = 0
    for doc in docs:
        dvid = doc["doc_version_id"]
        counts = _get_new_counts_for_doc(conn, dvid, run_id)
        new_counts_by_doc.append(counts)
        if any(
            v not in (None, 0, "")
            for stage_counts in counts
            if stage_counts
            for v in stage_counts.values()
        ):
            docs_with_new_data += 1

    log.info(
        "  Per-doc new counts: %d/%d documents have any new-in-run data "
        "(diff mode would show only these)",
        docs_with_new_data, len(docs),
    )

    # ── 6. Write JSON ────────────────────────────────────────────────
    payload = {
        "run_id": run_id,
        "run_stage_status": run_stage_status,
        "new_counts_by_doc": new_counts_by_doc,
        "run_artifact_counts": run_artifact_counts,
    }
    _write_json(out / "runs" / f"{run_id}.json", payload)


# ── Export: trace/<doc_version_id>.json ─────────────────────────────────

def _sample_rows(conn: sqlite3.Connection, sql: str, params: tuple, limit: int = SAMPLE_LIMIT) -> list[dict]:
    """Fetch up to `limit` sample rows, returned as dicts."""
    rows = conn.execute(f"{sql} LIMIT ?", (*params, limit)).fetchall()
    return [_row_to_dict(r) for r in rows]


def export_trace(conn: sqlite3.Connection, dvid: str, out: Path) -> None:
    """Export base trace for a single document (run-independent)."""
    # Document header
    header_row = conn.execute("""
        SELECT
            dv.doc_version_id, dv.document_id, d.publisher_id,
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

    # Samples per stage
    stage_samples: dict[str, dict[str, list[dict]]] = {}

    # Stage 01: ingest/document provenance
    stage_samples["stage_01_ingest"] = {
        "document_version": _sample_rows(conn,
            """SELECT doc_version_id, document_id, content_hash_clean,
                      content_length_raw, content_length_clean, boilerplate_ratio,
                      content_quality_score, primary_language, cleaning_spec_version,
                      created_in_run_id
               FROM document_version WHERE doc_version_id = ?""",
            (dvid,)),
        "scrape_record": _sample_rows(conn,
            """SELECT sr.scrape_id, sr.publisher_id, sr.source_id, sr.url_raw,
                      sr.scraped_at, sr.source_published_at, sr.source_title,
                      sr.source_language, sr.scrape_kind, sr.processing_status
               FROM scrape_record sr
               JOIN document_version dv ON dv.scrape_id = sr.scrape_id
               WHERE dv.doc_version_id = ?""",
            (dvid,)),
    }

    # Stage 02
    stage_samples["stage_02_parse"] = {
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

    # Stage 03
    stage_samples["stage_03_metadata"] = {
        "metadata": _sample_rows(conn,
            "SELECT title, published_at, detected_document_class, title_source, published_at_source FROM doc_metadata WHERE doc_version_id = ?",
            (dvid,)),
    }

    # Stage 04
    stage_samples["stage_04_mentions"] = {
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

    # Stage 05: embeddings (exclude the blob itself)
    stage_samples["stage_05_embeddings"] = {
        "chunk_embeddings": _sample_rows(conn,
            """SELECT ce.chunk_id, ce.embedding_dim, ce.model_version,
                      ce.language_used, ce.created_in_run_id
               FROM chunk_embedding ce
               JOIN chunk c ON c.chunk_id = ce.chunk_id
               WHERE c.doc_version_id = ? ORDER BY ce.chunk_id""",
            (dvid,)),
    }

    # Stage 06
    stage_samples["stage_06_taxonomy"] = {
        "facets": _sample_rows(conn,
            "SELECT facet_id, facet_type, facet_value, confidence FROM facet_assignment WHERE doc_version_id = ? ORDER BY confidence DESC",
            (dvid,)),
    }

    # Stage 07
    stage_samples["stage_07_novelty"] = {
        "novelty": _sample_rows(conn,
            "SELECT label, similarity_score, confidence FROM novelty_label WHERE doc_version_id = ?",
            (dvid,)),
    }

    # Stage 08
    stage_samples["stage_08_events"] = {
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

    payload = {
        "header": header,
        "stage_samples": stage_samples,
    }
    _write_json(out / "trace" / f"{dvid}.json", payload)


# ── Export: runs/<run_id>/impact/<doc_version_id>.json ──────────────────

def export_impact(conn: sqlite3.Connection, dvid: str, run_id: str, out: Path) -> None:
    """Export run-specific downstream impact for a document."""
    impact: dict[str, Any] = {}

    # Story cluster membership
    impact["clusters"] = _sample_rows(conn,
        """SELECT sc.story_id, sc.summary_text, scm.score, scm.role
           FROM story_cluster_member scm
           JOIN story_cluster sc ON sc.run_id = scm.run_id AND sc.story_id = scm.story_id
           WHERE scm.run_id = ? AND scm.doc_version_id = ?""",
        (run_id, dvid))

    # Metric points sourced from this doc
    impact["metric_points"] = _sample_rows(conn,
        """SELECT msp.series_id, ms.metric_name, msp.period_start, msp.value_norm
           FROM metric_series_point msp
           JOIN metric_series ms ON ms.run_id = msp.run_id AND ms.series_id = msp.series_id
           WHERE msp.run_id = ? AND msp.source_doc_version_id = ?""",
        (run_id, dvid))

    # Alerts referencing this doc
    impact["alerts"] = _sample_rows(conn,
        """SELECT alert_id, triggered_at, trigger_payload_json
           FROM alert
           WHERE run_id = ?
           AND json_valid(doc_version_ids)
           AND EXISTS (SELECT 1 FROM json_each(doc_version_ids) j WHERE j.value = ?)""",
        (run_id, dvid))

    # Digest items referencing this doc
    impact["digest_items"] = _sample_rows(conn,
        """SELECT item_id, digest_date, section, item_type
           FROM digest_item
           WHERE run_id = ?
           AND json_valid(doc_version_ids)
           AND EXISTS (SELECT 1 FROM json_each(doc_version_ids) j WHERE j.value = ?)""",
        (run_id, dvid))

    # Timeline items referencing this doc
    impact["timeline_items"] = _sample_rows(conn,
        """SELECT item_id, entity_id, item_type, event_time, summary_text
           FROM entity_timeline_item
           WHERE run_id = ? AND ref_doc_version_id = ?""",
        (run_id, dvid))

    # Validation failures for this doc
    impact["validation_failures"] = _sample_rows(conn,
        """SELECT failure_id, stage, check_name, severity, details
           FROM validation_failure
           WHERE run_id = ? AND doc_version_id = ?""",
        (run_id, dvid))

    # Only write if there's actual impact data
    has_data = any(len(v) > 0 for v in impact.values() if isinstance(v, list))
    if has_data:
        _write_json(out / "runs" / run_id / "impact" / f"{dvid}.json", impact)


# ── Main ────────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:  # noqa: D103
    parser = argparse.ArgumentParser(description="Export processed_posts.db → static JSON for Run Explorer")
    parser.add_argument("--db", default=Path("../../../database/processed_posts.db"), help="Path to processed_posts.db")
    parser.add_argument("--out", default=Path("../../../docs/data/"), help="Output directory (e.g. docs/data)")
    parser.add_argument("--runs", type=int, default=None, help="Max runs to export (default: all non-aborted)")
    parser.add_argument("--skip-traces", action="store_true", help="Skip per-document trace export (faster)")
    parser.add_argument("--completed-only", action="store_true",
                        help="Export only completed runs (default: export running + completed)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    out = Path(args.out)
    conn = _connect(args.db)

    try:
        log.info("Exporting meta.json...")
        meta = export_meta(conn, out)

        log.info("Exporting docs.json...")
        docs = export_docs(conn, out)
        log.info("Exported %d documents", len(docs))

        # Export run data — by default include running + completed (not failed/aborted)
        # A "running" run has useful partial data; excluding it leaves all run-scoped
        # stage columns empty, which is the most common reason for blank columns.
        exportable_statuses = {"completed"} if args.completed_only else {"completed", "running"}
        exportable_runs = [r for r in meta["runs"] if r["status"] in exportable_statuses]
        if args.runs is not None:
            exportable_runs = exportable_runs[:args.runs]

        if not exportable_runs:
            all_statuses = [r["status"] for r in meta["runs"]]
            log.warning(
                "No exportable runs found (%d total runs in DB, statuses: %s). "
                "Run-scoped stage columns (05 Index, 07 Cluster, 09–11) will be "
                "empty for ALL publications. If runs exist with status='running', "
                "omit --completed-only to include them.",
                len(meta["runs"]),
                ", ".join(f'{s}={all_statuses.count(s)}' for s in sorted(set(all_statuses))),
            )
        else:
            log.info(
                "Exporting %d run(s) (statuses: %s) out of %d total",
                len(exportable_runs),
                ", ".join(sorted(exportable_statuses)),
                len(meta["runs"]),
            )

        for run_info in exportable_runs:
            rid = run_info["run_id"]
            log.info("━━ Exporting run %s (status=%s) ━━", rid, run_info["status"])
            export_run(conn, rid, docs, out)

            # Export per-doc impact for this run
            if not args.skip_traces:
                for doc in docs:
                    export_impact(conn, doc["doc_version_id"], rid, out)

        # Export base traces
        if not args.skip_traces:
            log.info("Exporting %d document traces...", len(docs))
            for i, doc in enumerate(docs):
                export_trace(conn, doc["doc_version_id"], out)
                if (i + 1) % 100 == 0:
                    log.info("  traces: %d / %d", i + 1, len(docs))

        log.info("Export complete → %s", out)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())