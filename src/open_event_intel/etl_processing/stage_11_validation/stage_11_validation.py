"""
Stage 11 — Final cross-table validation.

Run-scoped integrity gate that verifies all outputs from stages 01–10 are
internally consistent, provenance-safe, and query-time safe before the run
can be marked ``completed``.

**Non-destructive**: MUST NOT modify ``evidence_span`` or ``event*`` tables.

**Required checks:**
- Evidence linkage completeness for surfaced rows
- `evidence_id` determinism: `SHA256(doc_version_id|span_start|span_end)` (sample verification)
- Span conformance: application slicing matches SQLite slicing
- Config coherence: no mixed `config_hash` in `eligible_docs`
- Event eligibility: all `doc_version_ids` in current revisions have `stage_08_events='ok'`
- ANN index file presence for `embedding_index` rows
- Orphaned event check: no `event` rows where `current_revision_id IS NULL` but `event_revision` rows exist for that `event_id`
- **Run-state hygiene (required):**
- Assert there is no `pipeline_run` row with `status='running'` other than the current `run_id`
    (should be guaranteed by `idx_pipeline_single_running`; log as `validation_failure` if violated).
- Detect abandoned/unfinished historical runs: any `pipeline_run` with `status IN ('failed','aborted')`.
    Log a `validation_failure` with `severity='warning'` including the run_ids and timestamps.
- Detect "non-completed run artifacts": for each of the following tables, verify that `created_in_run_id`
    references either (a) the current `run_id` or (b) a `pipeline_run` with `status='completed'`:
    `document_version`, `evidence_span`, `block`, `chunk`, `chunk_embedding`, `doc_metadata`, `table_extract`,
    `mention`, `mention_link`, `registry_update_proposal`,
    `facet_assignment`, `novelty_label`, `document_fingerprint`, `chunk_novelty`, `chunk_novelty_score`,
    `event`, `event_revision`, `event_entity_link`, `metric_observation`, `event_candidate`.
    Any rows referencing a non-completed, non-current run_id MUST be logged as `validation_failure` with `severity='error'`.
- **Export-readiness checks:**
    Verify that all prerequisite run-stage-status rows exist and that key artifact tables
    the static export pipeline depends on are populated.

.. note::
   SQL queries in ``Stage11DatabaseInterface`` should be migrated into
   ``database_interface.py`` in a future consolidation pass.
"""
import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel

from open_event_intel.etl_processing.config_interface import Config, get_config_version, load_config
from open_event_intel.etl_processing.database_interface import (
    ValidationFailureRow,
    compute_sha256_id,
)
from open_event_intel.etl_processing.stage_11_validation.database_stage_11_validation import DEFAULT_EVIDENCE_ID_SAMPLE_SIZE, DEFAULT_SPAN_CONFORMANCE_SAMPLE_SIZE, PREREQUISITE_STAGE, STAGE_NAME, \
    Stage11DatabaseInterface
from open_event_intel.logger import get_logger

logger = get_logger(__name__)

#Tables that carry created_in_run_id
TABLES_WITH_CREATED_IN_RUN_ID: tuple[str, ...] = (
    "document_version",
    "evidence_span",
    "block",
    "chunk",
    "chunk_embedding",
    "doc_metadata",
    "table_extract",
    "mention",
    "mention_link",
    "registry_update_proposal",
    "facet_assignment",
    "novelty_label",
    "document_fingerprint",
    # chunk_novelty omitted: model has no created_in_run_id column
    "chunk_novelty_score",
    "event",
    "event_revision",
    "event_entity_link",
    "metric_observation",
    "event_candidate",
)

# Doc-scoped and run-scoped stages the export pipeline expects
DOC_STAGE_IDS: tuple[str, ...] = (
    "stage_01_ingest",
    "stage_02_parse",
    "stage_03_metadata",
    "stage_04_mentions",
    "stage_05_embeddings",
    "stage_06_taxonomy",
    "stage_07_novelty",
    "stage_08_events",
)
RUN_STAGE_IDS: tuple[str, ...] = (
    "stage_05_embeddings_index",
    "stage_07_story_cluster",
    "stage_09_outputs",
    "stage_10_timeline",
    "stage_11_validation",
)

# Run-scoped artifact tables the export pipeline reads
EXPORT_RUN_ARTIFACT_TABLES: tuple[str, ...] = (
    "embedding_index",
    "story_cluster",
    "story_cluster_member",
    "metric_series",
    "metric_series_point",
    "alert",
    "alert_evidence",
    "digest_item",
    "digest_item_evidence",
    "entity_timeline_item",
    "entity_timeline_item_evidence",
    "validation_failure",
)

# Evidence-linkage pairs the export pipeline reads
_EVIDENCE_LINKAGE_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    # (parent_table, parent_id_col, evidence_table, evidence_fk_col)
    ("alert",                "alert_id", "alert_evidence",                "alert_id"),
    ("digest_item",          "item_id",  "digest_item_evidence",          "item_id"),
    ("entity_timeline_item", "item_id",  "entity_timeline_item_evidence", "item_id"),
)


class ValidationResult(BaseModel):
    """Outcome of a single validation check."""

    check_name: str
    severity: str  # "error" | "warning"
    details: str
    doc_version_id: str | None = None


# Validation check functions
def _make_failure_id(run_id: str, check_name: str, suffix: str) -> str:
    return compute_sha256_id(run_id, check_name, suffix)


def check_evidence_linkage(
    db: Stage11DatabaseInterface, run_id: str
) -> list[ValidationResult]:
    """Verify every surfaced row has required evidence linkage."""
    results: list[ValidationResult] = []

    # Alerts
    cnt = db.count_alerts_without_evidence(run_id)
    total_alerts = db.count_table_rows("alert", run_id=run_id)
    if cnt > 0:
        ids = db.get_alert_ids_without_evidence(run_id)
        results.append(ValidationResult(
            check_name="evidence_linkage_alerts",
            severity="error",
            details=f"{cnt}/{total_alerts} alert(s) without evidence. Sample IDs: {ids[:10]}",
        ))
        logger.warning("  FAIL evidence_linkage_alerts: %d/%d alert(s) missing evidence", cnt, total_alerts)
    else:
        logger.info("  PASS evidence_linkage_alerts: all %d alert(s) have evidence", total_alerts)

    # Digest items
    cnt = db.count_digest_items_without_evidence(run_id)
    total_digest = db.count_table_rows("digest_item", run_id=run_id)
    if cnt > 0:
        ids = db.get_digest_item_ids_without_evidence(run_id)
        results.append(ValidationResult(
            check_name="evidence_linkage_digest_items",
            severity="error",
            details=f"{cnt}/{total_digest} digest_item(s) without evidence. Sample IDs: {ids[:10]}",
        ))
        logger.warning("  FAIL evidence_linkage_digest_items: %d/%d missing evidence", cnt, total_digest)
    else:
        logger.info("  PASS evidence_linkage_digest_items: all %d digest item(s) have evidence", total_digest)

    # Timeline items
    cnt = db.count_timeline_items_without_evidence(run_id)
    total_timeline = db.count_table_rows("entity_timeline_item", run_id=run_id)
    if cnt > 0:
        ids = db.get_timeline_item_ids_without_evidence(run_id)
        results.append(ValidationResult(
            check_name="evidence_linkage_timeline_items",
            severity="error",
            details=f"{cnt}/{total_timeline} entity_timeline_item(s) without evidence. Sample IDs: {ids[:10]}",
        ))
        logger.warning("  FAIL evidence_linkage_timeline_items: %d/%d missing evidence", cnt, total_timeline)
    else:
        logger.info("  PASS evidence_linkage_timeline_items: all %d timeline item(s) have evidence", total_timeline)

    # Metric points
    cnt = db.count_metric_points_without_evidence(run_id)
    total_points = db.count_table_rows("metric_series_point", run_id=run_id)
    if cnt > 0:
        results.append(ValidationResult(
            check_name="evidence_linkage_metric_points",
            severity="error",
            details=f"{cnt}/{total_points} metric_series_point(s) with NULL evidence_id",
        ))
        logger.warning("  FAIL evidence_linkage_metric_points: %d/%d with NULL evidence_id", cnt, total_points)
    else:
        logger.info("  PASS evidence_linkage_metric_points: all %d point(s) have evidence_id", total_points)

    # Event revisions (current)
    cnt = db.count_event_revisions_without_evidence()
    if cnt > 0:
        results.append(ValidationResult(
            check_name="evidence_linkage_event_revisions",
            severity="error",
            details=f"{cnt} current event_revision(s) without evidence",
        ))
        logger.warning("  FAIL evidence_linkage_event_revisions: %d current revision(s) missing evidence", cnt)
    else:
        logger.info("  PASS evidence_linkage_event_revisions: all current revisions have evidence")

    # Facet assignments
    cnt = db.count_facet_assignments_without_evidence()
    if cnt > 0:
        results.append(ValidationResult(
            check_name="evidence_linkage_facet_assignments",
            severity="warning",
            details=f"{cnt} facet_assignment(s) without evidence rows",
        ))
        logger.warning("  WARN evidence_linkage_facet_assignments: %d without evidence", cnt)
    else:
        logger.info("  PASS evidence_linkage_facet_assignments: all have evidence")

    # Novelty labels
    cnt = db.count_novelty_labels_without_evidence()
    if cnt > 0:
        results.append(ValidationResult(
            check_name="evidence_linkage_novelty_labels",
            severity="warning",
            details=f"{cnt} novelty_label(s) without evidence rows",
        ))
        logger.warning("  WARN evidence_linkage_novelty_labels: %d without evidence", cnt)
    else:
        logger.info("  PASS evidence_linkage_novelty_labels: all have evidence")

    return results


def check_evidence_id_determinism(
    db: Stage11DatabaseInterface, sample_size: int = DEFAULT_EVIDENCE_ID_SAMPLE_SIZE
) -> list[ValidationResult]:
    """Sample-verify that ``evidence_id == SHA256(doc_version_id|span_start|span_end)``."""
    results: list[ValidationResult] = []
    rows = db.sample_evidence_spans(limit=sample_size)
    logger.info("  Sampling %d evidence_span rows (requested %d)", len(rows), sample_size)
    mismatches = 0
    for r in rows:
        expected = compute_sha256_id(r["doc_version_id"], r["span_start"], r["span_end"])
        if r["evidence_id"] != expected:
            mismatches += 1
            if mismatches <= 3:
                logger.debug(
                    "    Mismatch: evidence_id=%s expected=%s (doc=%s, %d:%d)",
                    r["evidence_id"][:16], expected[:16],
                    r["doc_version_id"][:16], r["span_start"], r["span_end"],
                )
    if mismatches:
        results.append(ValidationResult(
            check_name="evidence_id_determinism",
            severity="error",
            details=(
                f"{mismatches}/{len(rows)} sampled evidence_span(s) have non-deterministic "
                f"evidence_id (expected SHA256(doc_version_id|span_start|span_end))"
            ),
        ))
        logger.warning("  FAIL evidence_id_determinism: %d/%d mismatches", mismatches, len(rows))
    else:
        logger.info("  PASS evidence_id_determinism: %d/%d match", len(rows), len(rows))
    return results


def check_span_conformance(
    db: Stage11DatabaseInterface, sample_size: int = DEFAULT_SPAN_CONFORMANCE_SAMPLE_SIZE
) -> list[ValidationResult]:
    """Verify that stored span text matches SQLite ``substr()`` of clean_content."""
    results: list[ValidationResult] = []

    # Evidence spans
    rows = db.sample_evidence_spans(limit=sample_size)
    logger.info("  Checking %d evidence_span rows for text conformance", len(rows))
    mismatches = 0
    for r in rows:
        verification = db.verify_span_text_with_sqlite(
            r["doc_version_id"], r["span_start"], r["span_end"]
        )
        if verification and verification["stored_text"] != verification["sqlite_text"]:
            mismatches += 1
            if mismatches <= 3:
                stored_preview = (verification["stored_text"] or "")[:60]
                sqlite_preview = (verification["sqlite_text"] or "")[:60]
                logger.debug(
                    "    Span mismatch (doc=%s %d:%d): stored=%r vs sqlite=%r",
                    r["doc_version_id"][:16], r["span_start"], r["span_end"],
                    stored_preview, sqlite_preview,
                )
    if mismatches:
        results.append(ValidationResult(
            check_name="span_conformance_evidence",
            severity="error",
            details=(
                f"{mismatches}/{len(rows)} evidence_span(s): stored text != "
                f"SQLite substr(clean_content, start+1, length)"
            ),
        ))
        logger.warning("  FAIL span_conformance_evidence: %d/%d mismatches", mismatches, len(rows))
    else:
        logger.info("  PASS span_conformance_evidence: %d/%d match", len(rows), len(rows))

    # Chunk spans
    chunk_rows = db.sample_chunk_spans(limit=sample_size)
    logger.info("  Checking %d chunk rows for text conformance", len(chunk_rows))
    chunk_mismatches = 0
    for cr in chunk_rows:
        if cr["chunk_text"] != cr["sqlite_text"]:
            chunk_mismatches += 1
            if chunk_mismatches <= 3:
                logger.debug(
                    "    Chunk mismatch (chunk=%s): stored=%r vs sqlite=%r",
                    cr["chunk_id"][:16],
                    (cr["chunk_text"] or "")[:60],
                    (cr["sqlite_text"] or "")[:60],
                )
    if chunk_mismatches:
        results.append(ValidationResult(
            check_name="span_conformance_chunks",
            severity="error",
            details=(
                f"{chunk_mismatches}/{len(chunk_rows)} chunk(s): chunk_text != "
                f"SQLite substr(clean_content, start+1, length)"
            ),
        ))
        logger.warning("  FAIL span_conformance_chunks: %d/%d mismatches", chunk_mismatches, len(chunk_rows))
    else:
        logger.info("  PASS span_conformance_chunks: %d/%d match", len(chunk_rows), len(chunk_rows))

    return results


def check_config_coherence(db: Stage11DatabaseInterface) -> list[ValidationResult]:
    """Ensure no mixed ``config_hash`` values among eligible docs."""
    results: list[ValidationResult] = []
    hashes = db.get_eligible_doc_config_hashes()
    distinct_values = [h["config_hash"] for h in hashes]
    if len(distinct_values) > 1:
        results.append(ValidationResult(
            check_name="config_coherence",
            severity="error",
            details=f"Eligible docs have {len(distinct_values)} distinct config_hash values: {distinct_values}",
        ))
        logger.warning("  FAIL config_coherence: %d distinct config_hash values: %s",
                        len(distinct_values), distinct_values)
    elif len(distinct_values) == 1:
        logger.info("  PASS config_coherence: single config_hash=%s", distinct_values[0][:16])
    else:
        logger.info("  PASS config_coherence: no eligible docs (vacuously true)")
    return results


def check_event_eligibility(db: Stage11DatabaseInterface) -> list[ValidationResult]:
    """Verify all ``doc_version_ids`` in current revisions have ``stage_08_events='ok'``."""
    results: list[ValidationResult] = []
    rev_rows = db.get_current_revision_doc_version_ids()
    logger.info("  Inspecting %d event(s) with current revisions", len(rev_rows))
    bad_events: list[str] = []
    for rev in rev_rows:
        raw_ids = rev["doc_version_ids"]
        if isinstance(raw_ids, str):
            doc_ids: list[str] = json.loads(raw_ids)
        else:
            doc_ids = raw_ids
        statuses = db.get_doc_stage_status_for_ids(doc_ids, "stage_08_events")
        for dvid in doc_ids:
            if statuses.get(dvid) != "ok":
                bad_events.append(rev["event_id"])
                break
    if bad_events:
        results.append(ValidationResult(
            check_name="event_eligibility",
            severity="error",
            details=(
                f"{len(bad_events)} event(s) have current-revision doc_version_ids "
                f"without stage_08_events='ok'. Sample: {bad_events[:10]}"
            ),
        ))
        logger.warning("  FAIL event_eligibility: %d/%d event(s) have ineligible doc_version_ids",
                        len(bad_events), len(rev_rows))
    else:
        logger.info("  PASS event_eligibility: all %d event(s) have eligible docs", len(rev_rows))
    return results


def check_ann_index_files(
    db: Stage11DatabaseInterface, run_id: str
) -> list[ValidationResult]:
    """Verify ANN index files referenced by ``embedding_index`` exist on disk."""
    results: list[ValidationResult] = []
    rows = db.get_embedding_index_rows(run_id)
    logger.info("  Checking %d embedding_index row(s) for file presence", len(rows))
    missing: list[str] = []
    for r in rows:
        idx_path = Path(r["index_path"])
        if not idx_path.exists():
            missing.append(r["index_id"])
            logger.debug("    Missing index file: %s (index_id=%s)", r["index_path"], r["index_id"][:16])
    if missing:
        results.append(ValidationResult(
            check_name="ann_index_file_presence",
            severity="error",
            details=f"{len(missing)}/{len(rows)} embedding_index row(s) reference missing files. IDs: {missing[:10]}",
        ))
        logger.warning("  FAIL ann_index_file_presence: %d/%d files missing", len(missing), len(rows))
    else:
        logger.info("  PASS ann_index_file_presence: all %d index file(s) exist", len(rows))
    return results


def check_orphaned_events(db: Stage11DatabaseInterface) -> list[ValidationResult]:
    """Detect events with NULL ``current_revision_id`` that have revision rows."""
    results: list[ValidationResult] = []
    orphans = db.get_orphaned_events()
    if orphans:
        ids = [o["event_id"] for o in orphans]
        results.append(ValidationResult(
            check_name="orphaned_events",
            severity="error",
            details=(
                f"{len(orphans)} event(s) with current_revision_id IS NULL but "
                f"event_revision rows exist. IDs: {ids[:10]}"
            ),
        ))
        logger.warning("  FAIL orphaned_events: %d event(s) orphaned", len(orphans))
    else:
        logger.info("  PASS orphaned_events: no orphaned events found")
    return results


def check_run_state_hygiene(
    db: Stage11DatabaseInterface, current_run_id: str
) -> list[ValidationResult]:
    """Validate pipeline_run state and artifact provenance."""
    results: list[ValidationResult] = []

    other_running = db.get_other_running_runs(current_run_id)
    if other_running:
        ids = [r["run_id"] for r in other_running]
        results.append(ValidationResult(
            check_name="run_state_other_running",
            severity="error",
            details=f"Other pipeline_run(s) with status='running': {ids}",
        ))
        logger.warning("  FAIL run_state_other_running: %d other running run(s): %s",
                        len(ids), [i[:16] for i in ids])
    else:
        logger.info("  PASS run_state_other_running: no concurrent running runs")

    abandoned = db.get_abandoned_runs()
    if abandoned:
        summaries = [
            f"{r['run_id'][:16]}…({r['status']}, started={r['started_at']})"
            for r in abandoned
        ]
        results.append(ValidationResult(
            check_name="run_state_abandoned_runs",
            severity="warning",
            details=f"{len(abandoned)} abandoned run(s): {summaries[:10]}",
        ))
        logger.warning("  WARN run_state_abandoned_runs: %d abandoned run(s)", len(abandoned))
        for s in summaries[:5]:
            logger.warning("    %s", s)
    else:
        logger.info("  PASS run_state_abandoned_runs: no abandoned runs")

    logger.info("  Checking created_in_run_id provenance across %d tables", len(TABLES_WITH_CREATED_IN_RUN_ID))
    provenance_issues = 0
    for table in TABLES_WITH_CREATED_IN_RUN_ID:
        cnt = db.count_non_completed_run_artifacts(current_run_id, table)
        if cnt > 0:
            results.append(ValidationResult(
                check_name=f"non_completed_run_artifacts_{table}",
                severity="error",
                details=(
                    f"{cnt} row(s) in '{table}' reference a non-completed, "
                    f"non-current pipeline_run via created_in_run_id"
                ),
            ))
            logger.warning("    FAIL %s: %d row(s) with bad created_in_run_id", table, cnt)
            provenance_issues += 1
        else:
            logger.debug("    PASS %s: provenance OK", table)

    if provenance_issues == 0:
        logger.info("  PASS run_artifact_provenance: all %d tables clean", len(TABLES_WITH_CREATED_IN_RUN_ID))

    return results


def check_single_embedding_model(db: Stage11DatabaseInterface) -> list[ValidationResult]:
    """Validate that all ``chunk_embedding.model_version`` values are identical."""
    results: list[ValidationResult] = []
    versions = db.get_distinct_embedding_model_versions()
    if len(versions) > 1:
        results.append(ValidationResult(
            check_name="single_embedding_model",
            severity="error",
            details=f"Multiple embedding model versions found: {versions}",
        ))
        logger.warning("  FAIL single_embedding_model: %d versions: %s", len(versions), versions)
    elif len(versions) == 1:
        logger.info("  PASS single_embedding_model: model_version=%s", versions[0])
    else:
        logger.info("  PASS single_embedding_model: no embeddings (vacuously true)")
    return results


def check_export_readiness(
    db: Stage11DatabaseInterface, run_id: str
) -> list[ValidationResult]:
    """
    Verify that prerequisite run-stage-status rows exist and key artifact tables are populated so that ``export_pipeline_data.py`` can function.

    This check produces *warnings*, not errors, because export readiness is
    advisory (a run may be intentionally partial).
    """
    results: list[ValidationResult] = []

    # Check that all prerequisite doc stages have at least some 'ok' docs
    logger.info("  Checking doc-stage coverage for export readiness")
    for sid in DOC_STAGE_IDS:
        cnt = db.count_ok_docs_for_stage(sid)
        if cnt == 0:
            results.append(ValidationResult(
                check_name=f"export_readiness_doc_stage_{sid}",
                severity="warning",
                details=f"No doc_stage_status rows with status='ok' for {sid}; export will show no data for this stage",
            ))
            logger.warning("    WARN %s: 0 docs with status='ok'", sid)
        else:
            logger.info("    %s: %d doc(s) with status='ok'", sid, cnt)

    # Check run-stage-status rows that the export expects
    stage_statuses = db.get_run_stage_statuses_for_run(run_id)
    logger.info("  Checking run-stage-status rows for export readiness (found %d rows)", len(stage_statuses))
    # stage_11_validation won't exist yet (we're writing it), so exclude it
    expected_prior = [s for s in RUN_STAGE_IDS if s != STAGE_NAME]
    for sid in expected_prior:
        status = stage_statuses.get(sid)
        if status is None:
            results.append(ValidationResult(
                check_name=f"export_readiness_run_stage_{sid}",
                severity="warning",
                details=f"No run_stage_status row for {sid}; export may show this stage as 'not started'",
            ))
            logger.warning("    WARN %s: no run_stage_status row", sid)
        elif status != "ok":
            results.append(ValidationResult(
                check_name=f"export_readiness_run_stage_{sid}",
                severity="warning",
                details=f"run_stage_status for {sid} is '{status}' (not 'ok')",
            ))
            logger.warning("    WARN %s: status='%s' (expected 'ok')", sid, status)
        else:
            logger.info("    %s: status='ok'", sid)

    # Check key artifact table row counts the export reads
    logger.info("  Checking run-scoped artifact table counts for export readiness")
    empty_tables: list[str] = []
    for table in EXPORT_RUN_ARTIFACT_TABLES:
        # Some artifact tables are run-scoped (have run_id), some are not
        try:
            cnt = db.count_table_rows(table, run_id=run_id)
        except Exception:
            # Table may not have run_id column; count all rows
            cnt = db.count_table_rows(table)
        logger.info("    %-40s %d row(s)", table, cnt)
        if cnt == 0:
            empty_tables.append(table)

    if empty_tables:
        results.append(ValidationResult(
            check_name="export_readiness_empty_artifact_tables",
            severity="warning",
            details=(
                f"{len(empty_tables)} artifact table(s) empty for this run: {empty_tables}; "
                f"export will show no data for affected sections"
            ),
        ))
        logger.warning("  WARN export_readiness: %d empty artifact table(s)", len(empty_tables))
    else:
        logger.info("  PASS export_readiness: all artifact tables populated")

    return results


# Orchestration
def log_database_audit(db: Stage11DatabaseInterface, run_id: str) -> None:
    """Log a comprehensive row-count audit of the database for visual inspection."""
    logger.info("═══ Database audit (row counts) ═══")

    # Key tables with total and per-run counts
    audit_tables = [
        ("pipeline_run",              None),
        ("document",                  None),
        ("document_version",          "created_in_run_id"),
        ("scrape_record",             None),
        ("evidence_span",             "created_in_run_id"),
        ("block",                     "created_in_run_id"),
        ("chunk",                     "created_in_run_id"),
        ("chunk_embedding",           "created_in_run_id"),
        ("doc_metadata",              "created_in_run_id"),
        ("table_extract",             "created_in_run_id"),
        ("mention",                   "created_in_run_id"),
        ("mention_link",              "created_in_run_id"),
        ("facet_assignment",          "created_in_run_id"),
        ("novelty_label",             "created_in_run_id"),
        ("document_fingerprint",      "created_in_run_id"),
        ("chunk_novelty",             None),
        ("chunk_novelty_score",       "created_in_run_id"),
        ("event",                     "created_in_run_id"),
        ("event_revision",            "created_in_run_id"),
        ("event_entity_link",         "created_in_run_id"),
        ("metric_observation",        "created_in_run_id"),
        ("event_candidate",           "created_in_run_id"),
        ("embedding_index",           None),
        ("story_cluster",             None),
        ("story_cluster_member",      None),
        ("metric_series",             None),
        ("metric_series_point",       None),
        ("alert",                     None),
        ("digest_item",               None),
        ("entity_timeline_item",      None),
        ("validation_failure",        None),
    ]

    for table, run_col in audit_tables:
        try:
            total = db.count_table_rows(table)
            if run_col:
                in_run = db.count_table_rows_by_run(table, run_id, id_column=run_col)
                logger.info("  %-40s total=%-6d  this_run=%-6d", table, total, in_run)
            else:
                # Try run_id column for run-scoped tables
                try:
                    in_run = db.count_table_rows(table, run_id=run_id)
                    logger.info("  %-40s total=%-6d  this_run=%-6d", table, total, in_run)
                except Exception:
                    logger.info("  %-40s total=%-6d", table, total)
        except Exception as e:
            logger.warning("  %-40s ERROR: %s", table, e)

    logger.info("═══ End database audit ═══")


def run_all_checks(
    db: Stage11DatabaseInterface,
    run_id: str,
    evidence_sample_size: int = DEFAULT_EVIDENCE_ID_SAMPLE_SIZE,
    span_sample_size: int = DEFAULT_SPAN_CONFORMANCE_SAMPLE_SIZE,
) -> list[ValidationResult]:
    """Execute every validation check and return aggregated results."""
    all_results: list[ValidationResult] = []

    logger.info("─── Check 1/10: Evidence linkage completeness ───")
    all_results.extend(check_evidence_linkage(db, run_id))

    logger.info("─── Check 2/10: evidence_id determinism (sample=%d) ───", evidence_sample_size)
    all_results.extend(check_evidence_id_determinism(db, sample_size=evidence_sample_size))

    logger.info("─── Check 3/10: Span conformance (sample=%d) ───", span_sample_size)
    all_results.extend(check_span_conformance(db, sample_size=span_sample_size))

    logger.info("─── Check 4/10: Config coherence across eligible docs ───")
    all_results.extend(check_config_coherence(db))

    logger.info("─── Check 5/10: Event eligibility ───")
    all_results.extend(check_event_eligibility(db))

    logger.info("─── Check 6/10: ANN index file presence ───")
    all_results.extend(check_ann_index_files(db, run_id))

    logger.info("─── Check 7/10: Orphaned events ───")
    all_results.extend(check_orphaned_events(db))

    logger.info("─── Check 8/10: Run-state hygiene ───")
    all_results.extend(check_run_state_hygiene(db, run_id))

    logger.info("─── Check 9/10: Single embedding model constraint ───")
    all_results.extend(check_single_embedding_model(db))

    logger.info("─── Check 10/10: Export readiness ───")
    all_results.extend(check_export_readiness(db, run_id))

    return all_results


def persist_results(
    db: Stage11DatabaseInterface,
    run_id: str,
    results: list[ValidationResult],
) -> None:
    """Write ``ValidationFailureRow`` for each result."""
    for idx, r in enumerate(results):
        suffix = f"{r.check_name}:{idx}"
        failure_id = _make_failure_id(run_id, r.check_name, suffix)
        db.insert_validation_failure(ValidationFailureRow(
            failure_id=failure_id,
            run_id=run_id,
            stage=STAGE_NAME,
            doc_version_id=r.doc_version_id,
            check_name=r.check_name,
            details=r.details,
            severity=r.severity,
            auto_repaired=0,
        ))
    logger.info("Persisted %d validation_failure row(s) to database", len(results))


def run_stage(
    db: Stage11DatabaseInterface,
    run_id: str,
    config_version: str,
    prerequisite_stage: str = PREREQUISITE_STAGE,
    evidence_sample_size: int = DEFAULT_EVIDENCE_ID_SAMPLE_SIZE,
    span_sample_size: int = DEFAULT_SPAN_CONFORMANCE_SAMPLE_SIZE,
) -> int:
    """
    Execute stage 11 validation within a single transaction.

    :param db: Opened database interface.
    :param run_id: Current pipeline run ID.
    :param config_version: Config hash for ``run_stage_status``.
    :param prerequisite_stage: Stage that must be 'ok' before this stage runs.
    :param evidence_sample_size: Number of evidence spans to sample for determinism.
    :param span_sample_size: Number of spans to sample for text conformance.
    :return: 0 on success, 1 on fatal error.
    """
    logger.info("═══ Stage 11 Validation: prerequisite check ═══")
    logger.info("  Prerequisite stage: %s", prerequisite_stage)

    prerequisite = db.get_run_stage_status(run_id, prerequisite_stage)
    if prerequisite is None or prerequisite.status != "ok":
        status_str = prerequisite.status if prerequisite else "missing"
        msg = (
            f"Prerequisite {prerequisite_stage} not satisfied "
            f"(status={status_str})"
        )
        logger.error(msg)
        db.upsert_run_stage_status(
            run_id=run_id,
            stage=STAGE_NAME,
            config_hash=config_version,
            status="failed",
            error_message=msg,
        )
        return 1

    logger.info("  Prerequisite %s satisfied (status='ok')", prerequisite_stage)

    # Database audit before validation
    log_database_audit(db, run_id)

    with db.transaction():
        deleted = db.delete_validation_failures_for_run_stage(run_id, STAGE_NAME)
        if deleted:
            logger.info("Deleted %d prior validation_failure rows for this run/stage", deleted)
        else:
            logger.info("No prior validation_failure rows to delete")

        logger.info("═══ Running validation checks ═══")
        results = run_all_checks(
            db, run_id,
            evidence_sample_size=evidence_sample_size,
            span_sample_size=span_sample_size,
        )

        persist_results(db, run_id, results)

        errors = [r for r in results if r.severity == "error"]
        warnings = [r for r in results if r.severity == "warning"]

        logger.info("═══ Validation summary ═══")
        logger.info("  Total checks producing findings: %d", len(results))
        logger.info("  Errors:   %d", len(errors))
        logger.info("  Warnings: %d", len(warnings))

        if errors:
            logger.info("  Error details:")
            for e in errors:
                logger.info("    [ERROR] %s: %s", e.check_name, e.details)
        if warnings:
            logger.info("  Warning details:")
            for w in warnings:
                logger.info("    [WARN]  %s: %s", w.check_name, w.details)

        # Stage status is always 'ok' — errors are recorded in validation_failure
        # and consumers read validation_failure directly. This allows the pipeline
        # to complete and the export to include the validation results.
        details_json = json.dumps(
            {"errors": len(errors), "warnings": len(warnings), "total_checks": len(results)},
            sort_keys=True,
        )
        db.upsert_run_stage_status(
            run_id=run_id,
            stage=STAGE_NAME,
            config_hash=config_version,
            status="ok",
            details=details_json,
        )
        logger.info("  run_stage_status set to 'ok' (errors are advisory, stored in validation_failure)")

    if errors:
        logger.warning(
            "%d validation error(s) recorded — see validation_failure table for details",
            len(errors),
        )

    logger.info("═══ Stage 11 complete (exit_code=0) ═══")
    return 0


# CLI entry point
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stage 11: Final cross-table validation")
    parser.add_argument(
        "--run-id",
        type=str,
        default="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        help="Pipeline run ID (64-char hex SHA256). Required.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("../../../config/"),
        help="Directory containing config.yaml (default: ../../../config/)",
    )
    parser.add_argument(
        "--working-db",
        type=Path,
        default=Path("../../../database/processed_posts.db"),
        help="Path to the working database (default: ../../../database/processed_posts.db)",
    )
    parser.add_argument(
        "--prerequisite-stage",
        type=str,
        default=PREREQUISITE_STAGE,
        help=f"Stage that must be 'ok' before validation runs (default: {PREREQUISITE_STAGE})",
    )
    parser.add_argument(
        "--evidence-sample-size",
        type=int,
        default=DEFAULT_EVIDENCE_ID_SAMPLE_SIZE,
        help=f"Number of evidence spans to sample for determinism checks (default: {DEFAULT_EVIDENCE_ID_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--span-sample-size",
        type=int,
        default=DEFAULT_SPAN_CONFORMANCE_SAMPLE_SIZE,
        help=f"Number of spans to sample for text conformance checks (default: {DEFAULT_SPAN_CONFORMANCE_SAMPLE_SIZE})",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug-level logging")
    return parser.parse_args()


def main_stage_11_validation() -> int:
    """
    Set entry point for stage 11.

    :return: 0 on success, 1 on fatal error.
    """
    args = parse_args()

    config = load_config(args.config_dir / "config.yaml")
    config_version = get_config_version(config)

    # Log all input parameters
    logger.info("═══ Stage 11 Validation: starting ═══")
    logger.info("  run_id:              %s", args.run_id)
    logger.info("  config_dir:          %s", args.config_dir.resolve())
    logger.info("  working_db:          %s", args.working_db.resolve())
    logger.info("  config_version:      %s", config_version[:32] + "…" if len(config_version) > 32 else config_version)
    logger.info("  prerequisite_stage:  %s", args.prerequisite_stage)
    logger.info("  evidence_sample:     %d", args.evidence_sample_size)
    logger.info("  span_sample:         %d", args.span_sample_size)
    logger.info("  db exists:           %s", args.working_db.exists())

    db = Stage11DatabaseInterface(working_db_path=args.working_db)
    try:
        db.open()
        logger.info("  Database connection opened successfully")
        return run_stage(
            db,
            args.run_id,
            config_version,
            prerequisite_stage=args.prerequisite_stage,
            evidence_sample_size=args.evidence_sample_size,
            span_sample_size=args.span_sample_size,
        )
    except Exception:
        logger.exception("Fatal error in stage 11 validation")
        try:
            db.upsert_run_stage_status(
                run_id=args.run_id,
                stage=STAGE_NAME,
                config_hash=config_version,
                status="failed",
                error_message="Fatal exception; see logs.",
            )
        except Exception:
            logger.exception("Failed to record stage failure status")
        return 1
    finally:
        db.close()
        logger.info("Database connection closed")


if __name__ == "__main__":
    sys.exit(main_stage_11_validation())