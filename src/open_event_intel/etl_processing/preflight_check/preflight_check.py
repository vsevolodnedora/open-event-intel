"""
Pre-flight check: determine whether a new pipeline run is needed.

Compares publications in the source database (``preprocessed_posts.db``)
against already-ingested ``scrape_record`` rows in the working database
(``processed_posts.db``), **and** inspects ``doc_stage_status`` for
documents that were ingested but did not complete all pipeline stages.

Uses the same deterministic ID scheme as Stage 01 —
``scrape_id = SHA256(publisher_id | source_id | page)`` — so results
are exactly consistent with what Stage 01 would observe.

Exit codes:
    0: Pipeline **should** run (new publications and/or retryable documents).
    1: Fatal error (missing DB, bad schema, etc.).
    2: No new publications **and** no retryable documents — pipeline run
       **not** needed.

Usage::

    python preflight_should_run.py \\
        --source-db ../../database/preprocessed_posts.db \\
        --working-db ../../database/processed_posts.db

The script is intentionally **read-only** and never modifies either database.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from open_event_intel.etl_processing.database_interface import (
    DBError,
    DBSchemaError,
    compute_sha256_id,
)
from open_event_intel.etl_processing.preflight_check.database_preflight_check import (
    PreflightDatabaseInterface,
)
from open_event_intel.logger import get_logger

logger = get_logger(__name__)

SCRAPE_KIND = "page"

# Mapping kept in sync with Stage 01's PUBLISHER_TABLE_MAP.
PUBLISHER_TABLE_MAP: dict[str, str] = {
    "SMARD": "smard",
    "EEX": "eex",
    "ENTSOE": "entsoe",
    "ACER": "acer",
    "EC": "ec",
    "BNETZA": "bnetza",
    "TRANSNETBW": "transnetbw",
    "TENNET": "tennet",
    "FIFTY_HERTZ": "fifty_hertz",
    "AMPRION": "amprion",
    "ICIS": "icis",
    "AGORA": "agora",
    "ENERGY_WIRE": "energy_wire",
}


class PublisherDelta(BaseModel):
    """Per-publisher comparison between source and working databases."""

    publisher_id: str
    source_table: str
    source_count: int
    ingested_count: int
    new_count: int


class StageFailureSummary(BaseModel):
    """Per-stage count of problematic documents."""

    stage: str
    failed_count: int = 0
    blocked_count: int = 0


class PipelineHealthSummary(BaseModel):
    """Health overview of ingested documents across pipeline stages.

    Distinguishes *failed* documents (root-cause processing error at
    one or more stages) from *blocked-only* documents (all non-ok stages
    are ``'blocked'``, meaning they are waiting on a failed or skipped
    prerequisite and never attempted processing themselves).
    """

    model_config = ConfigDict(extra="forbid")

    total_documents: int
    retryable_count: int
    failed_count: int
    blocked_only_count: int
    stage_failures: list[StageFailureSummary]


class PreflightResult(BaseModel):
    """Aggregated result of the pre-flight check."""

    model_config = ConfigDict(extra="forbid")

    should_run: bool
    reason: str
    working_db_exists: bool
    has_completed_runs: bool
    publisher_deltas: list[PublisherDelta]
    total_source: int
    total_ingested: int
    total_new: int
    pipeline_health: PipelineHealthSummary | None
    has_retryable_work: bool


def compute_publisher_delta(
    db: PreflightDatabaseInterface,
    publisher_id: str,
    table_name: str,
) -> PublisherDelta:
    """Compare source publications against ingested records for one publisher.

    Computes the deterministic ``scrape_id`` for every source row and checks
    which IDs are missing from ``scrape_record``.

    :param db: Open database interface with both source and working connections.
    :param publisher_id: Canonical publisher key (e.g. ``"ENTSOE"``).
    :param table_name: Corresponding source-DB table name (e.g. ``"entsoe"``).
    :return: Delta summary for this publisher.
    """
    source_ids = db.get_source_publication_ids(table_name)
    expected_scrape_ids = {
        compute_sha256_id(publisher_id, sid, SCRAPE_KIND)
        for sid in source_ids
    }
    existing_scrape_ids = db.get_ingested_scrape_ids_for_publisher(publisher_id)
    new_ids = expected_scrape_ids - existing_scrape_ids

    return PublisherDelta(
        publisher_id=publisher_id,
        source_table=table_name,
        source_count=len(source_ids),
        ingested_count=len(existing_scrape_ids),
        new_count=len(new_ids),
    )


def analyse_pipeline_health(db: PreflightDatabaseInterface) -> PipelineHealthSummary:
    """Inspect ``doc_stage_status`` for retryable documents.

    :param db: Open database interface.
    :return: Structured health summary.
    """
    total_documents = db.get_total_document_count()
    failed_ids, blocked_only_ids = db.get_retryable_doc_version_ids()
    raw_counts = db.get_stage_failure_counts()

    stage_map: dict[str, StageFailureSummary] = {}
    for stage, status, cnt in raw_counts:
        if stage not in stage_map:
            stage_map[stage] = StageFailureSummary(stage=stage)
        entry = stage_map[stage]
        if status == "failed":
            stage_map[stage] = entry.model_copy(update={"failed_count": cnt})
        elif status == "blocked":
            stage_map[stage] = entry.model_copy(update={"blocked_count": cnt})

    stage_failures = sorted(stage_map.values(), key=lambda s: s.stage)

    return PipelineHealthSummary(
        total_documents=total_documents,
        retryable_count=len(failed_ids) + len(blocked_only_ids),
        failed_count=len(failed_ids),
        blocked_only_count=len(blocked_only_ids),
        stage_failures=stage_failures,
    )


def _build_reason(
    has_completed: bool,
    total_new: int,
    retryable: int,
) -> tuple[bool, str]:
    """Decide whether the pipeline should run and why.

    Only genuinely new (never-attempted) source publications trigger a run.
    Retryable documents (failed/blocked at downstream stages) already have
    ``scrape_record`` rows and will be skipped by Stage 01 as
    ``already_ingested``; they are reported as informational metadata but
    do **not** influence the should-run decision.

    :return: ``(should_run, human_readable_reason)``
    """
    if not has_completed:
        return True, "no_completed_runs"
    if total_new > 0:
        reason = f"new_publications={total_new}"
        if retryable > 0:
            reason += f" (also {retryable} retryable documents for downstream stages)"
        return True, reason
    return False, "up_to_date"


def run_preflight_check(
    source_db_path: Path,
    working_db_path: Path,
) -> PreflightResult:
    """Execute the full pre-flight comparison.

    :param source_db_path: Path to the read-only source database.
    :param working_db_path: Path to the working database.
    :return: Structured result indicating whether a pipeline run is needed.
    :raises FileNotFoundError: If the source database does not exist.
    :raises DBSchemaError: If the working database schema is invalid.
    """
    if not source_db_path.exists():
        raise FileNotFoundError(f"Source database not found: {source_db_path}")

    if not working_db_path.exists():
        logger.info("Working database does not exist — pipeline should run")
        return PreflightResult(
            should_run=True,
            reason="working_db_missing",
            working_db_exists=False,
            has_completed_runs=False,
            publisher_deltas=[],
            total_source=0,
            total_ingested=0,
            total_new=0,
            pipeline_health=None,
            has_retryable_work=False,
        )

    db = PreflightDatabaseInterface(
        working_db_path=working_db_path,
        source_db_path=source_db_path,
    )

    try:
        db.open()
    except DBSchemaError:
        raise
    except Exception as exc:
        raise DBError(f"Failed to open databases: {exc}") from exc

    try:
        has_completed = db.has_completed_runs()
        if not has_completed:
            logger.info("No completed runs in working DB — pipeline should run")

        available_tables = set(db.get_source_table_names())
        deltas: list[PublisherDelta] = []

        for publisher_id in sorted(PUBLISHER_TABLE_MAP):
            table_name = PUBLISHER_TABLE_MAP[publisher_id]
            if table_name not in available_tables:
                logger.debug(
                    "Source table '%s' not present, skipping publisher %s",
                    table_name,
                    publisher_id,
                )
                continue

            delta = compute_publisher_delta(db, publisher_id, table_name)
            deltas.append(delta)

            if delta.new_count > 0:
                logger.info(
                    "%-15s  source=%d  ingested=%d  new=%d",
                    publisher_id,
                    delta.source_count,
                    delta.ingested_count,
                    delta.new_count,
                )
            else:
                logger.debug(
                    "%-15s  source=%d  ingested=%d  new=0  (up to date)",
                    publisher_id,
                    delta.source_count,
                    delta.ingested_count,
                )

        total_source = sum(d.source_count for d in deltas)
        total_ingested = sum(d.ingested_count for d in deltas)
        total_new = sum(d.new_count for d in deltas)

        health = analyse_pipeline_health(db)
        _log_pipeline_health(health)

        should_run, reason = _build_reason(
            has_completed, total_new, health.retryable_count,
        )

        has_retryable_work = health.retryable_count > 0

        if has_retryable_work and total_new == 0 and has_completed:
            logger.info(
                "Note: %d retryable documents exist from downstream stages, "
                "but no new source publications to ingest — stage_01 would "
                "skip all publications as already_ingested",
                health.retryable_count,
            )

        return PreflightResult(
            should_run=should_run,
            reason=reason,
            working_db_exists=True,
            has_completed_runs=has_completed,
            publisher_deltas=deltas,
            total_source=total_source,
            total_ingested=total_ingested,
            total_new=total_new,
            pipeline_health=health,
            has_retryable_work=has_retryable_work,
        )
    finally:
        db.close()


def _log_pipeline_health(health: PipelineHealthSummary) -> None:
    """Emit structured log lines for the pipeline health summary."""
    if health.retryable_count == 0:
        logger.info(
            "Pipeline health: all %d documents fully processed (no retryable work)",
            health.total_documents,
        )
        return

    logger.info(
        "Pipeline health: %d total documents, %d retryable "
        "(%d failed, %d blocked-only)",
        health.total_documents,
        health.retryable_count,
        health.failed_count,
        health.blocked_only_count,
    )
    for sf in health.stage_failures:
        logger.info(
            "  %-25s  failed=%d  blocked=%d",
            sf.stage,
            sf.failed_count,
            sf.blocked_count,
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Pre-flight check: determine whether new source publications "
            "or retryable document failures require a pipeline run."
        ),
    )
    parser.add_argument(
        "--source-db",
        type=Path,
        default=Path("../../../../database/preprocessed_posts.db"),
        help="Path to preprocessed_posts.db (read-only source)",
    )
    parser.add_argument(
        "--working-db",
        type=Path,
        default=Path("../../../../database/processed_posts.db"),
        help="Path to processed_posts.db (working database)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    return parser.parse_args()


def main() -> int:
    """
    Entry point for the pre-flight check.

    :return: Exit code — 0 (should run), 1 (error), 2 (no update needed).
    """
    args = parse_args()

    if args.verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Pre-flight check: source=%s  working=%s", args.source_db, args.working_db)

    try:
        result = run_preflight_check(
            source_db_path=args.source_db,
            working_db_path=args.working_db,
        )
    except FileNotFoundError as exc:
        logger.error("Pre-flight check failed: %s", exc)
        return 1
    except DBSchemaError as exc:
        logger.error("Schema validation failed: %s", exc)
        return 1
    except DBError as exc:
        logger.error("Database error: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error during pre-flight check: %s", exc)
        return 1

    if result.should_run:
        health_msg = ""
        if result.pipeline_health is not None and result.has_retryable_work:
            h = result.pipeline_health
            health_msg = (
                f"  downstream_retryable={h.retryable_count} "
                f"(failed={h.failed_count}, blocked_only={h.blocked_only_count})"
            )
        logger.info(
            "SHOULD RUN [%s] — total_source=%d  ingested=%d  new=%d%s",
            result.reason,
            result.total_source,
            result.total_ingested,
            result.total_new,
            health_msg,
        )
        return 0

    # Not running — report reason
    retryable_msg = ""
    if result.has_retryable_work and result.pipeline_health is not None:
        retryable_msg = (
            f" ({result.pipeline_health.retryable_count} retryable documents "
            "exist for downstream stages but do not require stage_01 re-run)"
        )
    logger.info(
        "NO UPDATE NEEDED — all %d source publications ingested, "
        "0 new publications%s",
        result.total_source,
        retryable_msg,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())