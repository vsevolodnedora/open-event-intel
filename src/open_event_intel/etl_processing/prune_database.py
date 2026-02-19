"""
prune_database.py - Pre-pipeline database retention pruning.

Ensures the working database retains at most N latest completed pipeline runs
(default N=5). Designed to run BEFORE the pipeline creates a new run, so it
keeps N-1 completed runs, leaving room for the upcoming run to fill the Nth slot.

**Strategy:** copy retained data into a fresh database file and atomically swap
it in. This avoids slow/fragile in-place deletes, reclaims disk space without
VACUUM, and produces a clean, contiguous file.

**Document chain retention:** because the pipeline is incremental and cumulative
(run-scoped outputs reference ALL eligible documents across ALL prior runs), the
full document chain is always retained. Pruning targets run-scoped output tables
(metrics, alerts, digests, timelines, clusters, indices) and per-run logs. The
copy-and-swap still reclaims space from SQLite page fragmentation.

**Why ``pipeline_run`` rows are fully copied (not filtered):**
32 columns across the schema carry ``REFERENCES pipeline_run(run_id)`` FKs —
including ``scrape_record.ingest_run_id``, ``document_version.created_in_run_id``,
``evidence_span.created_in_run_id``, ``block.created_in_run_id``,
``chunk.created_in_run_id``, ``mention.created_in_run_id``, etc. Since the latest
completed run's outputs (metric_series_point, alerts, digests, timelines)
reference documents from ALL prior runs, deleting a ``pipeline_run`` row would
require cascading deletion of the entire transitive document chain — breaking
the kept runs' outputs. ``pipeline_run`` itself is one small row per run.

**Invariant:** the pruned database passes ``PRAGMA integrity_check`` and
``PRAGMA foreign_key_check`` with zero violations.

### Review

The script **does prune run-scoped output tables down to the latest runs**, but it **does not literally “leave only the latest N=5 runs” in the database**, and it has **two important operational risks** (WAL handling during backup/swap, and unbounded retention of non-completed runs).

---

### What it gets right (schema + design alignment)

* **Correct high-level approach for SQLite size control:** copy retained data into a fresh DB and swap it in. This **reclaims space** without relying on slow in-place deletes/VACUUM and reduces fragmentation.
* **Schema preservation is robust:** it **extracts DDL from `sqlite_master`**, recreates tables/indexes, then loads data, rebuilds FTS, and reapplies triggers. This generally keeps the pruned DB structurally consistent with the source DB.
* **Table coverage matches the provided schema:** every table in `database_schema.sql` is accounted for as either:

  * always-copied (document chain / cumulative tables),
  * run-filtered (run-scoped outputs/logs),
  * or join-filtered (evidence tables tied to retained parents),
    and `chunk_fts` is rebuilt.
* **Integrity verification is strong:** it runs `PRAGMA integrity_check` and `PRAGMA foreign_key_check` and fails hard on violations.

This matches the project’s “incremental and cumulative” intent: **retain the full document chain**, prune **derived per-run outputs**.

---

### Where it does *not* meet the stated “latest N=5 runs” claim

1. **It keeps *all* rows in `pipeline_run`.**
   `pipeline_run` is in `FULL_COPY_TABLES`, so even “pruned” completed runs remain present as rows with `status='completed'`.

   * Practically: you prune *outputs* for old runs, but the DB still “contains” those runs in metadata.
   * This is partly understandable because many cumulative tables reference `pipeline_run` via `created_in_run_id` / `ingest_run_id`, so **deleting old run rows would break FKs** unless you redesign those references.

2. **Off-by-one behavior is intentional but easy to misapply.**
   `keep_n = max_completed_runs - 1`, so with `--max-runs 5` it keeps **4 completed runs** *before* starting the next run (so the upcoming run becomes the 5th).

   * If someone runs this **after** a successful run, it will typically leave **only 4 completed runs’ outputs**, not 5.

**So:** it prunes to “N=5 after the next run,” not “N=5 right now,” and it prunes outputs rather than deleting run records.

---

### Risks / gaps that can break correctness or “manageable size”

1. **WAL/SHM deletion before backup can lose data / produce an incomplete backup.**
   `_swap_databases()` unlinks `original-wal` / `original-shm` **before** copying the original file to backup.

   * If the source DB was in WAL mode and not fully checkpointed, the main `processed_posts.db` file may not contain the latest committed state without its WAL.
   * Result: the “backup” may be stale/incomplete, and deleting WAL can discard data you expected to preserve.

2. **Non-completed runs are kept forever (and their run-scoped outputs are kept too).**
   `kept_run_ids = kept_completed ∪ all_non_completed`. If you accumulate many `failed/aborted` runs (possibly with heavy tables like `embedding_index`, clusters, etc.), size may still grow without bound.

3. **Nullable run_id rows are silently dropped in some “run-scoped” tables.**
   `llm_usage_log.run_id` and `validation_failure.run_id` are nullable in the schema. Any rows with `run_id IS NULL` will not match the `IN (...)` filter and will be lost.

4. **Database-only pruning may not reduce total disk usage materially.**
   Tables like `embedding_index` store paths to external index files; pruning DB rows **does not delete those external artifacts**, so overall storage can still grow.

---

### Practical recommendations (minimal changes, highest impact)

* **Fix WAL safety during swap/backup:**

  * checkpoint the *original* DB before copying/removing WAL, **or** copy the DB file *together with* its `-wal` and `-shm` as the backup set.
* **Make retention semantics explicit:**

  * rename flags/docs to “keep N after next run” or add a mode to keep **exactly N completed outputs now**.
* **Bound non-completed retention:**

  * keep at most the single `running` run (schema enforces max 1) and optionally the last K failed/aborted, or none.
* **Preserve NULL-run rows where intended:**

  * for tables with nullable `run_id`, copy `run_id IS NULL OR run_id IN (...)` if those rows are meaningful.
* **Consider additional size controls beyond runs:**

  * expired `llm_cache` cleanup and external artifact cleanup (indexes/exports) if “manageable size” is a real operational requirement.

---

### Bottom line

* **Correct for pruning *run-scoped outputs* in line with the “retain full document chain” design.**
* **Not correct if interpreted as “the DB contains only the latest 5 runs.”** It keeps all `pipeline_run` rows and all document-chain references.
* **Has a serious WAL/backup hazard** and **unbounded retention of non-completed runs** that can undermine correctness and long-term size control.

"""
import argparse
import logging
import os
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence

from open_event_intel.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PruneConfig:
    """Validated configuration for the pruning operation."""

    db_path: Path
    max_completed_runs: int = 5
    backup_suffix: str = ".pre_prune_backup"

    def __post_init__(self) -> None:  # noqa: D105
        if not isinstance(self.db_path, Path):
            object.__setattr__(self, "db_path", Path(self.db_path))
        if not self.db_path.is_file():
            raise ValueError(f"Database file does not exist: {self.db_path}")
        if self.max_completed_runs < 2:
            raise ValueError(
                f"max_completed_runs must be >= 2, got {self.max_completed_runs}"
            )


class RunStatus(str, Enum):  # noqa: D101
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


_VALID_STATUSES = frozenset(s.value for s in RunStatus)


@dataclass(frozen=True)
class PipelineRunRow:
    """Validated row from ``pipeline_run``."""

    run_id: str
    started_at: str
    completed_at: str | None
    config_version: str
    status: str

    def __post_init__(self) -> None:  # noqa: D105
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"Unexpected pipeline_run.status={self.status!r}")


@dataclass(frozen=True)
class RetentionDecision:
    """Outcome of the retention analysis."""

    kept_run_ids: frozenset[str]
    pruned_run_ids: frozenset[str]
    total_runs: int
    completed_before: int
    completed_after: int
    non_completed_kept: int


@dataclass
class PruneStats:
    """Metrics collected during the pruning operation."""

    bytes_before: int = 0
    bytes_after: int = 0
    rows_copied: dict[str, int] = field(default_factory=dict)
    tables_processed: int = 0
    elapsed_seconds: float = 0.0


# Tables where ALL rows are always retained (document chain + reference data).
# Order respects FK dependencies for safe insertion.
#
# pipeline_run is fully copied because 32 FK columns across the schema reference
# it (scrape_record.ingest_run_id, document_version.created_in_run_id, etc.).
# Deleting a pipeline_run row would cascade into the document chain and break
# the latest run's outputs. See module docstring for full rationale.
FULL_COPY_TABLES: Sequence[str] = (
    "pipeline_run",
    "entity_registry",
    "scrape_record",
    "document",
    "document_version",
    "evidence_span",
    "doc_stage_status",
    "block",
    "chunk",
    "table_extract",
    "doc_metadata",
    "mention",
    "mention_link",
    "chunk_embedding",
    "facet_assignment",
    "facet_assignment_evidence",
    "novelty_label",
    "novelty_label_evidence",
    "chunk_novelty",
    "chunk_novelty_score",
    "document_fingerprint",
    "event",
    "event_revision",
    "event_revision_evidence",
    "event_entity_link",
    "metric_observation",
    "event_candidate",
    "registry_update_proposal",
    "entity_registry_audit",
    "watchlist",
    "alert_rule",
    "llm_cache",
)

# Run-scoped tables filtered by kept run_ids.
# Tuple of (table_name, run_id_column).
RUN_SCOPED_TABLES: Sequence[tuple[str, str]] = (
    ("run_stage_status", "run_id"),
    ("embedding_index", "run_id"),
    ("story_cluster", "run_id"),
    ("story_cluster_member", "run_id"),
    ("metric_series", "run_id"),
    ("metric_series_point", "run_id"),
    ("alert", "run_id"),
    ("digest_item", "run_id"),
    ("entity_timeline_item", "run_id"),
    ("llm_usage_log", "run_id"),
    ("validation_failure", "run_id"),
)

# Evidence/join tables whose parent lives in a run-scoped table.
# Tuple of (child_table, child_fk_col, parent_table, parent_pk_col).
RUN_SCOPED_JOIN_TABLES: Sequence[tuple[str, str, str, str]] = (
    ("alert_evidence", "alert_id", "alert", "alert_id"),
    ("digest_item_evidence", "item_id", "digest_item", "item_id"),
    ("entity_timeline_item_evidence", "item_id", "entity_timeline_item", "item_id"),
)


def determine_retention(
    conn: sqlite3.Connection,
    max_completed: int,
) -> RetentionDecision:
    """
    Analyse ``pipeline_run`` and decide which runs to keep.

    :param conn: Read-only connection to the source database.
    :param max_completed: Target number of completed runs AFTER the upcoming
        pipeline run finishes. We keep ``max_completed - 1`` here.
    :return: A :class:`RetentionDecision` with kept/pruned sets.
    :raises ValueError: If the database has no pipeline_run rows or the query
        returns unexpected data.
    """
    rows = conn.execute(
        "SELECT run_id, started_at, completed_at, config_version, status "
        "FROM pipeline_run ORDER BY completed_at DESC NULLS FIRST, started_at DESC"
    ).fetchall()

    if not rows:
        raise ValueError("pipeline_run table is empty; nothing to prune")

    all_runs = [
        PipelineRunRow(
            run_id=r[0],
            started_at=r[1],
            completed_at=r[2],
            config_version=r[3],
            status=r[4],
        )
        for r in rows
    ]

    completed = [r for r in all_runs if r.status == RunStatus.COMPLETED]
    non_completed = [r for r in all_runs if r.status != RunStatus.COMPLETED]

    keep_n = max_completed - 1  # off-by-one: leave room for the upcoming run

    # Sort completed runs by completed_at descending (deterministic tiebreak).
    completed_sorted = sorted(
        completed,
        key=lambda r: (r.completed_at or "", r.run_id),
        reverse=True,
    )

    kept_completed = completed_sorted[:keep_n]
    pruned_completed = completed_sorted[keep_n:]

    # Non-completed runs (running/failed/aborted) are always kept:
    # - 'running' is the active run (at most 1, enforced by unique index)
    # - 'failed'/'aborted' may be resumed (spec §6.2 step 3)
    # These are single rows in pipeline_run and not a size concern.
    kept_ids = frozenset(r.run_id for r in kept_completed) | frozenset(
        r.run_id for r in non_completed
    )
    pruned_ids = frozenset(r.run_id for r in pruned_completed)

    decision = RetentionDecision(
        kept_run_ids=kept_ids,
        pruned_run_ids=pruned_ids,
        total_runs=len(all_runs),
        completed_before=len(completed),
        completed_after=len(kept_completed),
        non_completed_kept=len(non_completed),
    )

    logger.info(
        "Retention decision: %d total runs, %d completed (%d kept, %d pruned), "
        "%d non-completed kept",
        decision.total_runs,
        decision.completed_before,
        decision.completed_after,
        len(pruned_ids),
        decision.non_completed_kept,
    )
    if kept_completed:
        logger.info(
            "Kept completed run_ids (by completed_at desc): %s",
            [r.run_id for r in kept_completed],
        )
    if pruned_completed:
        logger.info(
            "Pruned completed run_ids: %s",
            [r.run_id for r in pruned_completed],
        )

    return decision


def _extract_schema_ddl(
    conn: sqlite3.Connection,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Extract DDL statements from ``sqlite_master``, grouped by type.

    Skips FTS5 shadow tables (``chunk_fts_*``) which are internal to the virtual
    table engine and must not be recreated manually. The actual FTS maintenance
    triggers (``chunk_ai``, ``chunk_ad``, ``chunk_au``) do NOT match this prefix
    and are correctly included in the trigger list.

    :return: ``(table_ddls, index_ddls, trigger_ddls, virtual_table_ddls)``
    """
    tables: list[str] = []
    indexes: list[str] = []
    triggers: list[str] = []
    virtual_tables: list[str] = []

    rows = conn.execute(
        "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY rowid"
    ).fetchall()

    for obj_type, name, sql in rows:
        if name.startswith("sqlite_") or name.startswith("chunk_fts_"):
            continue

        if obj_type == "table":
            if "VIRTUAL TABLE" in sql.upper() or name == "chunk_fts":
                virtual_tables.append(sql)
            else:
                tables.append(sql)
        elif obj_type == "index":
            indexes.append(sql)
        elif obj_type == "trigger":
            triggers.append(sql)

    return tables, indexes, triggers, virtual_tables


def _create_schema_tables(dst: sqlite3.Connection, tables: list[str]) -> None:
    """Create table DDL only (indexes deferred until after bulk load)."""
    dst.execute("PRAGMA journal_mode = WAL")
    dst.execute("PRAGMA foreign_keys = OFF")

    for ddl in tables:
        dst.execute(ddl)

    dst.commit()


def _create_indexes(dst: sqlite3.Connection, indexes: list[str]) -> None:
    """Create indexes after bulk data loading for better performance."""
    for ddl in indexes:
        try:
            dst.execute(ddl)
        except sqlite3.OperationalError as exc:
            logger.warning("Index creation skipped (%s): %.120s", exc, ddl)
    dst.commit()
    logger.info("Created %d indexes", len(indexes))


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Return column names for a regular (non-virtual) table."""
    rows = conn.execute(f"PRAGMA table_info([{table}])").fetchall()
    return [r[1] for r in rows]


def _copy_table_full(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    table: str,
    stats: PruneStats,
) -> None:
    """Copy all rows from *table* in *src* to *dst*."""
    cols = _table_columns(src, table)
    if not cols:
        return

    col_list = ", ".join(f"[{c}]" for c in cols)
    placeholders = ", ".join("?" for _ in cols)
    select_sql = f"SELECT {col_list} FROM [{table}]"
    insert_sql = f"INSERT INTO [{table}] ({col_list}) VALUES ({placeholders})"

    cursor = src.execute(select_sql)
    batch: list[tuple] = []
    count = 0
    batch_size = 5000

    for row in cursor:
        batch.append(row)
        if len(batch) >= batch_size:
            dst.executemany(insert_sql, batch)
            count += len(batch)
            batch.clear()

    if batch:
        dst.executemany(insert_sql, batch)
        count += len(batch)

    stats.rows_copied[table] = count
    stats.tables_processed += 1


def _copy_table_filtered(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    table: str,
    run_id_col: str,
    kept_run_ids: frozenset[str],
    stats: PruneStats,
) -> None:
    """Copy only rows whose *run_id_col* is in *kept_run_ids*."""
    cols = _table_columns(src, table)
    if not cols:
        return

    col_list = ", ".join(f"[{c}]" for c in cols)
    placeholders = ", ".join("?" for _ in cols)
    id_placeholders = ", ".join("?" for _ in kept_run_ids)

    select_sql = (
        f"SELECT {col_list} FROM [{table}] "
        f"WHERE [{run_id_col}] IN ({id_placeholders})"
    )
    insert_sql = f"INSERT INTO [{table}] ({col_list}) VALUES ({placeholders})"

    cursor = src.execute(select_sql, list(kept_run_ids))
    batch: list[tuple] = []
    count = 0
    batch_size = 5000

    for row in cursor:
        batch.append(row)
        if len(batch) >= batch_size:
            dst.executemany(insert_sql, batch)
            count += len(batch)
            batch.clear()

    if batch:
        dst.executemany(insert_sql, batch)
        count += len(batch)

    stats.rows_copied[table] = count
    stats.tables_processed += 1


def _copy_join_table_filtered(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    child_table: str,
    child_fk_col: str,
    parent_table: str,
    parent_pk_col: str,
    stats: PruneStats,
) -> None:
    """Copy child rows whose parent exists in the already-populated *dst*."""
    child_cols = _table_columns(src, child_table)
    if not child_cols:
        return

    col_list = ", ".join(f"[{c}]" for c in child_cols)
    placeholders = ", ".join("?" for _ in child_cols)

    parent_ids = {
        r[0]
        for r in dst.execute(
            f"SELECT [{parent_pk_col}] FROM [{parent_table}]"
        ).fetchall()
    }

    if not parent_ids:
        stats.rows_copied[child_table] = 0
        stats.tables_processed += 1
        return

    select_sql = f"SELECT {col_list} FROM [{child_table}]"
    insert_sql = f"INSERT INTO [{child_table}] ({col_list}) VALUES ({placeholders})"

    fk_idx = child_cols.index(child_fk_col)
    cursor = src.execute(select_sql)
    batch: list[tuple] = []
    count = 0
    batch_size = 5000

    for row in cursor:
        if row[fk_idx] in parent_ids:
            batch.append(row)
            if len(batch) >= batch_size:
                dst.executemany(insert_sql, batch)
                count += len(batch)
                batch.clear()

    if batch:
        dst.executemany(insert_sql, batch)
        count += len(batch)

    stats.rows_copied[child_table] = count
    stats.tables_processed += 1


def _rebuild_fts(dst: sqlite3.Connection) -> None:
    """Create the FTS5 virtual table and populate from the chunk content table."""
    dst.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5("
        "    chunk_id UNINDEXED,"
        "    chunk_text,"
        "    heading_context,"
        "    content='chunk',"
        "    content_rowid='chunk_rowid'"
        ")"
    )
    dst.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('rebuild')")
    dst.commit()
    logger.info("FTS5 index rebuilt")


def _apply_triggers(dst: sqlite3.Connection, trigger_ddls: list[str]) -> None:
    """Apply trigger DDL statements to the destination database."""
    applied = 0
    for ddl in trigger_ddls:
        try:
            dst.execute(ddl)
            applied += 1
        except sqlite3.OperationalError as exc:
            logger.warning("Trigger skipped (%s): %.120s", exc, ddl)
    dst.commit()
    logger.info("Applied %d / %d triggers", applied, len(trigger_ddls))


def _validate_destination(dst: sqlite3.Connection) -> None:
    """
    Run integrity and foreign-key checks on the pruned database.

    :raises RuntimeError: If any check fails.
    """
    dst.execute("PRAGMA foreign_keys = ON")

    integrity = dst.execute("PRAGMA integrity_check").fetchone()
    if integrity is None or integrity[0] != "ok":
        raise RuntimeError(f"integrity_check failed: {integrity}")
    logger.info("PRAGMA integrity_check: ok")

    fk_violations = dst.execute("PRAGMA foreign_key_check").fetchall()
    if fk_violations:
        sample = fk_violations[:10]
        raise RuntimeError(
            f"foreign_key_check found {len(fk_violations)} violations, "
            f"sample: {sample}"
        )
    logger.info("PRAGMA foreign_key_check: 0 violations")


def _checkpoint_and_consolidate(db_path: Path) -> None:
    """Checkpoint WAL and switch to DELETE journal mode for portability."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.commit()
    finally:
        conn.close()


def _swap_databases(
    original: Path,
    new_db: Path,
    backup_suffix: str,
) -> Path:
    """
    Atomically swap *new_db* into *original*'s path.

    Checkpoints the original database's WAL before touching auxiliary files,
    ensuring no committed data is lost even if the original was in WAL mode.

    :return: Path to the backup copy of the original database.
    """
    backup = original.with_suffix(original.suffix + backup_suffix)

    # Checkpoint the ORIGINAL db's WAL before touching any aux files.
    # Without this, committed pages in -wal would be silently lost.
    _checkpoint_and_consolidate(original)

    for suffix in ("-wal", "-shm"):
        for base in (original, new_db):
            aux = Path(str(base) + suffix)
            if aux.exists():
                aux.unlink()

    shutil.copy2(str(original), str(backup))
    os.replace(str(new_db), str(original))
    logger.info("Swapped pruned DB into %s (backup at %s)", original, backup)
    return backup


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _fmt_bytes(n: int) -> str:
    """Format byte count as a human-readable string."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def build_pruned_database(config: PruneConfig) -> PruneStats:  # noqa: C901
    """
    Execute the full prune-and-swap workflow.

    :param config: Validated pruning configuration.
    :return: Statistics about the operation.
    :raises RuntimeError: If validation of the new database fails.
    """
    stats = PruneStats()
    stats.bytes_before = config.db_path.stat().st_size
    start = time.monotonic()

    src = sqlite3.connect(f"file:{config.db_path}?mode=ro", uri=True)
    src.execute("PRAGMA query_only = ON")

    try:
        if not _table_exists(src, "pipeline_run"):
            logger.info("No pipeline_run table found; nothing to prune")
            return stats

        decision = determine_retention(src, config.max_completed_runs)

        if not decision.pruned_run_ids:
            logger.info(
                "Only %d completed runs exist (target: %d); pruning not needed",
                decision.completed_before,
                config.max_completed_runs,
            )
            return stats

        table_ddls, index_ddls, trigger_ddls, _vt_ddls = _extract_schema_ddl(src)

        tmp_path = config.db_path.with_suffix(".tmp_prune")
        if tmp_path.exists():
            tmp_path.unlink()

        dst = sqlite3.connect(str(tmp_path))
        try:
            # Phase 1: create tables only (indexes deferred for faster bulk load).
            _create_schema_tables(dst, table_ddls)

            # Phase 2: bulk-load data in a single transaction.
            dst.execute("PRAGMA foreign_keys = OFF")
            dst.execute("BEGIN")

            for table in FULL_COPY_TABLES:
                if _table_exists(src, table):
                    _copy_table_full(src, dst, table, stats)
                else:
                    logger.debug("Table %s not found in source; skipped", table)

            for table, run_col in RUN_SCOPED_TABLES:
                if _table_exists(src, table):
                    _copy_table_filtered(
                        src, dst, table, run_col, decision.kept_run_ids, stats
                    )
                else:
                    logger.debug("Table %s not found in source; skipped", table)

            for child, child_fk, parent, parent_pk in RUN_SCOPED_JOIN_TABLES:
                if _table_exists(src, child):
                    _copy_join_table_filtered(
                        src, dst, child, child_fk, parent, parent_pk, stats
                    )
                else:
                    logger.debug("Table %s not found in source; skipped", child)

            dst.commit()
            logger.info(
                "Bulk load complete: %d tables, %d total rows",
                stats.tables_processed,
                sum(stats.rows_copied.values()),
            )

            # Phase 3: create indexes now that data is loaded.
            _create_indexes(dst, index_ddls)

            # Phase 4: rebuild FTS and apply triggers.
            if _table_exists(src, "chunk_fts"):
                _rebuild_fts(dst)

            _apply_triggers(dst, trigger_ddls)

            # Phase 5: validate the new database.
            _validate_destination(dst)

        finally:
            dst.close()

    finally:
        src.close()

    # Phase 6: consolidate the new DB's WAL and swap.
    _checkpoint_and_consolidate(tmp_path)

    stats.bytes_after = tmp_path.stat().st_size
    logger.info(
        "Size: %s -> %s (%.1f%% of original)",
        _fmt_bytes(stats.bytes_before),
        _fmt_bytes(stats.bytes_after),
        (stats.bytes_after / stats.bytes_before * 100) if stats.bytes_before else 0,
    )

    _swap_databases(config.db_path, tmp_path, config.backup_suffix)

    stats.elapsed_seconds = time.monotonic() - start
    logger.info("Pruning completed in %.1fs", stats.elapsed_seconds)

    return stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments and return an argparse.Namespace."""
    parser = argparse.ArgumentParser(
        description="Prune processed_posts.db to retain only the N latest completed runs.",
    )
    parser.add_argument(
        "--working-db",
        type=Path,
        default=Path("../../../database/processed_posts.db"),
        help="Path to processed_posts.db",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=5,
        help="Maximum completed runs to retain AFTER the upcoming run (default: 5)",
    )
    parser.add_argument(
        "--backup-suffix",
        type=str,
        default=".pre_prune_backup",
        help="Suffix appended to the original DB filename for the backup copy",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Set entry point.

    :return: 0 on success or no-op, 1 on fatal error.
    """
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.working_db.exists():
        logger.info("Database %s does not exist yet; nothing to prune", args.working_db)
        return 0

    try:
        config = PruneConfig(
            db_path=args.working_db,
            max_completed_runs=args.max_runs,
            backup_suffix=args.backup_suffix,
        )
    except ValueError:
        logger.exception("Invalid configuration")
        return 1

    try:
        stats = build_pruned_database(config)
        if stats.tables_processed > 0:
            for table, count in sorted(stats.rows_copied.items()):
                logger.debug("  %-45s %8d rows", table, count)
    except RuntimeError:
        logger.exception("Pruning failed during validation; original DB is untouched")
        _cleanup_tmp(config.db_path)
        return 1
    except Exception:
        logger.exception("Unexpected error during pruning; original DB is untouched")
        _cleanup_tmp(config.db_path)
        return 1

    return 0


def _cleanup_tmp(db_path: Path) -> None:
    """Remove partial temp file on failure."""
    tmp = db_path.with_suffix(".tmp_prune")
    if tmp.exists():
        try:
            tmp.unlink()
            logger.info("Cleaned up partial temp file %s", tmp)
        except OSError:
            logger.warning("Could not remove temp file %s", tmp)


if __name__ == "__main__":
    sys.exit(main())