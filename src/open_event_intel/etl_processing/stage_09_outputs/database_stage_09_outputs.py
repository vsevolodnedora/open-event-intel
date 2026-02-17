import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from open_event_intel.etl_processing.database_interface import AlertRuleRow, DatabaseInterface, EventRevisionRow, EventRow, MetricObservationRow, MetricSeriesPointRow, MetricSeriesRow, WatchlistRow
from open_event_intel.logger import get_logger

logger = get_logger(__name__)

STAGE_NAME = "stage_09_outputs"

# Per-doc stages that must all be "ok" for a document to be eligible.
# Ordered by pipeline sequence for readability.
PER_DOC_STAGES = (
    "stage_01_ingest",
    "stage_02_parse",
    "stage_03_metadata",
    "stage_04_mentions",
    "stage_05_embeddings",
    "stage_06_taxonomy",
    "stage_07_novelty",
    "stage_08_events",
)

# FK-safe deletion order for run-scoped output tables.
# Child/join tables first, then parent tables.
FK_SAFE_DELETE_ORDER = (
    "alert_evidence",
    "digest_item_evidence",
    "metric_series_point",
    "alert",
    "digest_item",
    "metric_series",
)

# SQLite max variable number is 999 by default; batch below that.
SQLITE_BATCH_SIZE = 900

class EligibleEvent(BaseModel):
    """An event together with its current revision, ready for output processing."""

    model_config = ConfigDict(frozen=True)

    event: EventRow
    revision: EventRevisionRow

class Stage09DatabaseInterface(DatabaseInterface):
    """
    Database adapter for stage 09 (outputs).

    SQL lives here; business logic stays in the stage module.
    """

    READS = {
        "event",
        "event_revision",
        "event_revision_evidence",
        "novelty_label",
        "mention",
        "mention_link",
        "facet_assignment",
        "metric_observation",
        "alert_rule",
        "watchlist",
        "doc_stage_status",
        "run_stage_status",
        "pipeline_run",
        "document_version",
        "document",
        "evidence_span",
        "entity_registry",
    }
    WRITES = {
        "metric_series",
        "metric_series_point",
        "alert",
        "alert_evidence",
        "digest_item",
        "digest_item_evidence",
        "evidence_span",
        "run_stage_status",
    }

    def __init__(self, working_db_path: Path, source_db_path: Path | None = None) -> None:
        """Initialize a Stage09DatabaseInterface."""
        super().__init__(working_db_path, source_db_path, STAGE_NAME)

    # reads

    def get_eligible_doc_version_ids(self) -> list[str]:
        """
        Return doc_version_ids where all per-doc stages through 08 are ``ok``.

        Ordered deterministically by doc_version_id.
        """
        self._check_read_access("doc_stage_status")
        placeholders = ",".join("?" for _ in PER_DOC_STAGES)
        sql = f"""
            SELECT doc_version_id
            FROM doc_stage_status
            WHERE stage IN ({placeholders})
              AND status = 'ok'
            GROUP BY doc_version_id
            HAVING COUNT(DISTINCT stage) = ?
            ORDER BY doc_version_id
        """  # noqa: S608
        rows = self._fetchall(sql, (*PER_DOC_STAGES, len(PER_DOC_STAGES)))
        return [r["doc_version_id"] for r in rows]

    def get_eligible_events(self, eligible_doc_ids: frozenset[str]) -> list[EligibleEvent]:
        """
        Return events whose current revision references only eligible docs.

        Uses explicit column aliases to prevent name collisions between the
        ``event`` and ``event_revision`` tables.  Both share ``event_id``,
        ``created_in_run_id``, ``created_at``; the event table additionally
        has ``event_type``, ``canonical_key``, ``current_revision_id`` which
        are absent from ``event_revision``.  A bare ``SELECT e.*, er.*``
        would put all of these into a single flat dict, and Pydantic's
        ``extra="forbid"`` on ``EventRevisionRow`` would reject the event-
        only columns.

        **FIX (v2):** explicit column aliases prevent the crash.

        :param eligible_doc_ids: Set of eligible ``doc_version_id`` values.
        :return: List of :class:`EligibleEvent` in deterministic order.
        """
        self._check_read_access("event")
        self._check_read_access("event_revision")

        # Explicit aliases avoid column-name collisions between the two
        # tables.  The ``e_`` prefix marks event columns; ``er_`` marks
        # revision columns.
        sql = """
            SELECT
                e.event_id            AS e_event_id,
                e.event_type          AS e_event_type,
                e.canonical_key       AS e_canonical_key,
                e.current_revision_id AS e_current_revision_id,
                e.created_in_run_id   AS e_created_in_run_id,
                e.created_at          AS e_created_at,

                er.revision_id        AS er_revision_id,
                er.event_id           AS er_event_id,
                er.revision_no        AS er_revision_no,
                er.slots_json         AS er_slots_json,
                er.doc_version_ids    AS er_doc_version_ids,
                er.confidence         AS er_confidence,
                er.extraction_method  AS er_extraction_method,
                er.extraction_tier    AS er_extraction_tier,
                er.supersedes_revision_id AS er_supersedes_revision_id,
                er.created_in_run_id  AS er_created_in_run_id,
                er.created_at         AS er_created_at
            FROM event e
            JOIN event_revision er ON er.revision_id = e.current_revision_id
            WHERE e.current_revision_id IS NOT NULL
            ORDER BY e.event_id
        """
        rows = self._fetchall(sql)
        logger.debug(
            "get_eligible_events: fetched %d event+revision rows from DB", len(rows),
        )

        result: list[EligibleEvent] = []
        skipped_ineligible = 0
        for r in rows:
            rd = dict(r)

            # Build revision dict from er_-prefixed columns only
            rev_dict = {
                "revision_id":            rd["er_revision_id"],
                "event_id":               rd["er_event_id"],
                "revision_no":            rd["er_revision_no"],
                "slots_json":             rd["er_slots_json"],
                "doc_version_ids":        rd["er_doc_version_ids"],
                "confidence":             rd["er_confidence"],
                "extraction_method":      rd["er_extraction_method"],
                "extraction_tier":        rd["er_extraction_tier"],
                "supersedes_revision_id": rd["er_supersedes_revision_id"],
                "created_in_run_id":      rd["er_created_in_run_id"],
                "created_at":             rd["er_created_at"],
            }
            try:
                revision = EventRevisionRow.model_validate(rev_dict)
            except Exception:
                logger.error(
                    "Failed to validate EventRevisionRow for revision_id=%s, "
                    "event_id=%s.  Raw revision dict: %r",
                    rd.get("er_revision_id", "?")[:16],
                    rd.get("e_event_id", "?")[:16],
                    {k: (str(v)[:40] if v is not None else None) for k, v in rev_dict.items()},
                    exc_info=True,
                )
                raise

            # Check eligibility: every doc in the revision must be eligible
            if not all(dvid in eligible_doc_ids for dvid in revision.doc_version_ids):
                skipped_ineligible += 1
                continue

            event = EventRow(
                event_id=rd["e_event_id"],
                event_type=rd["e_event_type"],
                canonical_key=rd["e_canonical_key"],
                current_revision_id=rd["e_current_revision_id"],
                created_in_run_id=rd["e_created_in_run_id"],
                created_at=rd["e_created_at"],
            )
            result.append(EligibleEvent(event=event, revision=revision))

        logger.info(
            "get_eligible_events: %d eligible, %d skipped (ineligible docs), "
            "%d total from DB",
            len(result), skipped_ineligible, len(rows),
        )
        return result

    def get_eligible_metric_observations(
        self, eligible_doc_ids: frozenset[str],
    ) -> list[MetricObservationRow]:
        """
        Return metric observations from eligible docs that carry evidence.

        Observations with ``evidence_id IS NULL`` are excluded (fail-closed).

        **FIX (v2):** SQL-level ``IN`` filtering avoids loading the entire
        table into Python.  Batches at SQLITE_BATCH_SIZE to stay within
        SQLite's default ``SQLITE_MAX_VARIABLE_NUMBER`` of 999.
        """
        self._check_read_access("metric_observation")
        if not eligible_doc_ids:
            logger.info("get_eligible_metric_observations: 0 eligible docs → 0 observations")
            return []

        all_obs: list[MetricObservationRow] = []
        doc_list = sorted(eligible_doc_ids)

        for i in range(0, len(doc_list), SQLITE_BATCH_SIZE):
            batch = doc_list[i : i + SQLITE_BATCH_SIZE]
            placeholders = ",".join("?" for _ in batch)
            sql = f"""
                SELECT * FROM metric_observation
                WHERE evidence_id IS NOT NULL
                  AND doc_version_id IN ({placeholders})
                ORDER BY metric_id
            """  # noqa: S608
            rows = self._fetchall(sql, tuple(batch))
            for row in rows:
                all_obs.append(MetricObservationRow.model_validate(dict(row)))

        # Log how many were excluded for missing evidence (diagnostic)
        null_ev = self._fetchone(
            "SELECT COUNT(*) AS cnt FROM metric_observation WHERE evidence_id IS NULL",
        )
        null_count = null_ev["cnt"] if null_ev else 0
        if null_count > 0:
            logger.info(
                "get_eligible_metric_observations: %d observations globally lack evidence "
                "(excluded per fail-closed rule)",
                null_count,
            )
        logger.info(
            "get_eligible_metric_observations: returning %d observations "
            "(%d eligible docs, %d batches)",
            len(all_obs), len(eligible_doc_ids),
            (len(doc_list) + SQLITE_BATCH_SIZE - 1) // SQLITE_BATCH_SIZE,
        )
        return all_obs

    def get_revision_evidence_ids(self, revision_id: str) -> list[str]:
        """
        Return *unique* evidence_ids linked to an event revision.

        The ``event_revision_evidence`` table may contain multiple rows for
        the same ``(revision_id, evidence_id)`` pair with different
        ``purpose`` values.  We need distinct evidence IDs here because
        downstream consumers assign their own constant purpose (e.g.
        ``PURPOSE_WATCHLIST_TRIGGER``), and inserting duplicates would violate
        the UNIQUE constraint on ``(alert_id, evidence_id, purpose)`` in
        ``alert_evidence`` (and similarly for ``digest_item_evidence``).
        """
        self._check_read_access("event_revision_evidence")
        rows = self._fetchall(
            "SELECT DISTINCT evidence_id FROM event_revision_evidence WHERE revision_id = ? ORDER BY evidence_id",
            (revision_id,),
        )
        return [r["evidence_id"] for r in rows]

    def get_novelty_label(self, doc_version_id: str) -> str | None:
        """Return novelty label string for a document, or None."""
        self._check_read_access("novelty_label")
        row = self._fetchone(
            "SELECT label FROM novelty_label WHERE doc_version_id = ?", (doc_version_id,),
        )
        return row["label"] if row else None

    def get_facet_values(self, doc_version_id: str, facet_type: str) -> list[str]:
        """Return facet values for a doc and facet type."""
        self._check_read_access("facet_assignment")
        rows = self._fetchall(
            "SELECT facet_value FROM facet_assignment "
            "WHERE doc_version_id = ? AND facet_type = ? ORDER BY facet_value",
            (doc_version_id, facet_type),
        )
        return [r["facet_value"] for r in rows]

    def get_document_publisher(self, doc_version_id: str) -> str | None:
        """Return publisher_id for a doc_version_id."""
        self._check_read_access("document_version")
        self._check_read_access("document")
        row = self._fetchone(
            """
            SELECT d.publisher_id
            FROM document_version dv
            JOIN document d ON d.document_id = dv.document_id
            WHERE dv.doc_version_id = ?
            """,
            (doc_version_id,),
        )
        return row["publisher_id"] if row else None

    def get_mention_entity_ids(self, doc_version_id: str) -> list[str]:
        """Return distinct linked entity_ids for a document's mentions."""
        self._check_read_access("mention")
        self._check_read_access("mention_link")
        rows = self._fetchall(
            """
            SELECT DISTINCT ml.entity_id
            FROM mention m
            JOIN mention_link ml ON ml.mention_id = m.mention_id
            WHERE m.doc_version_id = ?
              AND ml.entity_id IS NOT NULL
            ORDER BY ml.entity_id
            """,
            (doc_version_id,),
        )
        return [r["entity_id"] for r in rows]

    def build_entity_name_to_id_map(self) -> dict[str, str]:
        """
        Build a lookup from entity names/aliases → entity_id.

        Includes ``canonical_name`` and each alias from ``entity_registry``.
        If two entities share a name the first (by entity_id sort order)
        wins, matching the deterministic ordering requirement.

        **Added in v2** to fix the watchlist entity-name → entity-id
        mismatch.
        """
        self._check_read_access("entity_registry")
        rows = self._fetchall(
            "SELECT entity_id, canonical_name, aliases FROM entity_registry ORDER BY entity_id"
        )
        mapping: dict[str, str] = {}
        for r in rows:
            eid = r["entity_id"]
            cname = r["canonical_name"]
            # canonical_name → entity_id
            if cname not in mapping:
                mapping[cname] = eid
            # aliases → entity_id
            raw_aliases = r["aliases"]
            if raw_aliases:
                try:
                    aliases = json.loads(raw_aliases) if isinstance(raw_aliases, str) else raw_aliases
                except (json.JSONDecodeError, TypeError):
                    aliases = []
                for alias in (aliases or []):
                    if alias not in mapping:
                        mapping[alias] = eid
        logger.debug("build_entity_name_to_id_map: %d name→id mappings", len(mapping))
        return mapping

    def list_active_alert_rules(self) -> list[AlertRuleRow]:
        """Return all active alert rules."""
        self._check_read_access("alert_rule")
        rows = self._fetchall("SELECT * FROM alert_rule WHERE active = 1 ORDER BY rule_id")
        return [AlertRuleRow.model_validate(dict(r)) for r in rows]

    def list_active_watchlists(self) -> list[WatchlistRow]:
        """Return all active watchlists."""
        self._check_read_access("watchlist")
        rows = self._fetchall("SELECT * FROM watchlist WHERE active = 1 ORDER BY watchlist_id")
        return [WatchlistRow.model_validate(dict(r)) for r in rows]

    def get_recent_alert_ids_for_rule(
        self, rule_id: str, event_id: str, window_hours: int,
    ) -> list[str]:
        """
        Return alert_ids for a rule+event within the suppression window.

        Used for suppression-window deduplication: if a matching alert was
        already generated within ``window_hours`` for the same rule and
        event, the new alert should be suppressed.
        """
        self._check_read_access("alert")
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()
        rows = self._fetchall(
            """SELECT alert_id FROM alert
               WHERE rule_id = ?
                 AND json_extract(trigger_payload_json, '$.canonical_key') =
                     (SELECT json_extract(trigger_payload_json, '$.canonical_key')
                      FROM alert WHERE alert_id = ?)
                 AND triggered_at >= ?
               ORDER BY triggered_at DESC""",
            (rule_id, event_id, cutoff),
        )
        return [r["alert_id"] for r in rows]

    # writes

    def insert_metric_series(self, row: MetricSeriesRow) -> None:
        """Insert a metric series."""
        self._check_write_access("metric_series")
        self._execute(
            """INSERT INTO metric_series
            (run_id, series_id, metric_name, geography, period_granularity, unit_norm)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (row.run_id, row.series_id, row.metric_name, row.geography,
             row.period_granularity, row.unit_norm),
        )

    def insert_metric_series_point(self, row: MetricSeriesPointRow) -> None:
        """Insert a metric series point."""
        self._check_write_access("metric_series_point")
        self._execute(
            """INSERT INTO metric_series_point
            (run_id, series_id, period_start, period_end, value_norm,
             source_doc_version_id, evidence_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (row.run_id, row.series_id, row.period_start.isoformat(),
             row.period_end.isoformat() if row.period_end else None,
             row.value_norm, row.source_doc_version_id, row.evidence_id),
        )

    def delete_stage09_outputs_for_run(self, run_id: str) -> dict[str, int]:
        """
        Delete all stage 09 run-scoped outputs in FK-safe order.

        **FIX (v2):** ``cursor.rowcount`` is now captured for every table
        (the v1 code had dead ``if False`` branches that always returned 0
        for alert_evidence and digest_item_evidence).

        :return: Mapping of table name to deleted row count.
        """
        counts: dict[str, int] = {}
        for table in FK_SAFE_DELETE_ORDER:
            self._check_write_access(table)

        # alert_evidence (via subquery on alert.run_id)
        cursor = self._execute(
            "DELETE FROM alert_evidence WHERE alert_id IN "
            "(SELECT alert_id FROM alert WHERE run_id = ?)",
            (run_id,),
        )
        counts["alert_evidence"] = cursor.rowcount

        # digest_item_evidence (via subquery on digest_item.run_id)
        cursor = self._execute(
            "DELETE FROM digest_item_evidence WHERE item_id IN "
            "(SELECT item_id FROM digest_item WHERE run_id = ?)",
            (run_id,),
        )
        counts["digest_item_evidence"] = cursor.rowcount

        # metric_series_point
        cursor = self._execute("DELETE FROM metric_series_point WHERE run_id = ?", (run_id,))
        counts["metric_series_point"] = cursor.rowcount

        # alert
        cursor = self._execute("DELETE FROM alert WHERE run_id = ?", (run_id,))
        counts["alert"] = cursor.rowcount

        # digest_item
        cursor = self._execute("DELETE FROM digest_item WHERE run_id = ?", (run_id,))
        counts["digest_item"] = cursor.rowcount

        # metric_series
        cursor = self._execute("DELETE FROM metric_series WHERE run_id = ?", (run_id,))
        counts["metric_series"] = cursor.rowcount

        return counts

    def delete_run_stage_status_for_stage(self, run_id: str, stage: str) -> None:
        """Delete run_stage_status row so upsert can recreate it cleanly."""
        self._check_write_access("run_stage_status")
        self._execute(
            "DELETE FROM run_stage_status WHERE run_id = ? AND stage = ?",
            (run_id, stage),
        )
