from pathlib import Path

from open_event_intel.etl_processing.database_interface import DatabaseInterface
from open_event_intel.logger import get_logger

logger = get_logger(__name__)

STAGE_NAME = "stage_01_ingest"

class Stage01DatabaseInterface(DatabaseInterface):
    """Database interface for Stage 01 Ingest."""

    READS = {
        "pipeline_run",
        "scrape_record",
        "document",
        "document_version",
        "doc_stage_status",
        "run_stage_status",
        "entity_registry",
        "alert_rule",
        "watchlist",
    }
    WRITES = {
        "pipeline_run",
        "scrape_record",
        "document",
        "document_version",
        "doc_stage_status",
        "run_stage_status",
        "entity_registry",
        "alert_rule",
        "watchlist",
    }

    def __init__(self, working_db_path: Path, source_db_path: Path) -> None:
        """Initialize Stage 01 Ingest."""
        super().__init__(working_db_path, source_db_path, STAGE_NAME)

    def has_completed_runs(self) -> bool:
        """Check if any completed pipeline runs exist."""
        self._check_read_access("pipeline_run")
        row = self._fetchone(
            "SELECT COUNT(*) as cnt FROM pipeline_run WHERE status = 'completed'"
        )
        return row is not None and row["cnt"] > 0

    def scrape_record_exists(self, scrape_id: str) -> bool:
        """Check if a scrape record already exists by scrape_id (primary key)."""
        self._check_read_access("scrape_record")
        row = self._fetchone(
            "SELECT 1 FROM scrape_record WHERE scrape_id = ?", (scrape_id,)
        )
        return row is not None

    def scrape_record_exists_by_natural_key(
        self,
        publisher_id: str,
        url_normalized: str,
        source_published_at: str,
        scrape_kind: str = "page",
    ) -> bool:
        """
        Check if a scrape record exists by its natural UNIQUE key.

        The ``scrape_record`` table carries a UNIQUE constraint on
        ``(publisher_id, url_normalized, source_published_at, scrape_kind)``.
        Two source publications with different ``source_id`` values can
        still collide on this natural key after URL normalization.  This
        method lets callers detect such collisions *before* attempting an
        INSERT, avoiding ``DBConstraintError`` for a predictable case.

        :return: True if a record with this natural key already exists.
        """
        self._check_read_access("scrape_record")
        row = self._fetchone(
            "SELECT scrape_id FROM scrape_record "
            "WHERE publisher_id = ? AND url_normalized = ? "
            "AND source_published_at = ? AND scrape_kind = ?",
            (publisher_id, url_normalized, source_published_at, scrape_kind),
        )
        if row is not None:
            logger.debug(
                "Natural-key collision: publisher=%s url_normalized=%s "
                "source_published_at=%s scrape_kind=%s "
                "-> existing scrape_id=%s",
                publisher_id,
                url_normalized[:120],
                source_published_at,
                scrape_kind,
                row["scrape_id"][:16],
            )
        return row is not None

    def count_entity_registry(self) -> int:
        """Count entries in entity_registry."""
        self._check_read_access("entity_registry")
        row = self._fetchone("SELECT COUNT(*) as cnt FROM entity_registry")
        return row["cnt"] if row else 0

    def count_alert_rules(self) -> int:
        """Count entries in alert_rule."""
        self._check_read_access("alert_rule")
        row = self._fetchone("SELECT COUNT(*) as cnt FROM alert_rule")
        return row["cnt"] if row else 0

    def count_watchlists(self) -> int:
        """Count entries in watchlist."""
        self._check_read_access("watchlist")
        row = self._fetchone("SELECT COUNT(*) as cnt FROM watchlist")
        return row["cnt"] if row else 0