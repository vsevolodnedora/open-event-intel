"""
Stage 00: Pipeline Setup and Run Acquisition.

Acquires the pipeline run lock, validates configuration, and ensures
the working database is initialized with the correct schema. This stage
MUST run before any other pipeline stage.

Responsibilities:
    - Ensure working DB exists and schema is valid (via DatabaseInterface)
    - Load and validate configuration
    - Compute config_version hash
    - Acquire run lock (create or resume pipeline_run)
    - Enforce config stability vs latest completed run

Exit codes:
    0: Success
    1: Fatal error (config drift, completed run reuse, another run active, etc.)
"""
import argparse
import logging
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from open_event_intel.etl_processing.config_interface import Config, get_config_version, load_config
from open_event_intel.etl_processing.database_interface import AnotherRunActiveError, DatabaseInterface, DBConstraintError, DBError, DBSchemaError, PipelineRunRow
from open_event_intel.etl_processing.stage_00_setup.database_00_setup import Stage00DatabaseInterface
from open_event_intel.logger import get_logger

logger = get_logger(__name__)


class RunAcquisitionStatus(str, Enum):
    """Status of run acquisition attempt."""

    CREATED = "created"
    RESUMED = "resumed"
    IDEMPOTENT = "idempotent"


class RunAcquisitionResult(BaseModel):
    """Result of run acquisition attempt."""

    status: RunAcquisitionStatus
    run_id: str
    config_version: str
    is_fresh_db: bool


class ConfigDriftError(Exception):
    """Raised when config version differs from latest completed run."""

    def __init__(self, current_version: str, expected_version: str) -> None:
        self.current_version = current_version
        self.expected_version = expected_version
        super().__init__(
            f"Config drift detected. Current: {current_version}, "
            f"Latest completed: {expected_version}. Rebuild required."
        )


class CompletedRunReuseError(Exception):
    """Raised when attempting to reuse a completed run_id."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(
            f"Cannot reuse completed run_id '{run_id}'. "
            "Completed runs are immutable."
        )


def validate_source_db(source_db_path: Path) -> None:
    """
    Validate that the source database exists and is readable.

    :param source_db_path: Path to the source database.
    :raises FileNotFoundError: If source database does not exist.
    """
    if not source_db_path.exists():
        raise FileNotFoundError(f"Source database not found: {source_db_path}")
    if not source_db_path.is_file():
        raise FileNotFoundError(f"Source database is not a file: {source_db_path}")


def validate_config_stability(
    db: Stage00DatabaseInterface,
    current_config_version: str,
) -> bool:
    """
    Validate that config version matches the latest completed run.

    :param db: Database interface.
    :param current_config_version: Current config version hash.
    :return: True if this is a fresh DB (no completed runs).
    :raises ConfigDriftError: If config version differs from latest completed run.
    """
    latest_completed = db.get_latest_completed_run()
    if latest_completed is None:
        logger.info("No completed runs found - fresh database bootstrap allowed")
        return True

    if latest_completed.config_version != current_config_version:
        raise ConfigDriftError(
            current_version=current_config_version,
            expected_version=latest_completed.config_version,
        )

    logger.info(
        "Config version matches latest completed run: %s",
        current_config_version,
    )
    return False


def acquire_run(
    db: Stage00DatabaseInterface,
    run_id: str,
    config_version: str,
) -> RunAcquisitionResult:
    """
    Acquire or resume a pipeline run.

    Implements the five-case run acquisition logic:
    1. run_id doesn't exist → INSERT with status='running'
    2. run_id exists with status='running' → proceed (idempotent)
    3. run_id exists with status IN ('failed','aborted') → UPDATE to 'running'
    4. run_id exists with status='completed' → abort
    5. Another run is running → constraint fails, abort

    :param db: Database interface.
    :param run_id: Run ID to acquire.
    :param config_version: Config version hash for this run.
    :return: Result of run acquisition.
    :raises CompletedRunReuseError: If attempting to reuse a completed run.
    :raises AnotherRunActiveError: If another run is already active.
    """
    existing = db.get_pipeline_run(run_id)

    if existing is None:
        logger.info("Creating new pipeline run: %s", run_id)
        new_run = PipelineRunRow(
            run_id=run_id,
            started_at=datetime.now(),
            config_version=config_version,
            status="running",
        )
        try:
            with db.transaction():
                db.insert_pipeline_run(new_run)
        except DBConstraintError as e:
            running = db.get_any_running_run()
            if running and running.run_id != run_id:
                raise AnotherRunActiveError(run_id) from e
            raise

        is_fresh = db.get_latest_completed_run() is None
        return RunAcquisitionResult(
            status=RunAcquisitionStatus.CREATED,
            run_id=run_id,
            config_version=config_version,
            is_fresh_db=is_fresh,
        )

    if existing.status == "completed":
        raise CompletedRunReuseError(run_id)

    if existing.status == "running":
        logger.info(
            "Run %s already running - idempotent re-entry",
            run_id,
        )
        return RunAcquisitionResult(
            status=RunAcquisitionStatus.IDEMPOTENT,
            run_id=run_id,
            config_version=existing.config_version,
            is_fresh_db=False,
        )

    if existing.status in ("failed", "aborted"):
        logger.info(
            "Resuming %s run: %s",
            existing.status,
            run_id,
        )
        with db.transaction():
            db.resume_pipeline_run(run_id, config_version)

        return RunAcquisitionResult(
            status=RunAcquisitionStatus.RESUMED,
            run_id=run_id,
            config_version=config_version,
            is_fresh_db=False,
        )

    raise DBError(f"Unexpected run status: {existing.status}")


def ensure_log_directory(log_dir: Path) -> None:
    """
    Ensure log directory exists.

    :param log_dir: Path to log directory.
    """
    log_dir.mkdir(parents=True, exist_ok=True)


def setup_file_logging(log_dir: Path, run_id: str) -> None:
    """
    Set up file logging for the pipeline run.

    :param log_dir: Path to log directory.
    :param run_id: Run ID for log file naming.
    """
    ensure_log_directory(log_dir)
    log_file = log_dir / f"stage_00_{run_id[:8]}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logger.info("Logging to file: %s", log_file)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 00: Pipeline Setup and Run Acquisition"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        help="Pipeline run ID (SHA256 hex, required; reused for resumption)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("../../../config/"),
        help="Directory containing config.yaml",
    )
    parser.add_argument(
        "--source-db",
        type=Path,
        default=Path("../../../database/preprocessed_posts.db"),
        help="Path to source database (read-only)",
    )
    parser.add_argument(
        "--working-db",
        type=Path,
        default=Path("../../../database/processed_posts.db"),
        help="Path to working database (created if missing)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("../../../output/processed/logs/"),
        help="Directory for log files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def run_setup(
    run_id: str,
    config_dir: Path,
    source_db_path: Path,
    working_db_path: Path,
    log_dir: Path,
) -> int:
    """
    Execute Stage 00 setup.

    :param run_id: Pipeline run ID.
    :param config_dir: Path to config directory.
    :param source_db_path: Path to source database.
    :param working_db_path: Path to working database.
    :param log_dir: Path to log directory.
    :return: Exit code (0 for success, 1 for failure).
    """
    setup_file_logging(log_dir, run_id)
    logger.info("Starting Stage 00 Setup")
    logger.info("Run ID: %s", run_id)
    logger.info("Config dir: %s", config_dir)
    logger.info("Source DB: %s", source_db_path)
    logger.info("Working DB: %s", working_db_path)

    try:
        validate_source_db(source_db_path)
    except FileNotFoundError as e:
        logger.error("Source database validation failed: %s", e)
        return 1

    config_path = config_dir / "config.yaml"
    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        return 1

    try:
        config = load_config(config_path)
        logger.info("Configuration loaded and validated successfully")
    except Exception as e:
        logger.error("Configuration validation failed: %s", e)
        return 1

    config_version = get_config_version(config)
    logger.info("Config version: %s", config_version)

    working_db_path.parent.mkdir(parents=True, exist_ok=True)

    db = Stage00DatabaseInterface(
        working_db_path=working_db_path,
        source_db_path=source_db_path,
    )

    try:
        db.open()
        logger.info("Database connection established")
    except DBSchemaError as e:
        logger.error("Schema validation failed: %s", e)
        return 1
    except Exception as e:
        logger.error("Database connection failed: %s", e)
        return 1

    try:
        validate_config_stability(db, config_version)
    except ConfigDriftError as e:
        logger.error("Config stability check failed: %s", e)
        db.close()
        return 1

    try:
        result = acquire_run(db, run_id, config_version)
        logger.info(
            "Run acquisition successful: status=%s, is_fresh_db=%s",
            result.status.value,
            result.is_fresh_db,
        )
    except CompletedRunReuseError as e:
        logger.error("Run acquisition failed: %s", e)
        db.close()
        return 1
    except AnotherRunActiveError as e:
        logger.error("Run acquisition failed: %s", e)
        db.close()
        return 1
    except DBConstraintError as e:
        logger.error("Database constraint error during run acquisition: %s", e)
        db.close()
        return 1
    except Exception as e:
        logger.error("Unexpected error during run acquisition: %s", e)
        db.close()
        return 1

    db.close()
    logger.info("Stage 00 Setup completed successfully")
    return 0


def main_stage_00_setup() -> int:
    """
    Set main entry point for Stage 00 Setup.

    :return: Exit code (0 for success, 1 for failure).
    """
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    return run_setup(
        run_id=args.run_id,
        config_dir=args.config_dir,
        source_db_path=args.source_db,
        working_db_path=args.working_db,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    sys.exit(main_stage_00_setup())