from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from open_event_intel.etl_processing.database_interface import DatabaseInterface, DBError


class Stage00DatabaseInterface(DatabaseInterface):
    """
    Database interface for Stage 00 setup.

    Provides methods for run acquisition, config stability checks,
    and stale-run recovery (marking an abandoned ``'running'`` run
    as ``'failed'`` so a new run can be acquired).
    """

    READS: ClassVar[set[str]] = {"pipeline_run", "run_stage_status"}
    WRITES: ClassVar[set[str]] = {"pipeline_run"}

    def __init__(
        self,
        working_db_path: Path,
        source_db_path: Path | None = None,
    ) -> None:
        """Initialize."""
        super().__init__(
            working_db_path=working_db_path,
            source_db_path=source_db_path,
            stage_name="stage_00_setup",
        )

    # Stale-run recovery

    def get_last_stage_for_run(self, run_id: str) -> dict | None:
        """
        Return the furthest-reached ``run_stage_status`` row for *run_id*.

        Stages are lexicographically ordered (``stage_01_…`` through
        ``stage_11_…``), so ``ORDER BY stage DESC`` gives the last stage
        that was attempted.

        :param run_id: The pipeline run to inspect.
        :return: Dict with ``stage``, ``status``, ``error_message`` or ``None``.
        """
        self._check_read_access("run_stage_status")
        row = self._fetchone(
            """SELECT stage, status, error_message
               FROM run_stage_status
               WHERE run_id = ?
               ORDER BY stage DESC
               LIMIT 1""",
            (run_id,),
        )
        return dict(row) if row else None

    def fail_pipeline_run(self, run_id: str, error_message: str | None = None) -> None:
        """
        Transition a pipeline run from ``'running'`` to ``'failed'``.

        Sets ``completed_at`` to the current UTC timestamp.  Only updates
        rows whose current status is ``'running'`` to prevent clobbering
        a legitimately completed or already-failed run.

        :param run_id: The pipeline run ID to mark as failed.
        :param error_message: Optional reason string (not stored in
            ``pipeline_run`` itself, but logged by the caller).
        :raises DBError: If no row was updated.
        """
        self._check_write_access("pipeline_run")
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._execute(
            """UPDATE pipeline_run
               SET status = 'failed', completed_at = ?
               WHERE run_id = ? AND status = 'running'""",
            (now, run_id),
        )
        if cursor.rowcount == 0:
            raise DBError(
                f"Cannot fail pipeline run '{run_id}': "
                f"no running row found (already completed, failed, or missing)"
            )