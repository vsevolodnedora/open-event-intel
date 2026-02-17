from pathlib import Path
from typing import ClassVar

from open_event_intel.etl_processing.database_interface import DatabaseInterface


class Stage00DatabaseInterface(DatabaseInterface):
    """
    Database interface for Stage 00 setup.

    Provides methods for run acquisition and config stability checks.
    Writes only to pipeline_run table.
    """

    READS: ClassVar[set[str]] = {"pipeline_run"}
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
