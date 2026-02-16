#!/usr/bin/env python3
"""
Export scrape & preprocess pipeline data to static JSON for GitHub Pages.

Reads ``scraped_posts.db`` and ``preprocessed_posts.db``, computes per-publisher /
per-date counts and KPIs, and writes the ``scrape_data/`` folder structure
expected by the web frontend.

Output layout::

    scrape_data/
        meta.json
        overview.json
        matrix.json
        publications/
            <publisher>/
                <YYYY-MM-DD>.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from open_event_intel.logger import get_logger

logger = get_logger(__name__)


class Publisher(str, Enum):
    """Canonical publisher identifiers matching DB table names."""

    SMARD = "SMARD"
    EEX = "EEX"
    ENTSOE = "ENTSOE"
    ACER = "ACER"
    EC = "EC"
    BNETZA = "BNETZA"
    TRANSNETBW = "TRANSNETBW"
    TENNET = "TENNET"
    FIFTY_HERTZ = "FIFTY_HERTZ"
    AMPRION = "AMPRION"
    ICIS = "ICIS"
    AGORA = "AGORA"
    ENERGY_WIRE = "ENERGY_WIRE"


ALL_PUBLISHERS: list[Publisher] = list(Publisher)


class Publication(BaseModel):
    """A single publication row from either database."""

    id: str = Field(..., alias="ID")
    published_on: str
    title: str
    added_on: str
    url: str
    language: str
    post_length: int = Field(ge=0)

    model_config = {"populate_by_name": True}


class DateBucket(BaseModel):
    """Scraped and preprocessed publications for one publisher on one date."""

    scraped: list[Publication] = Field(default_factory=list)
    preprocessed: list[Publication] = Field(default_factory=list)


class PublisherData(BaseModel):
    """Aggregated data for a single publisher across both databases."""

    publisher: Publisher
    scraped: list[Publication] = Field(default_factory=list)
    preprocessed: list[Publication] = Field(default_factory=list)
    by_date: dict[str, DateBucket] = Field(default_factory=dict)

    @property
    def preprocessed_ids(self) -> set[str]:
        return {p.id for p in self.preprocessed}

    @property
    def preprocessed_by_id(self) -> dict[str, Publication]:
        return {p.id: p for p in self.preprocessed}


class PipelineData(BaseModel):
    """Complete pipeline snapshot across all publishers."""

    publishers: dict[Publisher, PublisherData] = Field(default_factory=dict)


class CLIConfig(BaseModel):
    """Validated CLI arguments."""

    scraped_db: Path
    preprocessed_db: Path
    output_dir: Path
    window_days: int = Field(default=14, ge=1)

    @field_validator("scraped_db", "preprocessed_db")
    @classmethod
    def _db_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Database file not found: {v}")
        return v


class MetaOutput(BaseModel):
    """Schema for ``meta.json``."""

    generated_at_utc: str
    timezone: str = "CET"
    publishers: list[str]
    available_dates: dict[str, str | None]


class OverviewTotals(BaseModel):
    publishers: int
    publications_scraped: int
    publications_preprocessed: int


class OverviewYesterday(BaseModel):
    date: str | None
    publications_scraped: int


class OverviewLengths(BaseModel):
    largest_after_preprocessing: int = 0
    smallest_after_preprocessing: int = 0
    largest_before_preprocessing: int = 0
    smallest_before_preprocessing: int = 0


class OverviewOutput(BaseModel):
    """Schema for ``overview.json``."""

    totals: OverviewTotals
    yesterday: OverviewYesterday
    length_chars: OverviewLengths


class MatrixCell(BaseModel):
    n_scraped: int
    n_preprocessed: int


class MatrixRow(BaseModel):
    publisher: str
    total_scraped: int
    cells: dict[str, MatrixCell]


class MatrixOutput(BaseModel):
    """Schema for ``matrix.json``."""

    dates: list[str]
    rows: list[MatrixRow]


class PublicationDetail(BaseModel):
    """A single publication entry inside a detail JSON file."""

    id: str
    title: str
    published_on: str
    scraped_on: str
    language: str
    length_before_preprocessing: int
    length_after_preprocessing: int | None
    url: str


class PublicationDetailOutput(BaseModel):
    """Schema for ``publications/<publisher>/<date>.json``."""

    publisher: str
    published_on: str
    publications: list[PublicationDetail]


def parse_date(value: Any) -> str | None:
    """
    Extract ``YYYY-MM-DD`` from a timestamp string.

    :param value: Raw timestamp from the database.
    :returns: Date string or *None* if unparseable.

    Handles ``YYYY-MM-DD HH:MM:SS``, ISO-8601, and bare date formats.
    Falls back to a prefix check when no format matches.
    """
    if value is None:
        return None
    s = str(value)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:19], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return None


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check whether *table_name* exists in the connected SQLite database.

    :param conn: Open SQLite connection.
    :param table_name: Table name to probe.
    """
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cur.fetchone() is not None


def query_publications(
    conn: sqlite3.Connection, publisher: Publisher
) -> list[Publication]:
    """Read all publications for *publisher* from an SQLite database.

    :param conn: Open SQLite connection.
    :param publisher: Publisher whose table to query.
    :returns: Validated :class:`Publication` list (empty when table is missing).
    """
    table = publisher.value.lower()
    if not table_exists(conn, table):
        logger.debug("Table '%s' not found — skipping.", table)
        return []

    cur = conn.execute(
        f'SELECT ID, published_on, title, added_on, url, language, '
        f'length(post) AS post_length FROM "{table}"'
    )
    columns = [desc[0] for desc in cur.description]
    publications: list[Publication] = []
    skipped = 0
    for row in cur.fetchall():
        raw = dict(zip(columns, row))
        try:
            publications.append(Publication.model_validate(raw))
        except Exception:
            skipped += 1
    if skipped:
        logger.warning(
            "Publisher %s: skipped %d rows that failed validation.", publisher.value, skipped
        )
    return publications


def collect_data(
    scraped_conn: sqlite3.Connection,
    preprocessed_conn: sqlite3.Connection,
) -> PipelineData:
    """Query both databases and assemble a :class:`PipelineData` snapshot.

    :param scraped_conn: Connection to ``scraped_posts.db``.
    :param preprocessed_conn: Connection to ``preprocessed_posts.db``.
    """
    pipeline = PipelineData()

    for publisher in ALL_PUBLISHERS:
        logger.info("Collecting %s …", publisher.value)

        scraped = query_publications(scraped_conn, publisher)
        preprocessed = query_publications(preprocessed_conn, publisher)

        by_date: dict[str, DateBucket] = defaultdict(DateBucket)

        for pub in scraped:
            date_str = parse_date(pub.published_on)
            if date_str:
                by_date[date_str].scraped.append(pub)

        for pub in preprocessed:
            date_str = parse_date(pub.published_on)
            if date_str:
                by_date[date_str].preprocessed.append(pub)

        pd = PublisherData(
            publisher=publisher,
            scraped=scraped,
            preprocessed=preprocessed,
            by_date=dict(by_date),
        )
        pipeline.publishers[publisher] = pd

        logger.info(
            "  %s → %d scraped, %d preprocessed, %d unique dates.",
            publisher.value,
            len(scraped),
            len(preprocessed),
            len(by_date),
        )

    return pipeline


def validate_pipeline(data: PipelineData) -> list[str]:
    """Run consistency checks and return a list of warning messages.

    :param data: Fully populated pipeline data.

    Checks performed:
    * preprocessed count ≤ scraped count (global and per-date).
    * every preprocessed ID has a matching scraped ID.
    """
    warnings: list[str] = []

    for publisher, pd in sorted(data.publishers.items(), key=lambda kv: kv[0].value):
        n_scraped = len(pd.scraped)
        n_preprocessed = len(pd.preprocessed)

        if n_preprocessed > n_scraped:
            warnings.append(
                f"{publisher.value}: {n_preprocessed} preprocessed > "
                f"{n_scraped} scraped (global)"
            )

        for date_str, bucket in sorted(pd.by_date.items()):
            ns = len(bucket.scraped)
            np_ = len(bucket.preprocessed)
            if np_ > ns:
                warnings.append(
                    f"{publisher.value}/{date_str}: {np_} preprocessed > {ns} scraped"
                )

        scraped_ids = {p.id for p in pd.scraped}
        orphans = pd.preprocessed_ids - scraped_ids
        if orphans:
            warnings.append(
                f"{publisher.value}: {len(orphans)} preprocessed IDs with no scraped match"
            )

    return warnings


def _all_dates(data: PipelineData) -> list[str]:
    """Return all unique dates across every publisher, sorted ascending."""
    dates: set[str] = set()
    for pd in data.publishers.values():
        dates.update(pd.by_date.keys())
    return sorted(dates)


def _today_cet() -> datetime:
    """Return the current date in CET/CEST (Europe/Berlin).

    Uses a fixed UTC+1 offset as a safe fallback; the one-hour
    difference to CEST is acceptable for date-level granularity.
    """
    cet = timezone(timedelta(hours=1))
    return datetime.now(cet)


def _calendar_dates(end: datetime, n_days: int) -> list[str]:
    """Return *n_days* contiguous ``YYYY-MM-DD`` strings ending at *end* (inclusive)."""
    return [
        (end - timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(n_days - 1, -1, -1)
    ]


def build_meta(data: PipelineData) -> MetaOutput:
    """Build the ``meta.json`` payload.

    :param data: Pipeline snapshot.
    """
    sorted_dates = _all_dates(data)
    today_str = _today_cet().strftime("%Y-%m-%d")
    active = [
        p.value for p in ALL_PUBLISHERS
        if data.publishers.get(p) and data.publishers[p].by_date
    ]

    return MetaOutput(
        generated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        publishers=active if active else [p.value for p in ALL_PUBLISHERS],
        available_dates={
            "min": sorted_dates[0] if sorted_dates else None,
            "max": today_str,
        },
    )


def build_overview(data: PipelineData) -> OverviewOutput:
    """
    Build the ``overview.json`` payload (KPI strip).

    :param data: Pipeline snapshot.
    """
    total_scraped = 0
    total_preprocessed = 0
    publisher_count = 0
    lengths_before: list[int] = []
    lengths_after: list[int] = []

    for pd in data.publishers.values():
        ns = len(pd.scraped)
        np_ = len(pd.preprocessed)
        total_scraped += ns
        total_preprocessed += np_
        if ns > 0 or np_ > 0:
            publisher_count += 1

        lengths_before.extend(p.post_length for p in pd.scraped if p.post_length > 0)
        lengths_after.extend(p.post_length for p in pd.preprocessed if p.post_length > 0)

    yesterday_date = (_today_cet() - timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_scraped = 0
    for pd in data.publishers.values():
        bucket = pd.by_date.get(yesterday_date)
        if bucket:
            yesterday_scraped += len(bucket.scraped)

    return OverviewOutput(
        totals=OverviewTotals(
            publishers=publisher_count,
            publications_scraped=total_scraped,
            publications_preprocessed=total_preprocessed,
        ),
        yesterday=OverviewYesterday(
            date=yesterday_date,
            publications_scraped=yesterday_scraped,
        ),
        length_chars=OverviewLengths(
            largest_after_preprocessing=max(lengths_after) if lengths_after else 0,
            smallest_after_preprocessing=min(lengths_after) if lengths_after else 0,
            largest_before_preprocessing=max(lengths_before) if lengths_before else 0,
            smallest_before_preprocessing=min(lengths_before) if lengths_before else 0,
        ),
    )


def build_matrix(data: PipelineData, window_days: int = 14) -> MatrixOutput:
    """
    Build the ``matrix.json`` payload (publisher × date grid).

    :param data: Pipeline snapshot.
    :param window_days: Ignored (kept for API compatibility). All available
        dates are exported so the frontend can window freely (e.g. "Last month").

    All dates present in any publisher's data are included so the
    frontend's date-window controls (presets, manual range) can display
    any historical period without gaps.
    """
    # Collect every date that has data across all publishers
    all_dates = _all_dates(data)

    rows: list[MatrixRow] = []
    for publisher in ALL_PUBLISHERS:
        pd = data.publishers.get(publisher)
        if pd is None:
            rows.append(MatrixRow(publisher=publisher.value, total_scraped=0, cells={}))
            continue

        cells: dict[str, MatrixCell] = {}
        for date_str in all_dates:
            bucket = pd.by_date.get(date_str)
            if bucket:
                ns = len(bucket.scraped)
                np_ = len(bucket.preprocessed)
                if ns > 0 or np_ > 0:
                    cells[date_str] = MatrixCell(n_scraped=ns, n_preprocessed=np_)

        rows.append(
            MatrixRow(
                publisher=publisher.value,
                total_scraped=len(pd.scraped),
                cells=cells,
            )
        )

    return MatrixOutput(dates=all_dates, rows=rows)


def build_publication_detail(
    publisher: Publisher, date_str: str, pd: PublisherData
) -> PublicationDetailOutput | None:
    """
    Build a single ``publications/<publisher>/<date>.json`` payload.

    :param publisher: Target publisher.
    :param date_str: Date string (``YYYY-MM-DD``).
    :param pd: Publisher's aggregated data.
    :returns: Output model, or *None* if no scraped data exists for that date.
    """
    bucket = pd.by_date.get(date_str)
    if not bucket or not bucket.scraped:
        return None

    pre_by_id = pd.preprocessed_by_id
    details: list[PublicationDetail] = []

    for pub in bucket.scraped:
        preprocessed = pre_by_id.get(pub.id)
        details.append(
            PublicationDetail(
                id=pub.id,
                title=pub.title,
                published_on=pub.published_on,
                scraped_on=pub.added_on,
                language=pub.language,
                length_before_preprocessing=pub.post_length,
                length_after_preprocessing=preprocessed.post_length if preprocessed else None,
                url=pub.url,
            )
        )

    return PublicationDetailOutput(
        publisher=publisher.value,
        published_on=date_str,
        publications=details,
    )


def write_json(path: Path, model: BaseModel) -> None:
    """
    Serialize a Pydantic model to a JSON file.

    :param path: Destination file path (parent dirs are created automatically).
    :param model: Pydantic model to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.model_dump(mode="json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.debug("Wrote %s (%s bytes).", path, f"{path.stat().st_size:,}")


def export_all(data: PipelineData, output_dir: Path, window_days: int = 14) -> None:
    """
    Write every output file to *output_dir*.

    :param data: Validated pipeline snapshot.
    :param output_dir: Root output directory.
    :param window_days: Date window forwarded to :func:`build_matrix`.
    """
    logger.info("Writing output to %s/", output_dir)

    meta = build_meta(data)
    write_json(output_dir / "meta.json", meta)
    logger.info("Exported meta.json (%d publishers).", len(meta.publishers))

    overview = build_overview(data)
    write_json(output_dir / "overview.json", overview)
    logger.info(
        "Exported overview.json — %d scraped, %d preprocessed.",
        overview.totals.publications_scraped,
        overview.totals.publications_preprocessed,
    )

    matrix = build_matrix(data, window_days=window_days)
    write_json(output_dir / "matrix.json", matrix)
    logger.info("Exported matrix.json (%d dates × %d publishers).", len(matrix.dates), len(matrix.rows))

    detail_count = 0
    for publisher in ALL_PUBLISHERS:
        pd = data.publishers.get(publisher)
        if pd is None:
            continue
        for date_str in sorted(pd.by_date.keys()):
            detail = build_publication_detail(publisher, date_str, pd)
            if detail:
                detail_path = output_dir / "publications" / publisher.value / f"{date_str}.json"
                write_json(detail_path, detail)
                detail_count += 1

    logger.info("Exported %d publication detail files.", detail_count)


def log_several_day_summary(data: PipelineData, n=30) -> None:
    """Log a table of scraped / preprocessed counts per publisher per day (last N days).

    :param data: Pipeline snapshot.

    The table is printed via the logger so it appears in the standard log
    stream alongside other pipeline output.
    """
    all_dates = _all_dates(data)
    if not all_dates:
        logger.info(f"No dates available — skipping {n}-day summary table.")
        return

    today = _today_cet()
    last_n: list[str] = _calendar_dates(today, n)

    pub_col_width = max(len(p.value) for p in ALL_PUBLISHERS)
    date_col_width = 11  # "MM-DD s/p" fits in per-date columns

    header_dates = [d[5:] for d in last_n]  # "MM-DD" for compactness
    header = f"{'Publisher':<{pub_col_width}}  " + "  ".join(
        f"{'s':>3}/{'p':>3} {hd}" for hd in header_dates
    )
    separator = "-" * len(header)

    lines: list[str] = [
        "",
        f"{n}-Day Pipeline Summary (s = scraped, p = preprocessed)",
        separator,
        header,
        separator,
    ]

    for publisher in ALL_PUBLISHERS:
        pd = data.publishers.get(publisher)
        cells: list[str] = []
        for date_str in last_n:
            if pd:
                bucket = pd.by_date.get(date_str)
                ns = len(bucket.scraped) if bucket else 0
                np_ = len(bucket.preprocessed) if bucket else 0
            else:
                ns, np_ = 0, 0
            cells.append(f"{ns:>3}/{np_:>3}     ")
        row = f"{publisher.value:<{pub_col_width}}  " + "  ".join(cells)
        lines.append(row)

    lines.append(separator)

    totals: list[str] = []
    for date_str in last_n:
        ts = sum(
            len(pd.by_date.get(date_str, DateBucket()).scraped)
            for pd in data.publishers.values()
        )
        tp = sum(
            len(pd.by_date.get(date_str, DateBucket()).preprocessed)
            for pd in data.publishers.values()
        )
        totals.append(f"{ts:>3}/{tp:>3}     ")
    lines.append(f"{'TOTAL':<{pub_col_width}}  " + "  ".join(totals))
    lines.append(separator)

    for line in lines:
        logger.info(line)


def run(config: CLIConfig) -> None:
    """
    Top-level orchestrator: collect → validate → export → summarise.

    :param config: Validated CLI configuration.
    """
    logger.info("Opening databases …")
    logger.info("  scraped_db      = %s", config.scraped_db.resolve())
    logger.info("  preprocessed_db = %s", config.preprocessed_db.resolve())
    logger.info("  output_dir      = %s", config.output_dir.resolve())
    logger.info("  window_days     = %d", config.window_days)

    scraped_conn = sqlite3.connect(str(config.scraped_db))
    preprocessed_conn = sqlite3.connect(str(config.preprocessed_db))

    try:
        logger.info("Collecting data from databases …")
        data = collect_data(scraped_conn, preprocessed_conn)

        logger.info("Validating pipeline consistency …")
        warnings = validate_pipeline(data)
        for w in warnings:
            logger.warning(w)
        if not warnings:
            logger.info("Validation passed — no inconsistencies found.")

        logger.info("Exporting JSON files …")
        export_all(data, config.output_dir, window_days=config.window_days)

        log_several_day_summary(data)

        total_s = sum(len(pd.scraped) for pd in data.publishers.values())
        total_p = sum(len(pd.preprocessed) for pd in data.publishers.values())
        logger.info(
            "Done — %d scraped, %d preprocessed across %d publishers. %d warnings.",
            total_s,
            total_p,
            len(ALL_PUBLISHERS),
            len(warnings),
        )
    finally:
        scraped_conn.close()
        preprocessed_conn.close()
        logger.info("Database connections closed.")


def main() -> None:
    """CLI entry point — parse arguments, validate, and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Export scrape & preprocess pipeline data to static JSON."
    )
    parser.add_argument(
        "--scraped-db",
        type=Path,
        default=Path("../../../database/scraped_posts.db"),
        help="Path to scraped_posts.db",
    )
    parser.add_argument(
        "--preprocessed-db",
        type=Path,
        default=Path("../../../database/preprocessed_posts.db"),
        help="Path to preprocessed_posts.db",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../../../docs/scrape_data"),
        help="Output directory for JSON files",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=60,
        help="Number of days to include in the matrix (default: 60)",
    )

    args = parser.parse_args()

    try:
        config = CLIConfig(
            scraped_db=args.scraped_db,
            preprocessed_db=args.preprocessed_db,
            output_dir=args.output_dir,
            window_days=args.window_days,
        )
    except Exception as exc:
        logger.error("Invalid configuration: %s", exc)
        sys.exit(1)

    run(config)


if __name__ == "__main__":
    main()