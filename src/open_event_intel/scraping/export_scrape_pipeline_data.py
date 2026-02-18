#!/usr/bin/env python3
"""
Export scrape & preprocess pipeline data to a single SQLite database.

Reads ``scraped_posts.db`` and ``preprocessed_posts.db``, computes per-publisher /
per-date counts and KPIs, and writes ``scrape_data/sqlite/scrape.sqlite``.

Output layout::

    scrape_data/sqlite/
        scrape.sqlite   — meta, overview, matrix, publications (single file)
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
    id: str = Field(..., alias="ID")
    published_on: str
    title: str
    added_on: str
    url: str
    language: str
    post_length: int = Field(ge=0)
    model_config = {"populate_by_name": True}


class DateBucket(BaseModel):
    scraped: list[Publication] = Field(default_factory=list)
    preprocessed: list[Publication] = Field(default_factory=list)


class PublisherData(BaseModel):
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
    publishers: dict[Publisher, PublisherData] = Field(default_factory=dict)


class CLIConfig(BaseModel):
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


def parse_date(value: Any) -> str | None:
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
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None


def query_publications(conn: sqlite3.Connection, publisher: Publisher) -> list[Publication]:
    table = publisher.value.lower()
    if not table_exists(conn, table):
        logger.debug("Table '%s' not found — skipping.", table)
        return []
    cur = conn.execute(
        f'SELECT ID, published_on, title, added_on, url, language, '
        f'length(post) AS post_length FROM "{table}"')
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
        logger.warning("Publisher %s: skipped %d invalid rows.", publisher.value, skipped)
    return publications


def collect_data(scraped_conn: sqlite3.Connection, preprocessed_conn: sqlite3.Connection) -> PipelineData:
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
        pd = PublisherData(publisher=publisher, scraped=scraped,
                           preprocessed=preprocessed, by_date=dict(by_date))
        pipeline.publishers[publisher] = pd
        logger.info("  %s → %d scraped, %d preprocessed, %d dates.",
                     publisher.value, len(scraped), len(preprocessed), len(by_date))
    return pipeline


def validate_pipeline(data: PipelineData) -> list[str]:
    warnings: list[str] = []
    for publisher, pd in sorted(data.publishers.items(), key=lambda kv: kv[0].value):
        n_s = len(pd.scraped)
        n_p = len(pd.preprocessed)
        if n_p > n_s:
            warnings.append(f"{publisher.value}: {n_p} preprocessed > {n_s} scraped")
        for date_str, bucket in sorted(pd.by_date.items()):
            ns = len(bucket.scraped)
            np_ = len(bucket.preprocessed)
            if np_ > ns:
                warnings.append(f"{publisher.value}/{date_str}: {np_} preprocessed > {ns} scraped")
        scraped_ids = {p.id for p in pd.scraped}
        orphans = pd.preprocessed_ids - scraped_ids
        if orphans:
            warnings.append(f"{publisher.value}: {len(orphans)} orphan preprocessed IDs")
    return warnings


def _all_dates(data: PipelineData) -> list[str]:
    dates: set[str] = set()
    for pd in data.publishers.values():
        dates.update(pd.by_date.keys())
    return sorted(dates)


def _today_cet() -> datetime:
    cet = timezone(timedelta(hours=1))
    return datetime.now(cet)


def _j(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, default=str)


# ── SQLite export ───────────────────────────────────────────────────────

def _create_output_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    out = sqlite3.connect(str(path))
    out.execute("PRAGMA journal_mode = WAL")
    out.execute("PRAGMA synchronous = NORMAL")
    return out


def _init_scrape_schema(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE overview (id INTEGER PRIMARY KEY DEFAULT 1, data_json TEXT NOT NULL);
        CREATE TABLE matrix_date (date_str TEXT PRIMARY KEY);
        CREATE TABLE publisher_total (publisher TEXT PRIMARY KEY, total_scraped INTEGER NOT NULL);
        CREATE TABLE matrix_cell (
            publisher TEXT NOT NULL,
            date_str  TEXT NOT NULL,
            n_scraped      INTEGER NOT NULL,
            n_preprocessed INTEGER NOT NULL,
            PRIMARY KEY (publisher, date_str)
        );
        CREATE TABLE publication (
            publisher  TEXT NOT NULL,
            date_str   TEXT NOT NULL,
            id         TEXT NOT NULL,
            title      TEXT,
            published_on TEXT,
            scraped_on TEXT,
            language   TEXT,
            length_before INTEGER,
            length_after  INTEGER,
            url        TEXT,
            PRIMARY KEY (publisher, date_str, id)
        );
        CREATE INDEX idx_pub_date ON publication(publisher, date_str);
    """)


def export_sqlite(data: PipelineData, output_path: Path, window_days: int = 14) -> None:
    db = _create_output_db(output_path)
    _init_scrape_schema(db)

    # Meta
    sorted_dates = _all_dates(data)
    today_str = _today_cet().strftime("%Y-%m-%d")
    active = [p.value for p in ALL_PUBLISHERS
              if data.publishers.get(p) and data.publishers[p].by_date]
    publishers_list = active if active else [p.value for p in ALL_PUBLISHERS]

    db.execute("INSERT INTO meta VALUES (?, ?)",
               ("generated_at_utc", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
    db.execute("INSERT INTO meta VALUES (?, ?)", ("timezone", "CET"))
    db.execute("INSERT INTO meta VALUES (?, ?)", ("publishers", _j(publishers_list)))
    db.execute("INSERT INTO meta VALUES (?, ?)", ("available_dates_min",
               sorted_dates[0] if sorted_dates else ""))
    db.execute("INSERT INTO meta VALUES (?, ?)", ("available_dates_max", today_str))

    # Overview
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

    overview = {
        "totals": {
            "publishers": publisher_count,
            "publications_scraped": total_scraped,
            "publications_preprocessed": total_preprocessed,
        },
        "yesterday": {
            "date": yesterday_date,
            "publications_scraped": yesterday_scraped,
        },
        "length_chars": {
            "largest_after_preprocessing": max(lengths_after) if lengths_after else 0,
            "smallest_after_preprocessing": min(lengths_after) if lengths_after else 0,
            "largest_before_preprocessing": max(lengths_before) if lengths_before else 0,
            "smallest_before_preprocessing": min(lengths_before) if lengths_before else 0,
        },
    }
    db.execute("INSERT INTO overview VALUES (1, ?)", (_j(overview),))

    # Matrix
    all_dates = _all_dates(data)
    for d in all_dates:
        db.execute("INSERT INTO matrix_date VALUES (?)", (d,))

    detail_count = 0
    for publisher in ALL_PUBLISHERS:
        pd = data.publishers.get(publisher)
        total = len(pd.scraped) if pd else 0
        db.execute("INSERT INTO publisher_total VALUES (?, ?)", (publisher.value, total))

        if pd is None:
            continue

        for date_str in all_dates:
            bucket = pd.by_date.get(date_str)
            if not bucket:
                continue
            ns = len(bucket.scraped)
            np_ = len(bucket.preprocessed)
            if ns > 0 or np_ > 0:
                db.execute("INSERT INTO matrix_cell VALUES (?, ?, ?, ?)",
                           (publisher.value, date_str, ns, np_))

        # Publications detail
        pre_by_id = pd.preprocessed_by_id
        for date_str, bucket in pd.by_date.items():
            if not bucket.scraped:
                continue
            for pub in bucket.scraped:
                preprocessed = pre_by_id.get(pub.id)
                db.execute(
                    "INSERT OR IGNORE INTO publication VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (publisher.value, date_str, pub.id, pub.title, pub.published_on,
                     pub.added_on, pub.language, pub.post_length,
                     preprocessed.post_length if preprocessed else None, pub.url))
                detail_count += 1

    db.commit()
    logger.info("Exported %d matrix dates, %d publication rows", len(all_dates), detail_count)

    db.execute("PRAGMA journal_mode = DELETE")
    db.execute("VACUUM")
    db.close()
    size_kb = output_path.stat().st_size / 1024
    logger.info("Wrote %s (%.1f KB)", output_path, size_kb)


def run(config: CLIConfig) -> None:
    logger.info("Opening databases …")
    scraped_conn = sqlite3.connect(str(config.scraped_db))
    preprocessed_conn = sqlite3.connect(str(config.preprocessed_db))

    try:
        logger.info("Collecting data …")
        data = collect_data(scraped_conn, preprocessed_conn)

        logger.info("Validating …")
        warnings = validate_pipeline(data)
        for w in warnings:
            logger.warning(w)

        output_path = config.output_dir / "sqlite" / "scrape.sqlite"
        logger.info("Exporting to %s …", output_path)
        export_sqlite(data, output_path, window_days=config.window_days)

        total_s = sum(len(pd.scraped) for pd in data.publishers.values())
        total_p = sum(len(pd.preprocessed) for pd in data.publishers.values())
        logger.info("Done — %d scraped, %d preprocessed, %d warnings.",
                     total_s, total_p, len(warnings))
    finally:
        scraped_conn.close()
        preprocessed_conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export scrape & preprocess data to SQLite.")
    parser.add_argument("--scraped-db", type=Path,
                        default=Path("../../../database/scraped_posts.db"))
    parser.add_argument("--preprocessed-db", type=Path,
                        default=Path("../../../database/preprocessed_posts.db"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("../../../docs/scrape_data"))
    parser.add_argument("--window-days", type=int, default=60)
    args = parser.parse_args()

    try:
        config = CLIConfig(scraped_db=args.scraped_db,
                           preprocessed_db=args.preprocessed_db,
                           output_dir=args.output_dir,
                           window_days=args.window_days)
    except Exception as exc:
        logger.error("Invalid configuration: %s", exc)
        sys.exit(1)

    run(config)


if __name__ == "__main__":
    main()
