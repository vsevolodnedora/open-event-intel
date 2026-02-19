#!/usr/bin/env python3
"""
Deduplicate source-database publications that would cause natural-key
collisions in stage_01_ingest.

stage_01_ingest inserts into ``scrape_record`` with a UNIQUE constraint on
``(publisher_id, url_normalized, source_published_at, scrape_kind)``.
Two source rows with different ``ID`` values can collide after URL
normalisation when they share the same normalised URL and ``published_on``.

This script finds those collisions and drops the row with the *smaller*
decompressed ``post`` content, keeping the richer publication.

Usage:
    # Dry-run (default): only reports, changes nothing
    python dedup_source_db.py --source-db path/to/preprocessed_posts.db

    # Apply deletes
    python dedup_source_db.py --source-db path/to/preprocessed_posts.db --apply

    # Backup before applying (recommended)
    python dedup_source_db.py --source-db path/to/preprocessed_posts.db --apply --backup
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import sys
import zlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse, urlunparse

from open_event_intel.logger import get_logger

logger = get_logger(__name__)

# Publisher → source table mapping (must match stage_01_ingest)
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

# Reverse: table name → publisher id
TABLE_TO_PUBLISHER: dict[str, str] = {v: k for k, v in PUBLISHER_TABLE_MAP.items()}

# Publisher URL normalisation rules (extracted from config.yaml)
# Only fields that affect normalisation are included.
_URL_NORM_RULES: dict[str, dict] = {
    "SMARD": {
        "canonical_host": "www.smard.de",
        "strip_params": ["utm_source", "utm_medium", "utm_campaign", "ref", "bust"],
        "preserve_params": ["lang", "date", "region", "topic"],
    },
    "EEX": {
        "canonical_host": "www.eex.com",
        "strip_params": ["tx_news_pi1", "cHash", "utm_source", "utm_medium", "utm_campaign"],
        "preserve_params": [],
    },
    "ENTSOE": {
        "canonical_host": "www.entsoe.eu",
        "strip_params": ["utm_source", "utm_medium", "utm_campaign"],
        "preserve_params": [],
    },
    "ACER": {
        "canonical_host": "www.acer.europa.eu",
        "strip_params": [],
        "preserve_params": [],
    },
    "EC": {
        "canonical_host": "energy.ec.europa.eu",
        "strip_params": [],
        "preserve_params": [],
    },
    "BNETZA": {
        "canonical_host": "www.bundesnetzagentur.de",
        "strip_params": ["nn", "utm_source", "utm_medium"],
        "preserve_params": [],
    },
    "TRANSNETBW": {
        "canonical_host": "www.transnetbw.de",
        "strip_params": [],
        "preserve_params": ["lang"],
    },
    "TENNET": {
        "canonical_host": "www.tennet.eu",
        "strip_params": [],
        "preserve_params": ["lang"],
    },
    "FIFTY_HERTZ": {
        "canonical_host": "www.50hertz.com",
        "strip_params": [],
        "preserve_params": ["lang"],
    },
    "AMPRION": {
        "canonical_host": "www.amprion.net",
        "strip_params": [],
        "preserve_params": [],
    },
    "ICIS": {
        "canonical_host": "www.icis.com",
        "strip_params": [],
        "preserve_params": [],
    },
    "AGORA": {
        "canonical_host": "www.agora-energiewende.de",
        "strip_params": [],
        "preserve_params": [],
    },
    "ENERGY_WIRE": {
        "canonical_host": "www.cleanenergywire.org",
        "strip_params": [],
        "preserve_params": [],
    },
}

# RFC 3986 §2.3 unreserved characters
_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)


# URL normalisation (replicates stage_01_ingest §1.4.5 exactly)
def _decode_unreserved(s: str) -> str:
    """Decode percent-encoded unreserved characters (RFC 3986 §2.3)."""
    decoded = unquote(s)
    result: list[str] = []
    for ch in decoded:
        if ch in _UNRESERVED or ch == "/":
            result.append(ch)
        else:
            result.append(quote(ch, safe=""))
    return "".join(result)


def _normalize_path(path: str) -> str:
    """Normalize path segments (remove ., resolve ..)."""
    if not path:
        return "/"
    segments = path.split("/")
    output: list[str] = []
    for seg in segments:
        if seg == ".":
            continue
        elif seg == "..":
            if output and output[-1] != "":
                output.pop()
        else:
            output.append(seg)
    result = "/".join(output)
    if not result.startswith("/"):
        result = "/" + result
    return result if result else "/"


def normalize_url(url_raw: str, publisher_id: str | None) -> str: # noqa: C901
    """
    Normalise a URL exactly as stage_01_ingest does.

    Uses the hardcoded ``_URL_NORM_RULES`` that mirror config.yaml.
    """
    try:
        parsed = urlparse(url_raw)
    except Exception:
        return url_raw

    scheme = parsed.scheme.lower()
    netloc = parsed.hostname or ""
    netloc = netloc.lower()

    port = parsed.port
    if port == 80 and scheme == "http":
        port = None
    if port == 443 and scheme == "https":
        port = None
    if port:
        netloc = f"{netloc}:{port}"

    path = _decode_unreserved(parsed.path)
    path = _normalize_path(path)
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    query_params = parse_qs(parsed.query, keep_blank_values=True)

    norm = _URL_NORM_RULES.get(publisher_id or "") if publisher_id else None
    if norm:
        if norm.get("canonical_host"):
            netloc = norm["canonical_host"].lower()
        for param in norm.get("strip_params", []):
            query_params.pop(param, None)
        preserve = norm.get("preserve_params", [])
        if preserve:
            query_params = {k: v for k, v in query_params.items() if k in preserve}

    sorted_params = sorted(query_params.items())
    query_parts = []
    for key, values in sorted_params:
        for val in sorted(values):
            query_parts.append(f"{key}={val}")
    query = "&".join(query_parts)

    return urlunparse((scheme, netloc, path, "", query, ""))


# Blob decompression (best-effort, mirrors the scraping pipeline)
def _decompress_blob(blob: bytes) -> str:
    """
    Decompress a publication blob to text.

    Tries zlib (raw deflate and zlib-wrapped), then falls back to
    decoding as plain UTF-8.
    """
    if not blob:
        return ""
    # Try zlib (wbits=15 for zlib-wrapped, wbits=-15 for raw deflate,
    # wbits=31 for gzip)
    for wbits in (15, -15, 31):
        try:
            return zlib.decompress(blob, wbits).decode("utf-8", errors="replace")
        except (zlib.error, UnicodeDecodeError):
            continue
    # Fallback: treat as raw text
    try:
        return blob.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _content_length(blob: bytes) -> int:
    """Return the *decompressed* text length of a post blob."""
    return len(_decompress_blob(blob))


# Core dedup logic
def find_duplicates(
    conn: sqlite3.Connection,
    table_name: str,
    publisher_id: str,
) -> list[tuple[str, list[tuple[str, str, str, int]]]]:
    """
    Find duplicate groups in a single publisher table.

    Groups rows by ``(url_normalized, published_on)`` and returns only
    groups with more than one member.

    Returns a list of ``(natural_key_repr, members)`` where each member
    is ``(row_id, url_raw, title, decompressed_content_length)``.
    """
    sql = f'SELECT ID, published_on, url, title, post FROM "{table_name}"'  # noqa: S608
    cursor = conn.execute(sql)

    # natural key → list of (ID, url_raw, title, content_length)
    groups: dict[str, list[tuple[str, str, str, int]]] = defaultdict(list)

    for row_id, published_on, url, title, blob in cursor.fetchall():
        url_norm = normalize_url(url, publisher_id)
        key = f"{url_norm}||{published_on}"
        content_len = _content_length(blob)
        groups[key].append((row_id, url, title, content_len))

    # Keep only groups with duplicates
    return [
        (key, members)
        for key, members in groups.items()
        if len(members) > 1
    ]


def resolve_duplicates(
    conn: sqlite3.Connection,
    table_name: str,
    publisher_id: str,
    apply: bool = False,
) -> tuple[int, int]:
    """
    Find and optionally delete duplicate publications.

    For each collision group, keeps the row with the largest content and
    marks the rest for deletion.

    :return: ``(groups_found, rows_deleted)``
    """
    dup_groups = find_duplicates(conn, table_name, publisher_id)
    if not dup_groups:
        return 0, 0

    rows_to_delete: list[str] = []

    for key, members in dup_groups:
        # Sort by content length descending; ties broken by ID for determinism
        members.sort(key=lambda m: (-m[3], m[0]))
        keeper = members[0]
        losers = members[1:]

        url_norm, published_on = key.rsplit("||", 1)
        logger.info(
            "  DUPLICATE GROUP: url_normalized=%s, published_at=%s (%d rows)",
            url_norm[:120],
            published_on,
            len(members),
        )
        logger.info(
            "    KEEP:   id=%s  content_len=%d  title=%r",
            keeper[0][:24],
            keeper[3],
            keeper[2][:80] if keeper[2] else "N/A",
        )
        for loser in losers:
            logger.info(
                "    DROP:   id=%s  content_len=%d  title=%r  url_raw=%s",
                loser[0][:24],
                loser[3],
                loser[2][:80] if loser[2] else "N/A",
                loser[1][:120] if loser[1] else "N/A",
            )
            rows_to_delete.append(loser[0])

    deleted = 0
    if apply and rows_to_delete:
        # Use parameterised DELETE in batches (SQLite variable limit is 999)
        batch_size = 500
        for i in range(0, len(rows_to_delete), batch_size):
            batch = rows_to_delete[i : i + batch_size]
            placeholders = ",".join("?" * len(batch))
            conn.execute(
                f'DELETE FROM "{table_name}" WHERE ID IN ({placeholders})',  # noqa: S608
                batch,
            )
        conn.commit()
        deleted = len(rows_to_delete)
        logger.info(
            "  Deleted %d duplicate rows from '%s'",
            deleted,
            table_name,
        )
    elif rows_to_delete:
        logger.info(
            "  [DRY-RUN] Would delete %d rows from '%s'",
            len(rows_to_delete),
            table_name,
        )

    return len(dup_groups), deleted if apply else len(rows_to_delete)


# Main
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Remove source-DB duplicate publications that would cause "
            "natural-key UNIQUE constraint errors in stage_01_ingest."
        ),
    )
    parser.add_argument(
        "--source-db",
        type=Path,
        default=Path("../../../database/preprocessed_posts.db"),
        help="Path to the source (preprocessed) database (default: %(default)s)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=True,
        help="Actually delete duplicates (default is dry-run)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create a timestamped backup before modifying the database",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


def main() -> int:  # noqa: C901
    """Set main entry point."""
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    db_path: Path = args.source_db
    if not db_path.exists():
        logger.error("Source database not found: %s", db_path)
        return 1

    mode_label = "APPLY" if args.apply else "DRY-RUN"
    logger.info("=== Source DB dedup (%s) ===", mode_label)
    logger.info("Database: %s", db_path.resolve())

    # Backup
    if args.apply and args.backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = db_path.with_suffix(f".pre_dedup_{ts}.db")
        logger.info("Creating backup: %s", backup_path)
        shutil.copy2(db_path, backup_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 30000")

    # Discover tables
    cursor = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    available_tables = {row[0] for row in cursor.fetchall()}
    logger.info("Tables in source DB: %s", sorted(available_tables))

    total_groups = 0
    total_would_delete = 0

    for publisher_id, table_name in sorted(PUBLISHER_TABLE_MAP.items()):
        if table_name not in available_tables:
            logger.debug("Table '%s' not present, skipping %s", table_name, publisher_id)
            continue

        # Get row count
        cnt = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]  # noqa: S608
        logger.info(
            "Processing %s (table='%s', rows=%d)", publisher_id, table_name, cnt
        )

        groups, deleted = resolve_duplicates(
            conn, table_name, publisher_id, apply=args.apply
        )
        total_groups += groups
        total_would_delete += deleted

    # Summary
    logger.info("=== Summary (%s) ===", mode_label)
    logger.info("  Duplicate groups found: %d", total_groups)
    if args.apply:
        logger.info("  Rows deleted: %d", total_would_delete)
    else:
        logger.info("  Rows that would be deleted: %d", total_would_delete)
        if total_would_delete > 0:
            logger.info("  Re-run with --apply to execute deletions")

    conn.close()

    # Verify (only when changes were made)
    if args.apply and total_would_delete > 0:
        logger.info("Verifying no remaining duplicates...")
        conn2 = sqlite3.connect(str(db_path))
        remaining = 0
        for publisher_id, table_name in sorted(PUBLISHER_TABLE_MAP.items()):
            if table_name not in available_tables:
                continue
            groups = find_duplicates(conn2, table_name, publisher_id)
            if groups:
                remaining += len(groups)
                logger.error(
                    "VERIFY FAILED: %d duplicate groups remain in '%s'",
                    len(groups),
                    table_name,
                )
        conn2.close()
        if remaining:
            logger.error("Verification failed: %d duplicate groups remain", remaining)
            return 1
        logger.info("Verification passed: no remaining natural-key duplicates")

    return 0


if __name__ == "__main__":
    sys.exit(main())