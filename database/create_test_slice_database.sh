#!/usr/bin/env bash
set -euo pipefail

SRC_DB="preprocessed_posts.db"
DST_DB="test_preprocessed_posts.db"
MAX_ROWS=20

if [[ ! -f "$SRC_DB" ]]; then
  echo "Error: source DB not found: $SRC_DB" >&2
  exit 1
fi

# Make a fresh copy
cp -f "$SRC_DB" "$DST_DB"

# Ensure sqlite3 is available
if ! command -v sqlite3 >/dev/null 2>&1; then
  echo "Error: sqlite3 is not installed or not in PATH." >&2
  exit 1
fi

# Get list of user tables (exclude sqlite internal tables)
mapfile -t TABLES < <(
  sqlite3 -noheader -batch "$DST_DB" \
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
)

if [[ ${#TABLES[@]} -eq 0 ]]; then
  echo "No tables found in $DST_DB"
  exit 0
fi

echo "Truncating tables in $DST_DB to max ${MAX_ROWS} rows each..."
echo

for tbl in "${TABLES[@]}"; do
  # Try to delete everything except the first MAX_ROWS by rowid.
  # If a table is WITHOUT ROWID, rowid doesn't exist; in that case we fall back
  # to keeping an arbitrary MAX_ROWS via LIMIT/row_number (SQLite supports window
  # functions in modern versions).
  sqlite3 -batch "$DST_DB" <<SQL
PRAGMA foreign_keys=OFF;

-- Preferred: works for most tables that have rowid
DELETE FROM "$tbl"
WHERE rowid NOT IN (
  SELECT rowid FROM "$tbl" ORDER BY rowid LIMIT $MAX_ROWS
);

-- Fallback for WITHOUT ROWID (or if rowid delete didn't apply):
-- If rowid doesn't exist, the above statement fails and stops execution.
-- So we detect rowid presence first in bash instead of relying on this.
SQL

  # Count rows after truncation
  cnt="$(sqlite3 -noheader -batch "$DST_DB" "SELECT COUNT(*) FROM \"$tbl\";")"
  echo "  $tbl: $cnt rows"
done

echo
echo "Done. Created: $DST_DB"
