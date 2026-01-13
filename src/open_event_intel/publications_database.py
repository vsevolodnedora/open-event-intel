import csv
import hashlib
import json
import os
import re
import sqlite3
import zlib
from datetime import datetime
from typing import List

from pydantic import ValidationError

from open_event_intel.data_models import Publication
from open_event_intel.logger import get_logger

logger = get_logger(__name__)


class PostsDatabase:
    """Connects to the Posts database."""

    def __init__(self, db_path: str) -> None:
        """Initialize the database connection."""
        self.db_path = db_path

        # Check if database file already exists
        db_existed = os.path.exists(self.db_path)

        # Ensure parent directory exists (if any)
        parent_dir = os.path.dirname(self.db_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        # Connect and enable parsing of timestamps
        self.conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        self.conn.execute("PRAGMA foreign_keys = ON;")

        if db_existed:
            logger.info(f"Connected to existing database: {self.db_path}")
        else:
            logger.info(f"Database did not exist. Created new database at: {self.db_path}")

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def check_create_table(self, table_name: str) -> None:
        """Check if the given table exists in the database."""
        # Validate table name
        if not re.match(r"^[A-Za-z0-9_]+$", table_name):
            raise ValueError(f"Invalid table name: {table_name}")
        sql = f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            ID TEXT PRIMARY KEY,
            published_on TIMESTAMP NOT NULL,
            title TEXT NOT NULL,
            added_on TIMESTAMP NOT NULL,
            url TEXT NOT NULL,
            language TEXT NOT NULL,
            post BLOB NOT NULL
        );
        """
        self.conn.execute(sql)
        self.conn.commit()
        logger.info(f"Ensured table exists: {table_name}")

    def is_table(self, table_name: str) -> bool:
        """Return whether the given table is present in the database."""
        if not re.match(r"^[A-Za-z0-9_]+$", table_name):
            return False
        cursor = self.conn.execute(
            "SELECT count(name) FROM sqlite_master WHERE type='table' AND name=?;", (table_name,)
        )
        exists = cursor.fetchone()[0] > 0
        return exists

    def create_publication_id(self, post_url: str) -> str:
        """Create a new post id for the given URL which is assumed to be unique."""
        return hashlib.sha256(post_url.encode("utf-8")).hexdigest()

    def compress_publication_text(self, article_id: str, text: str) -> bytes:
        """Compress the given article ID and text into bytes."""
        logger.debug(f"Compressing article ID: {article_id}")
        return zlib.compress(text.encode("utf-8"), level=6)

    def decompress_publication_text(self, article_id: str, text: bytes) -> str:
        """Decompress the given article ID and text into bytes."""
        logger.debug(f"Decompressing article ID: {article_id}")
        try:
            return zlib.decompress(text).decode("utf-8")
        except zlib.error:
            # fallback: assume text is plain bytes
            return text.decode("utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"Failed to decompress article ID {article_id}: {e}")
            raise ValueError(
                f"Failed to decompress article (ID={article_id})."
            ) from e

    def is_publication(self, table_name: str, publication_id: str) -> bool:
        """Return whether the given article ID is present in the database in the given table."""
        if not self.is_table(table_name):
            return False
        cursor = self.conn.execute(
            f"SELECT COUNT(ID) FROM \"{table_name}\" WHERE ID = ?;", (publication_id,)
        )
        return cursor.fetchone()[0] > 0

    def add_publication(
        self,
        table_name: str,
        published_on: datetime,
        title: str,
        post_url: str,
        language: str,
        post: str,
        overwrite: bool = False
    ) -> None:
        """Add post to the given table, overwriting existing one if needed, compressing before adding."""
        logger.debug(f"Adding post to table: {table_name}")
        # ensure table exists
        if not self.is_table(table_name):
            raise ValueError(f"Table {table_name} does not exist.")
        # determine post ID
        post_id = self.create_publication_id(post_url)
        exists = self.is_publication(table_name, post_id)
        if exists and not overwrite:
            logger.debug(
                f"Post exists in {table_name} (url={post_url}, id={post_id}), skipping."
            )
            return

        # timestamp for insertion
        added_dt = datetime.now()
        compressed = self.compress_publication_text(post_id, post)
        if exists and overwrite:
            sql = f"""
            UPDATE "{table_name}"
               SET published_on = ?, title = ?, added_on = ?, url = ?, language = ?, post = ?
             WHERE ID = ?;
            """  # noqa: S608
            params = (
                published_on,
                title,
                added_dt,
                post_url,
                language,
                compressed,
                post_id,
            )
        else:
            sql = f"""
            INSERT INTO "{table_name}"
                   (ID, published_on, title, added_on, url, language, post)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """  # noqa: S608
            params = (
                post_id,
                published_on,
                title,
                added_dt,
                post_url,
                language,
                compressed,
            )
        self.conn.execute(sql, params)
        self.conn.commit()
        logger.debug(
            f"Post {'updated' if exists else 'added'} in {table_name}: id={post_id}, title={title}"
        )

    def get_all_publication_dates(self, table_name: str) -> list[datetime]:
        """Return a list of all post dates in the given table."""
        if not self.is_table(table_name):
            raise ValueError(f"Table {table_name} does not exist.")

        # Query to retrieve all the published_on dates
        cursor = self.conn.execute(f'SELECT published_on FROM "{table_name}";')

        # The values are already datetime objects due to detect_types
        dates = [row[0] for row in cursor.fetchall()]

        logger.info(f"Retrieved {len(dates)} post dates from table {table_name}")

        return dates

    def get_publication(self, table_name: str, publication_id: str) -> Publication:
        """Return a single Publication object from the given table and post ID."""
        logger.debug("Retrieving post from %s with id: %s", table_name, publication_id)

        if not self.is_table(table_name):
            raise ValueError(f"Table {table_name} does not exist.")
        if not self.is_publication(table_name, publication_id):
            raise ValueError(f"Post id {publication_id} does not exist in table {table_name}.")

        sql = f'SELECT ID, published_on, title, added_on, url, post FROM "{table_name}" WHERE ID = ?;'
        cursor = self.conn.execute(sql, (publication_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"No data retrieved for post id {publication_id} in table {table_name}.")

        pid, pub_dt, title, add_dt, url, language, blob = row
        text = self.decompress_publication_text(pid, blob)

        try:
            return Publication(
                id=str(pid),
                url=url,
                text=text,
                publisher=table_name,  # adjust if you store publisher elsewhere
                published_on=pub_dt,
                added_on=add_dt,
                language=language,
                title=title,
            )
        except ValidationError as e:
            logger.error("Failed to build Publication for ID %s: %s", pid, e)
            raise ValueError(f"Could not construct Publication for ID {pid}: {e}") from e

    def list_publications(self, table_name: str, sort_date: bool = False) -> List[Publication]:
        """Return all posts from the given table as Publication objects."""
        if not self.is_table(table_name):
            raise ValueError(f"Table {table_name} does not exist.")

        logger.debug("Listing all posts in table: %s", table_name)

        sql = f'SELECT ID, published_on, title, added_on, url, language, post FROM "{table_name}"' # noqa S608
        if sort_date:
            sql += " ORDER BY published_on DESC"
        sql += ";"

        publications: List[Publication] = []

        cursor = self.conn.execute(sql)
        for pid, pub_dt, title, add_dt, url, language, blob in cursor.fetchall():
            text = self.decompress_publication_text(pid, blob)
            try:
                publications.append(
                    Publication(
                        id=str(pid),
                        url=url,
                        text=text,
                        publisher=table_name,  # change if you store publisher elsewhere
                        published_on=pub_dt,
                        added_on=add_dt,
                        language=language,
                        title=title,
                    )
                )
            except ValidationError as e:
                logger.error("Failed to build Publication for ID %s: %s", pid, e)
                # Optionally skip or re-raise; here we skip the bad row.
                continue

        return publications

    def delete_publication(self, table_name: str, publication_id: str) -> None:
        """
        Delete a publication from the given table.

        Args:
            table_name: The name of the table (publisher name)
            publication_id: The unique identifier of the post to delete

        Raises:
            ValueError: If the table doesn't exist or the post_id is not found
        """
        logger.debug(f"Attempting to delete post from {table_name} with id: {publication_id}")

        # Validate table exists
        if not self.is_table(table_name):
            raise ValueError(f"Table {table_name} does not exist.")

        # Check if post exists
        if not self.is_publication(table_name, publication_id):
            raise ValueError(f"Post id {publication_id} does not exist in table {table_name}.")

        # Delete the post
        sql = f'DELETE FROM "{table_name}" WHERE ID = ?;'
        self.conn.execute(sql, (publication_id,))
        self.conn.commit()

        logger.info(f"Successfully deleted post from {table_name}: id={publication_id}")

    def dump_publications_as_markdown(self, table_name: str, out_dir: str) -> None:
        """Save each Publication as a markdown file in out_dir."""
        if not self.is_table(table_name):
            raise ValueError(f"Table {table_name} does not exist.")

        logger.debug("Dumping publications from %s to markdown in directory: %s", table_name, out_dir)
        os.makedirs(out_dir, exist_ok=True)

        # Now returns List[Publication]
        publications: List[Publication] = self.list_publications(table_name)

        for article in publications:
            # article is a Publication instance
            pub_dt: datetime = article.published_on

            # sanitize title for filename
            title = article.title or "untitled"
            safe_title = re.sub(r"[^A-Za-z0-9_-]", "_", title).strip("_")
            if not safe_title:
                safe_title = "untitled"

            date_str = pub_dt.strftime("%Y-%m-%d_%H-%M")
            # prefer article.publisher if present; fall back to table_name
            publisher = getattr(article, "publisher", table_name) or table_name
            safe_publisher = re.sub(r"[^A-Za-z0-9_-]", "_", str(publisher)).strip("_") or "publisher"

            fname = f"{date_str}__{safe_publisher}__{safe_title}.md"
            path = os.path.join(out_dir, fname)

            # write markdown (original code mentions front matter but only wrote content; keep that behavior)
            content = article.text
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

    def export_all_publications_metadata(self, out_dir: str, format: str = "json", filename: str = "all_publications") -> None:
        """
        Export metadata from all publications across all publishers to a single CSV or JSON file.

        For each publication, exports: ID, url, length (of text), publisher,
        published_on, added_on, and title.

        Args:
            out_dir: Output directory path
            format: Export format - either 'csv' or 'json' (default: 'json')
            filename: Base filename without extension (default: 'all_publications')

        Raises:
            ValueError: If format is invalid
        """
        if format not in ["csv", "json"]:
            raise ValueError(f"Format must be 'csv' or 'json', got: {format}")

        logger.debug("Exporting all publications to %s in directory: %s", format, out_dir)
        os.makedirs(out_dir, exist_ok=True)

        # Get all table names from the database
        cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        table_names = [row[0] for row in cursor.fetchall()]

        if not table_names:
            logger.warning("No tables found in database")
            return

        logger.info(f"Found {len(table_names)} publisher tables: {', '.join(table_names)}")

        # Collect metadata from all publishers
        all_metadata = []
        total_count = 0

        for table_name in table_names:
            try:
                publications: List[Publication] = self.list_publications(table_name, sort_date=True)

                for pub in publications:
                    metadata = {
                        "id": pub.id,
                        "url": pub.url,
                        "length": len(pub.text),
                        "publisher": pub.publisher,
                        "published_on": pub.published_on.isoformat(),
                        "added_on": pub.added_on.isoformat(),
                        "title": pub.title or "",
                        "language": pub.language or "",
                    }
                    all_metadata.append(metadata)

                logger.debug(f"Collected {len(publications)} publications from {table_name}")
                total_count += len(publications)

            except Exception as e:
                logger.error(f"Error processing table {table_name}: {e}")
                continue

        if not all_metadata:
            logger.warning(f"No publications found (for all tables: {table_names}) for metadata (.json) saving. Nothing will be saved.")
            return

        # Sort all metadata by published_on date (newest first)
        all_metadata.sort(key=lambda x: x["published_on"], reverse=True)

        # Export based on format
        if format == "csv":
            output_path = os.path.join(out_dir, f"{filename}.csv")
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                fieldnames = ["id", "url", "length", "publisher", "published_on", "added_on", "title", "language"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_metadata)
            logger.info("Exported %d publications from %d publishers to CSV: %s", total_count, len(table_names), output_path)

        elif format == "json":
            output_path = os.path.join(out_dir, f"{filename}.json")
            export_data = {"total_publications": total_count, "total_publishers": len(table_names), "publishers": table_names, "publications": all_metadata}
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info("Exported %d publications from %d publishers to JSON: %s", total_count, len(table_names), output_path)