"""Database for output of the temporal knowledge graph - CORRECTED VERSION with Confidence Fields."""
import json
import os
import sqlite3
import struct
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, Iterable, Iterator, List, Optional, Tuple

from src.data_models import Publication
from src.logger import get_logger
from src.tkg.data_models import (
    Entity,
    Predicate,
    RawStatement,
    StatementType,
    TemporalConfidence,
    TemporalEvent,
    TemporalType,
    Triplet,
    parse_date_str,
)

logger = get_logger(__name__)

# ==================== Database Exceptions ====================

class TKGDatabaseError(Exception):
    """Base exception for all database errors."""
    pass


class ValidationError(TKGDatabaseError):
    """Raised when data validation fails."""
    pass


class IntegrityError(TKGDatabaseError):
    """Raised when referential integrity is violated."""
    pass


class NotFoundError(TKGDatabaseError):
    """Raised when a required entity is not found."""
    pass


# ==================== Database Version ====================

DB_VERSION = 3  # Incremented due to addition of confidence fields


# ==================== Production-Ready TKGDatabase ====================

class TKGDatabase:
    """
    Production-ready database interface for temporal knowledge graphs.

    Key Design Decision: Since there's always exactly one TemporalEvent per Statement,
    we use the same ID for both (event.id == statement.id). This simplifies the schema
    and eliminates redundancy.

    Features:
    - Proper transaction management
    - Efficient bulk operations
    - Pagination support
    - Comprehensive validation
    - Connection lifecycle management
    - Database versioning
    - Support for temporal confidence tracking
    """

    # Embedding serialization constants
    EMB_MAGIC = b"EMBD"
    EMB_VERSION = 1
    EMB_DTYPE_FLOAT32 = 1
    EMB_DTYPE_FLOAT64 = 2
    EMB_DTYPE_SIZES = {1: 4, 2: 8}
    EMB_DEFAULT_DTYPE = 1
    DEFAULT_EMBEDDING_DIM = 256

    def __init__(
            self,
            db_path: str,
            memory: bool = False,
            refresh: bool = False,
            enable_wal: bool = True,
    ) -> None:
        """
        Initialize database with proper configuration.

        Args:
            db_path: Path to SQLite database file
            memory: Use in-memory database (for testing)
            refresh: Drop and recreate all tables
            enable_wal: Enable Write-Ahead Logging for better concurrency
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._in_transaction = False

        # Connect and configure
        self._connect(memory=memory, refresh=refresh, enable_wal=enable_wal)

    def _connect(self, memory: bool, refresh: bool, enable_wal: bool) -> None:
        """Establish database connection with proper configuration."""
        if not memory and refresh and os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except PermissionError as e:
                raise TKGDatabaseError(
                    "Cannot delete database file. Ensure all connections are closed."
                ) from e

        # Create connection
        self._conn = (
            sqlite3.connect(":memory:", check_same_thread=False)
            if memory
            else sqlite3.connect(self.db_path, check_same_thread=False)
        )

        # Configure connection
        self._conn.execute("PRAGMA foreign_keys = ON;")
        if enable_wal and not memory:
            self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.execute("PRAGMA synchronous = NORMAL;")

        # Enable row factory for easier data access
        self._conn.row_factory = sqlite3.Row

        # Initialize schema
        current_version = self._get_db_version()
        if current_version == 0 or refresh:
            if refresh:
                self._drop_all_tables()
            self._create_all_tables()
            self._set_db_version(DB_VERSION)
        elif current_version != DB_VERSION:
            raise TKGDatabaseError(
                f"Database version mismatch: found {current_version}, expected {DB_VERSION}. "
                "Migration required."
            )

    def _get_db_version(self) -> int:
        """Get current database schema version."""
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT version FROM schema_version LIMIT 1;")
            row = cursor.fetchone()
            return row[0] if row else 0
        except sqlite3.OperationalError:
            return 0

    def _set_db_version(self, version: int) -> None:
        """Set database schema version."""
        self._conn.execute("DROP TABLE IF EXISTS schema_version;")
        self._conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL);")
        self._conn.execute("INSERT INTO schema_version (version) VALUES (?);", (version,))
        self._conn.commit()

    # ==================== Transaction Management ====================

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """
        Context manager for database transactions.

        Usage:
            with db.transaction():
                db.insert_event(...)
                db.insert_triplet(...)
                # Commits on success, rolls back on exception
        """
        if self._in_transaction:
            # Nested transaction - just pass through
            yield
            return

        self._in_transaction = True
        try:
            self._conn.execute("BEGIN")
            yield
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Transaction rolled back: {e}")
            raise
        finally:
            self._in_transaction = False

    # ==================== Schema Creation ====================

    def _drop_all_tables(self) -> None:
        """Drop all tables in the database."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table};")

        self._conn.commit()
        logger.info(f"Dropped {len(tables)} tables.")

    def _create_all_tables(self) -> None:
        """Create all tables with proper foreign keys and constraints."""
        cursor = self._conn.cursor()

        # Publications table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS publications (
                id           TEXT PRIMARY KEY,
                url          TEXT NOT NULL UNIQUE,
                text         TEXT NOT NULL,
                publisher    TEXT NOT NULL,
                published_on TEXT NOT NULL,
                added_on     TEXT NOT NULL,
                title        TEXT,
                CHECK (length(id) > 0)
            );
        """)

        # Statements table - base extracted statements with temporal_confidence
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS statements (
                id                   BLOB PRIMARY KEY,
                publication_id       TEXT NOT NULL,
                statement            TEXT NOT NULL,
                statement_type       TEXT NOT NULL,
                temporal_type        TEXT NOT NULL,
                temporal_confidence  TEXT NOT NULL DEFAULT 'MEDIUM',
                created_at           TEXT NOT NULL,
                FOREIGN KEY (publication_id) REFERENCES publications(id) ON DELETE CASCADE,
                CHECK (statement_type IN ('FACT', 'OPINION', 'PREDICTION')),
                CHECK (temporal_type IN ('ATEMPORAL', 'EVENT', 'STATE', 'FORECAST')),
                CHECK (temporal_confidence IN ('HIGH', 'MEDIUM', 'LOW'))
            );
        """)

        # Events tables (raw and cleaned) with confidence fields
        for table in ["raw_events", "events"]:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id                     BLOB PRIMARY KEY,
                    publication_id         TEXT NOT NULL,
                    statement              TEXT NOT NULL,
                    statement_type         TEXT NOT NULL,
                    temporal_type          TEXT NOT NULL,
                    temporal_confidence    TEXT NOT NULL DEFAULT 'MEDIUM',
                    created_at             TEXT NOT NULL,
                    valid_at               TEXT,
                    valid_at_confidence    TEXT NOT NULL DEFAULT 'LOW',
                    expired_at             TEXT,
                    invalid_at             TEXT,
                    invalid_at_confidence  TEXT NOT NULL DEFAULT 'LOW',
                    invalidated_by         BLOB,
                    embedding              BLOB,
                    FOREIGN KEY (id) REFERENCES statements(id) ON DELETE CASCADE,
                    FOREIGN KEY (publication_id) REFERENCES publications(id) ON DELETE CASCADE,
                    FOREIGN KEY (invalidated_by) REFERENCES {table}(id) ON DELETE SET NULL,
                    CHECK (statement_type IN ('FACT', 'OPINION', 'PREDICTION')),
                    CHECK (temporal_type IN ('ATEMPORAL', 'EVENT', 'STATE', 'FORECAST')),
                    CHECK (temporal_confidence IN ('HIGH', 'MEDIUM', 'LOW')),
                    CHECK (valid_at_confidence IN ('HIGH', 'MEDIUM', 'LOW')),
                    CHECK (invalid_at_confidence IN ('HIGH', 'MEDIUM', 'LOW')),
                    CHECK (valid_at IS NULL OR invalid_at IS NULL OR valid_at <= invalid_at)
                );
            """)

        # Triplets tables (raw and cleaned)
        for table in ["raw_triplets", "triplets"]:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id           BLOB PRIMARY KEY,
                    event_id     BLOB NOT NULL,
                    subject_name TEXT NOT NULL,
                    subject_id   BLOB NOT NULL,
                    predicate    TEXT NOT NULL,
                    object_name  TEXT NOT NULL,
                    object_id    BLOB NOT NULL,
                    value        TEXT,
                    FOREIGN KEY (event_id) REFERENCES statements(id) ON DELETE CASCADE,
                    CHECK (length(subject_name) > 0),
                    CHECK (length(object_name) > 0)
                );
            """)

        # Entities tables (raw and cleaned)
        for table in ["raw_entities", "entities"]:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id           BLOB PRIMARY KEY,
                    event_id     BLOB NOT NULL,
                    name         TEXT NOT NULL,
                    type         TEXT NOT NULL DEFAULT '',
                    description  TEXT NOT NULL DEFAULT '',
                    resolved_id  BLOB,
                    FOREIGN KEY (event_id) REFERENCES statements(id) ON DELETE CASCADE,
                    FOREIGN KEY (resolved_id) REFERENCES {table}(id) ON DELETE SET NULL,
                    CHECK (id != resolved_id),
                    CHECK (length(name) > 0)
                );
            """)

        self._create_indexes()
        self._conn.commit()
        logger.info("Created all tables with proper constraints and confidence fields.")

    def _create_indexes(self) -> None:
        """Create optimized indexes for common queries."""
        cursor = self._conn.cursor()

        indexes = [
            # Publications
            "CREATE INDEX IF NOT EXISTS idx_publications_published_on ON publications(published_on);",
            "CREATE INDEX IF NOT EXISTS idx_publications_publisher ON publications(publisher);",

            # Statements
            "CREATE INDEX IF NOT EXISTS idx_statements_publication_id ON statements(publication_id);",
            "CREATE INDEX IF NOT EXISTS idx_statements_created_at ON statements(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_statements_types ON statements(statement_type, temporal_type);",
            "CREATE INDEX IF NOT EXISTS idx_statements_temporal_confidence ON statements(temporal_confidence);",

            # Events (raw and cleaned)
            "CREATE INDEX IF NOT EXISTS idx_raw_events_publication_id ON raw_events(publication_id);",
            "CREATE INDEX IF NOT EXISTS idx_events_publication_id ON events(publication_id);",
            "CREATE INDEX IF NOT EXISTS idx_raw_events_invalidated_by ON raw_events(invalidated_by);",
            "CREATE INDEX IF NOT EXISTS idx_events_invalidated_by ON events(invalidated_by);",
            "CREATE INDEX IF NOT EXISTS idx_raw_events_temporal ON raw_events(valid_at, invalid_at);",
            "CREATE INDEX IF NOT EXISTS idx_events_temporal ON events(valid_at, invalid_at);",
            "CREATE INDEX IF NOT EXISTS idx_raw_events_confidence ON raw_events(temporal_confidence);",
            "CREATE INDEX IF NOT EXISTS idx_events_confidence ON events(temporal_confidence);",
            "CREATE INDEX IF NOT EXISTS idx_raw_events_valid_at_confidence ON raw_events(valid_at_confidence);",
            "CREATE INDEX IF NOT EXISTS idx_events_valid_at_confidence ON events(valid_at_confidence);",

            # Triplets
            "CREATE INDEX IF NOT EXISTS idx_raw_triplets_event_id ON raw_triplets(event_id);",
            "CREATE INDEX IF NOT EXISTS idx_triplets_event_id ON triplets(event_id);",
            "CREATE INDEX IF NOT EXISTS idx_raw_triplets_predicate ON raw_triplets(predicate);",
            "CREATE INDEX IF NOT EXISTS idx_triplets_predicate ON triplets(predicate);",
            "CREATE INDEX IF NOT EXISTS idx_raw_triplets_subject_id ON raw_triplets(subject_id);",
            "CREATE INDEX IF NOT EXISTS idx_triplets_subject_id ON triplets(subject_id);",
            "CREATE INDEX IF NOT EXISTS idx_raw_triplets_object_id ON raw_triplets(object_id);",
            "CREATE INDEX IF NOT EXISTS idx_triplets_object_id ON triplets(object_id);",

            # Entities
            "CREATE INDEX IF NOT EXISTS idx_raw_entities_event_id ON raw_entities(event_id);",
            "CREATE INDEX IF NOT EXISTS idx_entities_event_id ON entities(event_id);",
            "CREATE INDEX IF NOT EXISTS idx_raw_entities_resolved_id ON raw_entities(resolved_id);",
            "CREATE INDEX IF NOT EXISTS idx_entities_resolved_id ON entities(resolved_id);",
            "CREATE INDEX IF NOT EXISTS idx_raw_entities_name ON raw_entities(name);",
            "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);",
            "CREATE INDEX IF NOT EXISTS idx_raw_entities_type ON raw_entities(type);",
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);",
        ]

        for index_sql in indexes:
            cursor.execute(index_sql)

    def clear_table(self, table_name: str, raw: bool = False) -> None:
        """Remove all entries from the table leaving indexing intact."""
        if table_name not in ["events", "triplets", "entities"]:
            raise ValueError("Table name must be 'events', 'triplets', or 'entities'. Given: {}".format(table_name))

        table = f"raw_{table_name}" if raw else table_name

        cursor = self._conn.cursor()
        cursor.execute(f"DELETE FROM {table};")
        self._conn.commit()

        # Get the count of remaining rows to verify deletion
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]

        logger.info(f"Cleared table '{table}'. Remaining rows: {count}")

    # ==================== Type Conversion Utilities ====================

    @staticmethod
    def _uuid_to_blob(u: Optional[uuid.UUID]) -> Optional[bytes]:
        """Convert UUID to bytes for storage."""
        return u.bytes if isinstance(u, uuid.UUID) else None

    @staticmethod
    def _blob_to_uuid(blob: Any) -> Optional[uuid.UUID]:
        """Convert bytes blob to UUID."""
        if blob is None:
            return None
        if isinstance(blob, uuid.UUID):
            return blob
        if isinstance(blob, (bytes, bytearray, memoryview)):
            data = bytes(blob)
            if len(data) == 16:
                return uuid.UUID(bytes=data)
        if isinstance(blob, str):
            try:
                return uuid.UUID(blob)
            except ValueError:
                pass
        raise ValidationError(f"Cannot convert {type(blob)} to UUID")

    @staticmethod
    def _enum_to_text(value: Any) -> Optional[str]:
        """Convert enum to text for storage."""
        if value is None:
            return None
        return value.value if hasattr(value, "value") else str(value)

    @staticmethod
    def _text_to_enum(enum_cls, value: Any):
        """Convert text to enum with validation."""
        if value is None:
            return None
        if isinstance(value, enum_cls):
            return value
        try:
            return enum_cls(value)
        except (ValueError, KeyError) as e:
            raise ValidationError(
                f"Invalid {enum_cls.__name__}: {value}. "
                f"Must be one of {[e.value for e in enum_cls]}"
            ) from e

    @staticmethod
    def _datetime_to_text(dt: Optional[datetime]) -> Optional[str]:
        """Convert datetime to ISO format string."""
        return dt.isoformat() if dt else None

    @staticmethod
    def _text_to_datetime(text: Optional[str]) -> Optional[datetime]:
        """Convert ISO format string to datetime."""
        return parse_date_str(text) if text else None

    # ==================== Validation ====================

    def _validate_temporal_range(
            self, valid_at: Optional[datetime], invalid_at: Optional[datetime]
    ) -> None:
        """Validate temporal range is coherent."""
        if valid_at and invalid_at and valid_at > invalid_at:
            raise ValidationError(
                f"valid_at ({valid_at}) must be before invalid_at ({invalid_at})"
            )

    def _validate_embedding(self, embedding: Optional[List[float]]) -> None:
        """Validate embedding dimensions and values."""
        if embedding is None:
            return

        if len(embedding) != self.DEFAULT_EMBEDDING_DIM:
            raise ValidationError(
                f"Embedding must have {self.DEFAULT_EMBEDDING_DIM} dimensions, "
                f"got {len(embedding)}"
            )

        if not all(isinstance(x, (int, float)) for x in embedding):
            raise ValidationError("All embedding values must be numeric")

    def _validate_publication_exists(self, publication_id: str) -> None:
        """Ensure publication exists in database."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT 1 FROM publications WHERE id = ? LIMIT 1;", (publication_id,))
        if not cursor.fetchone():
            raise NotFoundError(f"Publication {publication_id} not found")

    def _validate_statement_exists(self, statement_id: uuid.UUID) -> None:
        """Ensure statement exists in database."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT 1 FROM statements WHERE id = ? LIMIT 1;",
            (self._uuid_to_blob(statement_id),)
        )
        if not cursor.fetchone():
            raise NotFoundError(f"Statement {statement_id} not found")

    # ==================== Embedding Serialization ====================

    def _pack_floats_be(self, values: Iterable[float], dtype_code: int) -> bytes:
        """Pack floats to big-endian bytes."""
        vals = list(values)
        dtype_to_char = {self.EMB_DTYPE_FLOAT32: "f", self.EMB_DTYPE_FLOAT64: "d"}

        if dtype_code not in dtype_to_char:
            raise ValidationError(f"Unsupported dtype code: {dtype_code}")

        fmt = ">" + (dtype_to_char[dtype_code] * len(vals))
        try:
            return struct.pack(fmt, *vals)
        except struct.error as e:
            raise ValidationError(
                "Cannot pack embedding values into target dtype"
            ) from e

    def _unpack_floats_be(self, buf: bytes, dtype_code: int) -> List[float]:
        """Unpack big-endian bytes to floats."""
        if dtype_code not in self.EMB_DTYPE_SIZES:
            raise ValidationError(f"Unsupported dtype code: {dtype_code}")

        size = self.EMB_DTYPE_SIZES[dtype_code]
        if len(buf) % size != 0:
            raise ValidationError("Buffer size is not a multiple of dtype size")

        count = len(buf) // size
        dtype_to_char = {self.EMB_DTYPE_FLOAT32: "f", self.EMB_DTYPE_FLOAT64: "d"}
        fmt = ">" + (dtype_to_char[dtype_code] * count)

        try:
            return list(struct.unpack(fmt, buf))
        except struct.error as e:
            raise ValidationError("Cannot unpack embedding buffer") from e

    def _serialize_embedding(
            self, vec: Optional[List[float]], dtype_code: Optional[int] = None
    ) -> Optional[bytes]:
        """Serialize embedding to self-describing BLOB with validation."""
        if vec is None:
            return None

        self._validate_embedding(vec)

        if dtype_code is None:
            dtype_code = self.EMB_DEFAULT_DTYPE

        dim = len(vec)
        header = self.EMB_MAGIC + struct.pack(">BBI", self.EMB_VERSION, dtype_code, dim)

        if dim == 0:
            return header

        payload = self._pack_floats_be(vec, dtype_code)
        return header + payload

    def _deserialize_embedding(self, blob: Optional[bytes]) -> Optional[List[float]]:
        """Deserialize embedding BLOB to list of floats."""
        if blob is None:
            return None

        if isinstance(blob, memoryview):
            blob = blob.tobytes()

        # Check for headered format
        if len(blob) >= 10 and blob[:4] == self.EMB_MAGIC:
            version, dtype_code, dim = struct.unpack(">BBI", blob[4:10])

            if version != self.EMB_VERSION:
                raise ValidationError(
                    f"Unsupported embedding version: {version} (expected {self.EMB_VERSION})"
                )

            if dtype_code not in self.EMB_DTYPE_SIZES:
                raise ValidationError(f"Unsupported dtype code: {dtype_code}")

            size = self.EMB_DTYPE_SIZES[dtype_code]
            payload = blob[10:]

            if len(payload) != dim * size:
                raise ValidationError(
                    f"Embedding payload size mismatch: expected {dim * size}, got {len(payload)}"
                )

            return self._unpack_floats_be(payload, dtype_code)

        # Legacy fallback: raw float32 (try big-endian only, no guessing)
        if len(blob) % 4 == 0:
            count = len(blob) // 4
            try:
                result = list(struct.unpack(">" + ("f" * count), blob))
                logger.warning("Loaded legacy embedding format (no header)")
                return result
            except struct.error:
                pass

        raise ValidationError("Unrecognized embedding BLOB format")

    # ==================== Publication Operations ====================

    def insert_publication(
            self, publication: Publication, overwrite: bool = False
    ) -> bool:
        """
        Insert or update a publication.

        Args:
            publication: Publication to insert
            overwrite: If True, update existing publication

        Returns:
            True if publication already existed, False otherwise

        Raises:
            ValidationError: If publication data is invalid
        """
        # Validate
        if not publication.id:
            raise ValidationError("Publication must have an ID")
        if not publication.url:
            raise ValidationError("Publication must have a URL")

        cursor = self._conn.cursor()
        cursor.execute("SELECT 1 FROM publications WHERE id = ?;", (publication.id,))
        existed = cursor.fetchone() is not None

        if existed and not overwrite:
            logger.warning(f"Publication {publication.id} already exists. Skipping.")
            return True

        with self.transaction():
            if existed:
                cursor.execute(
                    """
                    UPDATE publications
                    SET url = ?, text = ?, publisher = ?, published_on = ?,
                        added_on = ?, title = ?
                    WHERE id = ?;
                    """,
                    (
                        publication.url,
                        publication.text,
                        publication.publisher,
                        self._datetime_to_text(publication.published_on),
                        self._datetime_to_text(publication.added_on),
                        publication.title,
                        publication.id,
                    ),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO publications (id, url, text, publisher, published_on, added_on, title)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        publication.id,
                        publication.url,
                        publication.text,
                        publication.publisher,
                        self._datetime_to_text(publication.published_on),
                        self._datetime_to_text(publication.added_on),
                        publication.title,
                    ),
                )

        logger.info(f"Publication {publication.id} {'updated' if existed else 'inserted'}.")
        return existed

    def has_publication(self, publication_id: str) -> bool:
        """Check if publication exists by ID."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT 1 FROM publications WHERE id = ? LIMIT 1;", (publication_id,))
        return cursor.fetchone() is not None

    def get_publication_by_id(self, publication_id: str) -> Publication:
        """
        Get publication by ID.

        Raises:
            NotFoundError: If publication doesn't exist
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, url, text, publisher, published_on, added_on, title
            FROM publications
            WHERE id = ?;
            """,
            (publication_id,),
        )
        row = cursor.fetchone()

        if not row:
            raise NotFoundError(f"Publication {publication_id} not found")

        return Publication(
            id=row["id"],
            url=row["url"],
            text=row["text"],
            publisher=row["publisher"],
            published_on=self._text_to_datetime(row["published_on"]),
            added_on=self._text_to_datetime(row["added_on"]),
            title=row["title"],
        )

    def iter_publications(
            self, batch_size: int = 100, order_by: str = "published_on ASC"
    ) -> Iterator[Publication]:
        """
        Iterate over all publications in batches (memory efficient).

        Args:
            batch_size: Number of publications per batch
            order_by: SQL ORDER BY clause

        Yields:
            Publication objects
        """
        cursor = self._conn.cursor()
        cursor.execute(
            f"""
            SELECT id, url, text, publisher, published_on, added_on, title
            FROM publications
            ORDER BY {order_by};
            """  # noqa: S608
        )

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            for row in rows:
                yield Publication(
                    id=row["id"],
                    url=row["url"],
                    text=row["text"],
                    publisher=row["publisher"],
                    published_on=self._text_to_datetime(row["published_on"]),
                    added_on=self._text_to_datetime(row["added_on"]),
                    title=row["title"],
                )

    def get_all_publications(self) -> List[Publication]:
        """Get all publications (use iter_publications for large datasets)."""
        return list(self.iter_publications())

    # ==================== Statement Operations ====================

    def insert_statement(
            self, publication_id: str, statement: RawStatement
    ) -> uuid.UUID:
        """
        Insert a statement and return its ID.

        Args:
            publication_id: ID of the parent publication
            statement: Statement to insert

        Returns:
            UUID of inserted statement (which will also be the event ID)

        Raises:
            ValidationError: If data is invalid
            NotFoundError: If publication doesn't exist
        """
        # Validate
        self._validate_publication_exists(publication_id)

        if not statement.statement:
            raise ValidationError("Statement text cannot be empty")

        stmt_id = statement.id if hasattr(statement, "id") and statement.id else uuid.uuid4()
        created_at = datetime.utcnow()

        # Get temporal_confidence with default if not present
        temporal_confidence = getattr(statement, 'temporal_confidence', TemporalConfidence.MEDIUM)

        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO statements
                (id, publication_id, statement, statement_type, temporal_type, temporal_confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    self._uuid_to_blob(stmt_id),
                    publication_id,
                    statement.statement,
                    self._enum_to_text(statement.statement_type),
                    self._enum_to_text(statement.temporal_type),
                    self._enum_to_text(temporal_confidence),
                    self._datetime_to_text(created_at),
                ),
            )

        logger.debug(f"Inserted statement {stmt_id}")
        return stmt_id

    def get_statement_by_id(self, statement_id: uuid.UUID) -> Tuple[str, RawStatement]:
        """
        Get statement by ID.

        Returns:
            Tuple of (publication_id, statement)

        Raises:
            NotFoundError: If statement doesn't exist
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, publication_id, statement, statement_type, temporal_type, temporal_confidence
            FROM statements
            WHERE id = ?;
            """,
            (self._uuid_to_blob(statement_id),),
        )
        row = cursor.fetchone()

        if not row:
            raise NotFoundError(f"Statement {statement_id} not found")

        pub_id = row["publication_id"]
        stmt_id = self._blob_to_uuid(row["id"])

        statement = RawStatement(
            id=stmt_id,
            statement=row["statement"],
            statement_type=self._text_to_enum(StatementType, row["statement_type"]),
            temporal_type=self._text_to_enum(TemporalType, row["temporal_type"]),
            temporal_confidence=self._text_to_enum(TemporalConfidence, row["temporal_confidence"]),
            publication_id=pub_id,
        )

        return pub_id, statement

    def iter_statements_for_publication(
            self, publication_id: str, batch_size: int = 100
    ) -> Iterator[Tuple[uuid.UUID, RawStatement]]:
        """
        Iterate over statements for a publication (memory efficient).

        Yields:
            Tuples of (statement_id, statement)
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, statement, statement_type, temporal_type, temporal_confidence
            FROM statements
            WHERE publication_id = ?
            ORDER BY created_at ASC;
            """,
            (publication_id,),
        )

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            for row in rows:
                event_id = self._blob_to_uuid(row["id"])
                statement = RawStatement(
                    id=event_id,
                    statement=row["statement"],
                    statement_type=self._text_to_enum(StatementType, row["statement_type"]),
                    temporal_type=self._text_to_enum(TemporalType, row["temporal_type"]),
                    temporal_confidence=self._text_to_enum(TemporalConfidence, row["temporal_confidence"]),
                    publication_id=publication_id,
                )
                yield event_id, statement

    def get_statements_for_publication(
            self, publication_id: str
    ) -> List[Tuple[uuid.UUID, RawStatement]]:
        """Get all statements for a publication."""
        return list(self.iter_statements_for_publication(publication_id))

    # ==================== Event Operations ====================

    def insert_event(
            self,
            event: TemporalEvent,
            raw: bool = False,
    ) -> None:
        """
        Insert an event (event.id must equal the statement.id).

        Args:
            event: TemporalEvent to insert (event.id is used as both event and statement ID)
            raw: Insert into raw_events table

        Raises:
            ValidationError: If data is invalid
            NotFoundError: If referenced entities don't exist
        """
        # Validate that the statement exists with the same ID
        self._validate_statement_exists(event.id)
        self._validate_temporal_range(event.valid_at, event.invalid_at)
        self._validate_embedding(event.embedding)

        if event.invalidated_by is not None:
            # Check that invalidating event exists
            table = "raw_events" if raw else "events"
            cursor = self._conn.cursor()
            cursor.execute(
                f"SELECT 1 FROM {table} WHERE id = ? LIMIT 1;",  # noqa: S608
                (self._uuid_to_blob(event.invalidated_by),),
            )
            if not cursor.fetchone():
                raise NotFoundError(
                    f"Invalidating event {event.invalidated_by} not found"
                )

        table = "raw_events" if raw else "events"

        # Get confidence fields with defaults
        temporal_confidence = getattr(event, 'temporal_confidence', TemporalConfidence.MEDIUM)
        valid_at_confidence = getattr(event, 'valid_at_confidence', TemporalConfidence.LOW)
        invalid_at_confidence = getattr(event, 'invalid_at_confidence', TemporalConfidence.LOW)

        with self.transaction():
            # Insert event with id that matches statement_id
            self._conn.execute(
                f"""
                INSERT INTO {table}
                (id, publication_id, statement, statement_type, temporal_type, temporal_confidence,
                 created_at, valid_at, valid_at_confidence, expired_at, invalid_at, 
                 invalid_at_confidence, invalidated_by, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,  # noqa: S608
                (
                    self._uuid_to_blob(event.id),
                    event.publication_id,
                    event.statement,
                    self._enum_to_text(event.statement_type),
                    self._enum_to_text(event.temporal_type),
                    self._enum_to_text(temporal_confidence),
                    self._datetime_to_text(event.created_at),
                    self._datetime_to_text(event.valid_at),
                    self._enum_to_text(valid_at_confidence),
                    self._datetime_to_text(event.expired_at),
                    self._datetime_to_text(event.invalid_at),
                    self._enum_to_text(invalid_at_confidence),
                    self._uuid_to_blob(event.invalidated_by),
                    self._serialize_embedding(event.embedding),
                ),
            )

        logger.info(f"Inserted event {event.id} into {table}")

    def get_event_by_id(self, event_id: uuid.UUID, raw: bool = False) -> TemporalEvent:
        """
        Get event by ID with linked triplets.

        Args:
            event_id: ID of the event (same as statement ID)
            raw: Use raw_events table

        Returns:
            TemporalEvent with associated triplets

        Raises:
            NotFoundError: If event doesn't exist
        """
        table = "raw_events" if raw else "events"
        triplet_table = "raw_triplets" if raw else "triplets"

        cursor = self._conn.cursor()
        cursor.execute(
            f"""
            SELECT id, publication_id, statement, statement_type, temporal_type, temporal_confidence,
                   created_at, valid_at, valid_at_confidence, expired_at, invalid_at, 
                   invalid_at_confidence, invalidated_by, embedding
            FROM {table}
            WHERE id = ?;
            """,  # noqa: S608
            (self._uuid_to_blob(event_id),),
        )
        row = cursor.fetchone()

        if not row:
            raise NotFoundError(f"Event {event_id} not found")

        # Get linked triplet IDs
        cursor.execute(
            f"SELECT id FROM {triplet_table} WHERE event_id = ?;",  # noqa: S608
            (self._uuid_to_blob(event_id),),
        )
        triplet_ids = [self._blob_to_uuid(r["id"]) for r in cursor.fetchall()]

        return TemporalEvent(
            id=self._blob_to_uuid(row["id"]),
            publication_id=row["publication_id"],
            statement=row["statement"],
            triplets=triplet_ids,
            statement_type=self._text_to_enum(StatementType, row["statement_type"]),
            temporal_type=self._text_to_enum(TemporalType, row["temporal_type"]),
            temporal_confidence=self._text_to_enum(TemporalConfidence, row["temporal_confidence"]),
            created_at=self._text_to_datetime(row["created_at"]),
            valid_at=self._text_to_datetime(row["valid_at"]),
            valid_at_confidence=self._text_to_enum(TemporalConfidence, row["valid_at_confidence"]),
            expired_at=self._text_to_datetime(row["expired_at"]),
            invalid_at=self._text_to_datetime(row["invalid_at"]),
            invalid_at_confidence=self._text_to_enum(TemporalConfidence, row["invalid_at_confidence"]),
            invalidated_by=self._blob_to_uuid(row["invalidated_by"]),
            embedding=self._deserialize_embedding(row["embedding"]),
        )

    def update_events_batch(
            self, events: List[TemporalEvent], raw: bool = False
    ) -> None:
        """
        Batch update multiple events efficiently.

        Only updates temporal fields and invalidation status.
        """
        if not events:
            return

        table = "raw_events" if raw else "events"

        with self.transaction():
            self._conn.executemany(
                f"""
                UPDATE {table}
                SET invalid_at = ?, invalid_at_confidence = ?, expired_at = ?, invalidated_by = ?
                WHERE id = ?;
                """,  # noqa: S608
                [
                    (
                        self._datetime_to_text(event.invalid_at),
                        self._enum_to_text(getattr(event, 'invalid_at_confidence', TemporalConfidence.LOW)),
                        self._datetime_to_text(event.expired_at),
                        self._uuid_to_blob(event.invalidated_by),
                        self._uuid_to_blob(event.id),
                    )
                    for event in events
                ],
            )

        logger.info(f"Batch updated {len(events)} events in {table}")

    def iter_events(
            self, raw: bool = False, batch_size: int = 100
    ) -> Iterator[TemporalEvent]:
        """
        Iterate over all events efficiently (no N+1 queries).

        Uses a single query with JOIN to fetch events and their triplets.
        """
        table = "raw_events" if raw else "events"
        triplet_table = "raw_triplets" if raw else "triplets"

        # Single efficient query with LEFT JOIN
        cursor = self._conn.cursor()
        cursor.execute(
            f"""
            SELECT 
                e.id, e.publication_id, e.statement, e.statement_type,
                e.temporal_type, e.temporal_confidence, e.created_at, 
                e.valid_at, e.valid_at_confidence, e.expired_at, 
                e.invalid_at, e.invalid_at_confidence, e.invalidated_by, e.embedding,
                GROUP_CONCAT(HEX(t.id)) as triplet_ids
            FROM {table} e
            LEFT JOIN {triplet_table} t ON e.id = t.event_id
            GROUP BY e.id
            ORDER BY e.created_at ASC;
            """  # noqa: S608
        )

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            for row in rows:
                # Parse triplet IDs
                triplet_ids = []
                if row["triplet_ids"]:
                    triplet_ids = [
                        self._blob_to_uuid(bytes.fromhex(tid))
                        for tid in row["triplet_ids"].split(",")
                    ]

                yield TemporalEvent(
                    id=self._blob_to_uuid(row["id"]),
                    publication_id=row["publication_id"],
                    statement=row["statement"],
                    triplets=triplet_ids,
                    statement_type=self._text_to_enum(StatementType, row["statement_type"]),
                    temporal_type=self._text_to_enum(TemporalType, row["temporal_type"]),
                    temporal_confidence=self._text_to_enum(TemporalConfidence, row["temporal_confidence"]),
                    created_at=self._text_to_datetime(row["created_at"]),
                    valid_at=self._text_to_datetime(row["valid_at"]),
                    valid_at_confidence=self._text_to_enum(TemporalConfidence, row["valid_at_confidence"]),
                    expired_at=self._text_to_datetime(row["expired_at"]),
                    invalid_at=self._text_to_datetime(row["invalid_at"]),
                    invalid_at_confidence=self._text_to_enum(TemporalConfidence, row["invalid_at_confidence"]),
                    invalidated_by=self._blob_to_uuid(row["invalidated_by"]),
                    embedding=self._deserialize_embedding(row["embedding"]),
                )

    def get_all_events(self, raw: bool = False) -> List[TemporalEvent]:
        """Get all events (use iter_events for large datasets)."""
        return list(self.iter_events(raw=raw))

    def has_events(self, raw: bool = False, statement_type: StatementType=StatementType.FACT) -> bool:
        """Check if there are any events in the database."""
        table = "raw_events" if raw else "events"
        cursor = self._conn.cursor()
        cursor.execute(
            f"SELECT 1 FROM {table} WHERE statement_type = ? LIMIT 1;", # noqa: S608
            (statement_type.value,)
        )
        return cursor.fetchone() is not None

    def get_event_for_statement(
            self, statement_id: uuid.UUID, raw: bool = False
    ) -> TemporalEvent:
        """
        Get the event for a given statement.
        Since event.id == statement.id, this is equivalent to get_event_by_id.

        Args:
            statement_id: ID of the statement (same as event ID)
            raw: Use raw_events table

        Returns:
            TemporalEvent

        Raises:
            NotFoundError: If event doesn't exist
        """
        # Since event.id == statement.id, we can directly use get_event_by_id
        return self.get_event_by_id(statement_id, raw=raw)

    # ==================== Triplet Operations ====================

    def insert_triplet(
            self, triplet: Triplet, event_id: uuid.UUID, raw: bool = False
    ) -> None:
        """
        Insert a triplet linked to an event.

        Args:
            triplet: Triplet to insert
            event_id: ID of the associated event (same as statement ID)
            raw: Insert into raw_triplets table

        Raises:
            ValidationError: If data is invalid
            NotFoundError: If statement/event doesn't exist
        """
        # Validate that the statement/event exists
        self._validate_statement_exists(event_id)

        if not triplet.subject_name:
            raise ValidationError("Triplet subject_name cannot be empty")
        if not triplet.object_name:
            raise ValidationError("Triplet object_name cannot be empty")

        table = "raw_triplets" if raw else "triplets"

        with self.transaction():
            self._conn.execute(
                f"""
                INSERT INTO {table}
                (id, event_id, subject_name, subject_id, predicate,
                 object_name, object_id, value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,  # noqa: S608
                (
                    self._uuid_to_blob(triplet.id),
                    self._uuid_to_blob(event_id),
                    triplet.subject_name,
                    self._uuid_to_blob(triplet.subject_id),
                    self._enum_to_text(triplet.predicate),
                    triplet.object_name,
                    self._uuid_to_blob(triplet.object_id),
                    triplet.value,
                ),
            )

        logger.debug(f"Inserted triplet {triplet.id} into {table}")

    def iter_triplets(self, raw: bool = False, batch_size: int = 500) -> Iterator[Triplet]:
        """Iterate over all triplets efficiently."""
        table = "raw_triplets" if raw else "triplets"

        cursor = self._conn.cursor()
        cursor.execute(
            f"""
            SELECT id, event_id, subject_name, subject_id, predicate,
                   object_name, object_id, value
            FROM {table};
            """  # noqa: S608
        )

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            for row in rows:
                yield Triplet(
                    id=self._blob_to_uuid(row["id"]),
                    event_id=self._blob_to_uuid(row["event_id"]),
                    subject_name=row["subject_name"],
                    subject_id=self._blob_to_uuid(row["subject_id"]),
                    predicate=self._text_to_enum(Predicate, row["predicate"]),
                    object_name=row["object_name"],
                    object_id=self._blob_to_uuid(row["object_id"]),
                    value=row["value"],
                )

    def get_all_triplets(self, raw: bool = False) -> List[Triplet]:
        """Get all triplets (use iter_triplets for large datasets)."""
        return list(self.iter_triplets(raw=raw))

    def get_triplets_for_event(
            self, event_id: uuid.UUID, raw: bool = False
    ) -> List[Triplet]:
        """
        Get all triplets for an event.

        Args:
            event_id: ID of the event (same as statement ID)
            raw: Use raw_triplets table

        Returns:
            List of triplets associated with the event
        """
        table = "raw_triplets" if raw else "triplets"

        cursor = self._conn.cursor()
        cursor.execute(
            f"""
            SELECT id, event_id, subject_name, subject_id, predicate,
                   object_name, object_id, value
            FROM {table}
            WHERE event_id = ?;
            """,  # noqa: S608
            (self._uuid_to_blob(event_id),),
        )

        return [
            Triplet(
                id=self._blob_to_uuid(row["id"]),
                event_id=self._blob_to_uuid(row["event_id"]),
                subject_name=row["subject_name"],
                subject_id=self._blob_to_uuid(row["subject_id"]),
                predicate=self._text_to_enum(Predicate, row["predicate"]),
                object_name=row["object_name"],
                object_id=self._blob_to_uuid(row["object_id"]),
                value=row["value"],
            )
            for row in cursor.fetchall()
        ]

    def get_triplets_for_statement(
            self, statement_id: uuid.UUID, raw: bool = False
    ) -> List[Triplet]:
        """
        Get all triplets for a statement.
        Since event.id == statement.id, this is equivalent to get_triplets_for_event.

        Args:
            statement_id: ID of the statement (same as event ID)
            raw: Use raw_triplets table

        Returns:
            List of triplets
        """
        return self.get_triplets_for_event(statement_id, raw=raw)

    def get_all_unique_predicates(self, raw: bool = False) -> List[Predicate]:
        """Get all unique predicates in the database."""
        table = "raw_triplets" if raw else "triplets"
        cursor = self._conn.cursor()
        cursor.execute(f"SELECT DISTINCT predicate FROM {table};")  # noqa: S608
        return [
            self._text_to_enum(Predicate, row["predicate"])
            for row in cursor.fetchall()
        ]

    def batch_fetch_related_triplet_events(
        self,
        incoming_triplets: List[Triplet],
        predicate_groups: Optional[List[set]] = None,
        raw: bool = False,
    ) -> Tuple[List[Triplet], List[TemporalEvent]]:
        """
        Batch fetch all existing triplets and their events that are related to incoming triplets.

        Related means:
          - Share a subject or object entity
          - Predicate is in the same group (if predicate_groups provided)
          - Associated event is a FACT

        Args:
            incoming_triplets: List of triplets to find related triplets for
            predicate_groups: Optional list of predicate group sets. If None, all predicates match.
                             Example: [{Predicate.IS_A, Predicate.INSTANCE_OF}, {Predicate.HAS, Predicate.OWNS}]
            raw: Use raw tables if True

        Returns:
            Tuple of (triplets, events) where events are associated with the triplets
        """
        if not incoming_triplets:
            logger.info("No incoming triplets found. Returning empty list.")
            return [], []

        # 1. Build sets of all relevant entity IDs and predicate groups
        entity_id_blobs = set()
        predicate_to_group = {}

        if predicate_groups:
            for group in predicate_groups:
                group_list = list(group)
                for pred in group_list:
                    predicate_to_group[pred] = group_list

        relevant_predicates = set()
        for triplet in incoming_triplets:
            entity_id_blobs.add(self._uuid_to_blob(triplet.subject_id))
            entity_id_blobs.add(self._uuid_to_blob(triplet.object_id))

            if predicate_groups:
                group = predicate_to_group.get(triplet.predicate, [triplet.predicate])
            else:
                group = [triplet.predicate]

            relevant_predicates.update([self._enum_to_text(p) for p in group])

        if not relevant_predicates:
            logger.warning("No relevant predicates found for incoming triplets")
            return [], []

        logger.info(f"Searching for triplets with {len(relevant_predicates)} relevant predicates and {len(entity_id_blobs)} entity IDs")

        # 2. Prepare SQL query
        table_prefix = "raw_" if raw else ""
        entity_ids_list = list(entity_id_blobs)
        relevant_predicates_list = list(relevant_predicates)

        entity_placeholders = ",".join(["?"] * len(entity_ids_list))
        predicate_placeholders = ",".join(["?"] * len(relevant_predicates_list))

        query = f"""
            SELECT
                t.id,
                t.event_id,
                t.subject_name,
                t.subject_id,
                t.predicate,
                t.object_name,
                t.object_id,
                t.value,
                e.publication_id,
                e.statement,
                e.statement_type,
                e.temporal_type,
                e.temporal_confidence,
                e.created_at,
                e.valid_at,
                e.valid_at_confidence,
                e.expired_at,
                e.invalid_at,
                e.invalid_at_confidence,
                e.invalidated_by,
                e.embedding
            FROM {table_prefix}triplets t
            JOIN {table_prefix}events e ON t.event_id = e.id
            WHERE
                (t.subject_id IN ({entity_placeholders}) OR t.object_id IN ({entity_placeholders}))
                AND t.predicate IN ({predicate_placeholders})
                AND e.statement_type = ?
        """  # noqa: S608

        params = entity_ids_list + entity_ids_list + relevant_predicates_list + [self._enum_to_text(StatementType.FACT)]

        cursor = self._conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        if len(rows) == 0:
            logger.warning("No triplets found in the database that matched the predicates. Returning empty list.")
            return [], []

        # 3. Process results - build triplets and collect event data
        triplets = []
        events_by_id = {}
        event_triplet_ids = {}  # event_id -> list of triplet_ids

        for row in rows:
            # Construct triplet
            triplet_id = self._blob_to_uuid(row["id"])
            event_id = self._blob_to_uuid(row["event_id"])

            triplet = Triplet(
                id=triplet_id,
                event_id=event_id,
                subject_name=row["subject_name"],
                subject_id=self._blob_to_uuid(row["subject_id"]),
                predicate=self._text_to_enum(Predicate, row["predicate"]),
                object_name=row["object_name"],
                object_id=self._blob_to_uuid(row["object_id"]),
                value=row["value"],
            )
            triplets.append(triplet)

            # Track triplet IDs for each event
            if event_id not in event_triplet_ids:
                event_triplet_ids[event_id] = []
            event_triplet_ids[event_id].append(triplet_id)

            # Construct event if not already processed
            if event_id not in events_by_id:
                event = TemporalEvent(
                    id=event_id,
                    publication_id=row["publication_id"],
                    statement=row["statement"],
                    triplets=[],  # Will be filled below
                    statement_type=self._text_to_enum(StatementType, row["statement_type"]),
                    temporal_type=self._text_to_enum(TemporalType, row["temporal_type"]),
                    temporal_confidence=self._text_to_enum(TemporalConfidence, row["temporal_confidence"]),
                    created_at=self._text_to_datetime(row["created_at"]),
                    valid_at=self._text_to_datetime(row["valid_at"]),
                    valid_at_confidence=self._text_to_enum(TemporalConfidence, row["valid_at_confidence"]),
                    expired_at=self._text_to_datetime(row["expired_at"]),
                    invalid_at=self._text_to_datetime(row["invalid_at"]),
                    invalid_at_confidence=self._text_to_enum(TemporalConfidence, row["invalid_at_confidence"]),
                    invalidated_by=self._blob_to_uuid(row["invalidated_by"]),
                    embedding=self._deserialize_embedding(row["embedding"]),
                )
                events_by_id[event_id] = event

        # 4. Assign triplet IDs to events
        for event_id, event in events_by_id.items():
            event.triplets = event_triplet_ids.get(event_id, [])

        events = list(events_by_id.values())

        logger.info(f"Fetched {len(triplets)} related triplets and {len(events)} events")

        return triplets, events

    # ==================== Entity Operations ====================

    def insert_entity(
            self, entity: Entity, event_id: uuid.UUID, raw: bool = False
    ) -> None:
        """
        Insert an entity linked to an event.

        Args:
            entity: Entity to insert
            event_id: ID of the associated event (same as statement ID)
            raw: Insert into raw_entities table

        Raises:
            ValidationError: If data is invalid
            NotFoundError: If statement/event doesn't exist
        """
        # Validate that the statement/event exists
        self._validate_statement_exists(event_id)

        if not entity.name:
            raise ValidationError("Entity name cannot be empty")

        if entity.resolved_id == entity.id:
            raise ValidationError(f"Entity resolved_id cannot equal id. Given both = {entity.resolved_id}")

        table = "raw_entities" if raw else "entities"

        with self.transaction():
            self._conn.execute(
                f"""
                INSERT INTO {table} (id, event_id, name, type, description, resolved_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    event_id = excluded.event_id,
                    name = excluded.name,
                    type = excluded.type,
                    description = excluded.description,
                    resolved_id = excluded.resolved_id;
                """, # noqa: S608
                (
                    self._uuid_to_blob(entity.id),
                    self._uuid_to_blob(event_id),
                    entity.name,
                    entity.type or "",
                    entity.description or "",
                    self._uuid_to_blob(entity.resolved_id),
                ),
            )

        logger.debug(f"Inserted entity {entity.id} into {table}")

    def iter_entities(
            self, raw: bool = False, canonical_only: bool = False, batch_size: int = 500
    ) -> Iterator[Entity]:
        """
        Iterate over entities efficiently.

        Args:
            raw: Use raw_entities table
            canonical_only: Only return canonical entities (resolved_id IS NULL)
            batch_size: Number of entities per batch
        """
        table = "raw_entities" if raw else "entities"
        where_clause = "WHERE resolved_id IS NULL" if canonical_only else ""

        cursor = self._conn.cursor()
        cursor.execute(
            f"""
            SELECT id, event_id, name, type, description, resolved_id
            FROM {table}
            {where_clause};
            """  # noqa: S608
        )

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            for row in rows:
                yield Entity(
                    id=self._blob_to_uuid(row["id"]),
                    event_id=self._blob_to_uuid(row["event_id"]),
                    name=row["name"],
                    type=row["type"] or "",
                    description=row["description"] or "",
                    resolved_id=self._blob_to_uuid(row["resolved_id"]),
                )

    def get_all_entities(
            self, raw: bool = False, canonical_only: bool = True
    ) -> List[Entity]:
        """Get all entities (use iter_entities for large datasets)."""
        return list(self.iter_entities(raw=raw, canonical_only=canonical_only))

    def get_entities_for_event(
            self, event_id: uuid.UUID, raw: bool = False
    ) -> List[Entity]:
        """
        Get all entities for an event.

        Args:
            event_id: ID of the event (same as statement ID)
            raw: Use raw_entities table

        Returns:
            List of entities associated with the event
        """
        table = "raw_entities" if raw else "entities"

        cursor = self._conn.cursor()
        cursor.execute(
            f"""
            SELECT id, event_id, name, type, description, resolved_id
            FROM {table}
            WHERE event_id = ?;
            """,  # noqa: S608
            (self._uuid_to_blob(event_id),),
        )

        return [
            Entity(
                id=self._blob_to_uuid(row["id"]),
                event_id=self._blob_to_uuid(row["event_id"]),
                name=row["name"],
                type=row["type"] or "",
                description=row["description"] or "",
                resolved_id=self._blob_to_uuid(row["resolved_id"]),
            )
            for row in cursor.fetchall()
        ]

    def get_entities_for_statement(
            self, statement_id: uuid.UUID, raw: bool = False
    ) -> List[Entity]:
        """
        Get all entities for a statement.
        Since event.id == statement.id, this is equivalent to get_entities_for_event.

        Args:
            statement_id: ID of the statement (same as event ID)
            raw: Use raw_entities table

        Returns:
            List of entities
        """
        return self.get_entities_for_event(statement_id, raw=raw)

    def update_entity_resolved_id(
            self, entity_id: uuid.UUID, canonical_id: uuid.UUID, raw: bool = False
    ) -> None:
        """Update an entity's resolved_id to point to a canonical entity."""
        table = "raw_entities" if raw else "entities"

        with self.transaction():
            self._conn.execute(
                f"UPDATE {table} SET resolved_id = ? WHERE id = ?;",  # noqa: S608
                (self._uuid_to_blob(canonical_id), self._uuid_to_blob(entity_id)),
            )

        logger.debug(f"Updated entity {entity_id} to resolve to {canonical_id}")

    def update_entity_references_batch(
            self, entity_mapping: dict[uuid.UUID, uuid.UUID], raw: bool = True
    ) -> None:
        """
        Batch update entity references from old IDs to new IDs.

        Args:
            entity_mapping: Dict mapping old_id -> new_id
            raw: Update raw tables

        This is used during entity resolution to merge duplicate entities.
        """
        if not entity_mapping:
            logger.warning("No mapping found for updating entity references")
            return

        prefix = "raw_" if raw else ""

        with self.transaction():
            # Update resolved_id in entities table
            self._conn.executemany(
                f"UPDATE {prefix}entities SET resolved_id = ? WHERE resolved_id = ?;",  # noqa: S608
                [
                    (self._uuid_to_blob(new_id), self._uuid_to_blob(old_id))
                    for old_id, new_id in entity_mapping.items()
                ],
            )

            # Update subject_id in triplets table
            self._conn.executemany(
                f"UPDATE {prefix}triplets SET subject_id = ? WHERE subject_id = ?;",  # noqa: S608
                [
                    (self._uuid_to_blob(new_id), self._uuid_to_blob(old_id))
                    for old_id, new_id in entity_mapping.items()
                ],
            )

            # Update object_id in triplets table
            self._conn.executemany(
                f"UPDATE {prefix}triplets SET object_id = ? WHERE object_id = ?;",  # noqa: S608
                [
                    (self._uuid_to_blob(new_id), self._uuid_to_blob(old_id))
                    for old_id, new_id in entity_mapping.items()
                ],
            )

        logger.info(f"Batch updated {len(entity_mapping)} entity references")

    def remove_entity(self, entity_id: uuid.UUID, raw: bool = False) -> None:
        """Remove an entity from the database."""
        table = "raw_entities" if raw else "entities"

        with self.transaction():
            self._conn.execute(
                f"DELETE FROM {table} WHERE id = ?;",  # noqa: S608
                (self._uuid_to_blob(entity_id),),
            )

        logger.debug(f"Removed entity {entity_id}")

    # ==================== Query Operations ====================

    def dump_publication(
            self, fpath: str, publication_id: str, raw: bool = False
    ) -> None:
        """
        Dump a publication with all extracted data to a JSON file.

        Args:
            fpath: Output file path
            publication_id: ID of publication to dump
            raw: Use raw tables

        Raises:
            NotFoundError: If publication doesn't exist
        """
        # Get publication
        publication = self.get_publication_by_id(publication_id)

        # Get all statements with their data
        statements_data = []

        for stmt_id, statement in self.iter_statements_for_publication(publication_id):
            try:
                # Get event (using stmt_id since event.id == statement.id)
                event:TemporalEvent = self.get_event_by_id(stmt_id, raw=raw)

                # Get triplets
                triplets:list[Triplet] = self.get_triplets_for_event(stmt_id, raw=raw)

                # Get entities
                entities:list[Entity] = self.get_entities_for_event(stmt_id, raw=raw)

                # Format data
                stmt_dict = {
                    "statement": statement.statement,
                    "statement_type": statement.statement_type.value,
                    "temporal_type": statement.temporal_type.value,
                    "temporal_confidence": statement.temporal_confidence.value,
                    "valid_at": (
                        event.valid_at.strftime("%Y-%m-%d_%H:%M")
                        if event.valid_at
                        else None
                    ),
                    "valid_at_confidence": event.valid_at_confidence.value,
                    "invalid_at": (
                        event.invalid_at.strftime("%Y-%m-%d_%H:%M")
                        if event.invalid_at
                        else None
                    ),
                    "invalid_at_confidence": event.invalid_at_confidence.value,
                    "triplets": [
                        f"{t.subject_name} || {t.predicate.value} || {t.object_name}"
                        for t in triplets
                    ],
                    "entities": [f"{e.name} : {e.type}" for e in entities],
                }

                # Add "invalidated_by" only for non-raw exports
                if not raw:
                    stmt_dict["invalidated_by"] = (
                        str(event.invalidated_by) if event.invalidated_by else None
                    )

                statements_data.append(stmt_dict)

                statements_data.append(stmt_dict)

            except NotFoundError as e:
                logger.warning(f"Skipping statement {stmt_id}: {e}")
                continue

        # Write to file
        with open(fpath, "w", encoding="utf-8") as f:
            f.write('"""\n')
            f.write(publication.text)
            f.write('\n"""\n\n')

            statements_dict = {"statements": statements_data}
            f.write(json.dumps(statements_dict, indent=2, ensure_ascii=False))
            f.write("\n")

        logger.info(
            f"Dumped {len(statements_data)} statements for publication "
            f"{publication_id} to {fpath}"
        )

    def export_to_csv(self, output_path: str, raw: bool = False, batch_size: int = 100) -> None:
        """
        Export database content to CSV file with publication and event data.

        Creates a CSV with one row per event, including publication metadata
        and event-specific information (temporal data, counts, etc.).

        Args:
            output_path: Path to output CSV file
            raw: Use raw tables if True (default: False)
            batch_size: Number of publications to process at once (default: 100)

        Raises:
            TKGDatabaseError: If export fails
        """
        import csv

        rows = []
        total_events = 0

        logger.info(f"Starting CSV export to {output_path} (raw={raw})")

        try:
            # Iterate through all publications
            for publication in self.iter_publications(batch_size=batch_size):
                # Get publication-level data
                pub_data = {
                    "publisher": publication.publisher,
                    "publication_name": publication.title or "",
                    "publication_url": publication.url,
                    "publication_title": publication.title or "",
                    "published_on": self._datetime_to_text(publication.published_on) if publication.published_on else "",
                }

                # Iterate through statements for this publication
                for stmt_id, statement in self.iter_statements_for_publication(publication.id):
                    try:
                        # Get event (using stmt_id since event.id == statement.id)
                        event = self.get_event_by_id(stmt_id, raw=raw)

                        # Get triplets and entities for counts
                        triplets = self.get_triplets_for_event(stmt_id, raw=raw)
                        entities = self.get_entities_for_event(stmt_id, raw=raw)

                        # Create row with all data
                        row = {
                            **pub_data,
                            "event_id": str(event.id),
                            "statement_length": len(event.statement),
                            "temporal_type": event.temporal_type.value,
                            "statement_type": event.statement_type.value,
                            "temporal_confidence": event.temporal_confidence.value,
                            "valid_at": self._datetime_to_text(event.valid_at) if event.valid_at else "",
                            "valid_at_confidence": event.valid_at_confidence.value,
                            "invalid_at": self._datetime_to_text(event.invalid_at) if event.invalid_at else "",
                            "invalid_at_confidence": event.invalid_at_confidence.value,
                            "num_triplets": len(triplets),
                            "num_entities": len(entities),
                            "invalidated_by": str(event.invalidated_by) if event.invalidated_by else "",
                        }

                        rows.append(row)
                        total_events += 1

                    except NotFoundError as e:
                        logger.warning(f"Skipping statement {stmt_id} in publication {publication.id}: {e}")
                        continue

            # Write to CSV file
            if rows:
                fieldnames = [
                    "publisher",
                    "publication_name",
                    "publication_url",
                    "publication_title",
                    "published_on",
                    "event_id",
                    "statement_length",
                    "temporal_type",
                    "statement_type",
                    "temporal_confidence",
                    "valid_at",
                    "valid_at_confidence",
                    "invalid_at",
                    "invalid_at_confidence",
                    "num_triplets",
                    "num_entities",
                    "invalidated_by",
                ]

                with open(output_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

                logger.info(f"Successfully exported {total_events} events to {output_path}")
            else:
                logger.warning("No events found to export")
                # Create empty CSV with headers
                with open(output_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "publisher",
                            "publication_name",
                            "publication_url",
                            "publication_title",
                            "published_on",
                            "event_id",
                            "statement_length",
                            "temporal_type",
                            "statement_type",
                            "temporal_confidence",
                            "valid_at",
                            "valid_at_confidence",
                            "invalid_at",
                            "invalid_at_confidence",
                            "num_triplets",
                            "num_entities",
                            "invalidated_by",
                        ]
                    )

        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            raise TKGDatabaseError(f"CSV export failed: {e}") from e

    # ==================== Connection Management ====================

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            try:
                if self._in_transaction:
                    self._conn.rollback()
                self._conn.close()
                logger.info("Database connection closed.")
            except Exception as e:
                logger.error(f"Error closing database: {e}")
            finally:
                self._conn = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.close()
        return False

    def __del__(self):
        """Ensure connection is closed on garbage collection."""
        if self._conn:
            try:
                self.close()
            except Exception:
                logger.error(f"Error closing database: {traceback.format_exc()}")
                pass