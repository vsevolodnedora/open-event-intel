        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE stage (
            sort_order INTEGER PRIMARY KEY,
            stage_id   TEXT NOT NULL UNIQUE,
            scope      TEXT NOT NULL,
            label      TEXT NOT NULL
        );
        CREATE TABLE run (
            run_id                TEXT PRIMARY KEY,
            status                TEXT NOT NULL,
            started_at            TEXT,
            completed_at          TEXT,
            config_version        TEXT,
            prev_completed_run_id TEXT
        );
        CREATE TABLE publisher (publisher_id TEXT PRIMARY KEY);
        CREATE TABLE doc (
            doc_index             INTEGER PRIMARY KEY,
            doc_version_id        TEXT NOT NULL UNIQUE,
            document_id           TEXT,
            publisher_id          TEXT,
            title                 TEXT,
            url_normalized        TEXT,
            source_published_at   TEXT,
            created_in_run_id     TEXT,
            content_quality_score REAL,
            primary_language      TEXT
        );
        CREATE TABLE doc_stage_status (
            doc_index     INTEGER NOT NULL,
            stage_index   INTEGER NOT NULL,
            status        TEXT,
            attempt       INTEGER,
            last_run_id   TEXT,
            processed_at  TEXT,
            error_message TEXT,
            details       TEXT,
            PRIMARY KEY (doc_index, stage_index)
        );
        CREATE TABLE doc_totals (
            doc_index   INTEGER NOT NULL,
            stage_index INTEGER NOT NULL,
            counts_json TEXT,
            PRIMARY KEY (doc_index, stage_index)
        );