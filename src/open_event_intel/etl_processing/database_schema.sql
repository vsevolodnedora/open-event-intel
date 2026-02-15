-- 4. Working Database Schema (processed_posts.db)

PRAGMA foreign_keys = ON;

-- 4.1 Identity + Versioning Layer (Stage 1)

CREATE TABLE pipeline_run (
    run_id TEXT PRIMARY KEY,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    config_version TEXT NOT NULL,           -- hash of effective config (strict mode; immutable per DB)
    budget_spent REAL DEFAULT 0,            -- dollars spend on OpenAI calls (see 'budget' in `config.yaml`)
    doc_count_processed INTEGER DEFAULT 0,
    doc_count_skipped INTEGER DEFAULT 0,
    doc_count_failed INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running'  -- running|completed|failed|aborted

);
-- Enforce mutual exclusion at DB level: at most one running pipeline_run at a time.
-- This replaces runtime locks/heartbeats; if a prior run crashed, manual remediation is:
--   UPDATE pipeline_run SET status='aborted', completed_at=CURRENT_TIMESTAMP WHERE status='running';
CREATE UNIQUE INDEX idx_pipeline_single_running
ON pipeline_run(status) WHERE status='running';

CREATE TABLE scrape_record (
    scrape_id TEXT PRIMARY KEY,
    publisher_id TEXT NOT NULL,
    source_id TEXT NOT NULL,                -- ID from source table OR derived ID for resources
    url_raw TEXT NOT NULL,
    url_normalized TEXT NOT NULL,           -- normalized form used for deterministic IDs and dedup
    -- Copied from source DB row at ingest time
    scraped_at TIMESTAMP NOT NULL,          -- added_on from source OR download time for resources
    source_published_at TIMESTAMP NOT NULL, -- source.published_on (required; this core deliverable ingests only pages)
    source_title TEXT NOT NULL,             -- source.title
    source_language TEXT,                   -- source.language ('de'|'en'|...)
    raw_content BLOB NOT NULL,              -- Original bytes before any processing
    raw_encoding_detected TEXT,             -- e.g., 'utf-8', 'latin-1'
    scrape_kind TEXT NOT NULL DEFAULT 'page',   -- page (attachments/resources may be added later)
    ingest_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),  -- run that ingested this record (for scope filtering)
    processing_status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ingest idempotency contract:
--   - Each source row (publisher_id, source_id, scrape_kind) maps to exactly one scrape_record.
--   - Stage 1 MUST use deterministic scrape_id = SHA256(publisher_id || '|' || source_id || '|' || scrape_kind) (hex).
--   - Stage 1 MUST NOT overwrite raw_content or copied source fields. Re-runs must be idempotent.
--     Required behavior:
--       * If the row already exists, Stage 1 MUST NOT overwrite it and MUST NOT reprocess it.
--         Any later changes in the source DB for an already-ingested record are **ignored** by this core deliverable.
--         Recommended: detect mismatches and record a non-fatal validation_failure (severity='warning') for audit,
--         but continue the run.
 
CREATE UNIQUE INDEX idx_scrape_unique_source
ON scrape_record(publisher_id, source_id, scrape_kind);

CREATE INDEX idx_scrape_urlnorm
ON scrape_record(publisher_id, url_normalized, source_published_at, scrape_kind);

-- For a given publisher, canonical URL + source_published_at identifies a single publication.
-- This disambiguates rare collisions introduced by URL normalization.
-- If the source DB violates this, Stage 1 must abort and the scraper/source data must be corrected upstream.
-- Fail-fast dedup invariant (required for this core project):
CREATE UNIQUE INDEX idx_scrape_unique_url_pubdate
ON scrape_record(publisher_id, url_normalized, source_published_at, scrape_kind);

-- Audit immutability contract:
-- scrape_record is an immutable audit log for raw bytes and copied source fields.
-- Stages may update ONLY: processing_status.
-- Any attempt to mutate immutable columns aborts.
CREATE TRIGGER scrape_record_no_update_immutable
BEFORE UPDATE OF
    publisher_id, source_id, url_raw, url_normalized,
    scraped_at, source_published_at, source_title, source_language,
    raw_content, raw_encoding_detected, scrape_kind,
    ingest_run_id, created_at
ON scrape_record
BEGIN
    SELECT RAISE(ABORT, 'scrape_record is append-only for raw bytes and copied source fields; updates are not allowed');
END;

CREATE TABLE document (
    document_id TEXT PRIMARY KEY,
    publisher_id TEXT NOT NULL,
    url_normalized TEXT NOT NULL,
    source_published_at TIMESTAMP NOT NULL, -- copied from scrape_record.source_published_at; part of document identity
    url_raw_first_seen TEXT NOT NULL,
    document_class TEXT,                    -- optional early guess (authoritative class stored in doc_metadata)
    is_attachment INTEGER DEFAULT 0,        -- 1 if originated as linked resource
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(publisher_id, url_normalized, source_published_at)
);
-- ID policy (recommended): document.document_id = SHA256(publisher_id || '|' || url_normalized || '|' || source_published_at) (hex)

CREATE TABLE document_version (
    doc_version_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES document(document_id),
    scrape_id TEXT NOT NULL REFERENCES scrape_record(scrape_id),

    content_hash_raw TEXT NOT NULL,
    encoding_repairs_applied TEXT,          -- JSON array of repairs

    cleaning_spec_version TEXT NOT NULL,    -- e.g., "clean_v1"
    normalization_spec TEXT NOT NULL,       -- JSON: {"nfc": true, ...}

    pii_masking_enabled INTEGER NOT NULL DEFAULT 0,
    -- Use 'none' when pii_masking_enabled=0 to keep idempotency/uniqueness well-defined.
    scrape_kind TEXT NOT NULL DEFAULT 'page' CHECK (scrape_kind = 'page'),   -- this core deliverable supports pages only
    pii_mask_log TEXT,                      -- JSON: placeholders + counts

    content_hash_clean TEXT NOT NULL,       -- hash(clean_content)
    clean_content TEXT NOT NULL,            -- NFC-normalized immutable canonical string
    span_indexing TEXT NOT NULL DEFAULT 'unicode_codepoint',

    content_length_raw INTEGER,
    content_length_clean INTEGER,
    boilerplate_ratio REAL,
    content_quality_score REAL,             -- 0..1; <0.15 may skip downstream stages

    primary_language TEXT,                  -- 'de','en',...
    secondary_languages TEXT,               -- JSON array
    language_detection_confidence REAL,     -- 0..1
    created_in_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Core deliverable constraint: single cleaned version per canonical document (see §1.3).
    UNIQUE(document_id),
    -- Ingest idempotency: a scrape_record maps to exactly one cleaned document_version.
    UNIQUE(scrape_id),
    CHECK (encoding_repairs_applied IS NULL OR (json_valid(encoding_repairs_applied) AND json_type(encoding_repairs_applied) = 'array')),
    CHECK (json_valid(normalization_spec) AND json_type(normalization_spec) = 'object'),
    CHECK (pii_mask_log IS NULL OR json_valid(pii_mask_log))
);

CREATE INDEX idx_docver_document ON document_version(document_id);
CREATE INDEX idx_docver_content_hash ON document_version(content_hash_raw);

-- Document-version idempotency (required):
-- Stage 1 MUST use deterministic doc_version_id so reruns do not create duplicate document_version rows.
-- Core deliverable constraint (single cleaned version): doc_version_id = SHA256(scrape_id) (hex).
-- Additional uniqueness by cleaning/masking versions is intentionally omitted:
-- this core deliverable is strict (rebuild required for spec changes), and UNIQUE(scrape_id) is sufficient.

-- 4.2 Shared System Tables (Multi-writer; validated by Stage 11)
-- (Defined early to support FK references in later stage tables.)

-- Normalized evidence span store (shared across stages)
CREATE TABLE evidence_span (
    evidence_id TEXT PRIMARY KEY,
    doc_version_id TEXT NOT NULL REFERENCES document_version(doc_version_id),

    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,

    -- Stored for convenience, but MUST equal clean_content[span_start:span_end].
    -- IMPORTANT: application should not supply arbitrary text; use computed slice.
    text TEXT NOT NULL,

    purpose TEXT,                           -- optional human/debug label
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_in_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),

    CHECK (span_start >= 0),
    CHECK (span_end > span_start),
    CHECK (length(text) > 0),
    -- Minimal enforcement: canonical formatting only (hash equality is enforced by stage logic + Stage 11).
    CHECK (length(evidence_id) = 64),
    CHECK (evidence_id = lower(evidence_id)),
    CHECK (evidence_id GLOB '[0-9a-f]*'),
    UNIQUE(doc_version_id, span_start, span_end)
);
-- Deterministic evidence_id (normative; required for get-or-create across stages and Q&A):
-- evidence_id = SHA256(doc_version_id || '|' || span_start || '|' || span_end) (hex).
-- Stages MUST compute evidence_id deterministically and then use INSERT OR IGNORE (or equivalent) on
-- UNIQUE(doc_version_id, span_start, span_end), followed by SELECT to retrieve the evidence_id.
-- Determinism enforcement (normative, stage-level; fail-closed):
--   After get-or-create, the caller MUST recompute expected_evidence_id and verify the stored evidence_id equals it.
--   If the existing row’s evidence_id does not match the expected deterministic ID, the stage MUST abort with a
--   validation_failure (this indicates prior corruption/non-conforming writer; do not proceed).

CREATE INDEX idx_evidence_docver ON evidence_span(doc_version_id);
CREATE INDEX idx_evidence_span ON evidence_span(doc_version_id, span_start, span_end);

-- Enforce append-only for evidence_span
CREATE TRIGGER evidence_span_no_update
BEFORE UPDATE ON evidence_span
BEGIN
    SELECT RAISE(ABORT, 'evidence_span is append-only; updates are not allowed');
END;

CREATE TRIGGER evidence_span_no_delete
BEFORE DELETE ON evidence_span
BEGIN
    SELECT RAISE(ABORT, 'evidence_span is append-only; deletes are not allowed');
END;

-- Best-effort DB-level slice validation using SQLite substr().
-- Note: substr() is 1-based for start index; length is characters.
-- Assumes clean_content is NFC-normalized and indices are in Unicode codepoints/characters.
CREATE TRIGGER evidence_span_validate_insert
BEFORE INSERT ON evidence_span
BEGIN
    
    -- Enforce span bounds against the referenced document_version.clean_content length.
    -- (Prevents out-of-range coordinates that could otherwise pass due to substr() truncation.)
    SELECT
      CASE
        WHEN NEW.span_start >= (
          SELECT length(dv.clean_content)
          FROM document_version dv
          WHERE dv.doc_version_id = NEW.doc_version_id
        )
        THEN RAISE(ABORT, 'evidence_span.span_start must be < length(clean_content)')
      END;
    SELECT
      CASE
        WHEN NEW.span_end > (
          SELECT length(dv.clean_content)
          FROM document_version dv
          WHERE dv.doc_version_id = NEW.doc_version_id
        )
        THEN RAISE(ABORT, 'evidence_span.span_end must be <= length(clean_content)')
      END;
    SELECT
      CASE
        WHEN (
          SELECT substr(dv.clean_content, NEW.span_start + 1, NEW.span_end - NEW.span_start)
          FROM document_version dv
          WHERE dv.doc_version_id = NEW.doc_version_id
        ) != NEW.text
        THEN RAISE(ABORT, 'evidence_span.text must equal clean_content[span_start:span_end]')
      END;
END;

-- Per-document stage completion (multi-writer; global-latest per doc+stage)
CREATE TABLE doc_stage_status (
    doc_version_id TEXT NOT NULL REFERENCES document_version(doc_version_id),
    stage TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),  -- last run to update this row
    attempt INTEGER NOT NULL DEFAULT 1,                    -- increment on each retry for this doc+stage
    status TEXT NOT NULL,                   -- pending|ok|failed|blocked|skipped
    processed_at TIMESTAMP,
    -- For this core deliverable: config_hash MUST equal pipeline_run.config_version for run_id.
    config_hash TEXT NOT NULL,
    error_message TEXT,
    details TEXT,
    PRIMARY KEY (doc_version_id, stage)
);
-- Allowed status values (normative)
-- NOTE: `blocked` is retryable and indicates unmet prerequisites (no processing attempted).
--       `skipped` is a final policy skip (quality/out-of-scope).
CREATE TRIGGER doc_stage_status_validate_status
BEFORE INSERT ON doc_stage_status
BEGIN
  SELECT CASE
    WHEN NEW.status NOT IN ('pending','ok','failed','blocked','skipped')
    THEN RAISE(ABORT, 'doc_stage_status.status must be one of pending|ok|failed|blocked|skipped')
  END;
END;

CREATE TRIGGER doc_stage_status_validate_status_update
BEFORE UPDATE OF status ON doc_stage_status
BEGIN
  SELECT CASE
    WHEN NEW.status NOT IN ('pending','ok','failed','blocked','skipped')
    THEN RAISE(ABORT, 'doc_stage_status.status must be one of pending|ok|failed|blocked|skipped')
  END;
END;

CREATE INDEX idx_doc_stage_resume ON doc_stage_status(stage, status)
    WHERE status IN ('pending', 'failed', 'blocked');

-- Run-scoped stage completion (for stages that compute corpus-level derived artifacts)
-- Used by Stage 09 Outputs and Stage 10 Timeline (see §6.1 and §6.5).
CREATE TABLE run_stage_status (
    run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    stage TEXT NOT NULL,                     -- stage_09_outputs | stage_10_timeline | stage_11_validation (optional)
    attempt INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL,                    -- pending|ok|failed
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    -- For this core deliverable: config_hash MUST equal pipeline_run.config_version for run_id.
    config_hash TEXT NOT NULL,
    error_message TEXT,
    details TEXT,
    PRIMARY KEY (run_id, stage)
);
-- Allowed status values (normative; consistency with doc_stage_status)
CREATE TRIGGER run_stage_status_validate_status
BEFORE INSERT ON run_stage_status
BEGIN
  SELECT CASE
    WHEN NEW.status NOT IN ('pending','ok','failed')
    THEN RAISE(ABORT, 'run_stage_status.status must be one of pending|ok|failed')
  END;
END;

CREATE TRIGGER run_stage_status_validate_status_update
BEFORE UPDATE OF status ON run_stage_status
BEGIN
  SELECT CASE
    WHEN NEW.status NOT IN ('pending','ok','failed')
    THEN RAISE(ABORT, 'run_stage_status.status must be one of pending|ok|failed')
  END;
END;
-- LLM usage tracking (multi-writer: any stage that calls an LLM writes here)
CREATE TABLE llm_usage_log (
    log_id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES pipeline_run(run_id),
    stage TEXT NOT NULL,
    purpose TEXT NOT NULL,
    model TEXT NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    cost REAL NOT NULL,
    cached INTEGER DEFAULT 0,
    cache_key TEXT,
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- LLM response cache (multi-writer)
CREATE TABLE llm_cache (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    response TEXT NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    hit_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Validation failures (written by validation stage; may also be written by stages doing inline checks)
CREATE TABLE validation_failure (
    failure_id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES pipeline_run(run_id),
    stage TEXT NOT NULL,
    doc_version_id TEXT REFERENCES document_version(doc_version_id),
    check_name TEXT NOT NULL,
    details TEXT,
    severity TEXT,
    auto_repaired INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4.3 Structural Layer (Stage 2)

CREATE TABLE block (
    block_id TEXT PRIMARY KEY,
    doc_version_id TEXT NOT NULL REFERENCES document_version(doc_version_id),
    block_type TEXT NOT NULL,               -- HEADING|PARAGRAPH|LIST|TABLE|...
    block_level INTEGER,                    -- headings: 1..6
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    parse_confidence REAL,
    boilerplate_flag TEXT,                  -- NULL|rule_matched|...
    boilerplate_reason TEXT,
    parent_block_id TEXT,
    language_hint TEXT,
    created_in_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- ID policy (recommended): block.block_id = SHA256(doc_version_id || '|' || span_start || '|' || span_end || '|' || block_type || '|' || coalesce(block_level,'')) (hex)
-- Natural-key uniqueness (normative; prevents duplicate inserts for already-processed documents)
CREATE UNIQUE INDEX idx_block_natural_key
ON block(doc_version_id, span_start, span_end, block_type, IFNULL(block_level, -1));

-- Enforce: if parent_block_id is set, parent block must exist and belong to same doc_version_id
CREATE TRIGGER block_validate_parent_docver_insert
BEFORE INSERT ON block
WHEN NEW.parent_block_id IS NOT NULL
BEGIN
    SELECT CASE
      WHEN (SELECT COUNT(1) FROM block p WHERE p.block_id = NEW.parent_block_id) = 0
      THEN RAISE(ABORT, 'block.parent_block_id must reference an existing block')
    END;
    SELECT CASE
      WHEN (SELECT p.doc_version_id FROM block p WHERE p.block_id = NEW.parent_block_id) != NEW.doc_version_id
      THEN RAISE(ABORT, 'block.parent_block_id must belong to the same doc_version_id')
    END;
END;



CREATE TRIGGER block_validate_parent_docver_update
BEFORE UPDATE OF parent_block_id, doc_version_id ON block
WHEN NEW.parent_block_id IS NOT NULL
BEGIN
    SELECT CASE
     WHEN (SELECT COUNT(1) FROM block p WHERE p.block_id = NEW.parent_block_id) = 0
      THEN RAISE(ABORT, 'block.parent_block_id must reference an existing block')
    END;
    SELECT CASE
      WHEN (SELECT p.doc_version_id FROM block p WHERE p.block_id = NEW.parent_block_id) != NEW.doc_version_id
      THEN RAISE(ABORT, 'block.parent_block_id must belong to the same doc_version_id')
    END;
END;

-- Fail-fast span bounds for blocks (prevents invalid coordinates early; chunk/table extraction depend on blocks)
CREATE TRIGGER block_validate_span_insert
BEFORE INSERT ON block
BEGIN
    SELECT CASE
      WHEN NEW.span_start < 0 OR NEW.span_end <= NEW.span_start
      THEN RAISE(ABORT, 'block span must satisfy 0 <= start < end')
    END;
    SELECT CASE
      WHEN NEW.span_end > (
        SELECT length(dv.clean_content) FROM document_version dv WHERE dv.doc_version_id = NEW.doc_version_id
      )
      THEN RAISE(ABORT, 'block span must be within clean_content bounds')
    END;
END;

CREATE TRIGGER block_validate_span_update
BEFORE UPDATE OF span_start, span_end, doc_version_id ON block
BEGIN
    SELECT CASE
      WHEN NEW.span_start < 0 OR NEW.span_end <= NEW.span_start
      THEN RAISE(ABORT, 'block span must satisfy 0 <= start < end')
    END;
    SELECT CASE
      WHEN NEW.span_end > (
        SELECT length(dv.clean_content) FROM document_version dv WHERE dv.doc_version_id = NEW.doc_version_id
      )
      THEN RAISE(ABORT, 'block span must be within clean_content bounds')
    END;
END;
CREATE INDEX idx_block_docver ON block(doc_version_id);

CREATE TABLE chunk (
    chunk_rowid INTEGER PRIMARY KEY AUTOINCREMENT, -- required for FTS5 content_rowid
    chunk_id TEXT UNIQUE NOT NULL,
    doc_version_id TEXT NOT NULL REFERENCES document_version(doc_version_id),
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    -- Provenance anchor for retrieval: must match (doc_version_id, span_start, span_end).
    evidence_id TEXT NOT NULL REFERENCES evidence_span(evidence_id),
    chunk_type TEXT NOT NULL,               -- semantic|table_summary|...
    block_ids TEXT NOT NULL,                -- JSON array of block_id
    chunk_text TEXT NOT NULL,               -- MUST equal clean_content[span_start:span_end] (validated in app + Stage 11)
    heading_context TEXT,
    retrieval_exclude INTEGER DEFAULT 0,
    mention_boundary_safe INTEGER DEFAULT 1,
    token_count_approx INTEGER,
    created_in_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (json_valid(block_ids) AND json_type(block_ids) = 'array' AND json_array_length(block_ids) > 0)
);
-- ID policy (recommended): chunk.chunk_id = SHA256(doc_version_id || '|' || span_start || '|' || span_end || '|' || chunk_type) (hex)
-- Natural-key uniqueness (normative; supports per-document immutability)
CREATE UNIQUE INDEX idx_chunk_natural_key
ON chunk(doc_version_id, span_start, span_end, chunk_type);

-- Fail-fast span integrity for chunks
CREATE TRIGGER chunk_validate_insert
BEFORE INSERT ON chunk
BEGIN
    SELECT CASE
      WHEN NEW.span_start < 0 OR NEW.span_end <= NEW.span_start
      THEN RAISE(ABORT, 'chunk span must satisfy 0 <= start < end')
    END;
    SELECT CASE
      WHEN NEW.span_end > (
        SELECT length(dv.clean_content) FROM document_version dv WHERE dv.doc_version_id = NEW.doc_version_id
      )
      THEN RAISE(ABORT, 'chunk span must be within clean_content bounds')
    END;
    SELECT CASE
      WHEN (
        SELECT substr(dv.clean_content, NEW.span_start + 1, NEW.span_end - NEW.span_start)
        FROM document_version dv WHERE dv.doc_version_id = NEW.doc_version_id
      ) != NEW.chunk_text
      THEN RAISE(ABORT, 'chunk.chunk_text must equal clean_content[span_start:span_end]')
    END;
    -- Enforce: chunk.evidence_id must match the same doc_version_id and span.
    SELECT CASE
      WHEN (
        SELECT COUNT(1) FROM evidence_span es
        WHERE es.evidence_id = NEW.evidence_id
          AND es.doc_version_id = NEW.doc_version_id
          AND es.span_start = NEW.span_start
          AND es.span_end = NEW.span_end
      ) = 0
      THEN RAISE(ABORT, 'chunk.evidence_id must reference evidence_span for the same doc_version_id/span')
    END;
    -- Enforce: chunk.block_ids must reference existing blocks from the same doc_version_id.
    SELECT CASE
      WHEN (
        SELECT COUNT(1)
        FROM json_each(NEW.block_ids) j
        LEFT JOIN block b ON b.block_id = j.value
        WHERE b.block_id IS NULL OR b.doc_version_id != NEW.doc_version_id
      ) > 0
      THEN RAISE(ABORT, 'chunk.block_ids must reference existing block rows from the same doc_version_id')
    END;
END;

CREATE TRIGGER chunk_validate_update
BEFORE UPDATE OF span_start, span_end, chunk_text, doc_version_id, block_ids, evidence_id ON chunk
BEGIN
    SELECT CASE
      WHEN NEW.span_start < 0 OR NEW.span_end <= NEW.span_start
      THEN RAISE(ABORT, 'chunk span must satisfy 0 <= start < end')
    END;
    SELECT CASE
      WHEN NEW.span_end > (
        SELECT length(dv.clean_content) FROM document_version dv WHERE dv.doc_version_id = NEW.doc_version_id
      )
      THEN RAISE(ABORT, 'chunk span must be within clean_content bounds')
    END;
    SELECT CASE
      WHEN (
        SELECT substr(dv.clean_content, NEW.span_start + 1, NEW.span_end - NEW.span_start)
        FROM document_version dv WHERE dv.doc_version_id = NEW.doc_version_id
      ) != NEW.chunk_text
      THEN RAISE(ABORT, 'chunk.chunk_text must equal clean_content[span_start:span_end]')
    END;
    SELECT CASE
      WHEN (
        SELECT COUNT(1) FROM evidence_span es
        WHERE es.evidence_id = NEW.evidence_id
          AND es.doc_version_id = NEW.doc_version_id
          AND es.span_start = NEW.span_start
          AND es.span_end = NEW.span_end
      ) = 0
      THEN RAISE(ABORT, 'chunk.evidence_id must reference evidence_span for the same doc_version_id/span')
    END;
    -- Enforce: chunk.block_ids must reference existing blocks from the same doc_version_id.
    SELECT CASE
      WHEN (
        SELECT COUNT(1)
        FROM json_each(NEW.block_ids) j
        LEFT JOIN block b ON b.block_id = j.value
        WHERE b.block_id IS NULL OR b.doc_version_id != NEW.doc_version_id
      ) > 0
      THEN RAISE(ABORT, 'chunk.block_ids must reference existing block rows from the same doc_version_id')
    END;
END;


CREATE INDEX idx_chunk_docver ON chunk(doc_version_id);

CREATE TABLE table_extract (
    table_id TEXT PRIMARY KEY,
    block_id TEXT NOT NULL REFERENCES block(block_id),
    doc_version_id TEXT NOT NULL REFERENCES document_version(doc_version_id),
    row_count INTEGER,
    col_count INTEGER,
    headers_json TEXT,                      -- JSON array of header strings
    header_row_index INTEGER,
    parse_quality REAL,
    parse_method TEXT,                      -- markdown_pipe|html_table|...
    table_class TEXT,                       -- time_series|comparison|...
    period_granularity TEXT,
    units_detected TEXT,                    -- JSON array
    raw_table_text TEXT,
    created_in_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- One table extraction per structural table block in this core deliverable.
CREATE UNIQUE INDEX idx_table_extract_unique_block
ON table_extract(block_id);

-- Enforce: table_extract.doc_version_id must match the referenced block.doc_version_id
CREATE TRIGGER table_extract_validate_block_docver_insert
BEFORE INSERT ON table_extract
BEGIN
    SELECT CASE
      WHEN (SELECT b.doc_version_id FROM block b WHERE b.block_id = NEW.block_id) != NEW.doc_version_id
      THEN RAISE(ABORT, 'table_extract.doc_version_id must match block.doc_version_id')
    END;
END;

CREATE TRIGGER table_extract_validate_block_docver_update
BEFORE UPDATE OF block_id, doc_version_id ON table_extract
BEGIN
    SELECT CASE
      WHEN (SELECT b.doc_version_id FROM block b WHERE b.block_id = NEW.block_id) != NEW.doc_version_id
      THEN RAISE(ABORT, 'table_extract.doc_version_id must match block.doc_version_id')
    END;
END;

-- FTS5 virtual table for lexical search
CREATE VIRTUAL TABLE chunk_fts USING fts5(
    chunk_id UNINDEXED,
    chunk_text,
    heading_context,
    content='chunk',
    content_rowid='chunk_rowid'
);

CREATE TRIGGER chunk_ai AFTER INSERT ON chunk BEGIN
    INSERT INTO chunk_fts(rowid, chunk_id, chunk_text, heading_context)
    VALUES (new.chunk_rowid, new.chunk_id, new.chunk_text, new.heading_context);
END;

CREATE TRIGGER chunk_ad AFTER DELETE ON chunk BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, chunk_id, chunk_text, heading_context)
    VALUES ('delete', old.chunk_rowid, old.chunk_id, old.chunk_text, old.heading_context);
END;

CREATE TRIGGER chunk_au AFTER UPDATE ON chunk BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, chunk_id, chunk_text, heading_context)
    VALUES ('delete', old.chunk_rowid, old.chunk_id, old.chunk_text, old.heading_context);
    INSERT INTO chunk_fts(rowid, chunk_id, chunk_text, heading_context)
    VALUES (new.chunk_rowid, new.chunk_id, new.chunk_text, new.heading_context);
END;

-- 4.4 Metadata Layer (Stage 3)

CREATE TABLE doc_metadata (
    doc_version_id TEXT PRIMARY KEY REFERENCES document_version(doc_version_id),

    title TEXT,
    title_span_start INTEGER,
    title_span_end INTEGER,
    title_source TEXT NOT NULL,             -- content_span|source_db_field(scrape_record.source_title)|html_meta|url|unknown
    title_confidence REAL,
    title_evidence_id TEXT REFERENCES evidence_span(evidence_id),

    published_at TIMESTAMP,
    published_at_raw TEXT,
    published_at_format TEXT,
    published_at_span_start INTEGER,
    published_at_span_end INTEGER,
    published_at_source TEXT NOT NULL,      -- content_span|source_db_field(scrape_record.source_published_at)|html_meta|url|unknown
    published_at_confidence REAL,
    published_at_evidence_id TEXT REFERENCES evidence_span(evidence_id),

    detected_document_class TEXT,
    document_class_confidence REAL,
    metadata_extraction_log TEXT,
    created_in_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- If source is content_span, evidence_id + span fields must be present.
    -- If source is NOT content_span, span fields + evidence_id must be NULL (avoid ambiguous provenance).
    CHECK (title_source = 'content_span'
           OR (title_evidence_id IS NULL AND title_span_start IS NULL AND title_span_end IS NULL)),
    CHECK (title_source <> 'content_span'
           OR (title_evidence_id IS NOT NULL AND title_span_start IS NOT NULL AND title_span_end IS NOT NULL)),
    CHECK (published_at_source = 'content_span'
           OR (published_at_evidence_id IS NULL AND published_at_span_start IS NULL AND published_at_span_end IS NULL)),
    CHECK (published_at_source <> 'content_span'
           OR (published_at_evidence_id IS NOT NULL AND published_at_span_start IS NOT NULL AND published_at_span_end IS NOT NULL))
 
);

CREATE INDEX idx_docmeta_published_at ON doc_metadata(published_at);

-- Enforce: when *_source='content_span', the referenced evidence_span must:
-- (a) exist, (b) belong to the same doc_version_id, and (c) match the stored span_start/span_end.
CREATE TRIGGER doc_metadata_validate_title_span_insert
BEFORE INSERT ON doc_metadata
WHEN NEW.title_source = 'content_span'
BEGIN
    SELECT CASE
        WHEN (SELECT COUNT(1) FROM evidence_span es WHERE es.evidence_id = NEW.title_evidence_id) = 0
        THEN RAISE(ABORT, 'doc_metadata.title_evidence_id must reference an existing evidence_span')
    END;
    SELECT CASE
        WHEN (SELECT es.doc_version_id FROM evidence_span es WHERE es.evidence_id = NEW.title_evidence_id) != NEW.doc_version_id
        THEN RAISE(ABORT, 'doc_metadata.title_evidence_id must belong to the same doc_version_id')
    END;
    SELECT CASE
        WHEN (SELECT es.span_start FROM evidence_span es WHERE es.evidence_id = NEW.title_evidence_id) != NEW.title_span_start
        THEN RAISE(ABORT, 'doc_metadata.title_span_start must match evidence_span.span_start')
    END;
    SELECT CASE
        WHEN (SELECT es.span_end FROM evidence_span es WHERE es.evidence_id = NEW.title_evidence_id) != NEW.title_span_end
        THEN RAISE(ABORT, 'doc_metadata.title_span_end must match evidence_span.span_end')
    END;
END;

CREATE TRIGGER doc_metadata_validate_title_span_update
BEFORE UPDATE OF title_source, title_evidence_id, title_span_start, title_span_end ON doc_metadata
WHEN NEW.title_source = 'content_span'
BEGIN
    SELECT CASE
        WHEN (SELECT COUNT(1) FROM evidence_span es WHERE es.evidence_id = NEW.title_evidence_id) = 0
        THEN RAISE(ABORT, 'doc_metadata.title_evidence_id must reference an existing evidence_span')
    END;
    SELECT CASE
        WHEN (SELECT es.doc_version_id FROM evidence_span es WHERE es.evidence_id = NEW.title_evidence_id) != NEW.doc_version_id
        THEN RAISE(ABORT, 'doc_metadata.title_evidence_id must belong to the same doc_version_id')
    END;
    SELECT CASE
        WHEN (SELECT es.span_start FROM evidence_span es WHERE es.evidence_id = NEW.title_evidence_id) != NEW.title_span_start
        THEN RAISE(ABORT, 'doc_metadata.title_span_start must match evidence_span.span_start')
    END;
    SELECT CASE
        WHEN (SELECT es.span_end FROM evidence_span es WHERE es.evidence_id = NEW.title_evidence_id) != NEW.title_span_end
        THEN RAISE(ABORT, 'doc_metadata.title_span_end must match evidence_span.span_end')
    END;
END;

CREATE TRIGGER doc_metadata_validate_published_at_span_insert
BEFORE INSERT ON doc_metadata
WHEN NEW.published_at_source = 'content_span'
BEGIN
    SELECT CASE
        WHEN (SELECT COUNT(1) FROM evidence_span es WHERE es.evidence_id = NEW.published_at_evidence_id) = 0
        THEN RAISE(ABORT, 'doc_metadata.published_at_evidence_id must reference an existing evidence_span')
    END;
    SELECT CASE
        WHEN (SELECT es.doc_version_id FROM evidence_span es WHERE es.evidence_id = NEW.published_at_evidence_id) != NEW.doc_version_id
        THEN RAISE(ABORT, 'doc_metadata.published_at_evidence_id must belong to the same doc_version_id')
    END;
    SELECT CASE
        WHEN (SELECT es.span_start FROM evidence_span es WHERE es.evidence_id = NEW.published_at_evidence_id) != NEW.published_at_span_start
        THEN RAISE(ABORT, 'doc_metadata.published_at_span_start must match evidence_span.span_start')
    END;
    SELECT CASE
        WHEN (SELECT es.span_end FROM evidence_span es WHERE es.evidence_id = NEW.published_at_evidence_id) != NEW.published_at_span_end
        THEN RAISE(ABORT, 'doc_metadata.published_at_span_end must match evidence_span.span_end')
    END;
END;

CREATE TRIGGER doc_metadata_validate_published_at_span_update
BEFORE UPDATE OF published_at_source, published_at_evidence_id, published_at_span_start, published_at_span_end ON doc_metadata
WHEN NEW.published_at_source = 'content_span'
BEGIN
    SELECT CASE
        WHEN (SELECT COUNT(1) FROM evidence_span es WHERE es.evidence_id = NEW.published_at_evidence_id) = 0
        THEN RAISE(ABORT, 'doc_metadata.published_at_evidence_id must reference an existing evidence_span')
    END;
    SELECT CASE
        WHEN (SELECT es.doc_version_id FROM evidence_span es WHERE es.evidence_id = NEW.published_at_evidence_id) != NEW.doc_version_id
        THEN RAISE(ABORT, 'doc_metadata.published_at_evidence_id must belong to the same doc_version_id')
    END;
    SELECT CASE
        WHEN (SELECT es.span_start FROM evidence_span es WHERE es.evidence_id = NEW.published_at_evidence_id) != NEW.published_at_span_start
        THEN RAISE(ABORT, 'doc_metadata.published_at_span_start must match evidence_span.span_start')
    END;
    SELECT CASE
        WHEN (SELECT es.span_end FROM evidence_span es WHERE es.evidence_id = NEW.published_at_evidence_id) != NEW.published_at_span_end
        THEN RAISE(ABORT, 'doc_metadata.published_at_span_end must match evidence_span.span_end')
    END;
END;

-- 4.5 Mention Layer (Stage 4)

CREATE TABLE entity_registry (
    entity_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,              -- ORG|TSO|REGULATOR|...
    canonical_name TEXT NOT NULL,
    aliases TEXT,                           -- JSON array
    name_variants_de TEXT,
    name_variants_en TEXT,
    abbreviations TEXT,
    compound_forms TEXT,
    valid_from DATE,
    valid_to DATE,
    source_authority TEXT,
    disambiguation_hints TEXT,              -- JSON
    parent_entity_id TEXT REFERENCES entity_registry(entity_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE entity_registry_audit (
    audit_id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL REFERENCES entity_registry(entity_id),
    change_type TEXT NOT NULL,
    old_value_json TEXT,
    new_value_json TEXT NOT NULL,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    changed_by TEXT,
    run_id TEXT REFERENCES pipeline_run(run_id),
    reason TEXT
);

CREATE TABLE mention (
    mention_id TEXT PRIMARY KEY,
    doc_version_id TEXT NOT NULL REFERENCES document_version(doc_version_id),
    chunk_ids TEXT,                         -- JSON array of chunk_id
    mention_type TEXT NOT NULL,             -- ORG|LEGAL_REF|GEO_*|...
    surface_form TEXT NOT NULL,             -- MUST equal clean_content[span_start:span_end] (validated in app + Stage 11)
    normalized_value TEXT,
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    confidence REAL NOT NULL,
    extraction_method TEXT,                 -- regex|gazetteer|...
    context_window_start INTEGER,
    context_window_end INTEGER,
    rejection_reason TEXT,
    metadata TEXT,                          -- JSON type-specific
    created_in_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (chunk_ids IS NULL OR (json_valid(chunk_ids) AND json_type(chunk_ids) = 'array'))
);
-- ID policy (recommended): mention.mention_id = SHA256(doc_version_id || '|' || span_start || '|' || span_end || '|' || mention_type || '|' || coalesce(normalized_value,'')) (hex)

-- Natural-key uniqueness (normative; prevents duplicate mention inserts)
CREATE UNIQUE INDEX idx_mention_natural_key
ON mention(doc_version_id, span_start, span_end, mention_type, IFNULL(normalized_value, ''));

-- Fail-fast span integrity for mentions (do not rely only on Stage 11).
CREATE TRIGGER mention_validate_insert
BEFORE INSERT ON mention
BEGIN
    SELECT CASE
      WHEN NEW.span_start < 0 OR NEW.span_end <= NEW.span_start
      THEN RAISE(ABORT, 'mention span must satisfy 0 <= start < end')
    END;
    SELECT CASE
      WHEN NEW.span_end > (
        SELECT length(dv.clean_content) FROM document_version dv WHERE dv.doc_version_id = NEW.doc_version_id
      )
      THEN RAISE(ABORT, 'mention span must be within clean_content bounds')
    END;
    SELECT CASE
      WHEN (
        SELECT substr(dv.clean_content, NEW.span_start + 1, NEW.span_end - NEW.span_start)
        FROM document_version dv WHERE dv.doc_version_id = NEW.doc_version_id
      ) != NEW.surface_form
      THEN RAISE(ABORT, 'mention.surface_form must equal clean_content[span_start:span_end]')
    END;
    -- Enforce: mention.chunk_ids (if provided) must reference existing chunks from the same doc_version_id.
    SELECT CASE
      WHEN NEW.chunk_ids IS NOT NULL
       AND (
         SELECT COUNT(1)
         FROM json_each(NEW.chunk_ids) j
         LEFT JOIN chunk c ON c.chunk_id = j.value
         WHERE c.chunk_id IS NULL OR c.doc_version_id != NEW.doc_version_id
       ) > 0
      THEN RAISE(ABORT, 'mention.chunk_ids must reference existing chunk rows from the same doc_version_id')
    END;
END;

CREATE TRIGGER mention_validate_update
BEFORE UPDATE OF span_start, span_end, surface_form, doc_version_id, chunk_ids ON mention
BEGIN
    SELECT CASE
      WHEN NEW.span_start < 0 OR NEW.span_end <= NEW.span_start
      THEN RAISE(ABORT, 'mention span must satisfy 0 <= start < end')
    END;
    SELECT CASE
      WHEN NEW.span_end > (
        SELECT length(dv.clean_content) FROM document_version dv WHERE dv.doc_version_id = NEW.doc_version_id
      )
      THEN RAISE(ABORT, 'mention span must be within clean_content bounds')
    END;
    SELECT CASE
      WHEN (
        SELECT substr(dv.clean_content, NEW.span_start + 1, NEW.span_end - NEW.span_start)
        FROM document_version dv WHERE dv.doc_version_id = NEW.doc_version_id
      ) != NEW.surface_form
      THEN RAISE(ABORT, 'mention.surface_form must equal clean_content[span_start:span_end]')
    END;
    -- Enforce: mention.chunk_ids (if provided) must reference existing chunks from the same doc_version_id.
    SELECT CASE
      WHEN NEW.chunk_ids IS NOT NULL
       AND (
         SELECT COUNT(1)
         FROM json_each(NEW.chunk_ids) j
         LEFT JOIN chunk c ON c.chunk_id = j.value
         WHERE c.chunk_id IS NULL OR c.doc_version_id != NEW.doc_version_id
       ) > 0
      THEN RAISE(ABORT, 'mention.chunk_ids must reference existing chunk rows from the same doc_version_id')
    END;
END;

CREATE INDEX idx_mention_docver ON mention(doc_version_id);
CREATE INDEX idx_mention_type ON mention(mention_type);

CREATE TABLE mention_link (
    link_id TEXT PRIMARY KEY,
    mention_id TEXT NOT NULL REFERENCES mention(mention_id),
    entity_id TEXT REFERENCES entity_registry(entity_id),
    link_confidence REAL,
    link_method TEXT,
    created_in_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE registry_update_proposal (
    proposal_id TEXT PRIMARY KEY,
    surface_form TEXT NOT NULL,
    proposal_type TEXT NOT NULL,
    target_entity_id TEXT,
    inferred_type TEXT,
    evidence_doc_ids TEXT NOT NULL,         -- JSON array of doc_version_id (high-level pointer)
    occurrence_count INTEGER DEFAULT 1,
    status TEXT DEFAULT 'pending',
    review_notes TEXT,
    created_in_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (json_valid(evidence_doc_ids) AND json_type(evidence_doc_ids) = 'array' AND json_array_length(evidence_doc_ids) > 0)
);

-- 4.6 Embedding Layer (Stage 5)

CREATE TABLE chunk_embedding (
    chunk_id TEXT PRIMARY KEY REFERENCES chunk(chunk_id),
    embedding_vector BLOB NOT NULL,         -- serialized float32 array
    embedding_dim INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    language_used TEXT,
    created_in_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_embedding_model ON chunk_embedding(model_version);

CREATE TABLE embedding_index (
    index_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    model_version TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    method TEXT NOT NULL,                   -- hnswlib|faiss
    index_path TEXT NOT NULL,
    built_at TIMESTAMP NOT NULL,
    chunk_count INTEGER NOT NULL,
    build_params_json TEXT,
    checksum TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_embedding_index_run_model ON embedding_index(run_id, model_version);

-- 4.7 Taxonomy Layer (Stage 6)

CREATE TABLE facet_assignment (
    facet_id TEXT PRIMARY KEY,
    doc_version_id TEXT NOT NULL REFERENCES document_version(doc_version_id),
    facet_type TEXT NOT NULL,               -- topic|document_type|urgency
    facet_value TEXT NOT NULL,
    confidence REAL NOT NULL,
    signals_json TEXT,                      -- JSON signals/features
    created_in_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (signals_json IS NULL OR json_valid(signals_json))
);

CREATE INDEX idx_facet_docver ON facet_assignment(doc_version_id);

CREATE TABLE facet_assignment_evidence (
    facet_id TEXT NOT NULL REFERENCES facet_assignment(facet_id),
    evidence_id TEXT NOT NULL REFERENCES evidence_span(evidence_id),
    purpose TEXT NOT NULL,                  -- classifier signal label (required; part of PK)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (facet_id, evidence_id, purpose)
);

-- Evidence for facet assignments must belong to the same document.
CREATE TRIGGER facet_assignment_evidence_validate_doc
BEFORE INSERT ON facet_assignment_evidence
BEGIN
    SELECT CASE
      WHEN (
        SELECT fa.doc_version_id FROM facet_assignment fa WHERE fa.facet_id = NEW.facet_id
      ) != (
        SELECT es.doc_version_id FROM evidence_span es WHERE es.evidence_id = NEW.evidence_id
      )
      THEN RAISE(ABORT, 'facet_assignment_evidence.evidence_id must belong to facet_assignment.doc_version_id')
    END;
END;

-- 4.8 Novelty Layer (Stage 7)

CREATE TABLE novelty_label (
    doc_version_id TEXT PRIMARY KEY REFERENCES document_version(doc_version_id),
    label TEXT NOT NULL,                    -- NEW|UNCHANGED|UPDATE_*|RE_REPORT|...
    neighbor_doc_version_ids TEXT,          -- JSON array of doc_version_id
    similarity_score REAL,
    shared_mentions TEXT,                   -- JSON
    linking_window_days INTEGER,
    confidence REAL NOT NULL,
    created_in_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (neighbor_doc_version_ids IS NULL OR (json_valid(neighbor_doc_version_ids) AND json_type(neighbor_doc_version_ids) = 'array')),
    CHECK (shared_mentions IS NULL OR json_valid(shared_mentions))
);

CREATE TABLE novelty_label_evidence (
    doc_version_id TEXT NOT NULL REFERENCES novelty_label(doc_version_id),
    evidence_id TEXT NOT NULL REFERENCES evidence_span(evidence_id),
    purpose TEXT NOT NULL,                  -- diff_summary|changed_section|verbatim_reuse|...
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (doc_version_id, evidence_id, purpose)
);

-- Evidence for novelty labels must belong to the same document.
CREATE TRIGGER novelty_label_evidence_validate_doc
BEFORE INSERT ON novelty_label_evidence
BEGIN
    SELECT CASE
      WHEN NEW.doc_version_id != (
        SELECT es.doc_version_id FROM evidence_span es WHERE es.evidence_id = NEW.evidence_id
      )
      THEN RAISE(ABORT, 'novelty_label_evidence.evidence_id must belong to novelty_label.doc_version_id')
    END;
END;


CREATE TABLE chunk_novelty (
    chunk_id TEXT PRIMARY KEY REFERENCES chunk(chunk_id),
    novelty_label TEXT NOT NULL,            -- NEW|VERBATIM_REUSE|PARAPHRASE|BOILERPLATE
    source_chunk_ids TEXT,                  -- JSON array
    similarity_scores TEXT,                 -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chunk_novelty_score (
    chunk_id TEXT PRIMARY KEY REFERENCES chunk(chunk_id),
    novelty_score REAL NOT NULL,            -- 0..1
    method TEXT NOT NULL,                   -- diff|minhash|embedding
    created_in_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE document_fingerprint (
    doc_version_id TEXT PRIMARY KEY REFERENCES document_version(doc_version_id),
    minhash_signature BLOB NOT NULL,
    simhash_signature BLOB,
    created_in_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE story_cluster (
    run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    story_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cluster_method TEXT,
    seed_doc_version_id TEXT REFERENCES document_version(doc_version_id),
    summary_text TEXT,
    PRIMARY KEY (run_id, story_id)
);

CREATE TABLE story_cluster_member (
    run_id TEXT NOT NULL,
    story_id TEXT NOT NULL,
    doc_version_id TEXT NOT NULL REFERENCES document_version(doc_version_id),
    score REAL,
    role TEXT,
    PRIMARY KEY (run_id, story_id, doc_version_id),
    FOREIGN KEY (run_id, story_id) REFERENCES story_cluster(run_id, story_id)
);

-- 4.9 Event Layer (Stage 8)

-- Stable event identity
CREATE TABLE event (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    canonical_key TEXT NOT NULL,            -- MUST include event_type (e.g., "OUTAGE::DE::FiftyHertz::facility_x::2026-01")
    current_revision_id TEXT,               -- set by Stage 8 after first revision insert (NULL allowed only during creation)
    created_in_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(canonical_key)
);
-- ID policy (recommended): event.event_id = SHA256(canonical_key) (hex)

-- Stage 8 write sequence (required):
-- 1) INSERT event with current_revision_id = NULL
-- 2) INSERT first event_revision
-- 3) UPDATE event.current_revision_id to that revision_id (**MUST be the same DB transaction**)

-- Append-only event revisions
CREATE TABLE event_revision (
    revision_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES event(event_id),
    revision_no INTEGER NOT NULL,

    slots_json TEXT NOT NULL,               -- structured slots (dates, values, actors, etc.)
    doc_version_ids TEXT NOT NULL,          -- JSON array of doc_version_id contributing to revision
    confidence REAL NOT NULL,
    extraction_method TEXT,
    extraction_tier INTEGER,                -- 1=rule, 2=heuristic, 3=LLM

    supersedes_revision_id TEXT REFERENCES event_revision(revision_id),
    created_in_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(event_id, revision_no),
    CHECK (json_valid(slots_json) AND json_type(slots_json) = 'object'),
    CHECK (json_valid(doc_version_ids) AND json_type(doc_version_ids) = 'array' AND json_array_length(doc_version_ids) > 0)
 
);

-- Enforce: all doc_version_ids in the JSON array must reference existing document_version rows.
CREATE TRIGGER event_revision_validate_doc_refs
BEFORE INSERT ON event_revision
BEGIN
    SELECT CASE
      WHEN (
        SELECT COUNT(1)
        FROM json_each(NEW.doc_version_ids) j
        LEFT JOIN document_version dv ON dv.doc_version_id = j.value
        WHERE dv.doc_version_id IS NULL
      ) > 0
      THEN RAISE(ABORT, 'event_revision.doc_version_ids contains invalid doc_version_id reference')
    END;
END;

CREATE INDEX idx_revision_event ON event_revision(event_id);

-- Enforce append-only for event_revision
CREATE TRIGGER event_revision_no_update
BEFORE UPDATE ON event_revision
BEGIN
    SELECT RAISE(ABORT, 'event_revision is append-only; updates are not allowed');
END;

CREATE TRIGGER event_revision_no_delete
BEFORE DELETE ON event_revision
BEGIN
    SELECT RAISE(ABORT, 'event_revision is append-only; deletes are not allowed');
END;

-- Enforce: event.current_revision_id must exist and belong to same event_id (INSERT)
-- NOTE: Defined AFTER event_revision so schema creation is valid in SQLite.
CREATE TRIGGER event_current_revision_validate_insert
BEFORE INSERT ON event
WHEN NEW.current_revision_id IS NOT NULL
BEGIN
    -- Must exist
    SELECT CASE
        WHEN (SELECT COUNT(1) FROM event_revision er WHERE er.revision_id = NEW.current_revision_id) = 0
        THEN RAISE(ABORT, 'current_revision_id must reference an existing event_revision')
    END;
    -- Must belong to same event_id
    SELECT CASE
        WHEN (SELECT er.event_id FROM event_revision er WHERE er.revision_id = NEW.current_revision_id) != NEW.event_id
        THEN RAISE(ABORT, 'current_revision_id must reference a revision of the same event_id')
    END;
END;

-- Enforce: event.current_revision_id must exist and belong to same event_id
CREATE TRIGGER event_current_revision_validate
BEFORE UPDATE OF current_revision_id ON event
WHEN NEW.current_revision_id IS NOT NULL
BEGIN
    -- Must exist
    SELECT
      CASE
        WHEN (SELECT COUNT(1) FROM event_revision er WHERE er.revision_id = NEW.current_revision_id) = 0
        THEN RAISE(ABORT, 'current_revision_id must reference an existing event_revision')
      END;

    -- Must belong to same event_id
    SELECT
      CASE
        WHEN (SELECT er.event_id FROM event_revision er WHERE er.revision_id = NEW.current_revision_id) != NEW.event_id
        THEN RAISE(ABORT, 'current_revision_id must reference a revision of the same event_id')
      END;
END;

-- Prevent setting current_revision_id to a revision that is not the latest revision_no (optional safety)
-- (If undesired, remove this trigger.)
CREATE TRIGGER event_current_revision_must_be_max_revision_no
BEFORE UPDATE OF current_revision_id ON event
WHEN NEW.current_revision_id IS NOT NULL
BEGIN
    SELECT
      CASE
        WHEN (
          SELECT er.revision_no FROM event_revision er WHERE er.revision_id = NEW.current_revision_id
        ) != (
          SELECT MAX(er2.revision_no) FROM event_revision er2 WHERE er2.event_id = NEW.event_id
        )
        THEN RAISE(ABORT, 'current_revision_id must point to latest revision_no for the event')
      END;
END;
-- NOTE: This trigger enforces that current_revision_id always points to the highest revision_no.
-- Per-document atomicity (§6.4) ensures that event + revision + evidence writes either all commit
-- or all rollback; orphaned revisions cannot occur under normal operation. Stage 11 validates
-- that no orphans exist. If orphans are detected (indicating prior crash/corruption), manual
-- remediation is required before the pipeline can proceed.

CREATE TABLE event_revision_evidence (
    revision_id TEXT NOT NULL REFERENCES event_revision(revision_id),
    evidence_id TEXT NOT NULL REFERENCES evidence_span(evidence_id),
    purpose TEXT NOT NULL,                  -- slot:deadline|slot:effective_date|summary|justification|...
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (revision_id, evidence_id, purpose)
);
-- ID policy (recommended): event_revision.revision_id = SHA256(event_id || '|' || revision_no) (hex)

-- Evidence for an event revision must come from one of the revision's contributing documents.
CREATE TRIGGER event_revision_evidence_validate_docset
BEFORE INSERT ON event_revision_evidence
BEGIN
    SELECT CASE
      WHEN (
        SELECT COUNT(1)
        FROM event_revision er
        JOIN evidence_span es ON es.evidence_id = NEW.evidence_id
        JOIN json_each(er.doc_version_ids) j
        WHERE er.revision_id = NEW.revision_id
          AND json_valid(er.doc_version_ids)
          AND j.value = es.doc_version_id
      ) = 0
      THEN RAISE(ABORT, 'event_revision_evidence.evidence_id must refer to a doc_version_id listed in event_revision.doc_version_ids')
    END;
END;

CREATE TABLE event_entity_link (
    revision_id TEXT NOT NULL REFERENCES event_revision(revision_id),
    entity_id TEXT NOT NULL REFERENCES entity_registry(entity_id),
    role TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_in_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (revision_id, entity_id, role)
);
-- Revision-scoped (normative): links attach to a specific event_revision.
-- When a new event_revision is created, Stage 8 MUST insert the corresponding event_entity_link rows for that revision.
-- Links are never updated; they evolve by adding links on newer revisions.

CREATE TABLE metric_observation (
    metric_id TEXT PRIMARY KEY,
    doc_version_id TEXT NOT NULL REFERENCES document_version(doc_version_id),
    table_id TEXT REFERENCES table_extract(table_id),

    metric_name TEXT NOT NULL,
    value_raw TEXT NOT NULL,
    unit_raw TEXT,
    value_norm REAL,
    unit_norm TEXT,

    period_start DATE,
    period_end DATE,
    period_granularity TEXT,
    geography TEXT,

    table_row_index INTEGER,
    table_col_index INTEGER,

    evidence_id TEXT REFERENCES evidence_span(evidence_id),
    parse_quality REAL,

    created_in_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- ID policy (recommended): metric_observation.metric_id = SHA256(doc_version_id || '|' || coalesce(table_id,'') || '|' || metric_name || '|' || coalesce(period_start,'') || '|' || coalesce(period_end,'') || '|' || coalesce(geography,'') || '|' || coalesce(table_row_index,'') || '|' || coalesce(table_col_index,'')) (hex)

-- Natural-key uniqueness (recommended; prevents duplicate inserts on retry/bugs)
CREATE UNIQUE INDEX idx_metric_obs_natural_key
ON metric_observation(
  doc_version_id,
  IFNULL(table_id,''),
  metric_name,
  IFNULL(period_start,''),
  IFNULL(period_end,''),
  IFNULL(geography,''),
  IFNULL(table_row_index,-1),
  IFNULL(table_col_index,-1)
);

-- Concrete provenance integrity (recommended; aligns with surfaced-evidence rules):
-- If metric_observation.evidence_id is present, it must belong to metric_observation.doc_version_id.
CREATE TRIGGER metric_observation_validate_evidence_doc
BEFORE INSERT ON metric_observation
WHEN NEW.evidence_id IS NOT NULL
BEGIN
    SELECT CASE
      WHEN NEW.doc_version_id != (
        SELECT es.doc_version_id FROM evidence_span es WHERE es.evidence_id = NEW.evidence_id
      )
      THEN RAISE(ABORT, 'metric_observation.evidence_id must belong to metric_observation.doc_version_id')
    END;
END;

CREATE TRIGGER metric_observation_validate_evidence_doc_update
BEFORE UPDATE OF evidence_id, doc_version_id ON metric_observation
WHEN NEW.evidence_id IS NOT NULL
BEGIN
    SELECT CASE
      WHEN NEW.doc_version_id != (
        SELECT es.doc_version_id FROM evidence_span es WHERE es.evidence_id = NEW.evidence_id
      )
      THEN RAISE(ABORT, 'metric_observation.evidence_id must belong to metric_observation.doc_version_id')
    END;
END;

CREATE INDEX idx_metric_obs_docver ON metric_observation(doc_version_id);
CREATE INDEX idx_metric_obs_name ON metric_observation(metric_name);

CREATE TABLE event_candidate (
    candidate_id TEXT PRIMARY KEY,
    doc_version_id TEXT NOT NULL REFERENCES document_version(doc_version_id),
    event_type TEXT NOT NULL,
    partial_slots TEXT,
    confidence REAL,
    status TEXT NOT NULL DEFAULT 'candidate',  -- candidate|rejected|promoted
    rejection_reason TEXT,
    extraction_tier INTEGER,
    created_in_run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (
      (status = 'rejected' AND rejection_reason IS NOT NULL)
      OR (status <> 'rejected' AND rejection_reason IS NULL)
    )
);

-- 4.10 Output Layer (Stage 9)

CREATE TABLE metric_series (
    run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    series_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    geography TEXT,
    period_granularity TEXT NOT NULL,
    unit_norm TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, series_id),
    UNIQUE(run_id, metric_name, geography, period_granularity, unit_norm)
);

CREATE TABLE metric_series_point (
    run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),

    series_id TEXT NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE,
    value_norm REAL NOT NULL,

    source_doc_version_id TEXT NOT NULL REFERENCES document_version(doc_version_id),
    -- Surfaced record: evidence is mandatory. Stage 9 must fail-closed (skip) if no evidence is available.
    evidence_id TEXT NOT NULL REFERENCES evidence_span(evidence_id),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Allow multiple contributing documents for the same series/period (no silent overwrite).
    PRIMARY KEY (run_id, series_id, period_start, source_doc_version_id),
    FOREIGN KEY (run_id, series_id) REFERENCES metric_series(run_id, series_id)
);

CREATE INDEX idx_metric_series_point_run ON metric_series_point(run_id, series_id, period_start, source_doc_version_id);

-- Concrete provenance integrity: metric_series_point evidence must belong to source_doc_version_id.
CREATE TRIGGER metric_series_point_validate_evidence_doc
BEFORE INSERT ON metric_series_point
BEGIN
    SELECT CASE
      WHEN NEW.source_doc_version_id != (
        SELECT es.doc_version_id FROM evidence_span es WHERE es.evidence_id = NEW.evidence_id
      )
      THEN RAISE(ABORT, 'metric_series_point.evidence_id must belong to metric_series_point.source_doc_version_id')
    END;
END;

CREATE TRIGGER metric_series_point_validate_evidence_doc_update
BEFORE UPDATE OF evidence_id, source_doc_version_id ON metric_series_point
BEGIN
    SELECT CASE
      WHEN NEW.source_doc_version_id != (
        SELECT es.doc_version_id FROM evidence_span es WHERE es.evidence_id = NEW.evidence_id
      )
      THEN RAISE(ABORT, 'metric_series_point.evidence_id must belong to metric_series_point.source_doc_version_id')
    END;
END;


CREATE TABLE watchlist (
    watchlist_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_values TEXT NOT NULL,            -- JSON array or DSL (implementation-defined)
    track_events TEXT,
    track_topics TEXT,
    alert_severity TEXT DEFAULT 'info',
    active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE alert_rule (
    rule_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    conditions_json TEXT NOT NULL,
    severity TEXT NOT NULL,
    suppression_window_hours INTEGER DEFAULT 24,
    active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (json_valid(conditions_json) AND json_type(conditions_json) IN ('object','array'))
);

CREATE TABLE alert (
    alert_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    rule_id TEXT REFERENCES alert_rule(rule_id),
    triggered_at TIMESTAMP NOT NULL,
    trigger_payload_json TEXT NOT NULL,     -- non-span trigger details
    doc_version_ids TEXT NOT NULL,          -- JSON array of doc_version_id
    event_ids TEXT,                         -- JSON array of event_id (or NULL)
    acknowledged INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (json_valid(trigger_payload_json) AND json_type(trigger_payload_json) = 'object'),
    CHECK (json_valid(doc_version_ids) AND json_type(doc_version_ids) = 'array' AND json_array_length(doc_version_ids) > 0)
);

-- Enforce: all doc_version_ids in the JSON array must reference existing document_version rows.
CREATE TRIGGER alert_validate_doc_refs
BEFORE INSERT ON alert
BEGIN
    SELECT CASE
      WHEN (
        SELECT COUNT(1)
        FROM json_each(NEW.doc_version_ids) j
        LEFT JOIN document_version dv ON dv.doc_version_id = j.value
        WHERE dv.doc_version_id IS NULL
      ) > 0
      THEN RAISE(ABORT, 'alert.doc_version_ids contains invalid doc_version_id reference')
    END;
END;

CREATE INDEX idx_alert_run ON alert(run_id, triggered_at);

CREATE TABLE alert_evidence (
    alert_id TEXT NOT NULL REFERENCES alert(alert_id),
    evidence_id TEXT NOT NULL REFERENCES evidence_span(evidence_id),
    purpose TEXT NOT NULL,                  -- rule_match|metric_point|event_support|...
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (alert_id, evidence_id, purpose)
);
-- INVARIANT (application-enforced, validated by Stage 11):
-- Every alert row MUST have at least one alert_evidence row.
-- SQLite lacks deferred constraints; Stage 9 MUST insert alert + evidence atomically per §6.4.

-- Evidence for an alert must come from one of the alert's contributing documents.
CREATE TRIGGER alert_evidence_validate_docset
BEFORE INSERT ON alert_evidence
BEGIN
    SELECT CASE
      WHEN (
        SELECT COUNT(1)
        FROM alert a
        JOIN evidence_span es ON es.evidence_id = NEW.evidence_id
        JOIN json_each(a.doc_version_ids) j
        WHERE a.alert_id = NEW.alert_id
          AND j.value = es.doc_version_id
      ) = 0
      THEN RAISE(ABORT, 'alert_evidence.evidence_id must refer to a doc_version_id listed in alert.doc_version_ids')
    END;
END;

CREATE TABLE digest_item (
    item_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    digest_date DATE NOT NULL,
    section TEXT NOT NULL,
    item_type TEXT NOT NULL,
    doc_version_ids TEXT NOT NULL,          -- JSON array of contributing doc_version_id (required for evidence enforcement)
    payload_json TEXT NOT NULL,             -- structured output payload
    event_ids TEXT,                         -- JSON array of event_id (optional)
    novelty_label TEXT,
    language_original TEXT,
    language_presented TEXT,
    translation_status TEXT,
    translation_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (json_valid(payload_json) AND json_type(payload_json) = 'object'),
    CHECK (json_valid(doc_version_ids) AND json_type(doc_version_ids) = 'array' AND json_array_length(doc_version_ids) > 0),
    CHECK (event_ids IS NULL OR (json_valid(event_ids) AND json_type(event_ids) = 'array'))
);

-- Enforce: all doc_version_ids in the JSON array must reference existing document_version rows.
CREATE TRIGGER digest_item_validate_doc_refs
BEFORE INSERT ON digest_item
BEGIN
    SELECT CASE
      WHEN (
        SELECT COUNT(1)
        FROM json_each(NEW.doc_version_ids) j
        LEFT JOIN document_version dv ON dv.doc_version_id = j.value
        WHERE dv.doc_version_id IS NULL
      ) > 0
      THEN RAISE(ABORT, 'digest_item.doc_version_ids contains invalid doc_version_id reference')
    END;
END;

CREATE INDEX idx_digest_item_run ON digest_item(run_id, digest_date);

CREATE TABLE digest_item_evidence (
    item_id TEXT NOT NULL REFERENCES digest_item(item_id),
    evidence_id TEXT NOT NULL REFERENCES evidence_span(evidence_id),
    purpose TEXT NOT NULL,                  -- headline_quote|supporting_quote|metric_support|...
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (item_id, evidence_id, purpose)
);
-- INVARIANT (application-enforced, validated by Stage 11):
-- Every digest_item row MUST have at least one digest_item_evidence row.
-- SQLite lacks deferred constraints; Stage 9 MUST insert digest_item + evidence atomically per §6.4.

-- Evidence for a digest item must come from one of the digest item's contributing documents.
CREATE TRIGGER digest_item_evidence_validate_docset
BEFORE INSERT ON digest_item_evidence
BEGIN
    SELECT CASE
      WHEN (
        SELECT COUNT(1)
        FROM digest_item di
        JOIN evidence_span es ON es.evidence_id = NEW.evidence_id
        JOIN json_each(di.doc_version_ids) j
        WHERE di.item_id = NEW.item_id
          AND json_valid(di.doc_version_ids)
          AND j.value = es.doc_version_id
      ) = 0
      THEN RAISE(ABORT, 'digest_item_evidence.evidence_id must refer to a doc_version_id listed in digest_item.doc_version_ids')
    END;
END;

-- 4.11 Timeline Layer (Stage 10)

CREATE TABLE entity_timeline_item (
    item_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES pipeline_run(run_id),
    entity_id TEXT NOT NULL REFERENCES entity_registry(entity_id),
    
    item_type TEXT NOT NULL,                -- event_revision|mention|doc
    category TEXT NOT NULL,                 -- for dedup: event_type OR mention_type-class OR doc_class

    ref_revision_id TEXT REFERENCES event_revision(revision_id),
    ref_mention_id TEXT REFERENCES mention(mention_id),
    ref_doc_version_id TEXT REFERENCES document_version(doc_version_id),

    event_time TIMESTAMP,
    time_source TEXT,                       -- slot_effective_date|slot_deadline|doc_published_at|doc_fallback
    summary_text TEXT,

    context_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (context_json IS NULL OR json_valid(context_json)),

    CHECK (
      (ref_revision_id IS NOT NULL) +
      (ref_mention_id IS NOT NULL) +
      (ref_doc_version_id IS NOT NULL) = 1
    )
);

CREATE INDEX idx_timeline_entity_time ON entity_timeline_item(run_id, entity_id, event_time);
CREATE INDEX idx_timeline_dedup ON entity_timeline_item(run_id, entity_id, category, event_time);

CREATE TABLE entity_timeline_item_evidence (
    item_id TEXT NOT NULL REFERENCES entity_timeline_item(item_id),
    evidence_id TEXT NOT NULL REFERENCES evidence_span(evidence_id),
    purpose TEXT NOT NULL,                  -- summary_quote|time_support|supporting_quote|...
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (item_id, evidence_id, purpose)
);
-- 4.X Run-scoped immutability enforcement (UPDATE forbidden; delete -> rebuild remains allowed)
CREATE TRIGGER embedding_index_no_update BEFORE UPDATE ON embedding_index
BEGIN SELECT RAISE(ABORT, 'run-scoped table embedding_index is immutable; updates are not allowed'); END;
CREATE TRIGGER story_cluster_no_update BEFORE UPDATE ON story_cluster
BEGIN SELECT RAISE(ABORT, 'run-scoped table story_cluster is immutable; updates are not allowed'); END;
CREATE TRIGGER story_cluster_member_no_update BEFORE UPDATE ON story_cluster_member
BEGIN SELECT RAISE(ABORT, 'run-scoped table story_cluster_member is immutable; updates are not allowed'); END;

CREATE TRIGGER metric_series_no_update BEFORE UPDATE ON metric_series
BEGIN SELECT RAISE(ABORT, 'run-scoped table metric_series is immutable; updates are not allowed'); END;
CREATE TRIGGER metric_series_point_no_update BEFORE UPDATE ON metric_series_point
BEGIN SELECT RAISE(ABORT, 'run-scoped table metric_series_point is immutable; updates are not allowed'); END;

CREATE TRIGGER alert_no_update BEFORE UPDATE ON alert
BEGIN SELECT RAISE(ABORT, 'run-scoped table alert is immutable; updates are not allowed'); END;
CREATE TRIGGER alert_evidence_no_update BEFORE UPDATE ON alert_evidence
BEGIN SELECT RAISE(ABORT, 'run-scoped table alert_evidence is immutable; updates are not allowed'); END;

CREATE TRIGGER digest_item_no_update BEFORE UPDATE ON digest_item
BEGIN SELECT RAISE(ABORT, 'run-scoped table digest_item is immutable; updates are not allowed'); END;
CREATE TRIGGER digest_item_evidence_no_update BEFORE UPDATE ON digest_item_evidence
BEGIN SELECT RAISE(ABORT, 'run-scoped table digest_item_evidence is immutable; updates are not allowed'); END;

CREATE TRIGGER entity_timeline_item_no_update BEFORE UPDATE ON entity_timeline_item
BEGIN SELECT RAISE(ABORT, 'run-scoped table entity_timeline_item is immutable; updates are not allowed'); END;
CREATE TRIGGER entity_timeline_item_evidence_no_update BEFORE UPDATE ON entity_timeline_item_evidence
BEGIN SELECT RAISE(ABORT, 'run-scoped table entity_timeline_item_evidence is immutable; updates are not allowed'); END;
                                                                                                                    
-- INVARIANT (application-enforced, validated by Stage 11):
-- Every entity_timeline_item row MUST have at least one entity_timeline_item_evidence row.
-- SQLite lacks deferred constraints; Stage 10 MUST insert timeline_item + evidence atomically per §6.4.

-- Evidence for a timeline item must be compatible with its reference:
--   - ref_doc_version_id: evidence must belong to that doc_version_id
--   - ref_mention_id: evidence must belong to mention.doc_version_id
--   - ref_revision_id: evidence must belong to one of event_revision.doc_version_ids
CREATE TRIGGER entity_timeline_item_evidence_validate
BEFORE INSERT ON entity_timeline_item_evidence
BEGIN
    -- Case 1: item references a document directly
    SELECT CASE
      WHEN (
        SELECT eti.ref_doc_version_id FROM entity_timeline_item eti WHERE eti.item_id = NEW.item_id
      ) IS NOT NULL
      AND (
        SELECT eti.ref_doc_version_id FROM entity_timeline_item eti WHERE eti.item_id = NEW.item_id
      ) != (
        SELECT es.doc_version_id FROM evidence_span es WHERE es.evidence_id = NEW.evidence_id
      )
      THEN RAISE(ABORT, 'timeline evidence must belong to ref_doc_version_id')
    END;

    -- Case 2: item references a mention
    SELECT CASE
      WHEN (
        SELECT eti.ref_mention_id FROM entity_timeline_item eti WHERE eti.item_id = NEW.item_id
      ) IS NOT NULL
      AND (
        SELECT m.doc_version_id FROM mention m
        WHERE m.mention_id = (SELECT eti.ref_mention_id FROM entity_timeline_item eti WHERE eti.item_id = NEW.item_id)
      ) != (
        SELECT es.doc_version_id FROM evidence_span es WHERE es.evidence_id = NEW.evidence_id
      )
      THEN RAISE(ABORT, 'timeline evidence must belong to mention.doc_version_id')
    END;

    -- Case 3: item references an event revision
    SELECT CASE
      WHEN (
        SELECT eti.ref_revision_id FROM entity_timeline_item eti WHERE eti.item_id = NEW.item_id
      ) IS NOT NULL
      AND (
        SELECT COUNT(1)
        FROM entity_timeline_item eti
        JOIN event_revision er ON er.revision_id = eti.ref_revision_id
        JOIN evidence_span es ON es.evidence_id = NEW.evidence_id
        JOIN json_each(er.doc_version_ids) j
        WHERE eti.item_id = NEW.item_id
          AND json_valid(er.doc_version_ids)
          AND j.value = es.doc_version_id
      ) = 0
      THEN RAISE(ABORT, 'timeline evidence for ref_revision_id must belong to one of event_revision.doc_version_ids')
    END;
END;