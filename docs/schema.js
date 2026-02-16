/**
 * schema.js — Runtime validation for external JSON data.
 *
 * Every fetch boundary passes through these validators before data
 * enters the app state. Invalid data throws with actionable messages.
 *
 * @module schema
 */

/** @typedef {'ok'|'failed'|'blocked'|'skipped'|'pending'|'not_started'} DocStatus */
/** @typedef {'ok'|'failed'|'pending'} RunStatus */
/** @typedef {'doc'|'run'} StageScope */
/** @typedef {'green'|'yellow'|'red'|'gray'} ScrapeStatus */

const DOC_STATUSES = new Set(['ok', 'failed', 'blocked', 'skipped', 'pending']);
const RUN_STATUSES = new Set(['ok', 'failed', 'pending']);
const SCOPES = new Set(['doc', 'run']);

class SchemaError extends Error {
  /** @param {string} path @param {string} detail */
  constructor(path, detail) {
    super(`Schema validation failed at ${path}: ${detail}`);
    this.name = 'SchemaError';
    this.path = path;
  }
}

/**
 * @param {unknown} val
 * @param {string} path
 * @param {string} type
 * @returns {asserts val is NonNullable<unknown>}
 */
function assertType(val, path, type) {
  if (type === 'array') {
    if (!Array.isArray(val)) throw new SchemaError(path, `expected array, got ${typeof val}`);
    return;
  }
  if (typeof val !== type) {
    throw new SchemaError(path, `expected ${type}, got ${typeof val}`);
  }
}

/**
 * @param {unknown} val
 * @param {string} path
 * @returns {asserts val is string}
 */
function assertString(val, path) { assertType(val, path, 'string'); }

/**
 * @param {unknown} val
 * @param {string} path
 * @returns {asserts val is object}
 */
function assertObject(val, path) {
  if (val === null || typeof val !== 'object' || Array.isArray(val)) {
    throw new SchemaError(path, `expected object, got ${val === null ? 'null' : typeof val}`);
  }
}

// ── ETL validators ──────────────────────────────────────────────────

/**
 * Validate meta.json structure.
 * @param {unknown} raw
 * @returns {{ export_version: string, generated_at: string, stages: Array, runs: Array, publishers: Array }}
 */
export function validateMeta(raw) {
  assertObject(raw, 'meta');
  const meta = /** @type {Record<string, unknown>} */(raw);

  assertString(meta.export_version, 'meta.export_version');
  assertString(meta.generated_at, 'meta.generated_at');
  assertType(meta.stages, 'meta.stages', 'array');
  assertType(meta.runs, 'meta.runs', 'array');
  assertType(meta.publishers, 'meta.publishers', 'array');

  for (let i = 0; i < /** @type {Array} */(meta.stages).length; i++) {
    const s = /** @type {Record<string, unknown>} */(/** @type {Array} */(meta.stages)[i]);
    assertString(s.stage_id, `meta.stages[${i}].stage_id`);
    if (!SCOPES.has(/** @type {string} */(s.scope))) {
      throw new SchemaError(`meta.stages[${i}].scope`, `expected doc|run, got ${s.scope}`);
    }
  }

  for (let i = 0; i < /** @type {Array} */(meta.runs).length; i++) {
    const r = /** @type {Record<string, unknown>} */(/** @type {Array} */(meta.runs)[i]);
    assertString(r.run_id, `meta.runs[${i}].run_id`);
    assertString(r.status, `meta.runs[${i}].status`);
  }

  return /** @type {any} */(meta);
}

/**
 * Validate docs.json structure.
 * @param {unknown} raw
 * @returns {{ docs: Array, stage_status_by_doc: Array, totals_by_doc: Array }}
 */
export function validateDocs(raw) {
  assertObject(raw, 'docs');
  const data = /** @type {Record<string, unknown>} */(raw);

  assertType(data.docs, 'docs.docs', 'array');
  assertType(data.stage_status_by_doc, 'docs.stage_status_by_doc', 'array');
  assertType(data.totals_by_doc, 'docs.totals_by_doc', 'array');

  const docs = /** @type {Array} */(data.docs);
  if (docs.length > 0) {
    const d = docs[0];
    assertString(d.doc_version_id, 'docs.docs[0].doc_version_id');
    assertString(d.publisher_id, 'docs.docs[0].publisher_id');
  }

  // Length alignment check
  if (docs.length !== /** @type {Array} */(data.stage_status_by_doc).length) {
    throw new SchemaError('docs', 'docs and stage_status_by_doc length mismatch');
  }
  if (docs.length !== /** @type {Array} */(data.totals_by_doc).length) {
    throw new SchemaError('docs', 'docs and totals_by_doc length mismatch');
  }

  return /** @type {any} */(data);
}

/**
 * Validate runs/<run_id>.json structure.
 * @param {unknown} raw
 * @returns {{ run_id: string, run_stage_status: Array, new_counts_by_doc: Array, run_artifact_counts: object }}
 */
export function validateRunData(raw) {
  assertObject(raw, 'runData');
  const data = /** @type {Record<string, unknown>} */(raw);

  assertString(data.run_id, 'runData.run_id');
  assertType(data.run_stage_status, 'runData.run_stage_status', 'array');
  assertType(data.new_counts_by_doc, 'runData.new_counts_by_doc', 'array');
  assertObject(data.run_artifact_counts, 'runData.run_artifact_counts');

  return /** @type {any} */(data);
}

/**
 * Validate trace/<doc_version_id>.json structure.
 * @param {unknown} raw
 * @returns {{ header: object, stage_samples: object }}
 */
export function validateTrace(raw) {
  assertObject(raw, 'trace');
  const data = /** @type {Record<string, unknown>} */(raw);

  assertObject(data.header, 'trace.header');
  assertObject(data.stage_samples, 'trace.stage_samples');

  return /** @type {any} */(data);
}

/**
 * Validate impact file (may be absent / 404 → returns null).
 * @param {unknown} raw
 * @returns {object|null}
 */
export function validateImpact(raw) {
  if (raw === null || raw === undefined) return null;
  assertObject(raw, 'impact');
  return /** @type {any} */(raw);
}

/**
 * Get a resolved doc status, returning 'not_started' for missing entries.
 * @param {object|null|undefined} statusEntry
 * @returns {DocStatus}
 */
export function resolveDocStatus(statusEntry) {
  if (!statusEntry) return 'not_started';
  const s = /** @type {string} */(/** @type {Record<string,unknown>} */(statusEntry).status);
  return DOC_STATUSES.has(s) ? /** @type {DocStatus} */(s) : 'not_started';
}

// ── Scrape validators ───────────────────────────────────────────────

/**
 * Validate scrape meta.json structure.
 * @param {unknown} raw
 * @returns {{ generated_at_utc: string, timezone: string, publishers: Array<string>, available_dates: { min: string|null, max: string|null } }}
 */
export function validateScrapeMeta(raw) {
  assertObject(raw, 'scrapeMeta');
  const data = /** @type {Record<string, unknown>} */(raw);
  assertString(data.generated_at_utc, 'scrapeMeta.generated_at_utc');
  assertString(data.timezone, 'scrapeMeta.timezone');
  assertType(data.publishers, 'scrapeMeta.publishers', 'array');
  assertObject(data.available_dates, 'scrapeMeta.available_dates');
  return /** @type {any} */(data);
}

/**
 * Validate scrape overview.json structure.
 * @param {unknown} raw
 * @returns {{ totals: { publishers: number, publications_scraped: number, publications_preprocessed: number }, yesterday: { date: string|null, publications_scraped: number }, length_chars: object }}
 */
export function validateScrapeOverview(raw) {
  assertObject(raw, 'scrapeOverview');
  const data = /** @type {Record<string, unknown>} */(raw);
  assertObject(data.totals, 'scrapeOverview.totals');
  assertObject(data.yesterday, 'scrapeOverview.yesterday');
  assertObject(data.length_chars, 'scrapeOverview.length_chars');
  return /** @type {any} */(data);
}

/**
 * Validate scrape matrix.json structure.
 * @param {unknown} raw
 * @returns {{ dates: Array<string>, rows: Array<{ publisher: string, total_scraped: number, cells: Record<string, { n_scraped: number, n_preprocessed: number }> }> }}
 */
export function validateScrapeMatrix(raw) {
  assertObject(raw, 'scrapeMatrix');
  const data = /** @type {Record<string, unknown>} */(raw);
  assertType(data.dates, 'scrapeMatrix.dates', 'array');
  assertType(data.rows, 'scrapeMatrix.rows', 'array');

  const rows = /** @type {Array} */(data.rows);
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    assertString(row.publisher, `scrapeMatrix.rows[${i}].publisher`);
    assertObject(row.cells, `scrapeMatrix.rows[${i}].cells`);
  }

  return /** @type {any} */(data);
}

/**
 * Validate scrape publications detail file.
 * @param {unknown} raw
 * @returns {{ publisher: string, published_on: string, publications: Array }}
 */
export function validateScrapePublications(raw) {
  assertObject(raw, 'scrapePublications');
  const data = /** @type {Record<string, unknown>} */(raw);
  assertString(data.publisher, 'scrapePublications.publisher');
  assertString(data.published_on, 'scrapePublications.published_on');
  assertType(data.publications, 'scrapePublications.publications', 'array');
  return /** @type {any} */(data);
}

// ── Scrape cell status derivation ───────────────────────────────────

/**
 * Derive the scrape cell status from counts.
 * @param {number} nScraped
 * @param {number} nPreprocessed
 * @returns {ScrapeStatus}
 */
export function deriveScrapeStatus(nScraped, nPreprocessed) {
  if (nScraped === 0 && nPreprocessed === 0) return 'gray';
  if (nScraped > 0 && nPreprocessed === nScraped) return 'green';
  if (nScraped > 0 && nPreprocessed < nScraped) return 'yellow';
  if (nPreprocessed > nScraped) return 'red'; // inconsistent
  return 'gray';
}
