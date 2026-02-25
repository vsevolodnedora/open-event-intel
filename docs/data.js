/**
 * data.js — Data fetching via sql.js SQLite databases.
 *
 * Loads .sqlite files on demand, queries them, and returns objects
 * shaped identically to the original JSON-based data layer so all
 * downstream modules (matrix, trace, scrape_matrix, etc.) work unchanged.
 *
 * @module data
 */

import { openDatabase, queryAll, queryOne, queryScalar } from './db.js';

// ── Base paths ──────────────────────────────────────────────────────

let etlBasePath = './etl_data/sqlite';
let scrapeBasePath = './scrape_data/sqlite';

/** @param {string} path */
export function setBasePath(path) {
  etlBasePath = path.replace(/\/+$/, '');
}

/** @param {string} path */
export function setScrapeBasePath(path) {
  scrapeBasePath = path.replace(/\/+$/, '');
}

// ── Lazy DB handles ─────────────────────────────────────────────────

/** @type {any} */ let catalogDb = null;
/** @type {any} */ let tracesDb = null;
/** @type {any} */ let scrapeDb = null;
/** @type {Map<string, any>} */ const runDbs = new Map();

async function getCatalog() {
  if (!catalogDb) catalogDb = await openDatabase(`${etlBasePath}/catalog.sqlite`);
  return catalogDb;
}

async function getTraces() {
  if (!tracesDb) tracesDb = await openDatabase(`${etlBasePath}/traces.sqlite`);
  return tracesDb;
}

async function getRunDb(runId) {
  if (runDbs.has(runId)) return runDbs.get(runId);
  const db = await openDatabase(`${etlBasePath}/runs/${runId}.sqlite`);
  runDbs.set(runId, db);
  return db;
}

async function getScrape() {
  if (!scrapeDb) scrapeDb = await openDatabase(`${scrapeBasePath}/scrape.sqlite`);
  return scrapeDb;
}

// ── ETL data loaders ────────────────────────────────────────────────

/**
 * Load ETL meta from catalog.sqlite.
 * @returns {Promise<{ export_version: string, generated_at: string, stages: Array, runs: Array, publishers: Array }>}
 */
export async function loadMeta() {
  const db = await getCatalog();

  const export_version = /** @type {string} */(queryScalar(db,
    "SELECT value FROM meta WHERE key = 'export_version'"));
  const generated_at = /** @type {string} */(queryScalar(db,
    "SELECT value FROM meta WHERE key = 'generated_at'"));

  const stages = queryAll(db,
    "SELECT stage_id, scope, label FROM stage ORDER BY sort_order");

  const runs = queryAll(db,
    "SELECT run_id, status, started_at, completed_at, config_version, prev_completed_run_id FROM run ORDER BY rowid");

  const publishers = queryAll(db,
    "SELECT publisher_id FROM publisher ORDER BY publisher_id");

  return { export_version, generated_at, stages, runs, publishers };
}

/**
 * Load ETL docs from catalog.sqlite.
 * Returns the same shape as the old docs.json.
 * @returns {Promise<{ docs: Array, stage_status_by_doc: Array, totals_by_doc: Array }>}
 */
export async function loadDocs() {
  const db = await getCatalog();

  const docRows = queryAll(db,
    `SELECT doc_index, doc_version_id, document_id, publisher_id, title,
            url_normalized, source_published_at, created_in_run_id,
            content_quality_score, primary_language
     FROM doc ORDER BY doc_index`);

  const docs = docRows.map(r => ({
    doc_version_id: r.doc_version_id,
    document_id: r.document_id,
    publisher_id: r.publisher_id,
    title: r.title,
    url_normalized: r.url_normalized,
    source_published_at: r.source_published_at,
    created_in_run_id: r.created_in_run_id,
    content_quality_score: r.content_quality_score,
    primary_language: r.primary_language,
  }));

  // Stage status — reconstruct the 2D array [doc_index][stage_index]
  const statusRows = queryAll(db,
    "SELECT doc_index, stage_index, status, attempt, last_run_id, processed_at, error_message, details FROM doc_stage_status ORDER BY doc_index, stage_index");

  const stageCount = /** @type {number} */(queryScalar(db,
    "SELECT COUNT(*) FROM stage WHERE scope = 'doc'"));

  // Pre-fetch stage_id mapping
  const stageIds = queryAll(db, "SELECT sort_order, stage_id FROM stage WHERE scope = 'doc' ORDER BY sort_order");
  const stageIdMap = new Map(stageIds.map(s => [s.sort_order, s.stage_id]));

  /** @type {Map<number, Map<number, object>>} */
  const statusMap = new Map();
  for (const r of statusRows) {
    if (!statusMap.has(r.doc_index)) statusMap.set(r.doc_index, new Map());
    statusMap.get(r.doc_index).set(r.stage_index, {
      stage_id: stageIdMap.get(r.stage_index),
      status: r.status,
      attempt: r.attempt,
      last_run_id: r.last_run_id,
      processed_at: r.processed_at,
      error_message: r.error_message,
      details: r.details,
    });
  }

  const stage_status_by_doc = docs.map((_, di) => {
    const rowMap = statusMap.get(di);
    const arr = [];
    for (let si = 0; si < stageCount; si++) {
      arr.push(rowMap?.get(si) ?? null);
    }
    return arr;
  });

  // Totals — reconstruct the 2D array
  const totalRows = queryAll(db,
    "SELECT doc_index, stage_index, counts_json FROM doc_totals ORDER BY doc_index, stage_index");

  /** @type {Map<number, Map<number, object>>} */
  const totalsMap = new Map();
  for (const r of totalRows) {
    if (!totalsMap.has(r.doc_index)) totalsMap.set(r.doc_index, new Map());
    totalsMap.get(r.doc_index).set(r.stage_index, JSON.parse(/** @type {string} */(r.counts_json)));
  }

  const totals_by_doc = docs.map((_, di) => {
    const rowMap = totalsMap.get(di);
    const arr = [];
    for (let si = 0; si < stageCount; si++) {
      arr.push(rowMap?.get(si) ?? null);
    }
    return arr;
  });

  return { docs, stage_status_by_doc, totals_by_doc };
}

/**
 * Load a run's data.
 * @param {string} runId
 * @param {Record<string, object>} cache
 * @returns {Promise<{ run_id: string, run_stage_status: Array, new_counts_by_doc: Array, run_artifact_counts: object }>}
 */
export async function loadRunData(runId, cache) {
  if (cache[runId]) return /** @type {any} */(cache[runId]);

  const db = await getRunDb(runId);

  // Run stage status
  const stageRows = queryAll(db,
    "SELECT stage_index, status_json FROM run_stage_status ORDER BY stage_index");

  // We need to know how many run stages there are (from catalog)
  const catalogDb = await getCatalog();
  const runStageCount = /** @type {number} */(queryScalar(catalogDb,
    "SELECT COUNT(*) FROM stage WHERE scope = 'run'"));

  /** @type {Map<number, object>} */
  const stageMap = new Map();
  for (const r of stageRows) {
    stageMap.set(r.stage_index, JSON.parse(/** @type {string} */(r.status_json)));
  }

  const run_stage_status = [];
  for (let i = 0; i < runStageCount; i++) {
    run_stage_status.push(stageMap.get(i) ?? null);
  }

  // New counts per doc
  const docCount = /** @type {number} */(queryScalar(catalogDb, "SELECT COUNT(*) FROM doc"));
  const docStageCount = /** @type {number} */(queryScalar(catalogDb,
    "SELECT COUNT(*) FROM stage WHERE scope = 'doc'"));

  const countsRows = queryAll(db,
    "SELECT doc_index, stage_index, counts_json FROM new_counts ORDER BY doc_index, stage_index");

  /** @type {Map<number, Map<number, object>>} */
  const countsMap = new Map();
  for (const r of countsRows) {
    if (!countsMap.has(r.doc_index)) countsMap.set(r.doc_index, new Map());
    countsMap.get(r.doc_index).set(r.stage_index, JSON.parse(/** @type {string} */(r.counts_json)));
  }

  const new_counts_by_doc = [];
  for (let di = 0; di < docCount; di++) {
    const rowMap = countsMap.get(di);
    const arr = [];
    for (let si = 0; si < docStageCount; si++) {
      arr.push(rowMap?.get(si) ?? null);
    }
    new_counts_by_doc.push(arr);
  }

  // Artifact counts
  const acRows = queryAll(db, "SELECT key, value FROM run_artifact_counts");
  const run_artifact_counts = {};
  for (const r of acRows) {
    run_artifact_counts[r.key] = r.value;
  }

  const result = { run_id: runId, run_stage_status, new_counts_by_doc, run_artifact_counts };
  cache[runId] = result;
  return result;
}

/**
 * Load a document's base trace.
 * @param {string} docVersionId
 * @param {Record<string, object>} cache
 * @returns {Promise<{ header: object, stage_samples: object }>}
 */
export async function loadTrace(docVersionId, cache) {
  if (cache[docVersionId]) return /** @type {any} */(cache[docVersionId]);

  const db = await getTraces();

  const headerRow = queryOne(db,
    "SELECT header_json FROM trace_header WHERE doc_version_id = ?", [docVersionId]);
  if (!headerRow) throw new Error(`Trace not found: ${docVersionId}`);

  const header = JSON.parse(/** @type {string} */(headerRow.header_json));

  const sampleRows = queryAll(db,
    "SELECT stage_id, table_name, rows_json FROM trace_sample WHERE doc_version_id = ?",
    [docVersionId]);

  /** @type {Record<string, Record<string, Array>>} */
  const stage_samples = {};
  for (const r of sampleRows) {
    const sid = /** @type {string} */(r.stage_id);
    if (!stage_samples[sid]) stage_samples[sid] = {};
    stage_samples[sid][/** @type {string} */(r.table_name)] = JSON.parse(/** @type {string} */(r.rows_json));
  }

  const result = { header, stage_samples };
  cache[docVersionId] = result;
  return result;
}

/**
 * Load a document's run-specific impact data.
 * @param {string} docVersionId
 * @param {string} runId
 * @param {Record<string, object|null>} cache
 * @returns {Promise<object|null>}
 */
export async function loadImpact(docVersionId, runId, cache) {
  const cacheKey = `${runId}:${docVersionId}`;
  if (cacheKey in cache) return cache[cacheKey];

  const db = await getRunDb(runId);

  const impactRows = queryAll(db,
    "SELECT impact_key, rows_json FROM impact WHERE doc_version_id = ?",
    [docVersionId]);

  if (impactRows.length === 0) {
    cache[cacheKey] = null;
    return null;
  }

  const impact = {};
  for (const r of impactRows) {
    impact[/** @type {string} */(r.impact_key)] = JSON.parse(/** @type {string} */(r.rows_json));
  }

  cache[cacheKey] = impact;
  return impact;
}

// ── Scrape data loaders ─────────────────────────────────────────────

/**
 * Load scrape meta.
 * @returns {Promise<{ generated_at_utc: string, timezone: string, publishers: Array<string>, available_dates: { min: string|null, max: string|null } }>}
 */
export async function loadScrapeMeta() {
  const db = await getScrape();

  const generated_at_utc = /** @type {string} */(queryScalar(db,
    "SELECT value FROM meta WHERE key = 'generated_at_utc'"));
  const tz = /** @type {string} */(queryScalar(db,
    "SELECT value FROM meta WHERE key = 'timezone'"));
  const publishersJson = /** @type {string} */(queryScalar(db,
    "SELECT value FROM meta WHERE key = 'publishers'"));
  const dateMin = /** @type {string} */(queryScalar(db,
    "SELECT value FROM meta WHERE key = 'available_dates_min'"));
  const dateMax = /** @type {string} */(queryScalar(db,
    "SELECT value FROM meta WHERE key = 'available_dates_max'"));

  return {
    generated_at_utc,
    timezone: tz,
    publishers: JSON.parse(publishersJson),
    available_dates: {
      min: dateMin || null,
      max: dateMax || null,
    },
  };
}

/**
 * Load scrape overview.
 * @returns {Promise<{ totals: object, yesterday: object, length_chars: object }>}
 */
export async function loadScrapeOverview() {
  const db = await getScrape();
  const row = queryOne(db, "SELECT data_json FROM overview WHERE id = 1");
  return JSON.parse(/** @type {string} */(row.data_json));
}

/**
 * Load scrape matrix.
 * @returns {Promise<{ dates: Array<string>, rows: Array<{ publisher: string, total_scraped: number, cells: Record<string, { n_scraped: number, n_preprocessed: number }> }> }>}
 */
export async function loadScrapeMatrix() {
  const db = await getScrape();

  const dateRows = queryAll(db, "SELECT date_str FROM matrix_date ORDER BY date_str");
  const dates = dateRows.map(r => /** @type {string} */(r.date_str));

  const totalRows = queryAll(db, "SELECT publisher, total_scraped FROM publisher_total ORDER BY publisher");
  const cellRows = queryAll(db,
    "SELECT publisher, date_str, n_scraped, n_preprocessed FROM matrix_cell");

  /** @type {Map<string, Record<string, { n_scraped: number, n_preprocessed: number }>>} */
  const cellMap = new Map();
  for (const r of cellRows) {
    const pub = /** @type {string} */(r.publisher);
    if (!cellMap.has(pub)) cellMap.set(pub, {});
    cellMap.get(pub)[/** @type {string} */(r.date_str)] = {
      n_scraped: /** @type {number} */(r.n_scraped),
      n_preprocessed: /** @type {number} */(r.n_preprocessed),
    };
  }

  const rows = totalRows.map(r => ({
    publisher: /** @type {string} */(r.publisher),
    total_scraped: /** @type {number} */(r.total_scraped),
    cells: cellMap.get(/** @type {string} */(r.publisher)) || {},
  }));

  return { dates, rows };
}

/**
 * Load scrape publications detail for a publisher/date.
 * @param {string} publisher
 * @param {string} date
 * @param {Record<string, object>} cache
 * @returns {Promise<{ publisher: string, published_on: string, publications: Array }|null>}
 */
export async function loadScrapePublications(publisher, date, cache) {
  const cacheKey = `${publisher}:${date}`;
  if (cacheKey in cache) return /** @type {any} */(cache[cacheKey]);

  const db = await getScrape();

  const pubRows = queryAll(db,
    `SELECT id, title, published_on, scraped_on, language,
            length_before AS length_before_preprocessing,
            length_after AS length_after_preprocessing, url
     FROM publication WHERE publisher = ? AND date_str = ?
     ORDER BY date_str, published_on`,
    [publisher, date]);

  if (pubRows.length === 0) {
    cache[cacheKey] = /** @type {any} */(null);
    return null;
  }

  const result = {
    publisher,
    published_on: date,
    publications: pubRows,
  };
  cache[cacheKey] = result;
  return result;
}
