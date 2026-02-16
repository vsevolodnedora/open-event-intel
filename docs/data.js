/**
 * data.js — Data fetching, caching, and I/O boundary.
 *
 * All external JSON is fetched here, validated via schema.js, and
 * stored in the state cache. No raw/untyped data escapes this module.
 *
 * @module data
 */

import {
  validateMeta, validateDocs, validateRunData, validateTrace, validateImpact,
  validateScrapeMeta, validateScrapeOverview, validateScrapeMatrix, validateScrapePublications
} from './schema.js';

// ── ETL base path ───────────────────────────────────────────────────

/**
 * Base path for ETL JSON data files.
 * @type {string}
 */
let etlBasePath = './etl_data';

/**
 * Base path for scrape JSON data files.
 * @type {string}
 */
let scrapeBasePath = './scrape_data';

/**
 * Configure the base path for ETL data files.
 * @param {string} path
 */
export function setBasePath(path) {
  etlBasePath = path.replace(/\/+$/, '');
}

/**
 * Configure the base path for scrape data files.
 * @param {string} path
 */
export function setScrapeBasePath(path) {
  scrapeBasePath = path.replace(/\/+$/, '');
}

// ── Generic fetch helper ────────────────────────────────────────────

/**
 * Fetch JSON from a URL with error handling.
 * @param {string} url
 * @param {{ optional?: boolean }} [opts]
 * @returns {Promise<unknown>}
 */
async function fetchJSON(url, opts = {}) {
  try {
    const resp = await fetch(url);
    if (resp.status === 404 && opts.optional) return null;
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status} fetching ${url}`);
    }
    return await resp.json();
  } catch (err) {
    if (opts.optional && err instanceof TypeError) return null;
    throw err;
  }
}

// ── ETL data loaders ────────────────────────────────────────────────

/**
 * Load ETL meta.json and validate.
 * @returns {Promise<ReturnType<typeof validateMeta>>}
 */
export async function loadMeta() {
  const raw = await fetchJSON(`${etlBasePath}/meta.json`);
  return validateMeta(raw);
}

/**
 * Load ETL docs.json and validate.
 * @returns {Promise<ReturnType<typeof validateDocs>>}
 */
export async function loadDocs() {
  const raw = await fetchJSON(`${etlBasePath}/docs.json`);
  return validateDocs(raw);
}

/**
 * Load a run's data file, with in-memory caching.
 * @param {string} runId
 * @param {Record<string, object>} cache — state.runDataCache
 * @returns {Promise<ReturnType<typeof validateRunData>>}
 */
export async function loadRunData(runId, cache) {
  if (cache[runId]) return /** @type {any} */(cache[runId]);
  const raw = await fetchJSON(`${etlBasePath}/runs/${runId}.json`);
  const validated = validateRunData(raw);
  cache[runId] = validated;
  return validated;
}

/**
 * Load a document's base trace, with caching.
 * @param {string} docVersionId
 * @param {Record<string, object>} cache — state.traceCache
 * @returns {Promise<ReturnType<typeof validateTrace>>}
 */
export async function loadTrace(docVersionId, cache) {
  if (cache[docVersionId]) return /** @type {any} */(cache[docVersionId]);
  const raw = await fetchJSON(`${etlBasePath}/trace/${docVersionId}.json`);
  const validated = validateTrace(raw);
  cache[docVersionId] = validated;
  return validated;
}

/**
 * Load a document's run-specific impact data (may be absent).
 * @param {string} docVersionId
 * @param {string} runId
 * @param {Record<string, object|null>} cache — state.impactCache
 * @returns {Promise<object|null>}
 */
export async function loadImpact(docVersionId, runId, cache) {
  const cacheKey = `${runId}:${docVersionId}`;
  if (cacheKey in cache) return cache[cacheKey];
  const raw = await fetchJSON(`${etlBasePath}/runs/${runId}/impact/${docVersionId}.json`, { optional: true });
  const validated = validateImpact(raw);
  cache[cacheKey] = validated;
  return validated;
}

// ── Scrape data loaders ─────────────────────────────────────────────

/**
 * Load scrape meta.json.
 * @returns {Promise<ReturnType<typeof validateScrapeMeta>>}
 */
export async function loadScrapeMeta() {
  const raw = await fetchJSON(`${scrapeBasePath}/meta.json`);
  return validateScrapeMeta(raw);
}

/**
 * Load scrape overview.json.
 * @returns {Promise<ReturnType<typeof validateScrapeOverview>>}
 */
export async function loadScrapeOverview() {
  const raw = await fetchJSON(`${scrapeBasePath}/overview.json`);
  return validateScrapeOverview(raw);
}

/**
 * Load scrape matrix.json.
 * @returns {Promise<ReturnType<typeof validateScrapeMatrix>>}
 */
export async function loadScrapeMatrix() {
  const raw = await fetchJSON(`${scrapeBasePath}/matrix.json`);
  return validateScrapeMatrix(raw);
}

/**
 * Load scrape publications detail for a publisher/date, with caching.
 * @param {string} publisher
 * @param {string} date — YYYY-MM-DD
 * @param {Record<string, object>} cache — state.scrapeDetailCache
 * @returns {Promise<ReturnType<typeof validateScrapePublications>|null>}
 */
export async function loadScrapePublications(publisher, date, cache) {
  const cacheKey = `${publisher}:${date}`;
  if (cacheKey in cache) return /** @type {any} */(cache[cacheKey]);
  const raw = await fetchJSON(
    `${scrapeBasePath}/publications/${publisher}/${date}.json`,
    { optional: true }
  );
  if (raw === null) {
    cache[cacheKey] = /** @type {any} */(null);
    return null;
  }
  const validated = validateScrapePublications(raw);
  cache[cacheKey] = validated;
  return validated;
}
