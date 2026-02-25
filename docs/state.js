/**
 * state.js — Central app state and derived data helpers.
 *
 * State is a plain object; mutations go through update() which triggers
 * a render callback. Derived sets (visible docs, cell resolution) are
 * pure functions of state.
 *
 * @module state
 */

import { resolveDocStatus } from './schema.js';

/**
 * @typedef {object} AppState
 * @property {string} activeTab — 'etl' | 'scrape'
 *
 * ETL state
 * @property {string|null} selectedRunId
 * @property {boolean} diffModeEnabled
 * @property {string} selectedPublisherId
 * @property {string|null} selectedDocVersionId
 * @property {boolean} matrixCollapsed
 * @property {string} filterText
 * @property {object|null} meta
 * @property {object|null} docsData
 * @property {Record<string, object>} runDataCache
 * @property {Record<string, object>} traceCache
 * @property {Record<string, object|null>} impactCache
 *
 * Scrape state
 * @property {object|null} scrapeMeta
 * @property {object|null} scrapeOverview
 * @property {object|null} scrapeMatrix
 * @property {string|null} scrapeEndDate — window end-date (YYYY-MM-DD)
 * @property {number} scrapeWindowSize — number of days to show
 * @property {string|null} scrapeSelectedPublisher — publisher name of selected cell
 * @property {string|null} scrapeSelectedDate — date of selected cell
 * @property {Record<string, object>} scrapeDetailCache — publisher:date → detail data
 *
 * Shared
 * @property {boolean} loading
 * @property {string|null} error
 */

/** @returns {AppState} */
export function createInitialState() {
  return {
    activeTab: 'scrape',

    // ETL
    selectedRunId: null,
    diffModeEnabled: false,
    selectedPublisherId: 'ALL',
    selectedDocVersionId: null,
    matrixCollapsed: false,
    filterText: '',
    meta: null,
    docsData: null,
    runDataCache: {},
    traceCache: {},
    impactCache: {},

    // Scrape
    scrapeMeta: null,
    scrapeOverview: null,
    scrapeMatrix: null,
    scrapeEndDate: null,
    scrapeWindowSize: 14,
    scrapeSelectedPublisher: null,
    scrapeSelectedDate: null,
    scrapeDetailCache: {},

    // Shared
    loading: true,
    error: null,
  };
}

/**
 * Create a state store with update/subscribe pattern.
 * @param {() => void} onUpdate — called after every state change
 * @returns {{ state: AppState, update: (patch: Partial<AppState>) => void }}
 */
export function createStore(onUpdate) {
  const state = createInitialState();

  function update(patch) {
    Object.assign(state, patch);
    onUpdate();
  }

  return { state, update };
}

// ── ETL pure derivation functions ───────────────────────────────────

/**
 * Get the list of visible docs, applying publisher + diff + text filters.
 * Returns array of { doc, docIndex } where docIndex is the original index
 * into docsData.docs (needed for aligned lookups).
 *
 * @param {AppState} state
 * @returns {Array<{ doc: object, docIndex: number }>}
 */
export function getVisibleDocs(state) {
  if (!state.docsData) return [];

  const docs = /** @type {Array<Record<string, unknown>>} */(state.docsData.docs);
  const result = [];

  for (let i = 0; i < docs.length; i++) {
    const doc = docs[i];

    // Publisher filter
    if (state.selectedPublisherId !== 'ALL' &&
        doc.publisher_id !== state.selectedPublisherId) {
      continue;
    }

    // Text filter
    if (state.filterText) {
      const q = state.filterText.toLowerCase();
      const title = String(doc.title || '').toLowerCase();
      const url = String(doc.url_normalized || '').toLowerCase();
      if (!title.includes(q) && !url.includes(q)) continue;
    }

    // Diff mode: only docs with new data in selected run
    if (state.diffModeEnabled && state.selectedRunId) {
      const runData = state.runDataCache[state.selectedRunId];
      if (runData) {
        const newCounts = /** @type {Array} */(/** @type {any} */(runData).new_counts_by_doc)[i];
        if (!docHasAnyNewData(newCounts)) continue;
      }
    }

    result.push({ doc, docIndex: i });
  }

  return result;
}

/**
 * Check if a doc's new-in-run counts contain any non-zero/non-null values.
 * @param {Array<object|null>|undefined} countsArray — stage-ordered new counts
 * @returns {boolean}
 */
function docHasAnyNewData(countsArray) {
  if (!countsArray) return false;
  for (const entry of countsArray) {
    if (!entry) continue;
    for (const [key, val] of Object.entries(entry)) {
      if (val !== null && val !== undefined && val !== 0 && val !== '') return true;
    }
  }
  return false;
}

/**
 * Get the status for a matrix cell.
 * @param {AppState} state
 * @param {number} docIndex — original index into docsData
 * @param {number} stageIndex — index into doc stages (0–7)
 * @returns {import('./schema.js').DocStatus}
 */
export function getCellStatus(state, docIndex, stageIndex) {
  if (!state.docsData) return 'not_started';
  const statusRow = /** @type {Array} */(state.docsData.stage_status_by_doc)[docIndex];
  if (!statusRow) return 'not_started';
  return resolveDocStatus(statusRow[stageIndex]);
}

/**
 * Get the status entry object for a cell (for tooltip detail).
 * @param {AppState} state
 * @param {number} docIndex
 * @param {number} stageIndex
 * @returns {object|null}
 */
export function getCellStatusEntry(state, docIndex, stageIndex) {
  if (!state.docsData) return null;
  const statusRow = /** @type {Array} */(state.docsData.stage_status_by_doc)[docIndex];
  return statusRow?.[stageIndex] ?? null;
}

/**
 * Get counts for a cell (totals or diff-mode new counts).
 * @param {AppState} state
 * @param {number} docIndex
 * @param {number} stageIndex
 * @returns {object|null}
 */
export function getCellCounts(state, docIndex, stageIndex) {
  if (state.diffModeEnabled && state.selectedRunId) {
    const runData = state.runDataCache[state.selectedRunId];
    if (runData) {
      const newCounts = /** @type {Array} */(/** @type {any} */(runData).new_counts_by_doc)[docIndex];
      return newCounts?.[stageIndex] ?? null;
    }
  }
  if (!state.docsData) return null;
  return /** @type {Array} */(state.docsData.totals_by_doc)[docIndex]?.[stageIndex] ?? null;
}

/**
 * Get a compact summary string from a counts object.
 * @param {object|null} counts
 * @returns {string}
 */
export function countsToSummary(counts) {
  if (!counts) return '';
  const parts = [];
  for (const [k, v] of Object.entries(counts)) {
    if (v !== null && v !== undefined && v !== 0 && v !== '') {
      parts.push(`${k}: ${v}`);
    }
  }
  return parts.join(', ');
}

/**
 * Get the first numeric count from a counts object for compact display.
 * @param {object|null} counts
 * @returns {number|null}
 */
export function countsToNumber(counts) {
  if (!counts) return null;
  for (const v of Object.values(counts)) {
    if (typeof v === 'number' && v > 0) return v;
  }
  return null;
}

// ── Scrape derivation helpers ───────────────────────────────────────

/**
 * Get today's date in CET timezone (Europe/Berlin).
 * @returns {string} YYYY-MM-DD
 */
export function getTodayCET() {
  const formatter = new Intl.DateTimeFormat('sv-SE', {
    timeZone: 'Europe/Berlin',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  });
  return formatter.format(new Date()); // sv-SE gives YYYY-MM-DD format
}

/**
 * Compute the visible date columns for the scrape matrix.
 * @param {AppState} state
 * @returns {string[]} array of YYYY-MM-DD strings, ending at endDate
 */
export function getScrapeVisibleDates(state) {
  const endDate = state.scrapeEndDate || getTodayCET();
  const windowSize = state.scrapeWindowSize;

  // If we have matrix data, use the available dates filtered to the window
  if (state.scrapeMatrix) {
    const availableDates = new Set(state.scrapeMatrix.dates);

    const dates = [];
    const end = new Date(endDate + 'T12:00:00Z');
    for (let i = windowSize - 1; i >= 0; i--) {
        const d = new Date(end);
        d.setUTCDate(d.getUTCDate() - i);
        const iso = d.toISOString().slice(0, 10);
        dates.push(iso);
    }

    // If NONE of the generated dates have data, shift window to end at last available date
    if (dates.length > 0 && availableDates.size > 0 &&
        !dates.some(d => availableDates.has(d))) {
        const sortedAvail = [...availableDates].sort();
        const fallbackEnd = new Date(sortedAvail[sortedAvail.length - 1] + 'T12:00:00Z');
        dates.length = 0;
        for (let i = windowSize - 1; i >= 0; i--) {
            const d = new Date(fallbackEnd);
            d.setUTCDate(d.getUTCDate() - i);
            dates.push(d.toISOString().slice(0, 10));
        }
    }

    return dates;
  }

  // Fallback: generate date range
  const dates = [];
  const end = new Date(endDate + 'T12:00:00Z');
  for (let i = windowSize - 1; i >= 0; i--) {
    const d = new Date(end);
    d.setUTCDate(d.getUTCDate() - i);
    dates.push(d.toISOString().slice(0, 10));
  }
  return dates;
}
