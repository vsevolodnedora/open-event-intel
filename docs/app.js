/**
 * app.js — Main application entry point.
 *
 * Boot sequence: init sql.js → load scrape data → render.
 * Tab switching swaps between ETL and Scrape views.
 *
 * @module app
 */

import { createStore, getVisibleDocs, getTodayCET } from './state.js';
import { loadMeta, loadDocs, loadRunData, loadTrace, loadImpact, loadScrapeMeta, loadScrapeOverview, loadScrapeMatrix, loadScrapePublications } from './data.js';
import { initMatrix, renderMatrix } from './matrix.js';
import { initTrace, renderTrace, showTracePlaceholder } from './trace.js';
import { initTooltip } from './tooltip.js';
import { initScrapeMatrix, renderScrapeMatrix } from './scrape_matrix.js';
import { initScrapeDetail, renderScrapeDetail, showScrapeDetailPlaceholder } from './scrape_detail.js';
import { initDB } from './db.js';

// ── DOM refs ────────────────────────────────────────────────────────

/** @type {HTMLSelectElement} */ let runSelectEl;
/** @type {HTMLInputElement} */ let diffToggleEl;
/** @type {HTMLSelectElement} */ let publisherSelectEl;
/** @type {HTMLInputElement} */ let filterInputEl;
/** @type {HTMLElement} */ let statsEl;
/** @type {HTMLElement} */ let loadingOverlay;
/** @type {HTMLElement} */ let errorOverlay;
/** @type {HTMLElement} */ let errorMessageEl;
/** @type {HTMLButtonElement} */ let collapseBtn;
/** @type {HTMLElement} */ let matrixPanel;
/** @type {HTMLButtonElement} */ let traceCloseBtn;

// Scrape DOM refs
/** @type {HTMLInputElement} */ let scrapeDateSelectEl;
/** @type {HTMLSelectElement} */ let scrapeWindowSizeEl;
/** @type {HTMLButtonElement} */ let scrapeDetailCloseBtn;
/** @type {HTMLButtonElement} */ let presetLastWeekBtn;
/** @type {HTMLButtonElement} */ let presetLastMonthBtn;

// Tab DOM refs
/** @type {NodeListOf<HTMLButtonElement>} */ let tabButtons;
/** @type {NodeListOf<HTMLElement>} */ let tabPanels;

// ── State ───────────────────────────────────────────────────────────

const { state, update } = createStore(render);

let scrapeDataLoaded = false;
let etlDataLoaded = false;

// ── Boot ────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', boot);

async function boot() {
  bindDOMRefs();
  initTooltip();
  initMatrix({ onDocumentSelect: handleDocSelect });
  initTrace();
  initScrapeMatrix({ onCellSelect: handleScrapeCellSelect });
  initScrapeDetail();
  bindControls();

  try {
    // Pre-load sql.js WASM before any data fetching
    await initDB();
    await loadScrapeData();
    hideLoading();
  } catch (err) {
    console.error('Scrape boot failed:', err);
    hideLoading();
  }
}

// ── DOM binding ─────────────────────────────────────────────────────

function bindDOMRefs() {
  runSelectEl = /** @type {HTMLSelectElement} */(document.getElementById('run-select'));
  diffToggleEl = /** @type {HTMLInputElement} */(document.getElementById('diff-toggle'));
  publisherSelectEl = /** @type {HTMLSelectElement} */(document.getElementById('publisher-select'));
  filterInputEl = /** @type {HTMLInputElement} */(document.getElementById('filter-input'));
  statsEl = /** @type {HTMLElement} */(document.getElementById('topbar-stats'));
  loadingOverlay = /** @type {HTMLElement} */(document.getElementById('loading-overlay'));
  errorOverlay = /** @type {HTMLElement} */(document.getElementById('error-overlay'));
  errorMessageEl = /** @type {HTMLElement} */(document.getElementById('error-message'));
  collapseBtn = /** @type {HTMLButtonElement} */(document.getElementById('matrix-collapse-btn'));
  matrixPanel = /** @type {HTMLElement} */(document.getElementById('matrix-panel'));
  traceCloseBtn = /** @type {HTMLButtonElement} */(document.getElementById('trace-close-btn'));

  scrapeDateSelectEl = /** @type {HTMLInputElement} */(document.getElementById('scrape-date-select'));
  scrapeWindowSizeEl = /** @type {HTMLSelectElement} */(document.getElementById('scrape-window-size'));
  scrapeDetailCloseBtn = /** @type {HTMLButtonElement} */(document.getElementById('scrape-detail-close-btn'));
  presetLastWeekBtn = /** @type {HTMLButtonElement} */(document.getElementById('scrape-preset-last-week'));
  presetLastMonthBtn = /** @type {HTMLButtonElement} */(document.getElementById('scrape-preset-last-month'));

  tabButtons = document.querySelectorAll('.tab-bar__tab');
  tabPanels = document.querySelectorAll('.tab-panel');
}

function clampToAvailableRange(endDate, days) {
    const meta = state.scrapeMeta;
    if (!meta?.available_dates) return { endDate, days };

    const { min, max } = meta.available_dates;

    // Clamp end-date to available max
    if (max && endDate > max) endDate = max;

    // Ensure start-date doesn't go before available min
    if (min) {
        const start = new Date(endDate + 'T12:00:00Z');
        start.setUTCDate(start.getUTCDate() - (days - 1));
        const startStr = start.toISOString().slice(0, 10);
        if (startStr < min) {
            const minD = new Date(min + 'T12:00:00Z');
            const endD = new Date(endDate + 'T12:00:00Z');
            days = Math.round((endD - minD) / 86400000) + 1;
            if (days < 1) days = 1;
        }
    }

    return { endDate, days };
}

function bindControls() {
  // Tab switching
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      if (tab) switchTab(tab);
    });
  });

  // ETL controls
  runSelectEl.addEventListener('change', async () => {
    await selectRun(runSelectEl.value);
  });

  diffToggleEl.addEventListener('change', () => {
    update({ diffModeEnabled: diffToggleEl.checked });
  });

  publisherSelectEl.addEventListener('change', () => {
    update({ selectedPublisherId: publisherSelectEl.value });
  });

  let filterTimeout = 0;
  filterInputEl.addEventListener('input', () => {
    clearTimeout(filterTimeout);
    filterTimeout = window.setTimeout(() => {
      update({ filterText: filterInputEl.value });
    }, 200);
  });

  collapseBtn.addEventListener('click', () => {
    const collapsed = !state.matrixCollapsed;
    update({ matrixCollapsed: collapsed });
    matrixPanel.classList.toggle('collapsed', collapsed);
    collapseBtn.setAttribute('aria-expanded', String(!collapsed));
    collapseBtn.querySelector('span').textContent = collapsed ? '▶' : '▼';
  });

  traceCloseBtn.addEventListener('click', () => {
    update({ selectedDocVersionId: null });
    showTracePlaceholder();
  });

  // Scrape controls -- remove the premature set; let loadScrapeData() be the single source of truth
  // scrapeDateSelectEl.value = getTodayCET();

  scrapeDateSelectEl.addEventListener('change', () => {
    update({ scrapeEndDate: scrapeDateSelectEl.value || null });
  });

  scrapeWindowSizeEl.addEventListener('change', () => {
    update({ scrapeWindowSize: parseInt(scrapeWindowSizeEl.value, 10) });
  });

  scrapeDetailCloseBtn.addEventListener('click', () => {
    update({ scrapeSelectedPublisher: null, scrapeSelectedDate: null });
    showScrapeDetailPlaceholder();
  });



  // Scrape presets
  presetLastWeekBtn.addEventListener('click', () => {
    let { endDate, days } = getLastWeekRange();
    ({ endDate, days } = clampToAvailableRange(endDate, days));
    scrapeDateSelectEl.value = endDate;
    setScrapeWindowOption(days);
    update({ scrapeEndDate: endDate, scrapeWindowSize: days });
  });

  presetLastMonthBtn.addEventListener('click', () => {
    let { endDate, days } = getLastMonthRange();
    ({ endDate, days } = clampToAvailableRange(endDate, days));
    scrapeDateSelectEl.value = endDate;
    setScrapeWindowOption(days);
    update({ scrapeEndDate: endDate, scrapeWindowSize: days });
  });

  // Global keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (state.activeTab === 'etl' && state.selectedDocVersionId) {
        update({ selectedDocVersionId: null });
        showTracePlaceholder();
      } else if (state.activeTab === 'scrape' && state.scrapeSelectedPublisher) {
        update({ scrapeSelectedPublisher: null, scrapeSelectedDate: null });
        showScrapeDetailPlaceholder();
      }
    }
  });
}

// ── Tab switching ───────────────────────────────────────────────────

async function switchTab(tabId) {
  tabButtons.forEach(btn => {
    const active = btn.dataset.tab === tabId;
    btn.classList.toggle('tab-bar__tab--active', active);
    btn.setAttribute('aria-selected', String(active));
  });

  tabPanels.forEach(panel => {
    const active = panel.id === `tab-panel-${tabId}`;
    panel.classList.toggle('tab-panel--active', active);
    panel.hidden = !active;
  });

  update({ activeTab: tabId });

  if (tabId === 'etl' && !etlDataLoaded) {
    await loadETLData();
  }
  if (tabId === 'scrape' && !scrapeDataLoaded) {
    await loadScrapeData();
  }
}

// ── ETL data loading ────────────────────────────────────────────────

async function loadETLData() {
  try {
    const [meta, docsData] = await Promise.all([loadMeta(), loadDocs()]);
    etlDataLoaded = true;

    update({ meta, docsData, loading: false });

    populateRunSelect(meta.runs);
    populatePublisherSelect(meta.publishers);

    const completedRuns = meta.runs.filter(r => r.status === 'completed');
    if (completedRuns.length > 0) {
      const runId = completedRuns[0].run_id;
      runSelectEl.value = runId;
      await selectRun(runId);
    } else if (meta.runs.length > 0) {
      const runId = meta.runs[0].run_id;
      runSelectEl.value = runId;
      await selectRun(runId);
    }
  } catch (err) {
    console.error('Failed to load ETL data:', err);
    etlDataLoaded = true;
  }
}

// ── Scrape data loading ─────────────────────────────────────────────

async function loadScrapeData() {
  try {
    const [scrapeMeta, scrapeOverview, scrapeMatrix] = await Promise.all([
      loadScrapeMeta(),
      loadScrapeOverview(),
      loadScrapeMatrix(),
    ]);

    scrapeDataLoaded = true;

    const endDate = scrapeMeta.available_dates?.max || getTodayCET();

    update({ scrapeMeta, scrapeOverview, scrapeMatrix, scrapeEndDate: endDate });

    scrapeDateSelectEl.value = endDate;
    if (scrapeMeta.available_dates?.min) {
      scrapeDateSelectEl.min = scrapeMeta.available_dates.min;
    }
    if (scrapeMeta.available_dates?.max) {
      scrapeDateSelectEl.max = scrapeMeta.available_dates.max;
    }
  } catch (err) {
    console.error('Failed to load scrape data:', err);
    scrapeDataLoaded = true;
  }
}

// ── Control population ──────────────────────────────────────────────

function populateRunSelect(runs) {
  runSelectEl.innerHTML = '';
  for (const run of runs) {
    const opt = document.createElement('option');
    opt.value = run.run_id;
    const raw = (run.completed_at || run.started_at || '');
    const dateStr = raw ? raw.slice(0, 19).replace('T', ' ') + ' UTC' : '';
    const statusTag = run.status !== 'completed' ? ` [${run.status}]` : '';
    opt.textContent = `${run.run_id.slice(0, 8)}… — ${dateStr}${statusTag}`;
    runSelectEl.appendChild(opt);
  }
}

function populatePublisherSelect(publishers) {
  for (const pub of publishers) {
    const opt = document.createElement('option');
    opt.value = pub.publisher_id;
    opt.textContent = pub.publisher_id;
    publisherSelectEl.appendChild(opt);
  }
}

// ── ETL Actions ─────────────────────────────────────────────────────

async function selectRun(runId) {
  try {
    const runData = await loadRunData(runId, state.runDataCache);
    const cache = { ...state.runDataCache, [runId]: runData };
    update({ selectedRunId: runId, runDataCache: cache });

    if (state.selectedDocVersionId) {
      await loadAndRenderTrace(state.selectedDocVersionId);
    }
  } catch (err) {
    console.error(`Failed to load run ${runId}:`, err);
    update({ selectedRunId: runId });
  }
}

async function handleDocSelect(dvid, _docIndex) {
  update({ selectedDocVersionId: dvid });
  await loadAndRenderTrace(dvid);
}

async function loadAndRenderTrace(dvid) {
  try {
    const [traceData, impactData] = await Promise.all([
      loadTrace(dvid, state.traceCache),
      state.selectedRunId
        ? loadImpact(dvid, state.selectedRunId, state.impactCache)
        : Promise.resolve(null),
    ]);

    if (state.selectedDocVersionId === dvid) {
      renderTrace(state, traceData, impactData);
    }
  } catch (err) {
    console.error(`Failed to load trace for ${dvid}:`, err);
    showTracePlaceholder();
  }
}

// ── Scrape Actions ──────────────────────────────────────────────────

async function handleScrapeCellSelect(publisher, date) {
  update({ scrapeSelectedPublisher: publisher, scrapeSelectedDate: date });

  try {
    const detail = await loadScrapePublications(publisher, date, state.scrapeDetailCache);
    if (state.scrapeSelectedPublisher === publisher && state.scrapeSelectedDate === date) {
      renderScrapeDetail(publisher, date, detail);
    }
  } catch (err) {
    console.error(`Failed to load publications for ${publisher}/${date}:`, err);
    renderScrapeDetail(publisher, date, null);
  }
}

// ── Render ──────────────────────────────────────────────────────────

function render() {
  if (state.activeTab === 'etl') {
    renderETL();
  } else if (state.activeTab === 'scrape') {
    renderScrape();
  }
  updateStats();
}

function renderETL() {
  renderMatrix(state);

  if (state.selectedDocVersionId) {
    const traceData = state.traceCache[state.selectedDocVersionId];
    if (traceData) {
      const impactKey = state.selectedRunId ? `${state.selectedRunId}:${state.selectedDocVersionId}` : null;
      const impactData = impactKey ? state.impactCache[impactKey] ?? null : null;
      renderTrace(state, traceData, impactData);
    }
  }
}

function renderScrape() {
  renderScrapeMatrix(state);
  renderScrapeKPI(state);
}

function renderScrapeKPI(state) {
  if (!state.scrapeOverview) return;

  const ov = state.scrapeOverview;

  const setKPI = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  setKPI('kpi-publishers', String(ov.totals?.publishers ?? '—'));
  setKPI('kpi-scraped', formatCompact(ov.totals?.publications_scraped));
  setKPI('kpi-preprocessed', formatCompact(ov.totals?.publications_preprocessed));
  setKPI('kpi-yesterday', ov.yesterday?.publications_scraped != null
    ? `${formatCompact(ov.yesterday.publications_scraped)} (${ov.yesterday.date || ''})`
    : '—');

  const lc = ov.length_chars;
  if (lc) {
    const smallest = lc.smallest_after_preprocessing ?? 0;
    const largest = lc.largest_after_preprocessing ?? 0;
    setKPI('kpi-size-range', `${formatCompact(smallest)} — ${formatCompact(largest)}`);
  }
}

function updateStats() {
  if (state.activeTab === 'etl') {
    if (!state.meta || !state.docsData) {
      statsEl.textContent = '';
      return;
    }
    const total = /** @type {Array} */(state.docsData.docs).length;
    const visible = getVisibleDocs(state).length;
    const runInfo = state.selectedRunId ? state.selectedRunId.slice(0, 8) : '—';
    statsEl.textContent = `Run: ${runInfo} | ${visible}/${total} docs${state.diffModeEnabled ? ' (diff)' : ''}`;
  } else if (state.activeTab === 'scrape') {
    if (!state.scrapeMatrix) {
      statsEl.textContent = '';
      return;
    }
    const rows = state.scrapeMatrix.rows?.length ?? 0;
    statsEl.textContent = `Scrape & Preprocess | ${rows} publishers`;
  }
}

// ── Overlays ────────────────────────────────────────────────────────

function hideLoading() {
  loadingOverlay.hidden = true;
  document.getElementById('app')?.removeAttribute('aria-busy');
}

function showError(msg) {
  loadingOverlay.hidden = true;
  errorMessageEl.textContent = msg;
  errorOverlay.hidden = false;

  const retryBtn = /** @type {HTMLButtonElement} */(document.getElementById('error-retry-btn'));
  retryBtn.onclick = () => {
    errorOverlay.hidden = true;
    loadingOverlay.hidden = false;
    boot();
  };
}

// ── Helpers ─────────────────────────────────────────────────────────

function setScrapeWindowOption(days) {
  const val = String(days);
  const exists = Array.from(scrapeWindowSizeEl.options).some(o => o.value === val);
  if (!exists) {
    const prev = scrapeWindowSizeEl.querySelector('option[data-preset]');
    if (prev) prev.remove();
    const opt = document.createElement('option');
    opt.value = val;
    opt.textContent = `${days} days`;
    opt.dataset.preset = 'true';
    scrapeWindowSizeEl.appendChild(opt);
  }
  scrapeWindowSizeEl.value = val;
}

function getNowCETComponents() {
  const now = new Date();
  const opts = { timeZone: 'Europe/Berlin' };
  const year = Number(new Intl.DateTimeFormat('en', { ...opts, year: 'numeric' }).format(now));
  const month = Number(new Intl.DateTimeFormat('en', { ...opts, month: 'numeric' }).format(now));
  const day = Number(new Intl.DateTimeFormat('en', { ...opts, day: 'numeric' }).format(now));
  const iso = `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
  return { year, month, day, iso };
}

function getLastWeekRange() {
  const { year, month, day } = getNowCETComponents();
  const d = new Date(year, month - 1, day);
  const dow = d.getDay();
  const daysSinceLastSunday = dow === 0 ? 7 : dow;
  const lastSunday = new Date(year, month - 1, day - daysSinceLastSunday);
  const endDate = toISODate(lastSunday);
  return { endDate, days: 7 };
}

function getLastMonthRange() {
  const { year, month } = getNowCETComponents();
  const lastDayPrevMonth = new Date(year, month - 1, 0);
  const firstDayPrevMonth = new Date(year, month - 2, 1);
  const days = Math.round((lastDayPrevMonth.getTime() - firstDayPrevMonth.getTime()) / 86400000) + 1;
  const endDate = toISODate(lastDayPrevMonth);
  return { endDate, days };
}

function toISODate(d) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function formatCompact(val) {
  if (val == null) return '—';
  const n = Number(val);
  if (isNaN(n)) return '—';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}
