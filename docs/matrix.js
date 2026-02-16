/**
 * matrix.js — Overview Matrix rendering with virtual scrolling.
 *
 * Renders a document × stage grid. Uses simple virtual scrolling:
 * only visible rows are in the DOM, recycled on scroll.
 *
 * @module matrix
 */

import { getCellStatus, getCellStatusEntry, getCellCounts, countsToNumber, getVisibleDocs } from './state.js';
import { showTooltip, hideTooltip, buildCellTooltipHTML, buildRunStageTooltipHTML, escapeHtml } from './tooltip.js';

/** @typedef {import('./state.js').AppState} AppState */

// ── Constants ───────────────────────────────────────────────────────

const ROW_HEIGHT = 28; // must match CSS --cell-h
const OVERSCAN = 8;

// ── DOM references ──────────────────────────────────────────────────

/** @type {HTMLElement} */ let stageHeaderEl;
/** @type {HTMLElement} */ let viewportEl;
/** @type {HTMLElement} */ let contentEl;
/** @type {HTMLElement} */ let docCountEl;

/** @type {AppState} */ let currentState;
/** @type {Array<{ doc: object, docIndex: number }>} */ let visibleDocs = [];
/** @type {(dvid: string, docIndex: number) => void} */ let onSelectDoc;

// Track rendered range to minimize DOM churn
let renderedStart = -1;
let renderedEnd = -1;

/**
 * Initialize the matrix module.
 * @param {{ onDocumentSelect: (dvid: string, docIndex: number) => void }} callbacks
 */
export function initMatrix(callbacks) {
  stageHeaderEl = /** @type {HTMLElement} */(document.getElementById('matrix-stage-header'));
  viewportEl = /** @type {HTMLElement} */(document.getElementById('matrix-viewport'));
  contentEl = /** @type {HTMLElement} */(document.getElementById('matrix-content'));
  docCountEl = /** @type {HTMLElement} */(document.getElementById('matrix-doc-count'));
  onSelectDoc = callbacks.onDocumentSelect;

  viewportEl.addEventListener('scroll', handleScroll, { passive: true });
  contentEl.addEventListener('click', handleRowClick);
  contentEl.addEventListener('mouseover', handleCellHover);
  contentEl.addEventListener('mouseout', handleCellOut);
}

/**
 * Full re-render: stage headers + recompute visible docs + rows.
 * @param {AppState} state
 */
export function renderMatrix(state) {
  currentState = state;
  if (!state.meta || !state.docsData) return;

  renderStageHeader(state);
  visibleDocs = getVisibleDocs(state);
  docCountEl.textContent = `${visibleDocs.length} docs`;

  // Reset virtual scroll
  renderedStart = -1;
  renderedEnd = -1;
  contentEl.innerHTML = '';
  contentEl.style.height = `${visibleDocs.length * ROW_HEIGHT}px`;

  renderVisibleRows();
}

// ── Stage header ────────────────────────────────────────────────────

/**
 * @param {AppState} state
 */
function renderStageHeader(state) {
  stageHeaderEl.innerHTML = '';
  const stages = /** @type {Array<{stage_id: string, scope: string, label: string}>} */(state.meta.stages);

  for (const stage of stages) {
    const el = document.createElement('div');
    el.className = 'matrix__stage-col' + (stage.scope === 'run' ? ' matrix__stage-col--run' : '');
    el.textContent = stage.label;
    el.setAttribute('role', 'columnheader');

    // Tooltip for run-scoped stages
    if (stage.scope === 'run') {
      el.addEventListener('mouseover', () => {
        const runData = state.selectedRunId ? state.runDataCache[state.selectedRunId] : null;
        let statusEntry = null;
        if (runData) {
          const runStages = /** @type {Array} */(/** @type {any} */(runData).run_stage_status);
          const runStagesDef = stages.filter(s => s.scope === 'run');
          const idx = runStagesDef.findIndex(s => s.stage_id === stage.stage_id);
          statusEntry = runStages[idx] ?? null;
        }
        const html = buildRunStageTooltipHTML(stage.stage_id, statusEntry);
        showTooltip(el, html);
      });
      el.addEventListener('mouseout', hideTooltip);
    }

    stageHeaderEl.appendChild(el);
  }
}

// ── Virtual scroll ──────────────────────────────────────────────────

function handleScroll() {
  renderVisibleRows();
}

function renderVisibleRows() {
  if (!currentState?.meta) return;

  const scrollTop = viewportEl.scrollTop;
  const viewHeight = viewportEl.clientHeight;
  const totalRows = visibleDocs.length;

  let startIdx = Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN;
  let endIdx = Math.ceil((scrollTop + viewHeight) / ROW_HEIGHT) + OVERSCAN;
  startIdx = Math.max(0, startIdx);
  endIdx = Math.min(totalRows, endIdx);

  // Skip if same range
  if (startIdx === renderedStart && endIdx === renderedEnd) return;

  renderedStart = startIdx;
  renderedEnd = endIdx;

  // Rebuild rows in visible range
  contentEl.innerHTML = '';

  const stages = /** @type {Array<{stage_id: string, scope: string, label: string}>} */(currentState.meta.stages);
  const docStages = stages.filter(s => s.scope === 'doc');
  const runStages = stages.filter(s => s.scope === 'run');

  for (let i = startIdx; i < endIdx; i++) {
    const { doc, docIndex } = visibleDocs[i];
    const d = /** @type {Record<string, unknown>} */(doc);
    const rowEl = document.createElement('div');
    rowEl.className = 'matrix__row';
    rowEl.dataset.docIndex = String(docIndex);
    rowEl.dataset.dvid = String(d.doc_version_id);
    rowEl.style.position = 'absolute';
    rowEl.style.top = `${i * ROW_HEIGHT}px`;
    rowEl.style.width = '100%';
    rowEl.setAttribute('role', 'row');

    // Check if this is a new publisher group
    if (i === 0 || visibleDocs[i - 1].doc.publisher_id !== d.publisher_id) {
      rowEl.classList.add('matrix__row--publisher-start');
    }

    if (d.doc_version_id === currentState.selectedDocVersionId) {
      rowEl.classList.add('matrix__row--selected');
    }

    // Label
    const labelEl = document.createElement('div');
    labelEl.className = 'matrix__row-label';

    const pubEl = document.createElement('span');
    pubEl.className = 'matrix__row-publisher';
    pubEl.textContent = String(d.publisher_id || '');

    const titleEl = document.createElement('span');
    titleEl.className = 'matrix__row-title';
    titleEl.textContent = String(d.title || '(no title)');
    titleEl.title = String(d.title || '');

    const dateEl = document.createElement('span');
    dateEl.className = 'matrix__row-date';
    dateEl.textContent = formatDate(d.source_published_at);

    labelEl.appendChild(pubEl);
    labelEl.appendChild(titleEl);
    labelEl.appendChild(dateEl);
    rowEl.appendChild(labelEl);

    // Doc stage cells (0–7)
    for (let si = 0; si < docStages.length; si++) {
      const cellEl = document.createElement('div');
      cellEl.className = 'matrix__cell';
      cellEl.dataset.stageIndex = String(si);
      cellEl.dataset.stageId = docStages[si].stage_id;

      const status = getCellStatus(currentState, docIndex, si);
      const markerEl = document.createElement('span');
      markerEl.className = `status-marker status-marker--${status}`;
      cellEl.appendChild(markerEl);

      // Compact count
      const counts = getCellCounts(currentState, docIndex, si);
      const num = countsToNumber(counts);
      if (num !== null) {
        const cntEl = document.createElement('span');
        cntEl.className = 'cell-count';
        cntEl.textContent = String(num);
        cellEl.appendChild(cntEl);
      }

      rowEl.appendChild(cellEl);
    }

    // Run-scoped stage cells (same value for all rows)
    if (currentState.selectedRunId) {
      const runData = currentState.runDataCache[currentState.selectedRunId];
      const runStatusArr = runData ? /** @type {Array} */(/** @type {any} */(runData).run_stage_status) : [];

      for (let ri = 0; ri < runStages.length; ri++) {
        const cellEl = document.createElement('div');
        cellEl.className = 'matrix__cell matrix__cell--run';
        cellEl.dataset.runStageIndex = String(ri);
        cellEl.dataset.stageId = runStages[ri].stage_id;

        const entry = runStatusArr[ri];
        const status = entry ? String(/** @type {any} */(entry).status) : 'not_started';
        const markerEl = document.createElement('span');
        markerEl.className = `status-marker status-marker--${status}`;
        cellEl.appendChild(markerEl);

        rowEl.appendChild(cellEl);
      }
    }

    contentEl.appendChild(rowEl);
  }
}

// ── Event handlers ──────────────────────────────────────────────────

/** @param {MouseEvent} e */
function handleRowClick(e) {
  const rowEl = /** @type {HTMLElement} */(e.target).closest('.matrix__row');
  if (!rowEl) return;
  const dvid = rowEl.dataset.dvid;
  const docIndex = parseInt(rowEl.dataset.docIndex || '0', 10);
  if (dvid && onSelectDoc) {
    onSelectDoc(dvid, docIndex);
  }
}

/** @param {MouseEvent} e */
function handleCellHover(e) {
  const target = /** @type {HTMLElement} */(e.target);
  const cellEl = target.closest('.matrix__cell');
  if (!cellEl || !currentState) return;

  const rowEl = cellEl.closest('.matrix__row');
  if (!rowEl) return;

  const stageId = /** @type {HTMLElement} */(cellEl).dataset.stageId || '';
  const docIndex = parseInt(/** @type {HTMLElement} */(rowEl).dataset.docIndex || '0', 10);

  // Doc-stage cell
  if (/** @type {HTMLElement} */(cellEl).dataset.stageIndex !== undefined) {
    const si = parseInt(/** @type {HTMLElement} */(cellEl).dataset.stageIndex || '0', 10);
    const status = getCellStatus(currentState, docIndex, si);
    const statusEntry = getCellStatusEntry(currentState, docIndex, si);
    const counts = getCellCounts(currentState, docIndex, si);
    const html = buildCellTooltipHTML(stageId, status, statusEntry, counts);
    showTooltip(/** @type {HTMLElement} */(cellEl), html);
    return;
  }

  // Run-stage cell
  if (/** @type {HTMLElement} */(cellEl).dataset.runStageIndex !== undefined) {
    const ri = parseInt(/** @type {HTMLElement} */(cellEl).dataset.runStageIndex || '0', 10);
    const runData = currentState.selectedRunId ? currentState.runDataCache[currentState.selectedRunId] : null;
    const runStatusArr = runData ? /** @type {Array} */(/** @type {any} */(runData).run_stage_status) : [];
    const html = buildRunStageTooltipHTML(stageId, runStatusArr[ri] ?? null);
    showTooltip(/** @type {HTMLElement} */(cellEl), html);
  }
}

/** @param {MouseEvent} _e */
function handleCellOut(_e) {
  hideTooltip();
}

// ── Utilities ───────────────────────────────────────────────────────

/**
 * Format an ISO date string to compact form.
 * @param {unknown} val
 * @returns {string}
 */
function formatDate(val) {
  if (!val) return '';
  const s = String(val);
  // Return just YYYY-MM-DD
  return s.slice(0, 10);
}
