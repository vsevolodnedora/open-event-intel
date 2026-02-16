/**
 * scrape_matrix.js — Scrape & Preprocess matrix rendering.
 *
 * Renders a publisher × date grid as an HTML table.
 * Cell colors derived from n_scraped / n_preprocessed comparison.
 *
 * @module scrape_matrix
 */

import { deriveScrapeStatus } from './schema.js';
import { getScrapeVisibleDates, getTodayCET } from './state.js';
import { showTooltip, hideTooltip, escapeHtml } from './tooltip.js';

/** @typedef {import('./state.js').AppState} AppState */

// ── DOM refs ────────────────────────────────────────────────────────

/** @type {HTMLElement} */ let theadEl;
/** @type {HTMLElement} */ let tbodyEl;
/** @type {HTMLElement} */ let infoEl;
/** @type {(publisher: string, date: string) => void} */ let onCellSelect;

/**
 * Initialize the scrape matrix module.
 * @param {{ onCellSelect: (publisher: string, date: string) => void }} callbacks
 */
export function initScrapeMatrix(callbacks) {
  theadEl = /** @type {HTMLElement} */(document.getElementById('scrape-matrix-thead'));
  tbodyEl = /** @type {HTMLElement} */(document.getElementById('scrape-matrix-tbody'));
  infoEl = /** @type {HTMLElement} */(document.getElementById('scrape-matrix-info'));
  onCellSelect = callbacks.onCellSelect;

  tbodyEl.addEventListener('click', handleCellClick);
  tbodyEl.addEventListener('mouseover', handleCellHover);
  tbodyEl.addEventListener('mouseout', handleCellOut);
}

/**
 * Full render of the scrape matrix.
 * @param {AppState} state
 */
export function renderScrapeMatrix(state) {
  if (!state.scrapeMatrix) {
    theadEl.innerHTML = '';
    tbodyEl.innerHTML = '';
    infoEl.textContent = '';
    return;
  }

  const matrix = state.scrapeMatrix;
  const dates = getScrapeVisibleDates(state);
  const todayCET = getTodayCET();
  const rows = /** @type {Array<{ publisher: string, total_scraped: number, cells: Record<string, { n_scraped: number, n_preprocessed: number }> }>} */(matrix.rows);

  // ── Header ──────────────────────────────────────────────────────
  theadEl.innerHTML = '';
  const headerRow = document.createElement('tr');

  // Publisher column
  const pubTh = document.createElement('th');
  pubTh.textContent = 'Publisher';
  headerRow.appendChild(pubTh);

  // Total column
  const totalTh = document.createElement('th');
  totalTh.textContent = 'Total';
  headerRow.appendChild(totalTh);

  // Date columns
  for (const date of dates) {
    const th = document.createElement('th');
    // Show MM-DD for compact display
    th.textContent = date.slice(5);
    th.title = date;
    if (date === todayCET) {
      th.classList.add('scrape-matrix__date-today');
    }
    headerRow.appendChild(th);
  }

  theadEl.appendChild(headerRow);

  // ── Body ────────────────────────────────────────────────────────
  tbodyEl.innerHTML = '';

  for (const row of rows) {
    const tr = document.createElement('tr');

    // Publisher label
    const pubTd = document.createElement('td');
    pubTd.textContent = row.publisher;
    tr.appendChild(pubTd);

    // Total column
    const totalTd = document.createElement('td');
    totalTd.className = 'scrape-matrix__total-cell';
    totalTd.textContent = String(row.total_scraped);
    tr.appendChild(totalTd);

    // Date cells
    for (const date of dates) {
      const td = document.createElement('td');
      td.className = 'scrape-matrix__cell';
      td.dataset.publisher = row.publisher;
      td.dataset.date = date;

      const cellData = row.cells[date];
      const nScraped = cellData?.n_scraped ?? 0;
      const nPreprocessed = cellData?.n_preprocessed ?? 0;
      const status = deriveScrapeStatus(nScraped, nPreprocessed);

      td.classList.add(`scrape-matrix__cell--${status}`);

      // Store data for tooltip
      td.dataset.nScraped = String(nScraped);
      td.dataset.nPreprocessed = String(nPreprocessed);
      td.dataset.status = status;

      // Selected state
      if (row.publisher === state.scrapeSelectedPublisher &&
          date === state.scrapeSelectedDate) {
        td.classList.add('scrape-matrix__cell--selected');
      }

      // Marker + count label
      const marker = document.createElement('span');
      marker.className = `scrape-matrix__cell-marker scrape-matrix__cell-marker--${status}`;
      td.appendChild(marker);

      const countSpan = document.createElement('span');
      countSpan.className = 'scrape-matrix__cell-count';
      if (nScraped === 0 && nPreprocessed === 0) {
        countSpan.classList.add('scrape-matrix__cell-count--zero');
        countSpan.textContent = '—';
      } else {
        countSpan.textContent = String(nPreprocessed);
      }
      td.appendChild(countSpan);

      tr.appendChild(td);
    }

    tbodyEl.appendChild(tr);
  }

  infoEl.textContent = `${rows.length} publishers × ${dates.length} days`;
}

// ── Event handlers ──────────────────────────────────────────────────

/** @param {MouseEvent} e */
function handleCellClick(e) {
  const td = /** @type {HTMLElement} */(e.target).closest('.scrape-matrix__cell');
  if (!td) return;
  const publisher = /** @type {HTMLElement} */(td).dataset.publisher;
  const date = /** @type {HTMLElement} */(td).dataset.date;
  if (publisher && date && onCellSelect) {
    onCellSelect(publisher, date);
  }
}

/** @param {MouseEvent} e */
function handleCellHover(e) {
  const td = /** @type {HTMLElement} */(e.target).closest('.scrape-matrix__cell');
  if (!td) return;

  const el = /** @type {HTMLElement} */(td);
  const publisher = el.dataset.publisher || '';
  const date = el.dataset.date || '';
  const nScraped = el.dataset.nScraped || '0';
  const nPreprocessed = el.dataset.nPreprocessed || '0';
  const status = el.dataset.status || 'gray';

  const statusLabels = {
    green: 'OK — All preprocessed',
    yellow: 'Partial — Some not preprocessed',
    red: 'Inconsistent — preprocessed > scraped',
    gray: 'Empty — No data',
  };

  const html = [
    tooltipRow('Publisher', escapeHtml(publisher)),
    tooltipRow('Date', escapeHtml(date)),
    '<hr class="tooltip__separator">',
    tooltipRow('Scraped', escapeHtml(nScraped)),
    tooltipRow('Preprocessed', escapeHtml(nPreprocessed)),
    tooltipRow('Status', `<span class="tooltip__val--${status === 'green' ? 'ok' : status === 'red' ? 'failed' : status === 'yellow' ? 'blocked' : 'pending'}">${escapeHtml(statusLabels[status] || status)}</span>`),
  ].join('');

  showTooltip(el, html);
}

/** @param {MouseEvent} _e */
function handleCellOut(_e) {
  hideTooltip();
}

// ── Helpers ─────────────────────────────────────────────────────────

/**
 * @param {string} key
 * @param {string} valHtml
 * @returns {string}
 */
function tooltipRow(key, valHtml) {
  return `<div class="tooltip__row"><span class="tooltip__key">${escapeHtml(key)}</span><span class="tooltip__val">${valHtml}</span></div>`;
}
