/**
 * tooltip.js — Shared tooltip positioning and content rendering.
 *
 * Uses a single tooltip element, repositioned on hover via safe DOM APIs.
 *
 * @module tooltip
 */

/** @type {HTMLElement|null} */
let tooltipEl = null;
/** @type {HTMLElement|null} */
let contentEl = null;

/** Bind to DOM elements. Call once on init. */
export function initTooltip() {
  tooltipEl = document.getElementById('tooltip');
  contentEl = document.getElementById('tooltip-content');
}

/**
 * Show the tooltip near a target element with HTML-safe content.
 * @param {HTMLElement} anchor — element to position near
 * @param {string} html — pre-sanitized HTML content
 */
export function showTooltip(anchor, html) {
  if (!tooltipEl || !contentEl) return;

  contentEl.innerHTML = ''; // clear previous
  // Build content safely by setting innerHTML only with our own generated markup
  const wrapper = document.createElement('div');
  wrapper.innerHTML = html;
  contentEl.appendChild(wrapper);

  tooltipEl.hidden = false;

  // Position
  const rect = anchor.getBoundingClientRect();
  const ttRect = tooltipEl.getBoundingClientRect();
  const vw = window.innerWidth;
  const vh = window.innerHeight;

  let left = rect.left + rect.width / 2 - ttRect.width / 2;
  let top = rect.bottom + 6;

  // Flip up if near bottom
  if (top + ttRect.height > vh - 8) {
    top = rect.top - ttRect.height - 6;
  }
  // Clamp horizontal
  left = Math.max(8, Math.min(left, vw - ttRect.width - 8));
  top = Math.max(8, top);

  tooltipEl.style.left = `${left}px`;
  tooltipEl.style.top = `${top}px`;
}

/** Hide the tooltip. */
export function hideTooltip() {
  if (tooltipEl) tooltipEl.hidden = true;
}

// ── Content builders (return safe HTML strings) ─────────────────────

/**
 * Build tooltip HTML for a matrix cell.
 * @param {string} stageId
 * @param {string} status
 * @param {object|null} statusEntry — full status object from docs.json
 * @param {object|null} counts — totals or diff counts
 * @returns {string}
 */
export function buildCellTooltipHTML(stageId, status, statusEntry, counts) {
  const rows = [];
  rows.push(tooltipRow('Stage', escapeHtml(stageId)));
  rows.push(tooltipRow('Status', `<span class="tooltip__val--${status}">${escapeHtml(status)}</span>`));

  if (statusEntry) {
    const se = /** @type {Record<string, unknown>} */(statusEntry);
    if (se.attempt) rows.push(tooltipRow('Attempt', escapeHtml(String(se.attempt))));
    if (se.last_run_id) rows.push(tooltipRow('Last run', escapeHtml(truncate(String(se.last_run_id), 16))));
    if (se.processed_at) rows.push(tooltipRow('Processed', escapeHtml(String(se.processed_at))));

    if (se.error_message) {
      rows.push('<hr class="tooltip__separator">');
      if (status === 'blocked') {
        rows.push(`<div class="tooltip__key">Blocked reason:</div>`);
      } else {
        rows.push(`<div class="tooltip__key">Error:</div>`);
      }
      rows.push(`<div class="tooltip__error">${escapeHtml(String(se.error_message))}</div>`);
    }
  }

  if (counts) {
    rows.push('<hr class="tooltip__separator">');
    rows.push(`<div class="tooltip__key">Counts:</div>`);
    for (const [k, v] of Object.entries(counts)) {
      if (v !== null && v !== undefined) {
        rows.push(tooltipRow(k, escapeHtml(String(v))));
      }
    }
  }

  return rows.join('');
}

/**
 * Build tooltip HTML for a run-scoped stage cell.
 * @param {string} stageId
 * @param {object|null} statusEntry
 * @returns {string}
 */
export function buildRunStageTooltipHTML(stageId, statusEntry) {
  const rows = [];
  rows.push(tooltipRow('Stage', escapeHtml(stageId)));

  if (!statusEntry) {
    rows.push(tooltipRow('Status', '<span class="tooltip__val--pending">not started</span>'));
    return rows.join('');
  }

  const se = /** @type {Record<string, unknown>} */(statusEntry);
  const status = String(se.status || 'unknown');
  rows.push(tooltipRow('Status', `<span class="tooltip__val--${status}">${escapeHtml(status)}</span>`));
  if (se.attempt) rows.push(tooltipRow('Attempt', escapeHtml(String(se.attempt))));
  if (se.started_at) rows.push(tooltipRow('Started', escapeHtml(String(se.started_at))));
  if (se.completed_at) rows.push(tooltipRow('Completed', escapeHtml(String(se.completed_at))));

  if (se.error_message) {
    rows.push('<hr class="tooltip__separator">');
    rows.push(`<div class="tooltip__error">${escapeHtml(String(se.error_message))}</div>`);
  }

  return rows.join('');
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

/**
 * @param {string} str
 * @returns {string}
 */
export function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * @param {string} str
 * @param {number} max
 * @returns {string}
 */
function truncate(str, max) {
  return str.length > max ? str.slice(0, max) + '…' : str;
}
