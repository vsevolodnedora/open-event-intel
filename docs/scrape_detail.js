/**
 * scrape_detail.js — Scrape publications detail panel renderer.
 *
 * When a publisher×date cell is clicked in the scrape matrix,
 * this module renders a table of individual publications.
 *
 * @module scrape_detail
 */

import { escapeHtml } from './tooltip.js';

// ── DOM refs ────────────────────────────────────────────────────────

/** @type {HTMLElement} */ let placeholderEl;
/** @type {HTMLElement} */ let contentEl;
/** @type {HTMLElement} */ let headerEl;
/** @type {HTMLElement} */ let tbodyEl;

/** Initialize the scrape detail panel DOM references. */
export function initScrapeDetail() {
  placeholderEl = /** @type {HTMLElement} */(document.getElementById('scrape-detail-placeholder'));
  contentEl = /** @type {HTMLElement} */(document.getElementById('scrape-detail-content'));
  headerEl = /** @type {HTMLElement} */(document.getElementById('scrape-detail-header'));
  tbodyEl = /** @type {HTMLElement} */(document.getElementById('scrape-detail-tbody'));
}

/**
 * Render publications for a publisher×date cell.
 * @param {string} publisher
 * @param {string} date
 * @param {object|null} data — validated publications JSON
 */
export function renderScrapeDetail(publisher, date, data) {
  placeholderEl.hidden = true;
  contentEl.hidden = false;

  // Header
  headerEl.innerHTML = '';
  const titleEl = document.createElement('div');
  titleEl.className = 'trace__doc-title';
  titleEl.textContent = `${publisher} — ${date}`;
  headerEl.appendChild(titleEl);

  const metaEl = document.createElement('div');
  metaEl.className = 'trace__doc-meta';

  if (data && data.publications) {
    const pubs = /** @type {Array} */(data.publications);
    const s = document.createElement('span');
    s.textContent = `${pubs.length} publication${pubs.length !== 1 ? 's' : ''}`;
    metaEl.appendChild(s);
  }

  headerEl.appendChild(metaEl);

  // Table body
  tbodyEl.innerHTML = '';

  if (!data || !data.publications || data.publications.length === 0) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 7;
    td.style.textAlign = 'center';
    td.style.color = 'var(--c-text-dim)';
    td.style.fontStyle = 'italic';
    td.style.padding = '2rem';
    td.textContent = 'No publications found for this publisher and date.';
    tr.appendChild(td);
    tbodyEl.appendChild(tr);
    return;
  }

  const publications = /** @type {Array<Record<string, unknown>>} */(data.publications);

  for (const pub of publications) {
    const tr = document.createElement('tr');

    // Title
    const titleTd = document.createElement('td');
    titleTd.textContent = String(pub.title || '(no title)');
    titleTd.title = String(pub.title || '');
    tr.appendChild(titleTd);

    // Published on
    const pubOnTd = document.createElement('td');
    pubOnTd.textContent = formatTimestamp(pub.published_on);
    tr.appendChild(pubOnTd);

    // Scraped on
    const scrapedTd = document.createElement('td');
    scrapedTd.textContent = formatTimestamp(pub.scraped_on);
    tr.appendChild(scrapedTd);

    // Language
    const langTd = document.createElement('td');
    langTd.textContent = String(pub.language || '—');
    tr.appendChild(langTd);

    // Length before preprocessing
    const lenBeforeTd = document.createElement('td');
    const lenBefore = pub.length_before_preprocessing;
    lenBeforeTd.textContent = lenBefore != null ? formatNumber(Number(lenBefore)) : '—';
    tr.appendChild(lenBeforeTd);

    // Length after preprocessing
    const lenAfterTd = document.createElement('td');
    const lenAfter = pub.length_after_preprocessing;
    if (lenAfter == null) {
      lenAfterTd.textContent = '—';
      lenAfterTd.className = 'scrape-detail__len-missing';
    } else {
      lenAfterTd.textContent = formatNumber(Number(lenAfter));
      // Highlight if lengths differ significantly
      if (lenBefore != null && Number(lenAfter) > Number(lenBefore)) {
        lenAfterTd.className = 'scrape-detail__len-mismatch';
      }
    }
    tr.appendChild(lenAfterTd);

    // URL
    const urlTd = document.createElement('td');
    if (pub.url) {
      const a = document.createElement('a');
      a.href = String(pub.url);
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      a.textContent = truncateUrl(String(pub.url));
      a.title = String(pub.url);
      urlTd.appendChild(a);
    } else {
      urlTd.textContent = '—';
    }
    tr.appendChild(urlTd);

    tbodyEl.appendChild(tr);
  }
}

/** Show the placeholder (no cell selected). */
export function showScrapeDetailPlaceholder() {
  placeholderEl.hidden = false;
  contentEl.hidden = true;
}

// ── Helpers ─────────────────────────────────────────────────────────

/**
 * Format a timestamp to compact display.
 * @param {unknown} val
 * @returns {string}
 */
function formatTimestamp(val) {
  if (!val) return '—';
  const s = String(val);
  // Show YYYY-MM-DD HH:MM
  return s.slice(0, 16).replace('T', ' ');
}

/**
 * Format a number with locale separators.
 * @param {number} n
 * @returns {string}
 */
function formatNumber(n) {
  return n.toLocaleString();
}

/**
 * Truncate a URL for display.
 * @param {string} url
 * @returns {string}
 */
function truncateUrl(url) {
  try {
    const u = new URL(url);
    const path = u.pathname.length > 30 ? u.pathname.slice(0, 27) + '…' : u.pathname;
    return u.hostname + path;
  } catch {
    return url.length > 50 ? url.slice(0, 47) + '…' : url;
  }
}
