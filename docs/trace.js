/**
 * trace.js — Document Trace Graph rendering.
 *
 * Renders a left-to-right stage-aligned view for a single document,
 * showing status nodes, artifact samples, and run-scoped stage detail.
 *
 * Run-scoped stages (05 Index, 07 Cluster, 09 Outputs, 10 Timeline,
 * 11 Validation) are rendered as individual columns — not a single
 * "Run Impact" bucket — each with its own status node and relevant
 * per-doc impact data.
 *
 * @module trace
 */

import { escapeHtml } from './tooltip.js';

/** @typedef {import('./state.js').AppState} AppState */

// ── Impact → stage mapping ──────────────────────────────────────────
// Maps impact JSON keys (from runs/<run>/impact/<dvid>.json) to the
// run-scoped stage they belong to.

/** @type {Record<string, string>} */
const IMPACT_KEY_TO_STAGE = {
  clusters:              'stage_07_cluster',
  metric_points:         'stage_09_outputs',
  alerts:                'stage_09_outputs',
  digest_items:          'stage_09_outputs',
  timeline_items:        'stage_10_timeline',
  validation_failures:   'stage_11_validation',
};

// Maps run-scoped stage IDs to the run_artifact_counts keys they own.
/** @type {Record<string, string[]>} */
const STAGE_ARTIFACT_COUNT_KEYS = {
  stage_05_index:      ['embedding_indexes', 'chunk_count_total'],
  stage_07_cluster:    ['clusters', 'cluster_memberships'],
  stage_09_outputs:    ['metric_series', 'metric_points', 'alerts', 'digest_items'],
  stage_10_timeline:   ['timeline_items'],
  stage_11_validation: ['validation_failures'],
};

// ── DOM refs ────────────────────────────────────────────────────────

/** @type {HTMLElement} */ let placeholderEl;
/** @type {HTMLElement} */ let contentEl;
/** @type {HTMLElement} */ let headerEl;
/** @type {HTMLElement} */ let graphEl;

/** Initialize trace panel DOM references. */
export function initTrace() {
  placeholderEl = /** @type {HTMLElement} */(document.getElementById('trace-placeholder'));
  contentEl = /** @type {HTMLElement} */(document.getElementById('trace-content'));
  headerEl = /** @type {HTMLElement} */(document.getElementById('trace-header'));
  graphEl = /** @type {HTMLElement} */(document.getElementById('trace-graph'));
}

/**
 * Render the trace for a document.
 * @param {AppState} state
 * @param {object} traceData — validated trace JSON
 * @param {object|null} impactData — validated impact JSON (or null)
 */
export function renderTrace(state, traceData, impactData) {
  placeholderEl.hidden = true;
  contentEl.hidden = false;

  const trace = /** @type {{ header: Record<string, unknown>, stage_samples: Record<string, Record<string, Array>> }} */(traceData);

  // Header
  renderTraceHeader(trace.header);

  // Graph
  graphEl.innerHTML = '';

  const stages = /** @type {Array<{stage_id: string, scope: string, label: string}>} */(state.meta?.stages || []);
  const docStages = stages.filter(s => s.scope === 'doc');
  const runStages = stages.filter(s => s.scope === 'run');

  // Find the doc index for this doc
  let docIndex = -1;
  if (state.docsData) {
    const docs = /** @type {Array<Record<string, unknown>>} */(state.docsData.docs);
    docIndex = docs.findIndex(d => d.doc_version_id === trace.header.doc_version_id);
  }

  // ── Per-document stage columns (01–08) ────────────────────────────

  for (let si = 0; si < docStages.length; si++) {
    const stageDef = docStages[si];
    const stageId = stageDef.stage_id;

    const colEl = createStageColumn(stageDef.label);

    // Status node
    let status = 'not_started';
    let statusEntry = null;
    if (docIndex >= 0 && state.docsData) {
      const statusRow = /** @type {Array} */(state.docsData.stage_status_by_doc)[docIndex];
      statusEntry = statusRow?.[si] ?? null;
      if (statusEntry) {
        status = String(/** @type {Record<string, unknown>} */(statusEntry).status || 'not_started');
      }
    }

    const statusNode = createStatusNode(status, statusEntry);
    colEl.appendChild(statusNode);

    // Samples from trace data
    const samples = trace.stage_samples[stageId];
    if (samples) {
      for (const [tableName, rows] of Object.entries(samples)) {
        if (rows && rows.length > 0) {
          const artifactNode = createArtifactNode(tableName, rows);
          colEl.appendChild(artifactNode);
        }
      }
    }

    graphEl.appendChild(colEl);

    // Separator between doc stages (except last before run section)
    if (si < docStages.length - 1) {
      graphEl.appendChild(createDivider());
    }
  }

  // ── Run-scoped stage columns ──────────────────────────────────────

  if (runStages.length > 0 && state.selectedRunId) {
    // Prepare run data references
    const runData = state.runDataCache[state.selectedRunId] ?? null;
    const runStatusArr = runData ? /** @type {Array} */(/** @type {any} */(runData).run_stage_status) : [];
    const artifactCounts = runData ? /** @type {Record<string, unknown>} */(/** @type {any} */(runData).run_artifact_counts) : {};

    // Pre-sort impact data by stage
    /** @type {Record<string, Record<string, Array>>} */
    const impactByStage = {};
    if (impactData) {
      for (const [key, rows] of Object.entries(/** @type {Record<string, Array>} */(impactData))) {
        if (!Array.isArray(rows) || rows.length === 0) continue;
        const targetStage = IMPACT_KEY_TO_STAGE[key];
        if (targetStage) {
          if (!impactByStage[targetStage]) impactByStage[targetStage] = {};
          impactByStage[targetStage][key] = rows;
        }
      }
    }

    // Visual separator between doc-scoped and run-scoped sections
    graphEl.appendChild(createDivider(true));

    for (let ri = 0; ri < runStages.length; ri++) {
      const stageDef = runStages[ri];
      const stageId = stageDef.stage_id;

      const colEl = createStageColumn(stageDef.label);
      colEl.classList.add('trace__stage-col--run');

      // Run stage status node
      const entry = runStatusArr[ri] ?? null;
      const runStatus = entry ? String(/** @type {Record<string, unknown>} */(entry).status || 'not_started') : 'not_started';
      const statusNode = createStatusNode(runStatus, entry);
      colEl.appendChild(statusNode);

      // Per-doc impact for this stage
      const stageImpact = impactByStage[stageId];
      if (stageImpact) {
        for (const [key, rows] of Object.entries(stageImpact)) {
          if (rows.length > 0) {
            const node = createImpactNode(key, rows);
            colEl.appendChild(node);
          }
        }
      }

      // Run-level artifact counts for this stage
      const countKeys = STAGE_ARTIFACT_COUNT_KEYS[stageId];
      if (countKeys && artifactCounts) {
        const relevantCounts = {};
        let hasAny = false;
        for (const k of countKeys) {
          const v = artifactCounts[k];
          if (v != null && v !== 0) {
            relevantCounts[k] = v;
            hasAny = true;
          }
        }
        if (hasAny) {
          const node = createRunArtifactCountsNode(relevantCounts);
          colEl.appendChild(node);
        }
      }

      // Show "no data" if column is otherwise empty (only status node)
      if (colEl.querySelectorAll('.trace__node').length <= 1 && runStatus === 'not_started') {
        const emptyNode = document.createElement('div');
        emptyNode.className = 'trace__node trace__node--pending';
        const label = document.createElement('div');
        label.className = 'trace__node-label';
        label.textContent = 'Not yet executed';
        emptyNode.appendChild(label);
        colEl.appendChild(emptyNode);
      }

      graphEl.appendChild(colEl);

      // Separator between run stages (except last)
      if (ri < runStages.length - 1) {
        graphEl.appendChild(createDivider());
      }
    }
  }
}

/** Show the placeholder (no doc selected). */
export function showTracePlaceholder() {
  placeholderEl.hidden = false;
  contentEl.hidden = true;
}

// ── Internal builders ───────────────────────────────────────────────

/**
 * Render the trace header bar.
 * @param {Record<string, unknown>} header
 */
function renderTraceHeader(header) {
  headerEl.innerHTML = '';

  const titleEl = document.createElement('div');
  titleEl.className = 'trace__doc-title';
  titleEl.textContent = String(header.title || '(no title)');

  const metaEl = document.createElement('div');
  metaEl.className = 'trace__doc-meta';

  const metaItems = [
    ['Publisher', header.publisher_id],
    ['Published', header.source_published_at ? String(header.source_published_at).slice(0, 10) : null],
    ['Language', header.primary_language],
    ['Quality', header.content_quality_score != null ? Number(header.content_quality_score).toFixed(2) : null],
    ['ID', header.doc_version_id ? String(header.doc_version_id).slice(0, 12) + '…' : null],
  ];

  for (const [label, val] of metaItems) {
    if (val) {
      const s = document.createElement('span');
      const keyEl = document.createElement('em');
      keyEl.textContent = `${label}: `;
      s.appendChild(keyEl);
      s.appendChild(document.createTextNode(String(val)));
      metaEl.appendChild(s);
    }
  }

  headerEl.appendChild(titleEl);
  headerEl.appendChild(metaEl);
}

/**
 * Create a stage column container.
 * @param {string} label
 * @returns {HTMLElement}
 */
function createStageColumn(label) {
  const col = document.createElement('div');
  col.className = 'trace__stage-col';

  const labelEl = document.createElement('div');
  labelEl.className = 'trace__stage-label';
  labelEl.textContent = label;
  col.appendChild(labelEl);

  return col;
}

/**
 * Create a divider element between stages.
 * @param {boolean} [isSection] — true for the doc→run section break
 * @returns {HTMLElement}
 */
function createDivider(isSection = false) {
  const sep = document.createElement('div');
  sep.className = 'trace__stage-divider' + (isSection ? ' trace__stage-divider--section' : '');
  sep.setAttribute('aria-hidden', 'true');
  return sep;
}

/**
 * Create a status node.
 * @param {string} status
 * @param {object|null} statusEntry
 * @returns {HTMLElement}
 */
function createStatusNode(status, statusEntry) {
  const node = document.createElement('div');
  node.className = `trace__node trace__node--${status}`;

  const statusLine = document.createElement('div');
  statusLine.className = 'trace__node-status';

  const marker = document.createElement('span');
  marker.className = `status-marker status-marker--${status}`;
  statusLine.appendChild(marker);

  const txt = document.createElement('span');
  txt.textContent = status.toUpperCase();
  statusLine.appendChild(txt);

  node.appendChild(statusLine);

  if (statusEntry) {
    const se = /** @type {Record<string, unknown>} */(statusEntry);

    // Show synthesized-from-artifacts info
    if (se.details) {
      try {
        const details = typeof se.details === 'string' ? JSON.parse(se.details) : se.details;
        if (details.synthesized_from_artifacts) {
          const synthEl = document.createElement('div');
          synthEl.className = 'trace__node-label';
          const counts = Object.entries(details.synthesized_from_artifacts)
            .map(([k, v]) => `${k}: ${v}`).join(', ');
          synthEl.textContent = `Inferred from: ${counts}`;
          node.appendChild(synthEl);
        }
      } catch { /* ignore parse errors */ }
    }

    if (se.error_message) {
      const errEl = document.createElement('div');
      errEl.className = status === 'blocked' ? 'trace__node-label' : 'trace__node-error';
      errEl.textContent = String(se.error_message);
      node.appendChild(errEl);
    }

    if (se.attempt && Number(se.attempt) > 1) {
      const attEl = document.createElement('div');
      attEl.className = 'trace__node-label';
      attEl.textContent = `Attempt ${se.attempt}`;
      node.appendChild(attEl);
    }
  }

  return node;
}

/**
 * Create an artifact node showing sample data.
 * @param {string} tableName
 * @param {Array<Record<string, unknown>>} rows
 * @returns {HTMLElement}
 */
function createArtifactNode(tableName, rows) {
  const node = document.createElement('div');
  node.className = 'trace__node';

  const label = document.createElement('div');
  label.className = 'trace__node-label';
  label.textContent = `${tableName} (${rows.length}${rows.length >= 5 ? '+' : ''})`;
  node.appendChild(label);

  const samplesEl = document.createElement('div');
  samplesEl.className = 'trace__node-samples';

  for (const row of rows.slice(0, 5)) {
    const item = document.createElement('div');
    item.className = 'trace__sample-item';
    item.textContent = summarizeRow(row);
    samplesEl.appendChild(item);
  }

  node.appendChild(samplesEl);
  return node;
}

/**
 * Create an impact node.
 * @param {string} key
 * @param {Array<Record<string, unknown>>} rows
 * @returns {HTMLElement}
 */
function createImpactNode(key, rows) {
  const node = document.createElement('div');
  node.className = 'trace__node trace__node--impact';

  const label = document.createElement('div');
  label.className = 'trace__node-label';
  label.textContent = `${formatImpactKey(key)} (${rows.length})`;
  node.appendChild(label);

  const samplesEl = document.createElement('div');
  samplesEl.className = 'trace__node-samples';

  for (const row of rows.slice(0, 3)) {
    const item = document.createElement('div');
    item.className = 'trace__sample-item';
    item.textContent = summarizeRow(row);
    samplesEl.appendChild(item);
  }

  node.appendChild(samplesEl);
  return node;
}

/**
 * Create a node showing run artifact aggregate counts.
 * @param {Record<string, unknown>} counts
 * @returns {HTMLElement}
 */
function createRunArtifactCountsNode(counts) {
  const node = document.createElement('div');
  node.className = 'trace__node';

  const label = document.createElement('div');
  label.className = 'trace__node-label';
  label.textContent = 'Run Totals';
  node.appendChild(label);

  const samplesEl = document.createElement('div');
  samplesEl.className = 'trace__node-samples';

  for (const [k, v] of Object.entries(counts)) {
    if (v != null && v !== 0) {
      const item = document.createElement('div');
      item.className = 'trace__sample-item';
      item.textContent = `${k}: ${v}`;
      samplesEl.appendChild(item);
    }
  }

  node.appendChild(samplesEl);
  return node;
}

// ── Helpers ─────────────────────────────────────────────────────────

/**
 * Summarize a row object into a compact string for display.
 * @param {Record<string, unknown>} row
 * @returns {string}
 */
function summarizeRow(row) {
  const parts = [];
  for (const [k, v] of Object.entries(row)) {
    if (v === null || v === undefined) continue;
    let s = String(v);
    if (s.length > 40) s = s.slice(0, 37) + '…';
    parts.push(`${k}=${s}`);
    if (parts.length >= 3) break;
  }
  return parts.join(' | ');
}

/**
 * Format impact key to a human label.
 * @param {string} key
 * @returns {string}
 */
function formatImpactKey(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}