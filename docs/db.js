/**
 * db.js — sql.js wrapper with in-memory and persistent caching.
 *
 * Manages SQLite database lifecycle: init sql.js WASM, fetch .sqlite
 * files over HTTP, cache ArrayBuffers in memory + IndexedDB, and
 * provide query helpers that return plain objects.
 *
 * Cache validation: stored ETag / Last-Modified headers are sent as
 * conditional-request headers on subsequent loads.  A 304 reuses the
 * cache; a 200 replaces it; a network error falls back to stale cache.
 *
 * @module db
 */

/** @type {any} */ let SQL = null;
/** @type {Map<string, any>} */ const dbInstances = new Map();
/** @type {Map<string, ArrayBuffer>} */ const bufferCache = new Map();

const IDB_NAME = 'run-explorer-cache';
const IDB_STORE = 'sqlite-files';
const IDB_VERSION = 2; // bumped: values are now {buffer, etag, lastModified}

// ── sql.js init ─────────────────────────────────────────────────────

/** Initialize sql.js WASM. Idempotent. */
export async function initDB() {
  if (SQL) return;
  SQL = await window.initSqlJs({
    locateFile: f => `https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.10.3/${f}`
  });
}

// ── IndexedDB cache ─────────────────────────────────────────────────

/**
 * @typedef {Object} CacheEntry
 * @property {ArrayBuffer} buffer
 * @property {string|null} etag
 * @property {string|null} lastModified
 */

/** @returns {Promise<IDBDatabase>} */
function openIDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, IDB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      // Delete old store and recreate (schema changed from v1)
      if (db.objectStoreNames.contains(IDB_STORE)) {
        db.deleteObjectStore(IDB_STORE);
      }
      db.createObjectStore(IDB_STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/**
 * @param {string} key
 * @returns {Promise<CacheEntry|null>}
 */
async function idbGet(key) {
  try {
    const db = await openIDB();
    return new Promise((resolve) => {
      const tx = db.transaction(IDB_STORE, 'readonly');
      const req = tx.objectStore(IDB_STORE).get(key);
      req.onsuccess = () => {
        const val = req.result;
        if (!val) return resolve(null);
        // Discard legacy entries (plain ArrayBuffer without metadata)
        if (val instanceof ArrayBuffer) return resolve(null);
        if (val && val.buffer instanceof ArrayBuffer) return resolve(val);
        resolve(null);
      };
      req.onerror = () => resolve(null);
    });
  } catch { return null; }
}

/**
 * @param {string} key
 * @param {CacheEntry} entry
 */
async function idbPut(key, entry) {
  try {
    const db = await openIDB();
    const tx = db.transaction(IDB_STORE, 'readwrite');
    tx.objectStore(IDB_STORE).put(entry, key);
  } catch { /* non-fatal */ }
}

/** Clear the entire IndexedDB cache store. */
async function idbClearAll() {
  try {
    const db = await openIDB();
    const tx = db.transaction(IDB_STORE, 'readwrite');
    tx.objectStore(IDB_STORE).clear();
  } catch { /* non-fatal */ }
}

// ── Database loading ────────────────────────────────────────────────

/**
 * Fetch a .sqlite file, validating any cached version via conditional
 * HTTP requests (If-None-Match / If-Modified-Since).
 *
 * Flow:
 *   1. If an in-memory db instance exists, return it (same page session).
 *   2. Look up IDB cache entry (buffer + etag + lastModified).
 *   3. Issue a conditional GET.  On 304 → reuse cached buffer.
 *      On 200 → store new buffer + headers in IDB.
 *      On network error → fall back to stale cache if available.
 *   4. Create sql.js Database from buffer.
 *
 * @param {string} url — path to .sqlite file
 * @returns {Promise<any>} sql.js Database
 */
export async function openDatabase(url) {
  if (dbInstances.has(url)) return dbInstances.get(url);
  await initDB();

  // Retrieve cached entry (buffer + validation headers)
  let cached = null;
  const memBuf = bufferCache.get(url);
  if (memBuf) {
    cached = await idbGet(url);
    if (!cached) {
      // Memory buffer without IDB metadata — unvalidated
      cached = { buffer: memBuf, etag: null, lastModified: null };
    }
  } else {
    cached = await idbGet(url);
    if (cached) bufferCache.set(url, cached.buffer);
  }

  // Build conditional request headers
  const reqHeaders = {};
  if (cached?.etag) reqHeaders['If-None-Match'] = cached.etag;
  if (cached?.lastModified) reqHeaders['If-Modified-Since'] = cached.lastModified;

  let buf = null;

  try {
    const resp = await fetch(url, { headers: reqHeaders });

    if (resp.status === 304 && cached) {
      // Server confirms cache is still valid
      buf = cached.buffer;
    } else if (resp.ok) {
      // New or updated content
      buf = await resp.arrayBuffer();

      const newEntry = {
        buffer: buf,
        etag: resp.headers.get('ETag') || null,
        lastModified: resp.headers.get('Last-Modified') || null,
      };

      bufferCache.set(url, buf);
      idbPut(url, newEntry); // fire-and-forget
    } else {
      throw new Error(`HTTP ${resp.status} fetching ${url}`);
    }
  } catch (err) {
    // Network error — fall back to stale cache if we have one
    if (cached) {
      console.warn(`Network error for ${url}, using cached version:`, err.message);
      buf = cached.buffer;
    } else {
      throw err; // no cache to fall back on
    }
  }

  const db = new SQL.Database(new Uint8Array(buf));
  dbInstances.set(url, db);
  return db;
}

/**
 * Close and release a database instance + its cached buffer.
 * @param {string} url
 */
export function closeDatabase(url) {
  const db = dbInstances.get(url);
  if (db) {
    db.close();
    dbInstances.delete(url);
  }
  bufferCache.delete(url);
}

/**
 * Close ALL open database instances and clear every cache layer
 * (in-memory maps + IndexedDB).  Call before a full data reload.
 */
export async function clearAllCaches() {
  for (const [, db] of dbInstances) {
    try { db.close(); } catch { /* ignore */ }
  }
  dbInstances.clear();
  bufferCache.clear();
  await idbClearAll();
}

// ── Query helpers ───────────────────────────────────────────────────

/**
 * Run a query and return all rows as plain objects.
 * @param {any} db — sql.js Database
 * @param {string} sql
 * @param {Array} [params]
 * @returns {Array<Record<string, unknown>>}
 */
export function queryAll(db, sql, params = []) {
  const stmt = db.prepare(sql);
  stmt.bind(params);
  const rows = [];
  while (stmt.step()) {
    rows.push(stmt.getAsObject());
  }
  stmt.free();
  return rows;
}

/**
 * Run a query and return the first row (or null).
 * @param {any} db — sql.js Database
 * @param {string} sql
 * @param {Array} [params]
 * @returns {Record<string, unknown>|null}
 */
export function queryOne(db, sql, params = []) {
  const stmt = db.prepare(sql);
  stmt.bind(params);
  const row = stmt.step() ? stmt.getAsObject() : null;
  stmt.free();
  return row;
}

/**
 * Run a query and return a single scalar value.
 * @param {any} db — sql.js Database
 * @param {string} sql
 * @param {Array} [params]
 * @returns {unknown}
 */
export function queryScalar(db, sql, params = []) {
  const row = queryOne(db, sql, params);
  if (!row) return null;
  const keys = Object.keys(row);
  return keys.length > 0 ? row[keys[0]] : null;
}