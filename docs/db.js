/**
 * db.js — sql.js wrapper with in-memory and persistent caching.
 *
 * Manages SQLite database lifecycle: init sql.js WASM, fetch .sqlite
 * files over HTTP, cache ArrayBuffers in memory + IndexedDB, and
 * provide query helpers that return plain objects.
 *
 * @module db
 */

/** @type {any} */ let SQL = null;
/** @type {Map<string, any>} */ const dbInstances = new Map();
/** @type {Map<string, ArrayBuffer>} */ const bufferCache = new Map();

const IDB_NAME = 'run-explorer-cache';
const IDB_STORE = 'sqlite-files';
const IDB_VERSION = 1;

// ── sql.js init ─────────────────────────────────────────────────────

/** Initialize sql.js WASM. Idempotent. */
export async function initDB() {
  if (SQL) return;
  SQL = await window.initSqlJs({
    locateFile: f => `https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.10.3/${f}`
  });
}

// ── IndexedDB cache ─────────────────────────────────────────────────

/** @returns {Promise<IDBDatabase>} */
function openIDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, IDB_VERSION);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(IDB_STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/**
 * @param {string} key
 * @returns {Promise<ArrayBuffer|null>}
 */
async function idbGet(key) {
  try {
    const db = await openIDB();
    return new Promise((resolve) => {
      const tx = db.transaction(IDB_STORE, 'readonly');
      const req = tx.objectStore(IDB_STORE).get(key);
      req.onsuccess = () => resolve(req.result ?? null);
      req.onerror = () => resolve(null);
    });
  } catch { return null; }
}

/**
 * @param {string} key
 * @param {ArrayBuffer} value
 */
async function idbPut(key, value) {
  try {
    const db = await openIDB();
    const tx = db.transaction(IDB_STORE, 'readwrite');
    tx.objectStore(IDB_STORE).put(value, key);
  } catch { /* non-fatal */ }
}

// ── Database loading ────────────────────────────────────────────────

/**
 * Fetch a .sqlite file, using memory + IDB cache layers.
 * Returns a sql.js Database instance.
 *
 * @param {string} url — path to .sqlite file
 * @returns {Promise<any>} sql.js Database
 */
export async function openDatabase(url) {
  if (dbInstances.has(url)) return dbInstances.get(url);
  await initDB();

  let buf = bufferCache.get(url) ?? null;

  // Try IndexedDB cache
  if (!buf) {
    buf = await idbGet(url);
    if (buf) bufferCache.set(url, buf);
  }

  // Fetch from network
  if (!buf) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`);
    buf = await resp.arrayBuffer();
    bufferCache.set(url, buf);
    idbPut(url, buf); // fire-and-forget
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
