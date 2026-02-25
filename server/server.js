/**
 * server.js — local ingestion server
 *
 * Listens on http://localhost:7842
 * POST /ingest  — receives options data from the Chrome extension
 * GET  /data    — returns stored rows as JSON (for inspection / UI)
 *
 * Storage: node:sqlite (built-in since Node 22, stable in Node 24 — no install needed)
 * Run: node server.js
 */

const fs   = require('fs');
const http = require('http');
const path = require('path');
const { DatabaseSync } = require('node:sqlite');

const PORT = 7842;
const DB_PATH = path.join(__dirname, 'options.db');

// ---------------------------------------------------------------------------
// Database setup
// ---------------------------------------------------------------------------
const db = new DatabaseSync(DB_PATH);

db.exec(`
  CREATE TABLE IF NOT EXISTS options_raw (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        INTEGER NOT NULL,
    endpoint  TEXT    NOT NULL,
    payload   TEXT    NOT NULL,
    created   INTEGER NOT NULL DEFAULT (unixepoch())
  );

  CREATE INDEX IF NOT EXISTS idx_ts ON options_raw(ts);
`);

const insertRow = db.prepare(
  'INSERT INTO options_raw (ts, endpoint, payload) VALUES (?, ?, ?)'
);

// ---------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------
const server = http.createServer((req, res) => {
  // CORS — allow requests from the extension / local tools
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  if (req.method === 'GET' && req.url === '/') {
    fs.readFile(path.join(__dirname, 'viewer.html'), (err, data) => {
      if (err) { res.writeHead(404); res.end('viewer.html not found'); return; }
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(data);
    });
    return;
  }

  if (req.method === 'POST' && req.url === '/ingest') {
    let body = '';
    req.on('data', (chunk) => (body += chunk));
    req.on('end', () => {
      try {
        const { url, data, ts } = JSON.parse(body);

        // Derive a short endpoint label from the URL
        const endpoint = url.includes('marketdata/options')
          ? 'marketdata/options'
          : url.includes('options/instruments')
          ? 'options/instruments'
          : 'unknown';

        insertRow.run(ts ?? Date.now(), endpoint, JSON.stringify(data));

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: true }));
      } catch (err) {
        console.error('[server] ingest error:', err.message);
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: false, error: err.message }));
      }
    });
    return;
  }

  if (req.method === 'GET' && req.url.startsWith('/data')) {
    const params = new URL(req.url, `http://localhost:${PORT}`).searchParams;
    const limit = Math.min(parseInt(params.get('limit') ?? '100', 10), 1000);
    const endpoint = params.get('endpoint') ?? null;

    let rows;
    if (endpoint) {
      rows = db
        .prepare(
          'SELECT * FROM options_raw WHERE endpoint = ? ORDER BY ts DESC LIMIT ?'
        )
        .all(endpoint, limit);
    } else {
      rows = db
        .prepare('SELECT * FROM options_raw ORDER BY ts DESC LIMIT ?')
        .all(limit);
    }

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(rows));
    return;
  }

  res.writeHead(404);
  res.end('Not found');
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`[RH Tracker server] listening on http://127.0.0.1:${PORT}`);
  console.log(`  POST /ingest  — receive data from extension`);
  console.log(`  GET  /data    — query stored rows`);
  console.log(`  DB   ${DB_PATH}`);
});
