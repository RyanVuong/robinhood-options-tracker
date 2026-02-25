/**
 * background.js — service worker
 *
 * Receives OPTIONS_FETCH messages from relay.js and forwards
 * the payload to the local database server at http://localhost:7842/ingest.
 *
 * Also tracks per-session stats (captures, errors) accessible from popup.
 */

const SERVER_URL = 'http://localhost:7842/ingest';

let stats = {
  captured: 0,
  errors: 0,
  lastTs: null,
};

chrome.runtime.onMessage.addListener((message) => {
  if (!message.__rh_tracker || message.type !== 'OPTIONS_FETCH') return;

  const { url, data, ts } = message;

  fetch(SERVER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, data, ts }),
  })
    .then((res) => {
      if (!res.ok) throw new Error(`Server responded ${res.status}`);
      stats.captured += 1;
      stats.lastTs = ts;
    })
    .catch((err) => {
      stats.errors += 1;
      console.error('[RH Tracker] Failed to send to server:', err.message);
    });
});

// Popup requests current stats
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === 'GET_STATS') {
    sendResponse(stats);
  }
});
