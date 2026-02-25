/**
 * injector.js — runs in MAIN world (page context)
 *
 * Hooks XMLHttpRequest (used by superagent/Redux-Saga in Robinhood's stack)
 * to intercept options API responses.
 * Cannot use chrome.* APIs here — communicates via window.postMessage.
 *
 * Endpoints captured:
 *   - /marketdata/options/   → live quotes + Greeks
 *   - /options/instruments/  → instrument metadata (strike, expiry, call/put)
 *
 * For marketdata/options, only contracts within ATM_RANGE strikes of the
 * at-the-money strike (derived from delta) are forwarded.
 */
(function () {
  const TRACKED_ENDPOINTS = [
    'marketdata/options',
    'options/instruments',
  ];

  // How many strike points either side of ATM to capture.
  // For SPX (~5900) ±300 covers roughly the top 50 strikes each side.
  const ATM_RANGE = 300;

  // ── helpers ──────────────────────────────────────────────────────────────

  // Parse OCC symbol → { type: 'C'|'P', strike: number }
  // Format: "SPXW  260225C06965000" — last 15 chars are YYMMDD + type + strike*1000
  function parseOcc(occ) {
    if (!occ) return null;
    const tail = occ.trim().slice(-15);
    return { type: tail[6], strike: parseInt(tail.slice(7), 10) / 1000 };
  }

  // Filter a marketdata/options payload to near-money contracts only.
  // Passes options/instruments payloads through unchanged.
  function filterNearMoney(data, url) {
    if (!url.includes('marketdata/options')) return data;
    const results = data && data.results;
    if (!Array.isArray(results)) return data;

    // Step 1: find ATM strike — call with delta closest to 0.5
    let atmStrike = null;
    const callsWithDelta = [];
    for (const r of results) {
      const d = parseFloat(r.delta);
      if (isNaN(d)) continue;
      const st = parseOcc(r.occ_symbol);
      if (st && st.type === 'C') callsWithDelta.push({ delta: d, strike: st.strike });
    }
    if (callsWithDelta.length > 0) {
      callsWithDelta.sort((a, b) => Math.abs(a.delta - 0.5) - Math.abs(b.delta - 0.5));
      atmStrike = callsWithDelta[0].strike;
    }

    // Step 2: keep only contracts with Greeks and within ATM_RANGE of ATM
    const filtered = results.filter(r => {
      if (!r.delta || r.delta === '') return false;            // no Greeks → drop
      if (atmStrike === null) return true;                     // can't detect ATM → keep all
      const st = parseOcc(r.occ_symbol);
      return st && Math.abs(st.strike - atmStrike) <= ATM_RANGE;
    });

    return Object.assign({}, data, { results: filtered });
  }

  // Send filtered data to the relay
  function processAndPost(url, rawData) {
    const payload = filterNearMoney(rawData, url);
    // Skip if filtering removed everything (avoids storing empty batches)
    if (
      url.includes('marketdata/options') &&
      Array.isArray(payload && payload.results) &&
      payload.results.length === 0
    ) return;
    window.postMessage(
      { __rh_tracker: true, type: 'OPTIONS_FETCH', url, data: payload, ts: Date.now() },
      '*'
    );
  }

  // ── XHR hook (Robinhood uses superagent → XHR, not fetch) ────────────────
  const originalOpen = XMLHttpRequest.prototype.open;
  const originalSend = XMLHttpRequest.prototype.send;

  XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    this._rhUrl = typeof url === 'string' ? url : '';
    return originalOpen.apply(this, [method, url, ...rest]);
  };

  XMLHttpRequest.prototype.send = function (...args) {
    const url = this._rhUrl || '';
    const isTracked = TRACKED_ENDPOINTS.some((ep) => url.includes(ep));

    if (isTracked) {
      this.addEventListener('load', () => {
        if (this.status === 200) {
          try {
            const data = JSON.parse(this.responseText);
            processAndPost(url, data);
          } catch (e) {}
        }
      });
    }

    return originalSend.apply(this, args);
  };

  // ── fetch hook (belt-and-suspenders) ─────────────────────────────────────
  const originalFetch = window.fetch;

  window.fetch = async function (...args) {
    const response = await originalFetch.apply(this, args);

    const url =
      typeof args[0] === 'string'
        ? args[0]
        : args[0] instanceof Request
        ? args[0].url
        : '';

    const isTracked = TRACKED_ENDPOINTS.some((ep) => url.includes(ep));

    if (isTracked) {
      response
        .clone()
        .json()
        .then((data) => processAndPost(url, data))
        .catch(() => {});
    }

    return response;
  };

  console.log('[RH Tracker] XHR + fetch hooks installed (ATM_RANGE=±' + ATM_RANGE + ')');
})();
