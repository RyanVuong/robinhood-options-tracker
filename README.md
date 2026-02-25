# Robinhood Options Tracker

A personal real-time SPX options intelligence system. Captures live options data directly from Robinhood's browser network layer, stores it locally, and runs a self-improving ML signal engine that surfaces actionable trade signals with confidence scores and plain-English reasoning.

---

## Architecture

```
┌─────────────────────────────────────────┐
│          Chrome Extension               │
│  injector.js  (MAIN world)              │  hooks XHR at browser level
│  relay.js     (ISOLATED world)          │  bridges postMessage → chrome runtime
│  background.js (service worker)         │  POSTs data to local server
└──────────────────┬──────────────────────┘
                   │ HTTP POST /ingest
┌──────────────────▼──────────────────────┐
│       Node.js Server  :7842             │
│  server.js                              │  ingests + deduplicates tick data
│  options.db  (SQLite)                   │  persists all captured rows
│  viewer.html                            │  live dashboard (3 tabs)
└──────────────────┬──────────────────────┘
                   │ reads SQLite directly
┌──────────────────▼──────────────────────┐
│       Python Signal Engine  :7843       │
│  signal_server.py                       │  computes features + ML signals
└─────────────────────────────────────────┘
```

## How It Works

**Data capture**: The Chrome extension hooks into `XMLHttpRequest` at the browser level (Robinhood uses superagent/XHR, not fetch) to intercept live options responses before they reach the page. Each response is filtered to contracts within ±300 strikes of ATM, then forwarded to the local Node server.

**Storage**: The Node server ingests and persists every tick to a local SQLite database. The live dashboard at `http://127.0.0.1:7842` merges the last 30 seconds of ticks by contract, highlights ATM, and auto-refreshes every 2 seconds.

**Signal engine**: A Python server reads the full SQLite history, computes 18 features (momentum windows, Greeks, intraday context, SPX daily %) and applies Triple Barrier labeling. Starts with rule-based heuristics, automatically upgrades to LightGBM once 200 labeled samples accumulate — with async retraining that never blocks live signal delivery.

---

## Stack

| Layer | Tech |
|---|---|
| Extension | Chrome MV3, Manifest V3, content scripts |
| Server | Node.js 24 (no npm deps — uses built-in `node:sqlite`) |
| Dashboard | Vanilla JS, HTML/CSS |
| Signal engine | Python, LightGBM, SHAP, pandas, scikit-learn |
| Storage | SQLite |

---

## Setup

### Prerequisites
- Node.js 24+ (required for built-in `node:sqlite`)
- Python 3.9+
- Google Chrome

### 1. Install the Chrome extension

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked** → select the `extension/` folder

### 2. Start the Node server

```bash
cd server
node server.js
```

Dashboard available at `http://127.0.0.1:7842`

### 3. Start the signal engine

```bash
pip install -r analysis/requirements.txt
py analysis/signal_server.py
```

Signals available at `http://127.0.0.1:7843/signals`

### 4. Capture data

Navigate to Robinhood's options chain for SPX/SPXW. The extension automatically begins intercepting responses. Open the dashboard to see live data flowing in.

---

## Dashboard

Three tabs:

- **Live Snapshot** — merges the last 30s of ticks per contract, ATM highlighted, sortable by any column
- **Capture History** — raw log of all ingested rows
- **Signals** — ML signal cards with confidence bars, reasoning, and SPX daily context; polls port 7843

---

## Signal Engine

### Feature set (18 features)

| Category | Features |
|---|---|
| Momentum | `pct_30s`, `pct_60s`, `pct_120s` |
| Greeks | `delta`, `gamma`, `theta`, `vega`, `iv` |
| Market structure | `spread_pct`, `cop_long`, `type_c`, `moneyness_pct` |
| Intraday context | `spx_daily_pct`, `vs_yesterday_pct`, `intraday_range_pct`, `intraday_return_pct`, `price_position`, `intraday_vol` |

### Labeling

Triple Barrier method (López de Prado):
- **Profit barrier**: +5% → label 1
- **Stop barrier**: −2% → label 0
- **Time barrier**: 60 seconds → label 0

### Model phases

- **Phase 1** (< 200 labeled rows): rule-based momentum threshold + Laplace-smoothed empirical conditional probability
- **Phase 2** (≥ 200 labeled rows): LightGBM classifier (300 trees, 5-fold stratified CV) + SHAP feature attribution for per-signal reasoning

### Config (live, no restart)

```bash
curl -X POST http://127.0.0.1:7843/config \
  -H "Content-Type: application/json" \
  -d '{"momentum_thresh_pct": 2.5, "profit_barrier_pct": 5.0, "conf_display_min": 0.6}'
```

---

## Project Structure

```
robinhood-options-tracker/
├── extension/
│   ├── manifest.json       # Chrome MV3 manifest
│   ├── injector.js         # MAIN world — XHR hook, ATM filter
│   ├── relay.js            # ISOLATED world — postMessage bridge
│   ├── background.js       # Service worker — HTTP ingest relay
│   ├── popup.html          # Extension popup UI
│   └── popup.js            # Popup stats
├── server/
│   ├── server.js           # Node HTTP server (port 7842)
│   ├── viewer.html         # Live dashboard
│   └── package.json
└── analysis/
    ├── signal_server.py    # Python signal engine (port 7843)
    └── requirements.txt
```

---

## Notes

- `options.db` is excluded from version control (can grow to 100MB+ per session)
- Requires an active Robinhood account and SPX options chain open in Chrome
- All data stays local — nothing is sent to any external service
