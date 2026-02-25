"""
signal_server.py — Options trading signal engine

Serves http://127.0.0.1:7843
  GET  /signals  — current trading signals (JSON)
  GET  /stats    — model status summary
  POST /config   — update thresholds live (no restart needed)

Algorithm:
  1. Load all options_raw rows from SQLite, build per-contract price timeseries
  2. Compute features: momentum windows (30s/60s/120s), Greeks, spread
  3. Triple Barrier labeling (López de Prado):
       profit barrier (+5%), stop barrier (-2%), time barrier (60s)
       → label 1 if profit hit first, 0 if stop/time hit first
  4. Phase 1 (< MIN_TRAIN_SAMPLES labeled rows):
       rule-based momentum signals + empirical conditional probability
       with Laplace smoothing
  5. Phase 2 (>= MIN_TRAIN_SAMPLES):
       LightGBM predict_proba as confidence
       SHAP TreeExplainer for per-signal reasoning

Run: py signal_server.py
"""

import json
import logging
import math
import sqlite3
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

# ── Optional heavy deps — graceful degradation ────────────────────────────────
try:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_PATH     = Path(__file__).parent.parent / "server" / "options.db"
SIGNAL_PORT = 7843

# ── Mutable config (POST /config to update live) ─────────────────────────────
_config = {
    "momentum_thresh_pct": 2.0,   # % move in 30s to trigger a rule signal
    "profit_barrier_pct":  5.0,   # Triple Barrier: take-profit target %
    "stop_barrier_pct":    2.0,   # Triple Barrier: stop-loss %
    "lookahead_s":         60,    # Triple Barrier: max seconds to look forward
    "min_train_samples":   200,   # labeled rows needed before switching to ML
    "conf_display_min":    0.0,   # hide signals below this confidence (0..1)
    "refresh_s":           3,     # background refresh interval (seconds)
}

# ── Globals (updated by background thread, read by HTTP handler) ──────────────
_lock          = threading.Lock()
_signals       = []
_model_state   = {
    "phase":        "rule",
    "n_labeled":    0,
    "n_positive":   0,
    "base_rate":    0.0,
    "auc":          None,
    "last_trained": None,
    "last_refresh": None,
    "error":        None,
}
_clf           = None   # fitted LGBMClassifier or None
_medians       = {}     # per-feature medians from training set (for imputation)
_shap_expl     = None   # shap.TreeExplainer or None

_retrain_pending = threading.Event()
_last_trained_n  = 0
_spx_context     = {}   # updated each refresh; included in /signals response

FEATURE_COLS = [
    # Short-term momentum
    "pct_30s", "pct_60s", "pct_120s",
    # Greeks + pricing
    "delta", "gamma", "theta", "vega", "iv",
    "spread_pct", "cop_long", "type_c",
    # SPX macro context (same value for all contracts at a given moment)
    "spx_daily_pct",       # how much SPX has moved since today's open
    "vs_yesterday_pct",    # how far current SPX is from yesterday's close
    "moneyness_pct",       # (this strike - current ATM) / ATM * 100
    # Intraday contract-level stats
    "intraday_range_pct",  # (today_high - today_low) / today_open — measures choppiness
    "intraday_return_pct", # (current - today_open) / today_open — daily P&L so far
    "price_position",      # 0 = at today's low, 1 = at today's high
    "intraday_vol",        # std of tick-level % returns today — measures intraday volatility
]

logging.basicConfig(
    level=logging.INFO,
    format="[signal %(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("signal")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flt(v):
    """Safe float conversion; returns None on failure or non-finite."""
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _parse_occ(occ):
    """Decode OCC symbol → (type 'C'/'P', strike float) or (None, None)."""
    if not occ:
        return None, None
    tail = occ.strip()[-15:]
    if len(tail) < 15:
        return None, None
    try:
        return tail[6], int(tail[7:]) / 1000
    except (ValueError, IndexError):
        return None, None


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_time_series():
    """
    Read all options_raw rows (marketdata/options endpoint) from SQLite.
    Returns {occ_symbol: DataFrame} sorted ascending by ts.
    Only includes rows with valid mark_price > 0.05, delta, and iv.
    Deduplicates (occ, ts) pairs — latest record wins.
    """
    if not DB_PATH.exists():
        return {}

    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    try:
        rows = conn.execute(
            "SELECT ts, payload FROM options_raw "
            "WHERE endpoint='marketdata/options' ORDER BY ts"
        ).fetchall()
    finally:
        conn.close()

    raw = defaultdict(dict)   # occ → {ts_ms: record_dict}

    for ts_ms, payload_str in rows:
        try:
            payload  = json.loads(payload_str)
            contracts = payload.get("results", [])
            if isinstance(payload, list):
                contracts = payload
        except (json.JSONDecodeError, TypeError):
            continue

        for c in contracts:
            occ = c.get("occ_symbol")
            if not occ:
                continue
            mark  = _flt(c.get("mark_price"))
            delta = _flt(c.get("delta"))
            iv    = _flt(c.get("implied_volatility"))
            if mark is None or mark <= 0.05 or delta is None or iv is None:
                continue

            bid  = _flt(c.get("bid_price")) or 0.0
            ask  = _flt(c.get("ask_price")) or 0.0
            opt_type, strike = _parse_occ(occ)

            raw[occ][ts_ms] = {
                "ts":         ts_ms,
                "mark":       mark,
                "bid":        bid,
                "ask":        ask,
                "delta":      delta,
                "gamma":      _flt(c.get("gamma")),
                "theta":      _flt(c.get("theta")),
                "vega":       _flt(c.get("vega")),
                "iv":         iv,
                "cop_long":   _flt(c.get("chance_of_profit_long")),
                "spread_pct": (ask - bid) / mark if mark > 0 else None,
                "type_c":     1 if opt_type == "C" else 0,
                "strike":     strike,
            }

    result = {}
    for occ, ts_map in raw.items():
        df = pd.DataFrame(sorted(ts_map.values(), key=lambda x: x["ts"]))
        result[occ] = df.reset_index(drop=True)
    return result


# ── SPX proxy context ─────────────────────────────────────────────────────────

def compute_spx_context(series):
    """
    Derive a SPX price proxy from the options data itself:
    The call with delta closest to 0.50 at each moment = at-the-money strike = SPX.

    Returns a dict with:
        current           — current SPX proxy (latest ATM strike)
        today_open        — ATM strike at start of today's session
        yesterday_close   — ATM strike at end of yesterday's session (None if no prior day)
        daily_pct         — % change from today's open to now
        vs_yesterday_pct  — % change from yesterday's close to now (None if unavailable)
        today_open_ts     — epoch ms of today's first data point (used for intraday calcs)
    """
    # Build a flat list of (ts, delta, strike) for all call options
    all_rows = []
    for occ, df in series.items():
        opt_type, strike = _parse_occ(occ)
        if opt_type != "C" or strike is None:
            continue
        mask = df["delta"].notna()
        if not mask.any():
            continue
        sub = df.loc[mask, ["ts", "delta"]].copy()
        sub["strike"] = strike
        all_rows.append(sub)

    if not all_rows:
        return {}

    calls = pd.concat(all_rows, ignore_index=True)
    calls["delta_dist"] = (calls["delta"] - 0.50).abs()

    # For each timestamp, pick the call with delta closest to 0.50 → ATM = SPX proxy
    atm_idx = calls.groupby("ts")["delta_dist"].idxmin()
    atm = calls.loc[atm_idx, ["ts", "strike"]].rename(columns={"strike": "spx"})
    atm = atm.sort_values("ts").reset_index(drop=True)

    atm["date"] = pd.to_datetime(atm["ts"], unit="ms").dt.date
    today        = atm["date"].max()
    today_rows   = atm[atm["date"] == today]
    yest_rows    = atm[atm["date"] < today]

    current       = float(atm.iloc[-1]["spx"])
    today_open    = float(today_rows.iloc[0]["spx"]) if not today_rows.empty else current
    today_open_ts = int(today_rows.iloc[0]["ts"])   if not today_rows.empty else int(atm.iloc[0]["ts"])
    yest_close    = float(yest_rows.iloc[-1]["spx"]) if not yest_rows.empty else None

    daily_pct     = (current - today_open)  / today_open  * 100 if today_open  > 0 else 0.0
    vs_yest_pct   = (current - yest_close)  / yest_close  * 100 if yest_close  and yest_close > 0 else None

    return {
        "current":          round(current, 0),
        "today_open":       round(today_open, 0),
        "yesterday_close":  round(yest_close, 0) if yest_close else None,
        "daily_pct":        round(daily_pct, 2),
        "vs_yesterday_pct": round(vs_yest_pct, 2) if vs_yest_pct is not None else None,
        "today_open_ts":    today_open_ts,
    }


# ── Intraday contract stats ────────────────────────────────────────────────────

def compute_intraday_features(df, today_open_ts):
    """
    Add intraday stats to a per-contract DataFrame.
    All values computed from today's data only (rows with ts >= today_open_ts).

    Added columns:
        intraday_range_pct   — (today_high - today_low) / today_open * 100
                                High value = contract has swung a lot today (choppy / volatile)
        intraday_return_pct  — (current_mark - today_open_mark) / today_open_mark * 100
                                How much the contract has gained/lost since market open
        price_position       — 0 = at today's low, 1 = at today's high
                                Helps answer "is this near the top or bottom of today's range?"
        intraday_vol         — std of tick-level % returns since market open (as %)
                                Higher = more chaotic tick-by-tick movement
    """
    today = df[df["ts"] >= today_open_ts].copy()
    if today.empty:
        today = df.copy()   # fallback if no data within today's window

    open_p = float(today.iloc[0]["mark"])  if not today.empty else None
    high_p = float(today["mark"].max())    if not today.empty else None
    low_p  = float(today["mark"].min())    if not today.empty else None

    if open_p is None or open_p <= 0:
        for col in ["intraday_range_pct", "intraday_return_pct", "price_position", "intraday_vol"]:
            df[col] = np.nan
        return df

    range_pct = (high_p - low_p) / open_p * 100

    # Tick-level return volatility (std of % changes between consecutive ticks)
    tick_returns = today["mark"].pct_change().dropna()
    intraday_vol = float(tick_returns.std() * 100) if len(tick_returns) > 5 else np.nan

    df["intraday_range_pct"]  = range_pct
    df["intraday_return_pct"] = (df["mark"] - open_p) / open_p * 100
    df["price_position"]      = (
        (df["mark"] - low_p) / (high_p - low_p) if high_p > low_p else 0.5
    )
    df["intraday_vol"] = intraday_vol

    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def compute_features(df):
    """
    Add momentum features to a per-contract DataFrame (sorted by ts).
    Uses binary search (searchsorted) for O(N log N) total — no O(N²) loops.
    """
    ts_arr   = df["ts"].to_numpy(dtype=np.int64)
    mark_arr = df["mark"].to_numpy(dtype=np.float64)

    for col, window_ms in [("pct_30s", 30_000), ("pct_60s", 60_000), ("pct_120s", 120_000)]:
        pcts = np.full(len(df), np.nan)
        for i in range(len(df)):
            cutoff = ts_arr[i] - window_ms
            j = int(np.searchsorted(ts_arr, cutoff, side="left"))
            if j < i and mark_arr[j] > 0:
                pcts[i] = (mark_arr[i] - mark_arr[j]) / mark_arr[j] * 100.0
        df[col] = pcts

    return df


# ── Triple Barrier labeling ───────────────────────────────────────────────────

def triple_barrier_labels(df, profit_pct, stop_pct, lookahead_ms):
    """
    López de Prado's Triple Barrier Method:
      1 — mark price first hits +profit_pct%  (take-profit) → label 1
      0 — mark price first hits -stop_pct%    (stop-loss)   → label 0
      0 — lookahead_ms elapses without either  (time-stop)  → label 0
      NaN — no future data within lookahead window (unlabelable, dropped later)
    """
    ts_arr   = df["ts"].to_numpy(dtype=np.int64)
    mark_arr = df["mark"].to_numpy(dtype=np.float64)
    labels   = np.full(len(df), np.nan)

    for i in range(len(df)):
        m0 = mark_arr[i]
        if m0 <= 0:
            continue
        profit_level = m0 * (1.0 + profit_pct / 100.0)
        stop_level   = m0 * (1.0 - stop_pct   / 100.0)
        end_ts       = ts_arr[i] + lookahead_ms
        end_idx      = int(np.searchsorted(ts_arr, end_ts, side="right"))

        if end_idx <= i + 1:
            continue   # no future data → leave as NaN

        label_set = False
        for k in range(i + 1, end_idx):
            mk = mark_arr[k]
            if mk >= profit_level:
                labels[i] = 1.0
                label_set = True
                break
            if mk <= stop_level:
                labels[i] = 0.0
                label_set = True
                break
        if not label_set:
            labels[i] = 0.0   # time barrier

    df["label"] = labels
    return df


def build_dataset(series, config):
    """
    Compute features + Triple Barrier labels across all contracts.
    Returns (full_df, spx_ctx) where spx_ctx is the SPX context dict.
    """
    lookahead_ms  = int(config["lookahead_s"]) * 1000
    profit_pct    = config["profit_barrier_pct"]
    stop_pct      = config["stop_barrier_pct"]

    # Compute SPX context once for all contracts
    spx_ctx       = compute_spx_context(series) if series else {}
    today_open_ts = spx_ctx.get("today_open_ts", 0)
    current_spx   = spx_ctx.get("current", 0.0)
    daily_pct     = spx_ctx.get("daily_pct", 0.0)
    vs_yest       = spx_ctx.get("vs_yesterday_pct")

    frames = []
    for occ, df in series.items():
        if len(df) < 3:
            continue

        _, strike = _parse_occ(occ)

        df = compute_features(df.copy())
        df = compute_intraday_features(df, today_open_ts)

        # SPX-relative features (constant per contract per refresh cycle)
        df["spx_daily_pct"]    = daily_pct
        df["vs_yesterday_pct"] = vs_yest if vs_yest is not None else 0.0
        df["moneyness_pct"]    = (
            (strike - current_spx) / current_spx * 100
            if strike and current_spx > 0 else np.nan
        )

        df = triple_barrier_labels(df, profit_pct, stop_pct, lookahead_ms)
        df["occ"] = occ
        frames.append(df)

    if not frames:
        return pd.DataFrame(), spx_ctx
    return pd.concat(frames, ignore_index=True), spx_ctx


# ── LightGBM training ─────────────────────────────────────────────────────────

def train_lgbm(labeled_df, feature_cols):
    """
    Train a LightGBM binary classifier.
    Uses scale_pos_weight to handle severe class imbalance.
    Returns (fitted clf, cross-validated AUC, medians dict).
    """
    X_raw   = labeled_df[feature_cols].copy()
    medians = X_raw.median().to_dict()
    X       = X_raw.fillna(pd.Series(medians)).to_numpy(dtype=np.float32)
    y       = labeled_df["label"].astype(int).to_numpy()

    n_pos = y.sum()
    if n_pos < 5:
        raise ValueError(f"Only {n_pos} positive labels — need at least 5 to train")

    pos_weight = max(1.0, (len(y) - n_pos) / n_pos)

    clf = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        min_child_samples=5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    # Cross-validated AUC (stratified to preserve class ratio per fold)
    n_folds = min(5, max(2, int(n_pos // 3)))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X, y):
        cv_clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=pos_weight, min_child_samples=5,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        cv_clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = cv_clf.predict_proba(X[va_idx])[:, 1]

    auc = float(roc_auc_score(y, oof))

    # Final fit on full data
    clf.fit(X, y)
    return clf, auc, medians


def retrain_async(labeled_df, done_cb=None):
    """Run LightGBM training in a daemon thread; updates globals on completion."""
    global _clf, _medians, _shap_expl

    def _run():
        global _clf, _medians, _shap_expl
        try:
            log.info(f"Training LightGBM on {len(labeled_df)} labeled rows …")
            clf, auc, medians = train_lgbm(labeled_df, FEATURE_COLS)

            expl = None
            if SHAP_AVAILABLE:
                try:
                    expl = shap.TreeExplainer(clf.booster_)
                    log.info("SHAP TreeExplainer ready")
                except Exception as e:
                    log.warning(f"SHAP init failed: {e}")

            with _lock:
                _clf       = clf
                _medians   = medians
                _shap_expl = expl
                _model_state["auc"]          = round(auc, 3)
                _model_state["phase"]        = "ml"
                _model_state["last_trained"] = _now_iso()
                _model_state["error"]        = None
            log.info(f"Training done — AUC={auc:.3f}")
        except Exception as e:
            log.error(f"Training failed: {e}")
            with _lock:
                _model_state["error"] = str(e)
        finally:
            if done_cb:
                done_cb()

    threading.Thread(target=_run, daemon=True).start()


def _maybe_retrain(labeled_df, n_labeled, config):
    """Trigger async retraining when data crosses growth thresholds."""
    global _last_trained_n
    if not LGB_AVAILABLE:
        return
    if n_labeled < config["min_train_samples"]:
        return
    if _retrain_pending.is_set():
        return
    # First training, or ≥10% more data since last train (min 50 rows)
    if n_labeled - _last_trained_n >= max(50, int(_last_trained_n * 0.10)):
        _retrain_pending.set()
        _last_trained_n = n_labeled
        retrain_async(
            labeled_df[FEATURE_COLS + ["label"]].dropna(subset=FEATURE_COLS).copy(),
            done_cb=_retrain_pending.clear,
        )


# ── Empirical conditional probability (Phase 1) ───────────────────────────────

def empirical_prob(labeled_df, col, thresh, direction="up"):
    """
    P(label=1 | pct_30s >= thresh) with Laplace smoothing toward the base rate.
    Returns (probability, n_matching_rows, n_positive_rows).
    """
    if labeled_df.empty or col not in labeled_df.columns:
        return 0.05, 0, 0

    base_rate = float(labeled_df["label"].mean()) if len(labeled_df) > 0 else 0.01

    if direction == "up":
        subset = labeled_df[labeled_df[col] >= thresh]
    else:
        subset = labeled_df[labeled_df[col] <= -thresh]

    n     = len(subset)
    n_pos = int(subset["label"].sum()) if n > 0 else 0

    # Laplace smoothing: blend toward base rate when sample is small
    if n < 20:
        prob = (n_pos + base_rate * 10) / (n + 10)
    else:
        prob = n_pos / n

    return float(np.clip(prob, 0.001, 0.999)), n, n_pos


# ── Signal generation ─────────────────────────────────────────────────────────

def compute_signals(series, full_df, spx_ctx, config):
    """
    Generate trading signals from the most recent per-contract data.

    For each contract:
    - Take the latest reading within the last 5s (handles Robinhood batching)
    - Skip if |pct_30s| < momentum_thresh
    - Phase 1: empirical conditional probability as confidence
    - Phase 2: LightGBM predict_proba + SHAP top-3 feature reasoning
    - Skip if confidence < conf_display_min
    """
    momentum_thresh = config["momentum_thresh_pct"]
    conf_min        = config["conf_display_min"]
    now_ms          = int(time.time() * 1000)
    cutoff_5s       = now_ms - 5_000

    labeled_df = (
        full_df.dropna(subset=["label"]).copy()
        if not full_df.empty and "label" in full_df.columns
        else pd.DataFrame()
    )

    # Snapshot: latest reading per contract from last 5s
    latest_by_occ = {}
    for occ, df in series.items():
        recent = df[df["ts"] >= cutoff_5s]
        row    = recent.iloc[-1] if not recent.empty else (df.iloc[-1] if not df.empty else None)
        if row is not None:
            latest_by_occ[occ] = row

    signals_out = []
    clf_snap  = _clf       # CPython assignment is atomic — safe without lock
    expl_snap = _shap_expl
    med_snap  = _medians

    for occ, row in latest_by_occ.items():
        pct_30 = None if pd.isna(row.get("pct_30s", float("nan"))) else float(row["pct_30s"])
        if pct_30 is None or abs(pct_30) < momentum_thresh:
            continue

        direction = "LONG" if pct_30 > 0 else "SHORT"
        pct_60    = None if pd.isna(row.get("pct_60s",  float("nan"))) else float(row["pct_60s"])
        pct_120   = None if pd.isna(row.get("pct_120s", float("nan"))) else float(row["pct_120s"])

        conf              = None
        reasoning_parts   = []

        if clf_snap is not None:
            # ── Phase 2: LightGBM ────────────────────────────────────────────
            feat_arr = np.array([[
                float(row[c]) if c in row.index and not pd.isna(row[c]) else med_snap.get(c, 0.0)
                for c in FEATURE_COLS
            ]], dtype=np.float32)

            conf = float(clf_snap.predict_proba(feat_arr)[0][1])

            # SHAP top-3 feature contributions
            if expl_snap is not None and SHAP_AVAILABLE:
                try:
                    sv = expl_snap.shap_values(feat_arr)
                    sv_arr = sv[1][0] if isinstance(sv, list) else sv[0]
                    top_idx = np.argsort(np.abs(sv_arr))[::-1][:3]
                    for idx in top_idx:
                        fc  = FEATURE_COLS[idx]
                        fv  = feat_arr[0][idx]
                        sgn = "↑" if sv_arr[idx] > 0 else "↓"
                        reasoning_parts.append(f"{fc}={fv:.3f}{sgn}")
                except Exception:
                    pass

        else:
            # ── Phase 1: rule-based + empirical probability ───────────────────
            prob, n, n_pos = empirical_prob(
                labeled_df, "pct_30s", momentum_thresh,
                "up" if pct_30 > 0 else "down"
            )
            conf = prob
            if n > 0:
                reasoning_parts.append(f"Hist: {n_pos}/{n} similar moves hit target")
            else:
                reasoning_parts.append("No history yet — rule-based")

        if conf is None:
            conf = 0.05

        if conf < conf_min:
            continue

        # Extract intraday stats for this contract
        def _rnd(v, d=2):
            try:
                f = float(v)
                return round(f, d) if math.isfinite(f) else None
            except (TypeError, ValueError):
                return None

        intraday_range  = _rnd(row.get("intraday_range_pct"))
        intraday_return = _rnd(row.get("intraday_return_pct"))
        price_pos       = _rnd(row.get("price_position"), 3)
        intraday_v      = _rnd(row.get("intraday_vol"), 3)
        moneyness       = _rnd(row.get("moneyness_pct"))
        spx_day_pct     = spx_ctx.get("daily_pct")

        # Build human-readable reasoning
        move_str = f"{'↑' if pct_30 > 0 else '↓'} {abs(pct_30):.1f}% in 30s"
        if pct_60 is not None:
            move_str += f", {abs(pct_60):.1f}% in 60s"
        if pct_120 is not None:
            move_str += f", {abs(pct_120):.1f}% in 2min"

        context_parts = list(reasoning_parts)
        if intraday_range is not None:
            context_parts.append(f"day range {intraday_range:.1f}%")
        if intraday_return is not None:
            sign = "+" if intraday_return >= 0 else ""
            context_parts.append(f"day ret {sign}{intraday_return:.1f}%")
        if price_pos is not None:
            pos_label = "near high" if price_pos > 0.8 else "near low" if price_pos < 0.2 else "mid-range"
            context_parts.append(f"{pos_label} ({price_pos:.0%} of day range)")
        if spx_day_pct is not None:
            sign = "+" if spx_day_pct >= 0 else ""
            context_parts.append(f"SPX {sign}{spx_day_pct:.2f}% today")

        reasoning = move_str
        if context_parts:
            reasoning += " | " + ", ".join(context_parts)

        opt_type, strike = _parse_occ(occ)
        mark = float(row["mark"])

        signals_out.append({
            "occ_symbol":         occ,
            "type":               opt_type or "?",
            "strike":             row.get("strike"),
            "mark_price":         round(mark, 2),
            "bid_price":          round(float(row.get("bid") or 0), 2),
            "ask_price":          round(float(row.get("ask") or 0), 2),
            "spread_pct":         round(float(row.get("spread_pct") or 0) * 100, 1),
            "delta":              round(float(row.get("delta") or 0), 4),
            "iv":                 round(float(row.get("iv") or 0) * 100, 1),
            "direction":          direction,
            "confidence":         round(conf, 3),
            "pct_30s":            round(pct_30, 2),
            "pct_60s":            round(pct_60, 2) if pct_60 is not None else None,
            "pct_120s":           round(pct_120, 2) if pct_120 is not None else None,
            # Intraday context
            "intraday_range_pct":  intraday_range,
            "intraday_return_pct": intraday_return,
            "price_position":      price_pos,
            "intraday_vol":        intraday_v,
            "moneyness_pct":       moneyness,
            "reasoning":          reasoning,
            "ts":                 int(row.get("ts", now_ms)),
            "age_s":              round((now_ms - int(row.get("ts", now_ms))) / 1000, 1),
        })

    signals_out.sort(key=lambda s: s["confidence"], reverse=True)
    return signals_out


# ── Background refresh loop ───────────────────────────────────────────────────

def refresh_loop():
    global _spx_context
    while True:
        try:
            config  = dict(_config)
            series  = load_time_series()

            if series:
                full_df, spx_ctx = build_dataset(series, config)
            else:
                full_df, spx_ctx = pd.DataFrame(), {}

            labeled_df = (
                full_df.dropna(subset=["label"])
                if not full_df.empty and "label" in full_df.columns
                else pd.DataFrame()
            )
            n_labeled  = len(labeled_df)
            n_positive = int(labeled_df["label"].sum()) if n_labeled > 0 else 0
            base_rate  = n_positive / n_labeled if n_labeled > 0 else 0.0

            _maybe_retrain(labeled_df, n_labeled, config)

            signals = compute_signals(series, full_df, spx_ctx, config)

            with _lock:
                _signals[:] = signals
                _spx_context = spx_ctx
                _model_state.update({
                    "n_labeled":    n_labeled,
                    "n_positive":   n_positive,
                    "base_rate":    round(base_rate, 4),
                    "last_refresh": _now_iso(),
                    "error":        None,
                })
                if n_labeled < config["min_train_samples"] and _clf is None:
                    _model_state["phase"] = "rule"

        except Exception as e:
            log.error(f"Refresh error: {e}", exc_info=True)
            with _lock:
                _model_state["error"] = str(e)

        time.sleep(_config.get("refresh_s", 3))


# ── HTTP request handler ──────────────────────────────────────────────────────

class SignalHandler(BaseHTTPRequestHandler):

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, data, code=200):
        body = json.dumps(data, default=str).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        with _lock:
            sig  = list(_signals)
            stat = dict(_model_state)

        if self.path == "/signals":
            with _lock:
                spx = dict(_spx_context)
            self._json({
                "ts":           int(time.time() * 1000),
                "model_status": stat,
                "config":       dict(_config),
                "spx_context":  spx,
                "signals":      sig,
            })
        elif self.path == "/stats":
            self._json(stat)
        else:
            self.send_response(404)
            self._cors()
            self.end_headers()

    def do_POST(self):
        if self.path != "/config":
            self.send_response(404)
            self._cors()
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        try:
            updates = json.loads(body)
            for k, v in updates.items():
                if k in _config:
                    _config[k] = type(_config[k])(v)
            self._json({"ok": True, "config": dict(_config)})
        except Exception as e:
            self._json({"ok": False, "error": str(e)}, 400)

    def log_message(self, fmt, *args):
        pass   # suppress per-request access log


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    log.info(f"DB path : {DB_PATH}")
    log.info(f"LightGBM: {'OK' if LGB_AVAILABLE else 'NOT FOUND — rule-based only (pip install lightgbm)'}")
    log.info(f"SHAP    : {'OK' if SHAP_AVAILABLE else 'NOT FOUND — no explanation support (pip install shap)'}")

    if not DB_PATH.exists():
        log.warning("DB not found yet — will retry each refresh cycle")

    # Start background engine
    threading.Thread(target=refresh_loop, daemon=True).start()

    srv = HTTPServer(("127.0.0.1", SIGNAL_PORT), SignalHandler)
    log.info(f"Signal server ready → http://127.0.0.1:{SIGNAL_PORT}/signals")
    log.info("Press Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info("Stopped")


if __name__ == "__main__":
    main()
