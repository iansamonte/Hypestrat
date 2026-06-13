"""
scan_setups.py — Daily setup scanner for the "Dip Reversion 2.5R" manual strategy.

Scans the top-10 token basket against the 5-point entry checklist:
  1. RSI(14) <= 32 on 1H
  2. Price at/below lower Bollinger Band (20, 2) on 1H
  3. Price within 2% of recent support (60-bar swing low)
  4. Perp funding rate <= 0  (shorts paying longs = contrarian bullish)
  5. Bullish reversal candle on last closed 1H bar (hammer / engulfing)

Plus two gates that VETO all entries:
  - BTC master filter: no entries while BTC < falling 4H EMA-100
  - Per-token trend filter: token must be above its 4H EMA-100,
    or its 4H EMA-50 must be flat/rising

Usage:
    python scan_setups.py            # live scan via Hyperliquid public API
    python scan_setups.py --demo     # synthetic data (offline check of output)

Each $20 trade gets exact SL/TP prices printed per its volatility tier (all 2.5R).
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd

API_URL = "https://api.hyperliquid.xyz/info"

# token -> (tier, sl_pct, tp_pct)   all 2.5R
TIERS = {
    "BTC":  (1, 0.04, 0.10),
    "ETH":  (1, 0.04, 0.10),
    "SOL":  (2, 0.05, 0.125),
    "BNB":  (2, 0.05, 0.125),
    "XRP":  (2, 0.05, 0.125),
    "ADA":  (2, 0.05, 0.125),
    "LINK": (2, 0.05, 0.125),
    "DOGE": (3, 0.06, 0.15),
    "AVAX": (3, 0.06, 0.15),
    "HYPE": (3, 0.06, 0.15),
}
TRADE_SIZE_USDC = 20.0
CANDLE_HOURS = 600          # ~25 days of 1H bars (enough for 4H EMA-100)
MIN_CHECKS = 3              # checklist passes needed for a SETUP verdict
MAX_SIMULTANEOUS = 2        # market-wide dump rule: take only N strongest


# ─── Data fetch ───────────────────────────────────────────────────────────────

def fetch_candles_1h(coin: str) -> pd.DataFrame:
    """1H perp candles from the Hyperliquid public info API (no auth)."""
    import requests
    end = int(time.time() * 1000)
    start = end - CANDLE_HOURS * 3600 * 1000
    resp = requests.post(API_URL, json={
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": "1h",
                "startTime": start, "endTime": end},
    }, timeout=15)
    resp.raise_for_status()
    rows = resp.json()
    df = pd.DataFrame([{
        "ts": r["t"], "open": float(r["o"]), "high": float(r["h"]),
        "low": float(r["l"]), "close": float(r["c"]), "volume": float(r["v"]),
    } for r in rows])
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    return df


def fetch_funding_rates() -> dict:
    """Current hourly funding per coin from one metaAndAssetCtxs call."""
    import requests
    resp = requests.post(API_URL, json={"type": "metaAndAssetCtxs"}, timeout=15)
    resp.raise_for_status()
    meta, ctxs = resp.json()
    out = {}
    for asset, ctx in zip(meta["universe"], ctxs):
        try:
            out[asset["name"]] = float(ctx.get("funding", 0.0))
        except (TypeError, ValueError):
            out[asset["name"]] = 0.0
    return out


def demo_candles(coin: str, seed: int) -> pd.DataFrame:
    """Synthetic 1H candles so the scanner output can be checked offline."""
    rng = np.random.default_rng(seed)
    n = CANDLE_HOURS
    base = {"BTC": 100_000, "ETH": 3_500, "SOL": 180, "BNB": 650, "XRP": 2.2,
            "ADA": 0.9, "LINK": 18, "DOGE": 0.32, "AVAX": 35, "HYPE": 30}.get(coin, 10)
    r = rng.normal(0, 0.008, n)
    r[-30:] -= 0.004                       # recent dip → some setups fire
    c = base * np.exp(np.cumsum(r))
    o = np.roll(c, 1); o[0] = base
    wick = np.abs(rng.normal(0, 0.004, n))
    df = pd.DataFrame({
        "open": o, "high": np.maximum(o, c) * (1 + wick),
        "low": np.minimum(o, c) * (1 - wick), "close": c,
        "volume": np.abs(rng.normal(1e6, 2e5, n)),
    }, index=pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="1h"))
    return df


# ─── Indicators ───────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def bollinger_lower(close: pd.Series, window: int = 20, k: float = 2.0) -> float:
    m = close.rolling(window).mean().iloc[-1]
    s = close.rolling(window).std().iloc[-1]
    return float(m - k * s)


def near_support(df: pd.DataFrame, lookback: int = 60, tol: float = 0.02) -> bool:
    swing_low = df["low"].iloc[-lookback:-2].min()
    return df["close"].iloc[-1] <= swing_low * (1 + tol)


def reversal_candle(df: pd.DataFrame) -> bool:
    """Hammer or bullish engulfing on the last closed bar."""
    prev, cur = df.iloc[-2], df.iloc[-1]
    body = abs(cur["close"] - cur["open"])
    rng_ = cur["high"] - cur["low"]
    if rng_ <= 0:
        return False
    lower_wick = min(cur["close"], cur["open"]) - cur["low"]
    hammer = (lower_wick >= 2 * body
              and cur["close"] >= cur["low"] + 0.6 * rng_)
    engulf = (prev["close"] < prev["open"]              # prev red
              and cur["close"] > cur["open"]            # cur green
              and cur["close"] >= prev["open"]
              and cur["open"] <= prev["close"])
    return bool(hammer or engulf)


def trend_ok(df_1h: pd.DataFrame) -> bool:
    """Above 4H EMA-100, or 4H EMA-50 flat/rising."""
    c4 = df_1h["close"].resample("4h").last().dropna()
    if len(c4) < 60:
        return False
    ema100 = c4.ewm(span=100, adjust=False).mean()
    ema50 = c4.ewm(span=50, adjust=False).mean()
    above = c4.iloc[-1] > ema100.iloc[-1]
    rising = ema50.iloc[-1] >= ema50.iloc[-5] * 0.998
    return bool(above or rising)


def btc_master_ok(df_btc: pd.DataFrame) -> bool:
    """Veto everything if BTC is below a FALLING 4H EMA-100."""
    c4 = df_btc["close"].resample("4h").last().dropna()
    ema100 = c4.ewm(span=100, adjust=False).mean()
    below = c4.iloc[-1] < ema100.iloc[-1]
    falling = ema100.iloc[-1] < ema100.iloc[-5]
    return not (below and falling)


# ─── Error helpers ────────────────────────────────────────────────────────────

def _short_err(e: Exception) -> str:
    """One-line summary of a fetch exception (no stack spam)."""
    msg = str(e)
    if "403" in msg or "Forbidden" in msg:
        return "403 Forbidden (host likely blocked by network policy)"
    if "Name or service not known" in msg or "getaddrinfo" in msg or "Failed to resolve" in msg:
        return "DNS resolution failed (no network)"
    if "timed out" in msg.lower() or "timeout" in msg.lower():
        return "request timed out"
    return msg[:80]


def _network_help(errs):
    """Print an actionable diagnosis when no data could be fetched."""
    blocked = any("403" in str(e) or "Forbidden" in str(e) for _, e in errs)
    print("\n  ✗ Could not fetch any market data — cannot run the scan.")
    if blocked:
        print("    Cause: api.hyperliquid.xyz is blocked by this environment's network policy.")
        print("    Fix:")
        print("      • Run on your own machine (no egress filter), OR")
        print("      • Allowlist 'api.hyperliquid.xyz' in the web environment's egress settings.")
    else:
        print("    Cause: no network connectivity to api.hyperliquid.xyz.")
        print("    Fix: check your internet connection, then retry.")
    print("    Tip: 'python scan_setups.py --demo' verifies the logic offline.")


# ─── Scan ─────────────────────────────────────────────────────────────────────

def scan(demo: bool = False):
    print("\n" + "═" * 78)
    print("  DIP REVERSION 2.5R — Top-10 Setup Scanner"
          + ("   [DEMO DATA — not live prices]" if demo else ""))
    print("═" * 78)

    funding = {}
    if not demo:
        try:
            funding = fetch_funding_rates()
        except Exception as e:
            print(f"  ⚠ funding fetch failed ({_short_err(e)}) — check #4 will show '?'")

    candles = {}
    fetch_errs = []
    for i, coin in enumerate(TIERS):
        try:
            candles[coin] = demo_candles(coin, seed=i) if demo else fetch_candles_1h(coin)
        except Exception as e:
            fetch_errs.append((coin, e))
    if fetch_errs:
        print(f"  ⚠ {len(fetch_errs)}/{len(TIERS)} tokens failed to fetch "
              f"(first: {fetch_errs[0][0]} — {_short_err(fetch_errs[0][1])})")

    if "BTC" not in candles:
        _network_help(fetch_errs)
        sys.exit(1)

    master = btc_master_ok(candles["BTC"])
    print(f"\n  BTC MASTER FILTER: {'✓ entries allowed' if master else '✗ BTC below falling 4H EMA-100 — NO ENTRIES on any token'}")

    results = []
    for coin, df in candles.items():
        if len(df) < 120:
            continue
        px = float(df["close"].iloc[-1])
        checks = {
            "rsi≤32":    rsi(df["close"]) <= 32,
            "lowerBB":   px <= bollinger_lower(df["close"]),
            "support":   near_support(df),
            "funding≤0": (funding.get(coin, 0.0) <= 0) if (funding or demo) else None,
            "reversal":  reversal_candle(df),
        }
        score = sum(1 for v in checks.values() if v)
        t_ok = trend_ok(df)
        is_setup = master and t_ok and score >= MIN_CHECKS
        results.append((coin, px, t_ok, checks, score, is_setup))

    # Market-wide dump rule: keep only the strongest N setups
    setups = sorted([r for r in results if r[5]], key=lambda r: -r[4])
    demoted = set(r[0] for r in setups[MAX_SIMULTANEOUS:])

    print(f"\n  {'Token':<6} {'Price':>12}  {'Trend':<5} {'Checks':<38} {'✓':>2}  Verdict")
    print("  " + "─" * 74)
    for coin, px, t_ok, checks, score, is_setup in results:
        marks = " ".join(
            (k + ("✓" if v else "✗" if v is not None else "?")) for k, v in checks.items())
        if is_setup and coin not in demoted:
            verdict = "🟢 SETUP"
        elif is_setup:
            verdict = "🟡 queue (dump rule: top %d only)" % MAX_SIMULTANEOUS
        elif not master:
            verdict = "⛔ BTC veto"
        elif not t_ok:
            verdict = "✗ no trend"
        else:
            verdict = f"… wait ({score}/{MIN_CHECKS})"
        print(f"  {coin:<6} {px:>12,.4f}  {'✓' if t_ok else '✗':<5} {marks:<38} {score:>2}  {verdict}")

    live = [r for r in setups[:MAX_SIMULTANEOUS]]
    if live:
        print(f"\n  ── ORDERS (${TRADE_SIZE_USDC:.0f} per trade, limit entry slightly below market) ──")
        for coin, px, *_ in live:
            _, sl_pct, tp_pct = TIERS[coin]
            sl, tp = px * (1 - sl_pct), px * (1 + tp_pct)
            print(f"  {coin:<6} entry ≲ {px:,.4f}   SL {sl:,.4f} (-{sl_pct*100:.0f}%, risk ${TRADE_SIZE_USDC*sl_pct:.2f})"
                  f"   TP {tp:,.4f} (+{tp_pct*100:.1f}%, gain ${TRADE_SIZE_USDC*tp_pct:.2f})")
        print("  Move SL to entry once price is +1R. Never move SL down.")
    else:
        print("\n  No qualifying setups right now. Cash is a position — wait.")
    print("═" * 78 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic data, offline")
    scan(demo=ap.parse_args().demo)
