"""
Hypestrat — Vectorized Backtest + Forward Test Runner.
All signals precomputed as numpy arrays; bar loop is pure index lookup.
Runs 3 scenarios × 2 phases in ~30 seconds total.
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
os.environ.setdefault("HL_PRIVATE_KEY", "")
os.environ.setdefault("HL_WALLET_ADDRESS", "")
os.environ.setdefault("BACKTESTING", "1")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


# ─── Realistic HYPE/USDC price generator ──────────────────────────────────────

def gen_hype(n: int, start: float, seed: int, mode: str) -> pd.DataFrame:
    np.random.seed(seed)
    dt = 15 / (60 * 24)  # 15-min bar in days
    cfg = {
        "bull":  (0.010, 0.052, 0.0007, 0.13, 0.0010, 0.09),
        "bear":  (-0.006, 0.058, 0.0015, 0.15, 0.0009, 0.07),
        "mixed": (0.003,  0.062, 0.0010, 0.17, 0.0010, 0.10),
    }[mode]
    drift, vol, cp, cs, rp, rs = cfg

    r = np.random.normal(drift*dt, vol*np.sqrt(dt), n)
    # Regime events
    crash_mask = np.random.random(n) < cp
    rally_mask = np.random.random(n) < rp
    r -= cs * crash_mask * (0.6 + np.random.random(n)*0.4)
    r += rs * rally_mask * (0.6 + np.random.random(n)*0.4)
    # Momentum persistence
    mom = np.zeros(n)
    for i in range(1, n): mom[i] = 0.12*mom[i-1] + r[i]
    r = 0.75*r + 0.25*mom

    if mode == "mixed":
        mid = n//2
        r[mid:mid+n//3]  -= 0.004*dt
        r[mid+n//3:]     += 0.005*dt

    px  = start * np.exp(np.cumsum(r))
    rng = np.abs(np.random.normal(0, vol*np.sqrt(dt)*0.55, n))
    op  = np.roll(px, 1); op[0] = start
    hi  = np.maximum(px, op) * (1 + rng)
    lo  = np.minimum(px, op) * (1 - rng)
    vol_data = np.abs(np.random.normal(1_500_000, 400_000, n))

    df = pd.DataFrame({"open":op,"high":hi,"low":lo,"close":px,"volume":vol_data},
                      index=pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"))
    return df


# ─── Vectorized signal precomputation ────────────────────────────────────────

def ema(s: np.ndarray, span: int) -> np.ndarray:
    a = 2/(span+1); out = np.empty_like(s); out[0] = s[0]
    for i in range(1, len(s)): out[i] = a*s[i] + (1-a)*out[i-1]
    return out

def rma(s: np.ndarray, period: int) -> np.ndarray:
    a = 1/period; out = np.empty_like(s); out[0] = s[0]
    for i in range(1, len(s)): out[i] = a*s[i] + (1-a)*out[i-1]
    return out

def precompute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised precomputation of all signal components.
    Returns a DataFrame indexed same as df with one column per signal.
    All signals in [-1, +1].
    """
    c  = df["close"].values.astype(float)
    h  = df["high"].values.astype(float)
    l  = df["low"].values.astype(float)
    o  = df["open"].values.astype(float)
    v  = df["volume"].values.astype(float)
    n  = len(c)
    sig = pd.DataFrame(index=df.index)

    # ── ATR (stored as metadata) ─────────────────────────────────────────────
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(h-l, np.maximum(np.abs(h-prev_c), np.abs(l-prev_c)))
    atr14 = rma(tr, 14)
    sig["_atr"] = atr14

    # ── RSI ──────────────────────────────────────────────────────────────────
    delta = np.diff(c, prepend=c[0])
    gain  = rma(np.maximum(delta, 0), 14)
    loss  = rma(np.maximum(-delta, 0), 14)
    rs    = np.where(loss==0, 100, gain/(loss+1e-12))
    rsi   = 100 - 100/(1+rs)
    sig["rsi"] = np.clip((50-rsi)/50*1.4, -1, 1)  # +1 oversold, -1 overbought

    # ── MACD ─────────────────────────────────────────────────────────────────
    ema12 = ema(c, 12); ema26 = ema(c, 26)
    macd  = ema12 - ema26
    sig_line = ema(macd, 9)
    hist = macd - sig_line
    sig["macd"] = np.clip(hist/(c*0.01+1e-9)*5, -1, 1)

    # ── Bollinger %B ─────────────────────────────────────────────────────────
    bb_m  = pd.Series(c).rolling(20).mean().values
    bb_s  = pd.Series(c).rolling(20).std().values
    pct_b = (c - (bb_m - 2*bb_s)) / (4*bb_s + 1e-9)
    sig["bollinger"] = np.clip((0.5-pct_b)*2, -1, 1)

    # ── EMA cross (9/21/55) ───────────────────────────────────────────────────
    e9  = ema(c, 9); e21 = ema(c, 21); e55 = ema(c, 55)
    ema_score = ((e9>e21).astype(float) + (e21>e55).astype(float)) / 2
    sig["ema_cross"] = ema_score*2 - 1

    # ── ADX direction ─────────────────────────────────────────────────────────
    up   = np.diff(h, prepend=h[0]); down = -np.diff(l, prepend=l[0])
    pdm  = rma(np.where((up>down)&(up>0), up, 0).astype(float), 14)
    mdm  = rma(np.where((down>up)&(down>0), down, 0).astype(float), 14)
    pdi  = 100*pdm/(atr14+1e-9); mdi = 100*mdm/(atr14+1e-9)
    dx   = 100*np.abs(pdi-mdi)/(pdi+mdi+1e-9)
    adx  = rma(dx, 14)
    strength = np.clip(adx/50, 0, 1)
    direction = np.sign(pdi - mdi)
    sig["adx"] = direction * strength

    # ── VWAP deviation ────────────────────────────────────────────────────────
    typical = (h+l+c)/3
    cum_tpv = pd.Series(typical*v).rolling(20).sum().values
    cum_vol = pd.Series(v).rolling(20).sum().values
    vwap    = cum_tpv / (cum_vol+1e-9)
    vwap_std= pd.Series((typical-vwap)**2).rolling(20).mean().values**0.5
    vwap_dev= (c-vwap)/(vwap_std+1e-9)
    sig["vwap_mr"] = np.clip(-vwap_dev/2, -1, 1)   # mean-reverting

    # TECHNICAL composite (equal weight)
    t_cols = ["rsi","macd","bollinger","ema_cross","adx","vwap_mr"]
    sig["technical"] = sig[t_cols].mean(axis=1).clip(-1, 1)

    # ── Multi-timeframe momentum ──────────────────────────────────────────────
    # Use df["close"] (has datetime index) so pandas alignment works correctly
    close_s = df["close"]
    mom_sigs = []
    for w in [4, 8, 16, 32, 64, 96]:
        ret = close_s.pct_change(w)
        mom_sigs.append(ret.clip(-0.3, 0.3) * (3.33/np.log(w+1)))
    sig["momentum"] = pd.concat(mom_sigs, axis=1).mean(axis=1).clip(-1, 1)

    # ── Simple mean reversion (rolling z-score only — fast) ──────────────────
    log_c  = np.log(close_s)
    mr_mu  = log_c.rolling(60).mean()
    mr_std = log_c.rolling(60).std()
    zscore = (log_c - mr_mu) / (mr_std + 1e-9)
    # Divide by 3 so ±1σ → ±0.33 signal rather than saturating at ±1
    sig["mean_reversion"] = (-zscore / 3.0).clip(-1, 1)  # oversold → buy

    # ── CVD (cumulative volume delta proxy) ───────────────────────────────────
    vol_s    = df["volume"]
    buy_vol  = vol_s.where(close_s >= df["open"], 0.0)
    sell_vol = vol_s.where(close_s <  df["open"], 0.0)
    cvd      = (buy_vol - sell_vol).rolling(50).sum()
    cvd_norm = cvd / (vol_s.rolling(50).sum() + 1e-9)
    sig["orderflow"] = cvd_norm.clip(-1, 1)

    # ── Volume ratio (amplifier) ──────────────────────────────────────────────
    sig["_vol_ratio"] = (vol_s / (vol_s.rolling(20).mean() + 1e-9)).clip(0, 5)

    # ── Realised vol for position sizing (annualised from 1d rolling) ─────────
    rv = close_s.pct_change().rolling(96).std() * np.sqrt(252*96)
    sig["_rv_annual"] = rv.fillna(rv.median() if not rv.dropna().empty else 0.5)

    # ── Funding rate signal (synthetic: correlated with 24h momentum) ─────────
    mom24 = close_s.pct_change(96).fillna(0)
    implied_funding = 0.0001 * np.tanh(mom24 * 5)
    # Contrarian: high funding → sell signal
    sig["funding"] = (-implied_funding / 0.0005).clip(-1, 1)

    # ── Master composite ──────────────────────────────────────────────────────
    weights = {"technical":0.25, "momentum":0.22, "mean_reversion":0.18,
               "orderflow":0.22, "funding":0.13}
    comp = sum(sig[k]*w for k,w in weights.items())
    # Amplify when vol_ratio confirms (high volume direction)
    amp = np.where(sig["_vol_ratio"]>1.5, 1.15, 1.0)
    sig["composite"] = (comp * amp).clip(-1, 1)

    return sig.fillna(0)


# ─── Trade record ─────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_bar: int;  exit_bar: int
    entry_px: float; exit_px: float
    direction: int;  size_usdc: float; leverage: float
    sl: float; tp: float; signal: float
    gross: float; fees: float; net: float
    exit_why: str; r_mult: float = 0.0


# ─── Bar-by-bar simulation ────────────────────────────────────────────────────

def simulate(df: pd.DataFrame, sigs: pd.DataFrame,
             capital: float = 5000.0,
             maker_fee: float = 0.0002,
             taker_fee: float = 0.0005,
             slippage: float = 0.0002,
             leverage_sched: Dict = None,
             learner=None) -> Tuple[np.ndarray, List[Trade]]:

    if leverage_sched is None:
        leverage_sched = {0.0: 3., 1.0: 5., 5.0: 7., 10.0: 10.}

    close  = df["close"].values.astype(float)
    low_   = df["low"].values.astype(float)
    high_  = df["high"].values.astype(float)
    comp   = sigs["composite"].values
    atr    = sigs["_atr"].values
    rv     = sigs["_rv_annual"].values
    N      = len(close)

    # Signal columns for learner
    sig_cols = ["technical", "momentum", "mean_reversion", "orderflow", "funding", "composite"]
    sig_arr  = {c: sigs[c].values for c in sig_cols}

    cash   = capital
    peak   = capital
    equity_curve = np.empty(N, dtype=float)
    equity_curve[0] = capital
    trades: List[Trade] = []

    open_pos = None   # dict or None
    WARMUP   = 120

    def get_leverage(eq: float) -> float:
        mult = eq / capital
        lev  = 3.0
        for thr, l in sorted(leverage_sched.items()):
            if mult >= thr: lev = l
        return lev

    for i in range(1, N):
        px   = close[i]
        lo_i = low_[i]
        hi_i = high_[i]

        # Mark to market
        if open_pos:
            d     = open_pos["d"]
            unrl  = d*(px-open_pos["ep"])/open_pos["ep"] * open_pos["sz"]*open_pos["lev"]
            equity = cash + unrl
        else:
            equity = cash

        equity_curve[i] = equity
        if equity > peak: peak = equity

        # Drawdown gate
        dd = (equity - peak) / peak
        if dd < -0.20 and open_pos:
            # Hard halt — close everything
            d = open_pos["d"]
            ep2 = px*(1-slippage-taker_fee) if d>0 else px*(1+slippage+taker_fee)
            gross = d*(ep2-open_pos["ep"])/open_pos["ep"]*open_pos["sz"]*open_pos["lev"]
            fees  = open_pos["sz"]*(maker_fee+taker_fee)
            net   = gross-fees; cash+=net; equity=cash
            risk  = abs(open_pos["ep"]-open_pos["sl"])/open_pos["ep"]*open_pos["sz"]*open_pos["lev"]
            r_mult = net/risk if risk>0 else 0.0
            trades.append(Trade(open_pos["bar"],i,open_pos["ep"],ep2,d,
                                open_pos["sz"],open_pos["lev"],open_pos["sl"],open_pos["tp"],
                                open_pos["sig"],gross,fees,net,"dd_halt",r_mult))
            if learner and open_pos.get("tid") is not None:
                learner.on_exit(open_pos["tid"], ep2, "dd_halt", net, fees, r_mult,
                                i - open_pos["bar"])
            open_pos = None

        # Drawdown-adjusted size multiplier
        if dd < -0.15:   size_mult = 0.25
        elif dd < -0.10: size_mult = 0.50
        elif dd < -0.05: size_mult = 0.75
        else:            size_mult = 1.00

        # Drawdown-adjusted signal threshold (base from learner if available)
        base_thr = learner.threshold if learner else 0.08
        if dd < -0.15:   threshold = base_thr * 2.25
        elif dd < -0.10: threshold = base_thr * 1.75
        elif dd < -0.05: threshold = base_thr * 1.375
        else:            threshold = base_thr

        # Check stop/TP on open position
        if open_pos:
            d, sl, tp = open_pos["d"], open_pos["sl"], open_pos["tp"]
            trig = ep2 = None
            if d > 0:
                if lo_i <= sl: trig, ep2 = "stop_loss", sl
                elif hi_i >= tp: trig, ep2 = "take_profit", tp
            else:
                if hi_i >= sl: trig, ep2 = "stop_loss", sl
                elif lo_i <= tp: trig, ep2 = "take_profit", tp

            if trig:
                ep2 = ep2*(1-slippage-taker_fee) if d>0 else ep2*(1+slippage+taker_fee)
                gross = d*(ep2-open_pos["ep"])/open_pos["ep"]*open_pos["sz"]*open_pos["lev"]
                fees  = open_pos["sz"]*(maker_fee+taker_fee)
                bars_held = i - open_pos["bar"]
                fund_cost = d * 0.0001 * (bars_held/32) * open_pos["sz"]
                net = gross - fees - fund_cost
                cash += net; equity = cash; equity_curve[i] = equity
                risk  = abs(open_pos["ep"]-sl)/open_pos["ep"]*open_pos["sz"]*open_pos["lev"]
                r_mult = net/risk if risk>0 else 0.0
                trades.append(Trade(open_pos["bar"],i,open_pos["ep"],ep2,d,
                                    open_pos["sz"],open_pos["lev"],sl,tp,
                                    open_pos["sig"],gross,fees,net,trig,r_mult))
                if learner and open_pos.get("tid") is not None:
                    learner.on_exit(open_pos["tid"], ep2, trig, net, fees, r_mult, bars_held)
                open_pos = None

            # Trailing stop (every 8 bars)
            if open_pos and i % 8 == 0:
                a = atr[i]
                if open_pos["d"] > 0:
                    open_pos["sl"] = max(open_pos["sl"], px - 1.5*a)
                else:
                    open_pos["sl"] = min(open_pos["sl"], px + 1.5*a)

        # Entry logic (every 4 bars, only when flat, after warmup)
        if i >= WARMUP and i % 4 == 0 and open_pos is None and equity > 50:
            csig = comp[i]
            if abs(csig) >= threshold:
                direction = int(np.sign(csig))
                a = atr[i]
                sl_mult = learner.stop_atr_mult if learner else 2.0
                tp_mult = learner.tp_atr_mult if learner else 5.0
                sl = px - sl_mult*a if direction>0 else px + sl_mult*a
                tp = px + tp_mult*a if direction>0 else px - tp_mult*a
                rr = tp_mult / sl_mult  # reward/risk ratio

                # Kelly-inspired sizing
                win_prob = 0.50 + abs(csig)*0.18
                kelly = max(0, (win_prob*rr - (1-win_prob))/rr)
                kf = learner.kelly_fraction if learner else 0.25
                frac_kelly = min(kelly * kf, 0.30)

                stop_pct = max(abs(px-sl)/px, 0.004)
                risk_target = equity * 0.01
                size_from_risk = risk_target / stop_pct

                lev = get_leverage(equity)
                size_from_kelly = equity * frac_kelly * lev

                sz = min(size_from_kelly, size_from_risk) * size_mult
                sz = min(sz, equity * 0.90)
                if sz < 15: continue

                entry = px*(1+slippage+maker_fee) if direction>0 else px*(1-slippage-maker_fee)

                # Register entry with learner
                tid = None
                if learner:
                    snap = {c: float(sig_arr[c][i]) for c in sig_cols}
                    atr_pct = a / px if px > 0 else 0.0
                    tid = learner.on_entry(entry, direction, sz, lev, atr_pct,
                                           snap, regime="unknown", dd_at_entry=dd)
                open_pos = dict(bar=i, ep=entry, d=direction,
                                sz=sz, lev=lev, sl=sl, tp=tp, sig=csig, tid=tid)

    # Close remaining at final bar
    if open_pos:
        fp = close[-1]
        d  = open_pos["d"]
        ep2= fp*(1-slippage-taker_fee) if d>0 else fp*(1+slippage+taker_fee)
        gross = d*(ep2-open_pos["ep"])/open_pos["ep"]*open_pos["sz"]*open_pos["lev"]
        fees  = open_pos["sz"]*(maker_fee+taker_fee)
        net   = gross-fees; cash+=net
        risk  = abs(open_pos["ep"]-open_pos["sl"])/open_pos["ep"]*open_pos["sz"]*open_pos["lev"]
        r_mult = net/risk if risk>0 else 0.0
        trades.append(Trade(open_pos["bar"],N-1,open_pos["ep"],ep2,d,
                            open_pos["sz"],open_pos["lev"],open_pos["sl"],open_pos["tp"],
                            open_pos["sig"],gross,fees,net,"end_of_data",r_mult))
        if learner and open_pos.get("tid") is not None:
            learner.on_exit(open_pos["tid"], ep2, "end_of_data", net, fees, r_mult,
                            N-1 - open_pos["bar"])
        equity_curve[-1] = cash

    return equity_curve, trades


# ─── Metrics ─────────────────────────────────────────────────────────────────

def metrics(eq: np.ndarray, tr: List[Trade], ini: float, bh_ret: float) -> Dict:
    from utils.math_utils import (sharpe_ratio, sortino_ratio, calmar_ratio,
                                   omega_ratio, max_drawdown, var_historical,
                                   cvar_historical, drawdown_series)
    r    = np.diff(eq)/np.maximum(eq[:-1], 1)
    BPY  = 252*96
    n_yr = len(eq)/BPY
    fin  = float(eq[-1])
    tot  = (fin-ini)/ini
    cagr = (fin/ini)**(1/max(n_yr,1/252)) - 1 if fin>0 else -1.

    sr   = sharpe_ratio(r, BPY)
    so   = sortino_ratio(r, BPY)
    mdd,*_= max_drawdown(eq)
    cal  = calmar_ratio(eq, BPY)
    om   = omega_ratio(r)
    vol  = float(np.std(r)*np.sqrt(BPY))
    var  = var_historical(r,.99) if len(r)>=20 else 0.
    cvar = cvar_historical(r,.99) if len(r)>=20 else 0.

    dd_arr = drawdown_series(eq)
    mddur = cur = 0
    for x in (dd_arr<-.01):
        cur = cur+1 if x else 0
        mddur = max(mddur, cur)

    nets = [t.net for t in tr]
    wins = [p for p in nets if p>0]
    loss_= [p for p in nets if p<0]
    wr   = len(wins)/len(nets) if nets else 0
    pf   = sum(wins)/(sum(abs(l) for l in loss_)+1e-9) if wins else 0.
    avgr = float(np.mean([t.r_mult for t in tr])) if tr else 0.
    exp_ = float(np.mean(nets)) if nets else 0.
    fees = sum(t.fees for t in tr)
    hold = float(np.mean([t.exit_bar-t.entry_bar for t in tr]))/4 if tr else 0.  # hours
    exits= {}
    for t in tr: exits[t.exit_why] = exits.get(t.exit_why,0)+1

    return dict(ini=ini, fin=fin, tot=tot, cagr=cagr, vol=vol,
                sr=sr, so=so, cal=cal, om=om,
                mdd=mdd, mddur=mddur/96,
                var=var, cvar=cvar, bh=bh_ret,
                n=len(tr), wr=wr, pf=pf, avgr=avgr,
                exp=exp_, fees=fees, hold=hold, exits=exits)


def show(m: Dict, label: str, phase: str):
    bh_vs = m["tot"] - m["bh"]
    ok = "✓ BEAT B&H" if bh_vs > 0 else "✗ LAG  B&H"
    print(f"\n  ┌──────────────────────────────────────────────────────────┐")
    print(f"  │  {label:<30}  [{phase}]")
    print(f"  ├──────────────────────────────────────────────────────────┤")
    print(f"  │  Capital:    ${m['ini']:>9,.0f}  →  ${m['fin']:>12,.2f}")
    print(f"  │  Return:     {m['tot']*100:>+9.1f}%   B&H: {m['bh']*100:>+7.1f}%   {ok}")
    print(f"  │  CAGR:       {m['cagr']*100:>+9.1f}%   Ann.Vol: {m['vol']*100:.1f}%")
    print(f"  │  Sharpe:     {m['sr']:>9.2f}   Sortino: {m['so']:.2f}   Calmar: {m['cal']:.2f}")
    print(f"  │  Max DD:     {m['mdd']*100:>9.1f}%   DD Dur: {m['mddur']:.1f}d   Omega: {m['om']:.2f}")
    print(f"  │  VaR 99%:   {m['var']*100:>9.2f}%   CVaR:   {m['cvar']*100:.2f}%")
    print(f"  │  Trades:     {m['n']:>9}   WinRate: {m['wr']*100:.1f}%   PF: {m['pf']:.2f}")
    print(f"  │  Avg R:     {m['avgr']:>+9.2f}R   Expect: ${m['exp']:.2f}/trade   Hold: {m['hold']:.1f}h")
    print(f"  │  Fees Paid:  ${m['fees']:>8.2f}")
    ex = "  ".join(f"{k}={v}" for k,v in sorted(m["exits"].items(), key=lambda x:-x[1]))
    print(f"  │  Exits:      {ex}")
    print(f"  └──────────────────────────────────────────────────────────┘")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from strategy.learning.adaptive_learner import AdaptiveLearner

    CAPITAL = 5_000.0
    N       = 8000          # ~83 days  (IS=62d, OOS=21d)
    SPLIT   = 0.75          # 75/25 split
    IS      = int(N*SPLIT)
    OOS     = N - IS

    # Self-improving learner — loads persisted state if it exists
    learner = AdaptiveLearner(state_dir="state")

    SCENARIOS = [
        ("Bull Market  (HYPE rally)",    "bull",  8.0,  42),
        ("Bear Market  (crypto winter)",  "bear",  28.0, 99),
        ("Mixed Cycle  (full cycle)",     "mixed", 12.0,  7),
    ]

    print("\n" + "═"*63)
    print("  HYPESTRAT — HYPE/USDC Backtest + Forward Test Results")
    print(f"  Initial Capital: ${CAPITAL:,.0f}  |  {N} bars = {N*15//60//24} days")
    print(f"  In-Sample (backtest):  {IS} bars = {IS*15//60//24}d")
    print(f"  Out-of-Sample (OOS):   {OOS} bars = {OOS*15//60//24}d")
    print(f"  Costs: 0.02% maker + 0.05% taker + 2 bps slippage")
    print(f"  Signals: Technical · Momentum · Mean-Rev · OrderFlow · Funding")
    print(f"  Sizing:  Quarter-Kelly (25%) + Vol targeting + Trailing stops")
    print("═"*63)

    all_is, all_oos = [], []
    t_total = time.time()

    for label, mode, sp, seed in SCENARIOS:
        print(f"\n{'━'*63}")
        print(f"  ▶  {label}")
        print(f"{'━'*63}")

        df = gen_hype(N, sp, seed, mode)

        bh_full = df["close"].iloc[-1]/sp - 1
        bh_is   = df["close"].iloc[IS-1]/sp - 1
        bh_oos  = df["close"].iloc[-1]/df["close"].iloc[IS] - 1

        print(f"  Price: ${sp:.2f} → ${df['close'].iloc[-1]:.2f}  "
              f"|  Full B&H: {bh_full*100:+.1f}%  "
              f"|  IS B&H: {bh_is*100:+.1f}%  "
              f"|  OOS B&H: {bh_oos*100:+.1f}%")

        # Precompute all signals once
        t0 = time.time()
        print("  Precomputing signals...", end="", flush=True)
        all_sigs = precompute_signals(df)
        print(f" {time.time()-t0:.1f}s")

        # ── IN-SAMPLE ──────────────────────────────────────────────────────
        is_df   = df.iloc[:IS]
        is_sigs = all_sigs.iloc[:IS]
        t0 = time.time()
        print(f"  IS  ({IS} bars, {IS*15//60//24}d)...", end="", flush=True)
        eq_is, tr_is = simulate(is_df, is_sigs, CAPITAL, learner=learner)
        t1 = time.time()-t0
        print(f" {t1:.1f}s  →  ${eq_is[-1]:,.2f}  ({(eq_is[-1]/CAPITAL-1)*100:+.1f}%)")

        m_is = metrics(eq_is, tr_is, CAPITAL, bh_is)
        show(m_is, label, "IN-SAMPLE backtest")
        all_is.append(m_is)

        # ── OUT-OF-SAMPLE ──────────────────────────────────────────────────
        oos_cap  = float(eq_is[-1])
        oos_df   = df.iloc[IS:]
        oos_sigs = all_sigs.iloc[IS:]
        t0 = time.time()
        print(f"\n  OOS ({OOS} bars, {OOS*15//60//24}d)...", end="", flush=True)
        eq_oos, tr_oos = simulate(oos_df, oos_sigs, oos_cap, learner=learner)
        t1 = time.time()-t0
        print(f" {t1:.1f}s  →  ${eq_oos[-1]:,.2f}  ({(eq_oos[-1]/oos_cap-1)*100:+.1f}%)")

        m_oos = metrics(eq_oos, tr_oos, oos_cap, bh_oos)
        show(m_oos, label, "OUT-OF-SAMPLE forward test")
        all_oos.append(m_oos)

        # ── Full cycle summary ─────────────────────────────────────────────
        eq_full = np.concatenate([eq_is, eq_oos[1:]])
        tr_full = tr_is + tr_oos
        m_f = metrics(eq_full, tr_full, CAPITAL, bh_full)
        print(f"\n  ► FULL: {m_f['tot']*100:+.1f}% return  |  "
              f"Sharpe={m_f['sr']:.2f}  Calmar={m_f['cal']:.2f}  "
              f"MDD={m_f['mdd']*100:.1f}%  WR={m_f['wr']*100:.0f}%  "
              f"Trades={m_f['n']}  B&H={bh_full*100:+.1f}%")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def avg(lst, k): return float(np.mean([x[k] for x in lst]))

    print(f"\n{'═'*63}")
    print("  AGGREGATE — Mean Across 3 Scenarios")
    print(f"{'═'*63}")
    print(f"  {'Metric':<22}  {'In-Sample (backtest)':>20}  {'OOS (forward test)':>18}")
    print(f"  {'─'*62}")
    for k, lbl, pct in [
        ("tot",  "Total Return",    True),
        ("cagr", "CAGR",            True),
        ("sr",   "Sharpe Ratio",    False),
        ("so",   "Sortino Ratio",   False),
        ("cal",  "Calmar Ratio",    False),
        ("om",   "Omega Ratio",     False),
        ("mdd",  "Max Drawdown",    True),
        ("wr",   "Win Rate",        True),
        ("pf",   "Profit Factor",   False),
        ("avgr", "Avg R-Multiple",  False),
        ("exp",  "Expectancy/trade",False),
    ]:
        iv, ov = avg(all_is, k), avg(all_oos, k)
        if pct:
            print(f"  {lbl:<22}  {iv*100:>+19.1f}%  {ov*100:>+17.1f}%")
        else:
            print(f"  {lbl:<22}  {iv:>20.2f}  {ov:>18.2f}")

    is_oos_gap = avg(all_is,"tot") - avg(all_oos,"tot")
    print(f"\n  IS→OOS degradation:  {is_oos_gap*100:+.1f}%")
    print(f"  (Good if < 15% — indicates genuine edge, not curve-fitting)")

    # ── Compound Growth Projection ─────────────────────────────────────────────
    NEED_MONTHLY = (1_000_000/5_000)**(1/48) - 1
    avg_oos_m = ((1+avg(all_oos,"tot"))**(1/(OOS/(96*20)))) - 1
    avg_is_m  = ((1+avg(all_is,"tot"))**(1/(IS/(96*20)))) - 1

    print(f"\n{'─'*63}")
    print(f"  COMPOUND GROWTH PROJECTION — $5,000 → $1,000,000 in 48 months")
    print(f"  Required monthly return:  {NEED_MONTHLY*100:.2f}%")
    print(f"  Achieved IS  (backtest):  {avg_is_m*100:+.2f}%/month")
    print(f"  Achieved OOS (forward):   {avg_oos_m*100:+.2f}%/month")
    print(f"\n  {'Month':>5}  {'Target':>12}  {'@ IS rate':>12}  {'@ OOS rate':>12}")
    for mo in [0, 6, 12, 18, 24, 30, 36, 42, 48]:
        tgt  = 5000*(1+NEED_MONTHLY)**mo
        proj_is  = 5000*(1+avg_is_m)**mo
        proj_oos = 5000*(1+avg_oos_m)**mo
        print(f"  {mo:>5}  ${tgt:>11,.0f}  ${proj_is:>11,.0f}  ${proj_oos:>11,.0f}")

    print(f"\n  Total run time: {time.time()-t_total:.1f}s")
    print(f"{'═'*63}\n")

    # ── Self-improvement reports ──────────────────────────────────────────────
    learner.print_all_reports()
    print(learner.final_summary())
    learner.save()
