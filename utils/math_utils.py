"""
Core mathematical utilities for quantitative trading.

Implements:
  - Kelly Criterion (full & fractional)
  - Ornstein-Uhlenbeck parameter estimation
  - Rolling Z-score / normalization
  - Information Coefficient
  - VaR / CVaR (Expected Shortfall)
  - Maximum drawdown & recovery statistics
  - Compounding target calculator
  - Welch's t-test for signal significance
  - Entropy / mutual information for signal diversity
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from numba import njit
from typing import Tuple, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Kelly Criterion
# ──────────────────────────────────────────────────────────────────────────────

def kelly_criterion_discrete(win_prob: float, win_loss_ratio: float) -> float:
    """
    Kelly fraction for discrete bets.
    f* = (p*b - q) / b  where b = win/loss ratio, p = P(win), q = P(loss)
    """
    if win_prob <= 0 or win_prob >= 1 or win_loss_ratio <= 0:
        return 0.0
    q = 1.0 - win_prob
    kelly = (win_prob * win_loss_ratio - q) / win_loss_ratio
    return max(0.0, kelly)


def kelly_criterion_continuous(mu: float, sigma: float, risk_free: float = 0.0) -> float:
    """
    Kelly fraction for continuous returns.
    f* = (mu - r) / sigma^2
    """
    if sigma <= 0:
        return 0.0
    kelly = (mu - risk_free) / (sigma ** 2)
    return max(0.0, kelly)


def fractional_kelly(kelly_full: float, fraction: float = 0.25) -> float:
    """Apply fractional Kelly scaling for drawdown control."""
    return min(kelly_full * fraction, 1.0)


def compute_position_size(
    portfolio_value: float,
    kelly_full: float,
    kelly_fraction: float,
    max_pct: float,
    price: float,
    leverage: float = 1.0,
) -> float:
    """
    Compute final position size in units of the asset.
    Applies fractional Kelly capped at max_pct of portfolio, scaled by leverage.
    """
    frac = fractional_kelly(kelly_full, kelly_fraction)
    frac = min(frac, max_pct)
    dollar_size = portfolio_value * frac * leverage
    return dollar_size / price


# ──────────────────────────────────────────────────────────────────────────────
# Ornstein-Uhlenbeck Parameter Estimation
# ──────────────────────────────────────────────────────────────────────────────

def estimate_ou_params(series: pd.Series) -> Tuple[float, float, float, float]:
    """
    Estimate OU parameters via OLS on the Euler-discretised SDE.
    dX = theta*(mu - X)*dt + sigma*dW
    Regression: X[t] - X[t-1] = a + b*X[t-1] + eps
    Returns (theta, mu, sigma, half_life_bars)
    """
    y = series.diff().dropna().values
    x = series.shift(1).dropna().values
    if len(x) < 10:
        return 0.0, float(series.mean()), float(series.std()), np.inf

    slope, intercept, _, _, _ = stats.linregress(x, y)
    theta = -slope  # Mean-reversion speed
    if theta <= 0:
        return 0.0, float(series.mean()), float(series.std()), np.inf

    mu = intercept / theta
    residuals = y - (slope * x + intercept)
    sigma = float(np.std(residuals))
    half_life = np.log(2.0) / theta
    return theta, mu, sigma, half_life


def ou_z_score(price: float, ou_mu: float, ou_sigma: float) -> float:
    """Z-score of current price relative to OU equilibrium."""
    if ou_sigma <= 0:
        return 0.0
    return (price - ou_mu) / ou_sigma


# ──────────────────────────────────────────────────────────────────────────────
# Rolling Statistics
# ──────────────────────────────────────────────────────────────────────────────

def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling Z-score normalisation."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def robust_zscore(series: pd.Series, window: int) -> pd.Series:
    """Robust Z-score using median and MAD (outlier-resistant)."""
    med = series.rolling(window).median()
    mad = series.rolling(window).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    return (series - med) / (mad.replace(0, np.nan) * 1.4826)


# ──────────────────────────────────────────────────────────────────────────────
# Information Coefficient (Predictive Signal Quality)
# ──────────────────────────────────────────────────────────────────────────────

def information_coefficient(signals: np.ndarray, returns: np.ndarray) -> float:
    """
    Rank IC: Spearman correlation between signal ranks and forward return ranks.
    IC > 0.05 is considered economically meaningful for a daily signal.
    """
    if len(signals) < 5:
        return 0.0
    ic, _ = stats.spearmanr(signals, returns)
    return float(ic) if not np.isnan(ic) else 0.0


def rolling_ic(
    signals: pd.Series,
    returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Rolling Information Coefficient over a sliding window."""
    result = []
    for i in range(window, len(signals) + 1):
        s = signals.iloc[i - window : i].values
        r = returns.iloc[i - window : i].values
        result.append(information_coefficient(s, r))
    return pd.Series(result, index=signals.index[window - 1 :])


def ic_information_ratio(rolling_ic_series: pd.Series) -> float:
    """IC Information Ratio = mean(IC) / std(IC). Target > 0.5."""
    mu = rolling_ic_series.mean()
    sigma = rolling_ic_series.std()
    if sigma == 0:
        return 0.0
    return float(mu / sigma)


# ──────────────────────────────────────────────────────────────────────────────
# Risk Metrics
# ──────────────────────────────────────────────────────────────────────────────

def var_historical(returns: np.ndarray, confidence: float = 0.99) -> float:
    """Historical VaR at given confidence level (positive = loss)."""
    return float(-np.percentile(returns, (1 - confidence) * 100))


def cvar_historical(returns: np.ndarray, confidence: float = 0.99) -> float:
    """
    Conditional VaR (Expected Shortfall) — mean loss beyond VaR.
    CVaR is a coherent risk measure unlike VaR.
    """
    var = var_historical(returns, confidence)
    tail = returns[returns <= -var]
    if len(tail) == 0:
        return var
    return float(-tail.mean())


def var_parametric(returns: np.ndarray, confidence: float = 0.99) -> float:
    """Parametric (Gaussian) VaR assuming normal distribution."""
    mu = np.mean(returns)
    sigma = np.std(returns)
    z = norm.ppf(1 - confidence)
    return float(-(mu + z * sigma))


@njit
def max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Compute maximum drawdown, peak index, and trough index.
    Returns (mdd_fraction, peak_idx, trough_idx)
    Uses numba JIT for speed on large arrays.
    """
    peak = equity_curve[0]
    peak_idx = 0
    max_dd = 0.0
    trough_idx = 0
    for i in range(1, len(equity_curve)):
        if equity_curve[i] > peak:
            peak = equity_curve[i]
            peak_idx = i
        dd = (peak - equity_curve[i]) / peak
        if dd > max_dd:
            max_dd = dd
            trough_idx = i
    return max_dd, peak_idx, trough_idx


def drawdown_series(equity_curve: np.ndarray) -> np.ndarray:
    """Rolling drawdown from peak at each bar."""
    running_max = np.maximum.accumulate(equity_curve)
    return (equity_curve - running_max) / running_max


def calmar_ratio(equity_curve: np.ndarray, periods_per_year: int = 252 * 96) -> float:
    """
    Calmar Ratio = CAGR / |Max Drawdown|
    Target > 3.0 for elite strategies.
    """
    mdd, _, _ = max_drawdown(equity_curve)
    if mdd == 0:
        return np.inf
    n = len(equity_curve)
    cagr = (equity_curve[-1] / equity_curve[0]) ** (periods_per_year / n) - 1
    return float(cagr / mdd)


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252 * 96, risk_free: float = 0.0) -> float:
    """
    Annualised Sharpe Ratio.
    periods_per_year = 252*96 for 15-minute bars (96 bars/day).
    """
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    excess = returns - risk_free / periods_per_year
    sr = np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year)
    return float(sr)


def sortino_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252 * 96,
    mar: float = 0.0,
) -> float:
    """
    Sortino Ratio = (E[R] - MAR) / downside_std
    Only penalises downside volatility.
    """
    excess = returns - mar / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return np.inf
    ds = np.std(downside) * np.sqrt(periods_per_year)
    return float(np.mean(excess) * periods_per_year / ds)


def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Omega Ratio = E[max(R - T, 0)] / E[max(T - R, 0)]
    > 1 means more gains above threshold than losses below.
    """
    gains = np.maximum(returns - threshold, 0).sum()
    losses = np.maximum(threshold - returns, 0).sum()
    if losses == 0:
        return np.inf
    return float(gains / losses)


# ──────────────────────────────────────────────────────────────────────────────
# Compounding Targets
# ──────────────────────────────────────────────────────────────────────────────

def required_monthly_return(start: float, target: float, months: int) -> float:
    """Monthly return r such that start * (1+r)^months = target."""
    return (target / start) ** (1 / months) - 1


def compound_schedule(
    start: float,
    monthly_return: float,
    months: int,
) -> pd.DataFrame:
    """
    Generate the month-by-month compound growth schedule.
    Useful for tracking progress vs target.
    """
    values = [start * (1 + monthly_return) ** m for m in range(months + 1)]
    months_idx = range(months + 1)
    df = pd.DataFrame({"month": months_idx, "portfolio_value": values})
    df["cagr_to_date"] = df.apply(
        lambda row: (row.portfolio_value / start) ** (12 / max(row.month, 1)) - 1
        if row.month > 0 else 0.0,
        axis=1,
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Signal Significance Testing
# ──────────────────────────────────────────────────────────────────────────────

def signal_ttest(group_a: np.ndarray, group_b: np.ndarray) -> Tuple[float, float]:
    """Welch's t-test: is the mean return of group_a different from group_b?"""
    t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)
    return float(t_stat), float(p_val)


def binomial_win_rate_ci(wins: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score confidence interval for win rate."""
    z = norm.ppf((1 + confidence) / 2)
    p_hat = wins / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return float(center - margin), float(center + margin)


# ──────────────────────────────────────────────────────────────────────────────
# Realized Volatility Estimators
# ──────────────────────────────────────────────────────────────────────────────

def parkinson_volatility(high: np.ndarray, low: np.ndarray) -> float:
    """
    Parkinson (1980) high-low volatility estimator.
    More efficient than close-to-close: uses intrabar range.
    """
    return float(np.sqrt(np.mean((np.log(high / low)) ** 2) / (4 * np.log(2))))


def garman_klass_volatility(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> float:
    """
    Garman-Klass (1980) OHLC volatility estimator.
    Most efficient classical estimator using full OHLC data.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    return float(np.sqrt(np.mean(gk)))


def yang_zhang_volatility(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k: float = 0.34,
) -> float:
    """
    Yang-Zhang (2000) volatility: handles overnight jumps.
    sigma^2 = sigma_oc^2 + k*sigma_cc^2 + (1-k)*sigma_gk^2
    """
    n = len(close)
    log_oc = np.log(open_[1:] / close[:-1])
    log_co = np.log(close / open_)[1:]
    log_hl = np.log(high / low)[1:]

    sigma_oc2 = np.mean(log_oc**2)
    sigma_co2 = np.mean(log_co**2)
    sigma_gk2 = np.mean(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2)

    sigma2 = sigma_oc2 + k * sigma_co2 + (1 - k) * sigma_gk2
    return float(np.sqrt(max(sigma2, 0)))


# ──────────────────────────────────────────────────────────────────────────────
# Signal Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def weighted_signal(
    signals: dict[str, float],
    weights: dict[str, float],
    clip: float = 1.0,
) -> float:
    """
    Compute weighted average of named signals.
    Each signal in [-1, +1]. Clips output to [-clip, +clip].
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for name, sig in signals.items():
        w = weights.get(name, 0.0)
        if not np.isnan(sig) and not np.isnan(w):
            weighted_sum += sig * w
            total_weight += abs(w)
    if total_weight == 0:
        return 0.0
    raw = weighted_sum / total_weight
    return float(np.clip(raw, -clip, clip))


def signal_consensus(signals: dict[str, float], threshold: float = 0.1) -> int:
    """
    Count how many signals agree on direction.
    Returns positive int for longs, negative for shorts, 0 for mixed.
    """
    longs = sum(1 for s in signals.values() if s > threshold)
    shorts = sum(1 for s in signals.values() if s < -threshold)
    neutrals = len(signals) - longs - shorts
    if longs > shorts:
        return longs
    elif shorts > longs:
        return -shorts
    return 0
