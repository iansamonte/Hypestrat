"""
Market regime detector using Hidden Markov Models + Kalman Filter.

Architecture:
  1. Kalman Filter: Real-time price/trend estimation (removes noise)
  2. HMM (3-state Gaussian): Bull / Sideways / Bear regime classification
  3. Regime-conditional strategy weights: each regime → different signal weights

HMM features:
  - Log returns (primary)
  - Volatility (rolling std of returns)
  - Volume normalised (relative volume)
  - Trend strength (EMA ratio)

Kalman Filter:
  - State vector: [price, velocity] (position + trend)
  - Kalman gain automatically updates based on signal-to-noise
  - Outputs: filtered price + velocity (trend direction and speed)

References:
  - Hamilton (1989): Regime-switching models
  - Ang & Bekaert (2002): International asset allocation with regime shifts
  - Kalman (1960): New approach to linear filtering and prediction
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

warnings.filterwarnings("ignore")


class KalmanTrendFilter:
    """
    Kalman filter for extracting the latent trend from noisy price data.

    State: x = [level, slope]  (price level + rate of change)
    Observation: y = close price

    Measurement model: y_t = [1, 0] * x_t + v_t
    Transition model: x_t = [[1, dt], [0, 1]] * x_{t-1} + w_t
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_noise: float = 0.001,
        observation_noise: float = 1.0,
    ) -> None:
        # Transition matrix (constant velocity model)
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        # Observation matrix
        self.H = np.array([[1.0, 0.0]])
        # Process noise covariance
        self.Q = np.eye(2) * process_noise
        # Observation noise covariance
        self.R = np.array([[observation_noise]])
        # Initial state estimate
        self.x = None
        # Initial covariance
        self.P = np.eye(2) * 100.0

    def reset(self, initial_price: float) -> None:
        self.x = np.array([[initial_price], [0.0]])
        self.P = np.eye(2) * 100.0

    def update(self, observation: float) -> Tuple[float, float]:
        """
        Process one observation. Returns (filtered_price, filtered_slope).
        filtered_slope > 0 means uptrend, < 0 means downtrend.
        """
        if self.x is None:
            self.reset(observation)
            return observation, 0.0

        # Predict step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update step
        y = np.array([[observation]]) - self.H @ x_pred  # Innovation
        S = self.H @ P_pred @ self.H.T + self.R  # Innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        return float(self.x[0, 0]), float(self.x[1, 0])

    def filter_series(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Apply Kalman filter to entire price series. Returns (levels, slopes)."""
        if len(prices) == 0:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        self.reset(float(prices.iloc[0]))
        levels, slopes = [], []
        for price in prices:
            level, slope = self.update(float(price))
            levels.append(level)
            slopes.append(slope)

        return (
            pd.Series(levels, index=prices.index),
            pd.Series(slopes, index=prices.index),
        )


class HMMRegimeDetector:
    """
    3-regime Hidden Markov Model for market state classification.

    Regimes (labelled by mean return after sorting):
      0: Bear  — negative mean return, high volatility
      1: Sideways — near-zero return, medium volatility
      2: Bull  — positive mean return, lower volatility

    The Viterbi algorithm finds the most likely regime sequence.
    """

    def __init__(self, n_regimes: int = 3, n_iter: int = 200) -> None:
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self._model = None
        self._regime_labels: Dict[int, str] = {}
        self._trained = False

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build HMM feature matrix from OHLCV."""
        close = df["close"]
        returns = np.log(close).diff().fillna(0).values
        volatility = pd.Series(returns).rolling(5).std().fillna(0).values

        ema9 = close.ewm(span=9, adjust=False).mean()
        ema55 = close.ewm(span=55, adjust=False).mean()
        trend_strength = ((ema9 - ema55) / ema55).fillna(0).values

        volume = df.get("volume", pd.Series(1.0, index=df.index))
        vol_ratio = (volume / (volume.rolling(20).mean() + 1e-9)).fillna(1.0).values

        features = np.column_stack([returns, volatility, trend_strength, vol_ratio])
        return np.nan_to_num(features, nan=0.0)

    def fit(self, df: pd.DataFrame) -> None:
        """Fit HMM on provided data."""
        try:
            from hmmlearn import hmm
        except ImportError:
            self._trained = False
            return

        features = self._build_features(df)
        if len(features) < 50:
            return

        model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="diag",
            n_iter=self.n_iter,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(features)

        self._model = model

        # Label regimes by their mean log-return
        states = model.predict(features)
        means_by_state = {}
        for s in range(self.n_regimes):
            mask = states == s
            if mask.any():
                means_by_state[s] = float(features[mask, 0].mean())
            else:
                means_by_state[s] = 0.0

        sorted_states = sorted(means_by_state.keys(), key=lambda k: means_by_state[k])
        labels = ["bear", "sideways", "bull"]
        self._regime_labels = {sorted_states[i]: labels[i] for i in range(len(sorted_states))}

        self._trained = True

    def predict(self, df: pd.DataFrame) -> Tuple[int, str, np.ndarray]:
        """
        Predict current regime. Returns (state_int, label_str, state_probs).
        state_probs is [bear_prob, sideways_prob, bull_prob].
        """
        if not self._trained or self._model is None:
            return 1, "sideways", np.array([0.0, 1.0, 0.0])

        features = self._build_features(df)
        if len(features) < 20:
            return 1, "sideways", np.array([0.0, 1.0, 0.0])

        try:
            state = int(self._model.predict(features)[-1])
            # Posterior state probabilities
            _, posteriors = self._model.score_samples(features)
            current_probs = posteriors[-1]  # probabilities for last observation

            label = self._regime_labels.get(state, "sideways")

            # Re-sort probs into [bear, sideways, bull] order
            label_order = ["bear", "sideways", "bull"]
            sorted_probs = np.zeros(3)
            for raw_state, lbl in self._regime_labels.items():
                idx = label_order.index(lbl)
                if raw_state < len(current_probs):
                    sorted_probs[idx] = current_probs[raw_state]

            return state, label, sorted_probs

        except Exception:
            return 1, "sideways", np.array([0.0, 1.0, 0.0])

    def predict_sequence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict regime for all bars in df. Returns DataFrame with regime columns."""
        if not self._trained or self._model is None:
            n = len(df)
            return pd.DataFrame({
                "regime": ["sideways"] * n,
                "regime_int": [1] * n,
                "bear_prob": [0.0] * n,
                "sideways_prob": [1.0] * n,
                "bull_prob": [0.0] * n,
            }, index=df.index)

        features = self._build_features(df)
        try:
            states = self._model.predict(features)
            _, posteriors = self._model.score_samples(features)
            label_order = ["bear", "sideways", "bull"]

            bear_probs, sideways_probs, bull_probs = [], [], []
            for p in posteriors:
                sp = np.zeros(3)
                for raw_state, lbl in self._regime_labels.items():
                    idx = label_order.index(lbl)
                    if raw_state < len(p):
                        sp[idx] = p[raw_state]
                bear_probs.append(sp[0])
                sideways_probs.append(sp[1])
                bull_probs.append(sp[2])

            return pd.DataFrame({
                "regime": [self._regime_labels.get(s, "sideways") for s in states],
                "regime_int": states,
                "bear_prob": bear_probs,
                "sideways_prob": sideways_probs,
                "bull_prob": bull_probs,
            }, index=df.index)

        except Exception:
            n = len(df)
            return pd.DataFrame({
                "regime": ["sideways"] * n,
                "regime_int": [1] * n,
                "bear_prob": [0.0] * n,
                "sideways_prob": [1.0] * n,
                "bull_prob": [0.0] * n,
            }, index=df.index)


class RegimeDetector:
    """
    Combined regime detector: Kalman filter + HMM.

    Provides:
      - Kalman-smoothed price and trend velocity
      - HMM regime state probabilities
      - Signal weight multipliers per regime
    """

    # Signal weight multipliers for each regime
    # [momentum_mult, mean_reversion_mult, orderflow_mult, funding_mult, ml_mult]
    REGIME_WEIGHTS = {
        "bull": {
            "momentum": 1.5,
            "mean_reversion": 0.5,
            "technical": 1.2,
            "orderflow": 1.0,
            "funding": 0.8,
            "ml": 1.0,
        },
        "sideways": {
            "momentum": 0.6,
            "mean_reversion": 1.5,
            "technical": 1.0,
            "orderflow": 1.2,
            "funding": 1.2,
            "ml": 1.0,
        },
        "bear": {
            "momentum": 1.3,
            "mean_reversion": 0.7,
            "technical": 1.2,
            "orderflow": 1.0,
            "funding": 1.0,
            "ml": 1.2,
        },
    }

    def __init__(
        self,
        hmm_n_regimes: int = 3,
        hmm_n_iter: int = 200,
        kalman_process_noise: float = 0.001,
        kalman_obs_noise: float = 1.0,
        hmm_lookback: int = 500,
    ) -> None:
        self.kalman = KalmanTrendFilter(
            process_noise=kalman_process_noise,
            observation_noise=kalman_obs_noise,
        )
        self.hmm = HMMRegimeDetector(n_regimes=hmm_n_regimes, n_iter=hmm_n_iter)
        self.hmm_lookback = hmm_lookback
        self._hmm_trained = False

    def fit_hmm(self, df: pd.DataFrame) -> None:
        """Train HMM on historical data."""
        data = df.iloc[-self.hmm_lookback :] if len(df) > self.hmm_lookback else df
        self.hmm.fit(data)
        self._hmm_trained = True

    def detect(self, df: pd.DataFrame) -> Dict:
        """
        Run full regime detection on the most recent data.
        Returns comprehensive regime state dict.
        """
        if df.empty:
            return self._neutral_state()

        # Kalman filter
        levels, slopes = self.kalman.filter_series(df["close"])
        current_level = float(levels.iloc[-1])
        current_slope = float(slopes.iloc[-1])

        # Normalise slope as fraction of price
        trend_velocity = current_slope / (current_level + 1e-9)

        # HMM regime
        if not self._hmm_trained:
            self.fit_hmm(df)

        state_int, regime_label, probs = self.hmm.predict(df)

        bear_prob, sideways_prob, bull_prob = probs[0], probs[1], probs[2]

        # Smooth regime weights using probability-weighted combination
        weights = {}
        for signal_key in ["momentum", "mean_reversion", "technical", "orderflow", "funding", "ml"]:
            bear_w = self.REGIME_WEIGHTS["bear"][signal_key]
            sideways_w = self.REGIME_WEIGHTS["sideways"][signal_key]
            bull_w = self.REGIME_WEIGHTS["bull"][signal_key]
            weights[signal_key] = (
                bear_prob * bear_w + sideways_prob * sideways_w + bull_prob * bull_w
            )

        # Regime confidence (how certain is the HMM?)
        confidence = float(np.max(probs))

        return {
            "regime": regime_label,
            "regime_int": state_int,
            "bear_prob": float(bear_prob),
            "sideways_prob": float(sideways_prob),
            "bull_prob": float(bull_prob),
            "confidence": confidence,
            "kalman_price": current_level,
            "kalman_slope": current_slope,
            "trend_velocity": trend_velocity,
            "signal_weights": weights,
            # Composite trend signal from Kalman: +1 = strong uptrend, -1 = strong downtrend
            "trend_signal": float(np.clip(trend_velocity * 1000, -1, 1)),
        }

    def _neutral_state(self) -> Dict:
        return {
            "regime": "sideways",
            "regime_int": 1,
            "bear_prob": 0.0,
            "sideways_prob": 1.0,
            "bull_prob": 0.0,
            "confidence": 0.33,
            "kalman_price": 0.0,
            "kalman_slope": 0.0,
            "trend_velocity": 0.0,
            "signal_weights": {k: 1.0 for k in ["momentum", "mean_reversion", "technical", "orderflow", "funding", "ml"]},
            "trend_signal": 0.0,
        }
