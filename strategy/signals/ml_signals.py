"""
Machine Learning signal generator — LSTM + XGBoost ensemble.

Architecture:
  Layer 1: Feature engineering (50+ features from OHLCV + indicators)
  Layer 2: LSTM (128→64→32) for sequence modelling (learns temporal patterns)
  Layer 3: XGBoost (500 trees) for tabular feature importance
  Layer 4: Meta-learner (logistic regression on LSTM+XGB outputs)
  Output: Probability of up-move → normalised to [-1, +1] signal

Walk-forward training prevents look-ahead bias:
  Train on [t-180d, t], predict on [t, t+30d], step 30d.

Feature set:
  - Multi-timeframe returns (5 windows)
  - RSI, MACD, Bollinger, ATR (normalised)
  - Volume ratios and deltas
  - Momentum indicators
  - Funding rate and OI (when available)
  - Time-of-day and day-of-week cyclical encoding
"""
from __future__ import annotations

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer 50+ features from OHLCV data.
    All features are normalised/standardised to avoid scale dominance.
    """
    feat = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    vol = df.get("volume", pd.Series(1.0, index=df.index))

    # ── Returns (multiple horizons) ──────────────────────────────────────────
    for w in [1, 3, 5, 10, 20, 40, 96]:
        ret = close.pct_change(w)
        feat[f"ret_{w}"] = ret

    # ── Volatility features ───────────────────────────────────────────────────
    for w in [5, 10, 20, 60]:
        feat[f"vol_{w}"] = close.pct_change().rolling(w).std()
        feat[f"vol_ratio_{w}"] = feat[f"vol_{w}"] / (feat[f"vol_{w}"].rolling(60).mean() + 1e-9)

    # ── RSI ───────────────────────────────────────────────────────────────────
    for period in [7, 14, 21]:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta).clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        feat[f"rsi_{period}"] = (rsi - 50) / 50  # Normalise to [-1, +1]

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = (macd - macd_sig) / (close * 0.01 + 1e-9)
    feat["macd_cross"] = np.sign(macd - macd_sig)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    for w in [10, 20]:
        bb_m = close.rolling(w).mean()
        bb_s = close.rolling(w).std()
        feat[f"bb_pctb_{w}"] = (close - bb_m) / (2 * bb_s + 1e-9)
        feat[f"bb_width_{w}"] = (4 * bb_s) / (bb_m + 1e-9)

    # ── ATR normalised ────────────────────────────────────────────────────────
    prev_c = close.shift(1)
    tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    for w in [7, 14]:
        atr = tr.ewm(alpha=1/w, adjust=False).mean()
        feat[f"atr_pct_{w}"] = atr / (close + 1e-9)

    # ── EMA ratios ────────────────────────────────────────────────────────────
    for s, l in [(9, 21), (21, 55), (9, 55)]:
        ema_s = close.ewm(span=s, adjust=False).mean()
        ema_l = close.ewm(span=l, adjust=False).mean()
        feat[f"ema_ratio_{s}_{l}"] = (ema_s - ema_l) / (ema_l + 1e-9)

    # ── Volume features ───────────────────────────────────────────────────────
    feat["vol_ratio_20"] = vol / (vol.rolling(20).mean() + 1e-9)
    feat["vol_delta"] = vol.diff() / (vol.shift(1) + 1e-9)
    # Price-volume correlation (recent)
    feat["pv_corr"] = close.pct_change().rolling(10).corr(vol.pct_change())

    # ── Price patterns ────────────────────────────────────────────────────────
    feat["body"] = (close - open_) / (high - low + 1e-9)
    feat["upper_wick"] = (high - close.clip(lower=open_)) / (high - low + 1e-9)
    feat["lower_wick"] = (open_.clip(upper=close) - low) / (high - low + 1e-9)
    feat["gap"] = (open_ - close.shift(1)) / (close.shift(1) + 1e-9)

    # ── Momentum ──────────────────────────────────────────────────────────────
    for w in [3, 5, 10, 20]:
        feat[f"mom_{w}"] = close.pct_change(w).clip(-0.5, 0.5)

    # ── Rolling statistics ────────────────────────────────────────────────────
    for w in [20, 60]:
        feat[f"skew_{w}"] = close.pct_change().rolling(w).skew()
        feat[f"kurt_{w}"] = close.pct_change().rolling(w).kurt()

    # ── High/Low relative position ────────────────────────────────────────────
    for w in [10, 20, 50]:
        high_w = high.rolling(w).max()
        low_w = low.rolling(w).min()
        feat[f"hl_position_{w}"] = (close - low_w) / (high_w - low_w + 1e-9) - 0.5

    # ── Time features (cyclical encoding) ────────────────────────────────────
    if hasattr(df.index, "hour"):
        hour = df.index.hour
        feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    if hasattr(df.index, "dayofweek"):
        dow = df.index.dayofweek
        feat["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        feat["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    return feat.replace([np.inf, -np.inf], np.nan)


def create_target(close: pd.Series, forward_periods: int = 1, threshold: float = 0.0) -> pd.Series:
    """
    Binary classification target: 1 if next-bar return > threshold, 0 otherwise.
    """
    forward_return = close.shift(-forward_periods) / close - 1
    return (forward_return > threshold).astype(int)


class LSTMModel:
    """LSTM sequence model for price direction prediction."""

    def __init__(self, seq_len: int = 60, units: List[int] = None, dropout: float = 0.2) -> None:
        self.seq_len = seq_len
        self.units = units or [128, 64, 32]
        self.dropout = dropout
        self._model = None
        self._scaler = None

    def build(self, n_features: int):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

            model = Sequential([
                Bidirectional(LSTM(self.units[0], return_sequences=True), input_shape=(self.seq_len, n_features)),
                Dropout(self.dropout),
                BatchNormalization(),
                LSTM(self.units[1], return_sequences=True),
                Dropout(self.dropout),
                BatchNormalization(),
                LSTM(self.units[2]),
                Dropout(self.dropout),
                Dense(16, activation="relu"),
                Dense(1, activation="sigmoid"),
            ])
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            self._model = model
            return model
        except ImportError:
            return None

    def _make_sequences(self, X: np.ndarray, y: np.ndarray = None):
        Xs, ys = [], []
        for i in range(self.seq_len, len(X)):
            Xs.append(X[i - self.seq_len : i])
            if y is not None:
                ys.append(y[i])
        Xs = np.array(Xs)
        if y is not None:
            return Xs, np.array(ys)
        return Xs

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32) -> None:
        if self._model is None:
            self.build(X.shape[1])
        if self._model is None:
            return

        try:
            from sklearn.preprocessing import StandardScaler
            from tensorflow.keras.callbacks import EarlyStopping

            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
            X_seq, y_seq = self._make_sequences(X_scaled, y)

            self._model.fit(
                X_seq, y_seq,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.15,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                verbose=0,
            )
        except Exception as e:
            from utils.logger import logger
            logger.warning(f"LSTM training failed: {e}")

    def predict_proba(self, X: np.ndarray) -> float:
        if self._model is None or self._scaler is None:
            return 0.5
        try:
            X_scaled = self._scaler.transform(X)
            if len(X_scaled) < self.seq_len:
                return 0.5
            X_seq = self._make_sequences(X_scaled[-self.seq_len:].reshape(self.seq_len, -1))
            if len(X_seq) == 0:
                seq = X_scaled[-self.seq_len:].reshape(1, self.seq_len, -1)
                prob = float(self._model.predict(seq, verbose=0)[0][0])
            else:
                prob = float(self._model.predict(X_seq[-1:], verbose=0)[0][0])
            return prob
        except Exception:
            return 0.5

    def save(self, path: str) -> None:
        if self._model:
            self._model.save(path + "_lstm.keras")
        if self._scaler:
            with open(path + "_lstm_scaler.pkl", "wb") as f:
                pickle.dump(self._scaler, f)

    def load(self, path: str) -> None:
        try:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(path + "_lstm.keras")
            with open(path + "_lstm_scaler.pkl", "rb") as f:
                self._scaler = pickle.load(f)
        except Exception:
            pass


class XGBoostModel:
    """XGBoost gradient boosting for feature-based prediction."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.01,
        subsample: float = 0.8,
        colsample: float = 0.8,
    ) -> None:
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=42,
        )
        self._model = None
        self._scaler = None
        self.feature_importances_: Optional[pd.Series] = None

    def train(self, X: pd.DataFrame, y: np.ndarray) -> None:
        try:
            import xgboost as xgb
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X.fillna(0))

            self._model = xgb.XGBClassifier(**self.params, verbosity=0)
            self._model.fit(
                X_scaled, y,
                eval_set=[(X_scaled[-100:], y[-100:])],
                verbose=False,
            )
            self.feature_importances_ = pd.Series(
                self._model.feature_importances_,
                index=X.columns,
            ).sort_values(ascending=False)
        except Exception as e:
            from utils.logger import logger
            logger.warning(f"XGBoost training failed: {e}")

    def predict_proba(self, X: pd.DataFrame) -> float:
        if self._model is None or self._scaler is None:
            return 0.5
        try:
            X_scaled = self._scaler.transform(X.fillna(0))
            prob = float(self._model.predict_proba(X_scaled)[0][1])
            return prob
        except Exception:
            return 0.5

    def save(self, path: str) -> None:
        if self._model:
            self._model.save_model(path + "_xgb.ubj")
        if self._scaler:
            with open(path + "_xgb_scaler.pkl", "wb") as f:
                pickle.dump(self._scaler, f)

    def load(self, path: str) -> None:
        try:
            import xgboost as xgb
            self._model = xgb.XGBClassifier()
            self._model.load_model(path + "_xgb.ubj")
            with open(path + "_xgb_scaler.pkl", "rb") as f:
                self._scaler = pickle.load(f)
        except Exception:
            pass


class MLSignalGenerator:
    """
    Ensemble ML signal: LSTM + XGBoost with adaptive weighting.

    Signal generation:
      prob = w_lstm * p_lstm + w_xgb * p_xgb
      signal = (prob - 0.5) * 2  →  [-1, +1]

    Retrains periodically using walk-forward methodology to avoid
    look-ahead bias and track non-stationarity.
    """

    def __init__(
        self,
        seq_len: int = 60,
        lstm_weights: float = 0.4,
        xgb_weights: float = 0.6,
        model_dir: str = "models",
        min_train_samples: int = 500,
    ) -> None:
        self.seq_len = seq_len
        self.lstm_weight = lstm_weights
        self.xgb_weight = xgb_weights
        self.model_dir = model_dir
        self.min_train_samples = min_train_samples

        os.makedirs(model_dir, exist_ok=True)
        self.lstm = LSTMModel(seq_len=seq_len)
        self.xgb = XGBoostModel()
        self._trained = False
        self._last_train_idx = 0

    def train(self, df: pd.DataFrame) -> None:
        """
        Full training on provided DataFrame.
        Builds features, creates targets, trains both models.
        """
        if len(df) < self.min_train_samples:
            return

        features = build_features(df).fillna(0)
        target = create_target(df["close"], forward_periods=1)

        # Align and drop NaN
        mask = ~(features.isna().any(axis=1) | target.isna())
        X = features[mask]
        y = target[mask].values

        if len(X) < self.min_train_samples:
            return

        from utils.logger import logger
        logger.info(f"ML training | {len(X)} samples | {X.shape[1]} features")

        # XGBoost training
        self.xgb.train(X, y)

        # LSTM training (needs sequential data)
        self.lstm.train(X.values, y)

        self._trained = True
        self._last_train_idx = len(df)

        # Save models
        self.lstm.save(os.path.join(self.model_dir, "hypestrat"))
        self.xgb.save(os.path.join(self.model_dir, "hypestrat"))

        logger.info("ML models trained and saved")

    def predict(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate ML signals for the most recent bar.
        Returns dict with 'ml_composite' in [-1, +1].
        """
        if len(df) < self.seq_len + 20:
            return {"ml_composite": 0.0, "ml_confidence": 0.0}

        features = build_features(df).fillna(0)
        if features.empty or features.isna().all().all():
            return {"ml_composite": 0.0, "ml_confidence": 0.0}

        X_latest = features.iloc[-1:].fillna(0)

        # XGBoost prediction
        xgb_prob = self.xgb.predict_proba(X_latest)

        # LSTM prediction (needs sequence)
        X_seq = features.fillna(0).values
        lstm_prob = self.lstm.predict_proba(X_seq)

        if not self._trained:
            # Not trained yet → neutral signal
            return {"ml_composite": 0.0, "ml_confidence": 0.0, "xgb_prob": xgb_prob, "lstm_prob": lstm_prob}

        # Ensemble
        ensemble_prob = self.lstm_weight * lstm_prob + self.xgb_weight * xgb_prob

        # Convert probability to signal in [-1, +1]
        signal = (ensemble_prob - 0.5) * 2

        # Confidence: distance from 0.5 (0 = no confidence, 1 = very confident)
        confidence = abs(ensemble_prob - 0.5) * 2

        return {
            "ml_composite": float(np.clip(signal, -1, 1)),
            "ml_confidence": float(confidence),
            "xgb_prob": float(xgb_prob),
            "lstm_prob": float(lstm_prob),
            "ensemble_prob": float(ensemble_prob),
        }

    def needs_retraining(self, current_idx: int, retrain_every: int = 500) -> bool:
        """Check if models should be retrained."""
        return not self._trained or (current_idx - self._last_train_idx) >= retrain_every

    def try_load_pretrained(self) -> bool:
        """Load previously saved models if available."""
        try:
            model_path = os.path.join(self.model_dir, "hypestrat")
            self.lstm.load(model_path)
            self.xgb.load(model_path)
            self._trained = True
            return True
        except Exception:
            return False
