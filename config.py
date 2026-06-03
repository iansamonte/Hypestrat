"""
Central configuration for the Hypestrat HYPE/USDC trading system.

Growth target: $5,000 → $1,000,000 in 4 years
Required CAGR: (1,000,000 / 5,000)^(1/4) - 1 ≈ 276.1% per year
Required monthly return: (200)^(1/48) - 1 ≈ 11.34% per month
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ExchangeConfig:
    private_key: str = field(default_factory=lambda: os.getenv("HL_PRIVATE_KEY", ""))
    wallet_address: str = field(default_factory=lambda: os.getenv("HL_WALLET_ADDRESS", ""))
    mainnet: bool = field(default_factory=lambda: os.getenv("HL_MAINNET", "true").lower() == "true")
    base_url: str = "https://api.hyperliquid.xyz"
    ws_url: str = "wss://api.hyperliquid.xyz/ws"

    @property
    def api_url(self) -> str:
        return self.base_url if self.mainnet else "https://api.hyperliquid-testnet.xyz"


@dataclass
class AssetConfig:
    symbol: str = "HYPE"
    quote: str = "USDC"
    pair: str = "HYPE/USDC"
    # Hyperliquid perpetual contract specs
    min_order_size: float = 1.0
    tick_size: float = 0.001
    contract_size: float = 1.0
    max_leverage: float = 10.0


@dataclass
class CapitalConfig:
    initial_capital: float = field(default_factory=lambda: float(os.getenv("INITIAL_CAPITAL", "5000")))
    # Target: $5k → $1M in 4 years = 200x = 276.1% CAGR
    target_capital: float = 1_000_000.0
    target_years: int = 4
    target_monthly_return: float = 0.1134  # 11.34% per month
    target_daily_return: float = 0.003599  # ~0.36% per day compounded


@dataclass
class RiskConfig:
    # Kelly Criterion fractional multiplier (0.25 = quarter-Kelly for safety)
    kelly_fraction: float = field(default_factory=lambda: float(os.getenv("KELLY_FRACTION", "0.25")))
    # Maximum single position as fraction of portfolio
    max_position_pct: float = field(default_factory=lambda: float(os.getenv("MAX_POSITION_PCT", "0.30")))
    # Maximum total portfolio exposure
    max_portfolio_heat: float = field(default_factory=lambda: float(os.getenv("MAX_PORTFOLIO_HEAT", "0.60")))
    # Stop loss in ATR multiples
    stop_loss_atr: float = 2.0
    # Take profit in ATR multiples
    take_profit_atr: float = 5.0
    # Trailing stop in ATR multiples (activates after 1R profit)
    trailing_stop_atr: float = 1.5
    # Maximum drawdown before halting
    max_drawdown_pct: float = field(default_factory=lambda: float(os.getenv("MAX_DRAWDOWN_THRESHOLD", "0.20")))
    # Daily P&L limits
    daily_loss_limit_pct: float = field(default_factory=lambda: float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.05")))
    weekly_loss_limit_pct: float = field(default_factory=lambda: float(os.getenv("WEEKLY_LOSS_LIMIT_PCT", "0.10")))
    monthly_loss_limit_pct: float = 0.20
    # Minimum signals required to agree before taking a trade
    min_signal_consensus: int = 3
    # Signal strength threshold [-1, 1] to enter a trade
    signal_threshold: float = 0.35
    # VaR confidence level
    var_confidence: float = 0.99
    # Leverage schedule (grows with portfolio performance)
    leverage_schedule: Dict[float, float] = field(default_factory=lambda: {
        0.0: 3.0,    # Up to 100% of initial: 3x leverage
        1.0: 5.0,    # 1x initial (10k): 5x leverage
        5.0: 7.0,    # 5x initial (25k): 7x leverage
        10.0: 10.0,  # 10x initial (50k): 10x leverage
    })


@dataclass
class SignalConfig:
    # Timeframes for multi-timeframe analysis (in minutes)
    timeframes: List[int] = field(default_factory=lambda: [1, 5, 15, 60, 240, 1440])
    primary_timeframe: int = 15  # 15-minute primary
    # Technical indicator lookback periods
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    ema_short: int = 9
    ema_medium: int = 21
    ema_long: int = 55
    atr_period: int = 14
    adx_period: int = 14
    # Momentum windows
    momentum_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 40, 80])
    # Mean reversion lookback
    mr_lookback: int = 60
    ou_half_life_min: float = 2.0   # Min acceptable OU half-life (candles)
    ou_half_life_max: float = 50.0  # Max acceptable OU half-life (candles)
    z_score_entry: float = 2.0      # Enter mean reversion at this Z-score
    z_score_exit: float = 0.5       # Exit at this Z-score
    # Funding rate thresholds (per 8h)
    funding_long_threshold: float = -0.0005   # Very negative → buy signal
    funding_short_threshold: float = 0.0005   # Very positive → sell signal


@dataclass
class MLConfig:
    # LSTM settings
    lstm_sequence_len: int = field(default_factory=lambda: int(os.getenv("LSTM_SEQUENCE_LEN", "60")))
    lstm_units: List[int] = field(default_factory=lambda: [128, 64, 32])
    lstm_dropout: float = 0.2
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    lstm_retrain_hours: int = field(default_factory=lambda: int(os.getenv("LSTM_RETRAIN_INTERVAL_HOURS", "24")))
    # XGBoost settings
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.01
    xgb_subsample: float = 0.8
    xgb_colsample: float = 0.8
    xgb_retrain_hours: int = field(default_factory=lambda: int(os.getenv("XGBOOST_RETRAIN_INTERVAL_HOURS", "12")))
    # Minimum training samples
    min_train_samples: int = 500
    # Model ensemble weights (LSTM, XGBoost)
    ensemble_weights: List[float] = field(default_factory=lambda: [0.4, 0.6])
    # Walk-forward optimization
    train_window_days: int = 180
    test_window_days: int = 30
    step_days: int = 30


@dataclass
class RegimeConfig:
    # HMM settings
    n_regimes: int = 3   # Bull, Bear, Sideways
    hmm_n_iter: int = 200
    hmm_lookback: int = 500
    # Regime labels (assigned based on mean return)
    regime_names: List[str] = field(default_factory=lambda: ["Bear", "Sideways", "Bull"])
    # Kalman filter settings for trend estimation
    kalman_transition_cov: float = 0.001
    kalman_observation_cov: float = 1.0


@dataclass
class ExecutionConfig:
    # Order types
    use_limit_orders: bool = True
    limit_order_offset_bps: float = 2.0   # Place limit 2bps better than mid
    # TWAP settings
    twap_slices: int = 5
    twap_interval_seconds: int = 10
    # Maximum slippage tolerance
    max_slippage_bps: float = 10.0
    # Order timeout
    order_timeout_seconds: int = 30
    # Retry attempts
    max_retries: int = 3


@dataclass
class BacktestConfig:
    start_date: str = field(default_factory=lambda: os.getenv("BACKTEST_START", "2024-01-01"))
    end_date: str = field(default_factory=lambda: os.getenv("BACKTEST_END", "2025-01-01"))
    # Transaction costs
    maker_fee: float = 0.0002   # 0.02% maker fee on Hyperliquid
    taker_fee: float = 0.0005   # 0.05% taker fee on Hyperliquid
    slippage_bps: float = 2.0   # 2 bps slippage assumption
    # Funding rate impact
    include_funding: bool = True
    # Rebalance frequency
    rebalance_minutes: int = 15


@dataclass
class LogConfig:
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    file: str = field(default_factory=lambda: os.getenv("LOG_FILE", "logs/hypestrat.log"))
    rotation: str = "100 MB"
    retention: str = "30 days"


@dataclass
class HypestratConfig:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    asset: AssetConfig = field(default_factory=AssetConfig)
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    log: LogConfig = field(default_factory=LogConfig)

    def validate(self) -> None:
        if not self.exchange.private_key and not os.getenv("BACKTESTING"):
            raise ValueError("HL_PRIVATE_KEY must be set for live trading")
        if self.risk.kelly_fraction <= 0 or self.risk.kelly_fraction > 1:
            raise ValueError("KELLY_FRACTION must be in (0, 1]")
        if self.capital.initial_capital <= 0:
            raise ValueError("INITIAL_CAPITAL must be positive")


# Singleton config instance
CONFIG = HypestratConfig()
