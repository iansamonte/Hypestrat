"""Backtest portfolio simulator with mark-to-market and equity tracking."""
from __future__ import annotations

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


class BacktestPortfolio:
    """
    Simulates portfolio state during backtesting.
    Tracks equity, open P&L, and records the equity curve.
    """

    def __init__(
        self,
        initial_capital: float = 5000.0,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0005,
        slippage_bps: float = 2.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.cash = initial_capital
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_bps = slippage_bps

        self._equity_curve: List[float] = [initial_capital]
        self._timestamps: List[Any] = [None]
        self._open_pnl: float = 0.0

    def mark_to_market(self, current_price: float, position: Dict) -> None:
        """Update equity with unrealised P&L of open position."""
        if not position:
            self._open_pnl = 0.0
            return
        entry = position["entry_price"]
        size = position["size_usdc"]
        lev = position["leverage"]
        direction = position["direction"]
        price_ret = (current_price - entry) / entry
        self._open_pnl = direction * price_ret * size * lev
        self.equity = self.cash + self._open_pnl

    def record_trade(self, net_pnl: float, exit_price: float, position: Dict) -> None:
        """Record a closed trade and update cash/equity."""
        self.cash += net_pnl
        self._open_pnl = 0.0
        self.equity = self.cash
        self._equity_curve.append(self.equity)
        self._timestamps.append(None)

    def record_bar(self, timestamp: Any = None) -> None:
        """Record equity at end of each bar (for equity curve)."""
        self._equity_curve.append(self.equity)
        self._timestamps.append(timestamp)

    @property
    def equity_curve(self) -> np.ndarray:
        return np.array(self._equity_curve)

    @property
    def equity_series(self) -> pd.Series:
        return pd.Series(self._equity_curve, index=pd.RangeIndex(len(self._equity_curve)))

    @property
    def total_return(self) -> float:
        return (self.equity - self.initial_capital) / self.initial_capital

    def reset(self) -> None:
        self.equity = self.initial_capital
        self.cash = self.initial_capital
        self._equity_curve = [self.initial_capital]
        self._timestamps = [None]
        self._open_pnl = 0.0
