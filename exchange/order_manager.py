"""
Order lifecycle manager with state tracking, fill monitoring, and TWAP execution.

Manages:
  - Order state machine (PENDING → OPEN → FILLED / CANCELLED / REJECTED)
  - Partial fills and fill averaging
  - TWAP slicer for large orders
  - Stop-loss and take-profit order tracking
"""
from __future__ import annotations

import time
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from threading import Lock

import numpy as np

from utils.logger import logger
from exchange.hyperliquid_client import HyperliquidClient


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


@dataclass
class Order:
    order_id: str
    coin: str
    side: OrderSide
    order_type: OrderType
    size: float
    price: float
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    parent_trade_id: Optional[str] = None
    reduce_only: bool = False

    @property
    def remaining_size(self) -> float:
        return self.size - self.filled_size

    @property
    def fill_pct(self) -> float:
        return self.filled_size / self.size if self.size > 0 else 0.0

    def update_fill(self, fill_size: float, fill_price: float) -> None:
        if self.filled_size + fill_size > self.size:
            fill_size = self.size - self.filled_size
        # Running average fill price
        total_filled = self.filled_size + fill_size
        if total_filled > 0:
            self.avg_fill_price = (
                (self.avg_fill_price * self.filled_size + fill_price * fill_size) / total_filled
            )
        self.filled_size = total_filled
        self.status = OrderStatus.FILLED if self.remaining_size < 1e-9 else OrderStatus.PARTIALLY_FILLED
        self.updated_at = time.time()


@dataclass
class Trade:
    trade_id: str
    coin: str
    side: OrderSide
    target_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: float
    orders: List[Order] = field(default_factory=list)
    pnl_usdc: float = 0.0
    is_closed: bool = False
    opened_at: float = field(default_factory=time.time)
    closed_at: Optional[float] = None

    @property
    def position_size(self) -> float:
        filled = sum(o.filled_size for o in self.orders if o.order_type in (OrderType.LIMIT, OrderType.MARKET))
        return filled

    @property
    def avg_entry(self) -> float:
        orders = [o for o in self.orders if o.order_type in (OrderType.LIMIT, OrderType.MARKET) and o.filled_size > 0]
        if not orders:
            return self.entry_price
        total_val = sum(o.avg_fill_price * o.filled_size for o in orders)
        total_sz = sum(o.filled_size for o in orders)
        return total_val / total_sz if total_sz > 0 else self.entry_price

    def unrealised_pnl(self, current_price: float) -> float:
        sz = self.position_size
        if sz == 0:
            return 0.0
        direction = 1 if self.side == OrderSide.BUY else -1
        return direction * sz * (current_price - self.avg_entry) * self.leverage

    def realised_r_multiple(self) -> float:
        """Return trade profit in units of initial risk (R)."""
        risk_per_unit = abs(self.avg_entry - self.stop_loss)
        if risk_per_unit == 0:
            return 0.0
        actual_per_unit = self.pnl_usdc / self.position_size if self.position_size > 0 else 0.0
        direction = 1 if self.side == OrderSide.BUY else -1
        return direction * actual_per_unit / risk_per_unit


class OrderManager:
    """
    Central order management: places, tracks, and closes all orders.
    Thread-safe via internal lock.
    """

    def __init__(self, client: HyperliquidClient, max_slippage_bps: float = 10.0) -> None:
        self._client = client
        self._max_slippage_bps = max_slippage_bps
        self._orders: Dict[str, Order] = {}
        self._trades: Dict[str, Trade] = {}
        self._lock = Lock()
        self._trade_counter = 0
        self._order_counter = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def open_trade(
        self,
        coin: str,
        side: OrderSide,
        size: float,
        current_price: float,
        stop_loss: float,
        take_profit: float,
        leverage: float = 1.0,
        use_limit: bool = True,
        limit_offset_bps: float = 2.0,
    ) -> Optional[Trade]:
        """Open a new trade with entry, stop-loss, and take-profit orders."""
        with self._lock:
            trade_id = self._next_trade_id()
            entry_price = self._compute_entry_price(side, current_price, limit_offset_bps, use_limit)
            trade = Trade(
                trade_id=trade_id,
                coin=coin,
                side=side,
                target_size=size,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=leverage,
            )
            self._trades[trade_id] = trade

        # Set leverage
        self._client.set_leverage(coin, int(leverage))

        # Place entry order
        is_buy = (side == OrderSide.BUY)
        if use_limit:
            result = self._client.place_limit_order(coin, is_buy, size, entry_price)
        else:
            result = self._client.place_market_order(coin, is_buy, size, self._max_slippage_bps)

        order_id = self._extract_order_id(result) or self._next_order_id()
        order = Order(
            order_id=order_id,
            coin=coin,
            side=side,
            order_type=OrderType.LIMIT if use_limit else OrderType.MARKET,
            size=size,
            price=entry_price,
            parent_trade_id=trade_id,
        )
        with self._lock:
            self._orders[order_id] = order
            trade.orders.append(order)

        if not use_limit:
            # Simulate immediate fill for market orders
            order.update_fill(size, entry_price)

        logger.info(
            f"Trade {trade_id} opened | {side.value} {size:.4f} {coin} "
            f"@ {entry_price:.4f} | SL={stop_loss:.4f} TP={take_profit:.4f} | {leverage}x"
        )
        return trade

    def close_trade(self, trade_id: str, current_price: float) -> float:
        """Close a trade and return realised PnL in USDC."""
        with self._lock:
            trade = self._trades.get(trade_id)
            if trade is None or trade.is_closed:
                return 0.0

        # Cancel any open orders for this trade
        self._client.cancel_all_orders(trade.coin)

        # Close position
        pos = self._client.get_position(trade.coin)
        if pos:
            self._client.close_position(trade.coin)

        pnl = trade.unrealised_pnl(current_price)
        with self._lock:
            trade.pnl_usdc = pnl
            trade.is_closed = True
            trade.closed_at = time.time()

        logger.info(
            f"Trade {trade_id} closed | PnL={pnl:+.2f} USDC | "
            f"R={trade.realised_r_multiple():.2f}R"
        )
        return pnl

    def twap_order(
        self,
        coin: str,
        is_buy: bool,
        total_size: float,
        n_slices: int = 5,
        interval_seconds: float = 10.0,
    ) -> List[Dict]:
        """
        Execute a large order as TWAP (Time-Weighted Average Price).
        Splits into n_slices child orders placed interval_seconds apart.
        """
        slice_size = total_size / n_slices
        results = []
        logger.info(f"TWAP: {'BUY' if is_buy else 'SELL'} {total_size} {coin} in {n_slices} slices")

        for i in range(n_slices):
            price = self._client.get_mid_price(coin)
            slippage = self._max_slippage_bps / 10000
            limit_px = price * (1 + slippage) if is_buy else price * (1 - slippage)
            result = self._client.place_limit_order(coin, is_buy, slice_size, round(limit_px, 6))
            results.append(result)
            logger.debug(f"TWAP slice {i+1}/{n_slices}: size={slice_size:.4f} @ {limit_px:.4f}")
            if i < n_slices - 1:
                time.sleep(interval_seconds)

        return results

    def update_trailing_stop(
        self,
        trade: Trade,
        current_price: float,
        atr: float,
        atr_multiple: float = 1.5,
    ) -> None:
        """Update trailing stop loss based on current ATR and price."""
        is_long = trade.side == OrderSide.BUY
        if is_long:
            new_stop = current_price - atr * atr_multiple
            if new_stop > trade.stop_loss:
                trade.stop_loss = new_stop
                logger.debug(f"Trade {trade.trade_id}: trailing stop updated → {new_stop:.4f}")
        else:
            new_stop = current_price + atr * atr_multiple
            if new_stop < trade.stop_loss:
                trade.stop_loss = new_stop
                logger.debug(f"Trade {trade.trade_id}: trailing stop updated → {new_stop:.4f}")

    def check_stops(self, trade: Trade, current_price: float) -> Optional[str]:
        """
        Check if stop-loss or take-profit has been hit.
        Returns 'stop_loss', 'take_profit', or None.
        """
        if trade.is_closed:
            return None
        if trade.side == OrderSide.BUY:
            if current_price <= trade.stop_loss:
                return "stop_loss"
            if current_price >= trade.take_profit:
                return "take_profit"
        else:
            if current_price >= trade.stop_loss:
                return "stop_loss"
            if current_price <= trade.take_profit:
                return "take_profit"
        return None

    def get_open_trades(self) -> List[Trade]:
        with self._lock:
            return [t for t in self._trades.values() if not t.is_closed]

    def get_all_trades(self) -> List[Trade]:
        with self._lock:
            return list(self._trades.values())

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _next_trade_id(self) -> str:
        self._trade_counter += 1
        return f"T{self._trade_counter:06d}"

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"O{self._order_counter:08d}"

    @staticmethod
    def _compute_entry_price(
        side: OrderSide,
        mid_price: float,
        offset_bps: float,
        use_limit: bool,
    ) -> float:
        if not use_limit:
            return mid_price
        offset = offset_bps / 10000
        if side == OrderSide.BUY:
            return mid_price * (1 - offset)
        return mid_price * (1 + offset)

    @staticmethod
    def _extract_order_id(result: Dict) -> Optional[str]:
        try:
            statuses = result["response"]["data"]["statuses"]
            for s in statuses:
                for key in ("resting", "filled"):
                    if key in s:
                        return str(s[key]["oid"])
        except (KeyError, TypeError, IndexError):
            pass
        return None
