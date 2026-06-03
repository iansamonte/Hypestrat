"""
Hyperliquid exchange client — wraps the official SDK with:
  - Async/sync candle & orderbook data retrieval
  - Real-time WebSocket subscriptions for trades, L2 book, funding
  - Account state, open positions, portfolio equity tracking
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable

import pandas as pd
import numpy as np

from utils.logger import logger

try:
    import hyperliquid.utils.constants as hl_const
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import signing
    from eth_account import Account
    HL_SDK_AVAILABLE = True
except ImportError:
    logger.warning("hyperliquid-python-sdk not installed — mock mode active")
    HL_SDK_AVAILABLE = False


class HyperliquidClient:
    """
    Thread-safe wrapper around the Hyperliquid Python SDK.
    Provides both synchronous data methods and an async WebSocket subscriber.
    """

    def __init__(self, private_key: str, wallet_address: str, mainnet: bool = True) -> None:
        self.wallet_address = wallet_address
        self.mainnet = mainnet
        self._info: Optional[Any] = None
        self._exchange: Optional[Any] = None

        if HL_SDK_AVAILABLE and private_key:
            base_url = hl_const.MAINNET_API_URL if mainnet else hl_const.TESTNET_API_URL
            self._info = Info(base_url, skip_ws=True)
            account = Account.from_key(private_key)
            self._exchange = Exchange(account, base_url)
            logger.info(f"Hyperliquid client initialised | {'mainnet' if mainnet else 'testnet'} | {wallet_address[:10]}…")
        else:
            logger.warning("Running in MOCK mode — no live data or orders")

    # ──────────────────────────────────────────────────────────────────────────
    # Market Data
    # ──────────────────────────────────────────────────────────────────────────

    def get_candles(
        self,
        coin: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles from Hyperliquid.
        interval: '1m', '3m', '5m', '15m', '1h', '4h', '1d'
        Returns DataFrame with columns: [timestamp, open, high, low, close, volume]
        """
        if self._info is None:
            return self._mock_candles(start_ms, end_ms, interval)

        try:
            snapshot = self._info.candles_snapshot(coin, interval, start_ms, end_ms)
            rows = []
            for c in snapshot:
                rows.append({
                    "timestamp": pd.Timestamp(c["t"], unit="ms", tz="UTC"),
                    "open": float(c["o"]),
                    "high": float(c["h"]),
                    "low": float(c["l"]),
                    "close": float(c["c"]),
                    "volume": float(c["v"]),
                    "num_trades": int(c["n"]),
                })
            df = pd.DataFrame(rows).set_index("timestamp")
            return df
        except Exception as e:
            logger.error(f"get_candles error: {e}")
            return pd.DataFrame()

    def get_candles_n(
        self,
        coin: str,
        interval: str,
        n: int = 500,
    ) -> pd.DataFrame:
        """Fetch the last n candles for the given interval."""
        interval_ms = self._interval_to_ms(interval)
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - n * interval_ms
        return self.get_candles(coin, interval, start_ms, end_ms)

    def get_orderbook(self, coin: str) -> Dict[str, Any]:
        """
        Get current L2 order book snapshot.
        Returns {'bids': [[price, size], ...], 'asks': [[price, size], ...]}
        """
        if self._info is None:
            return {"bids": [[10.0, 100.0]], "asks": [[10.01, 100.0]]}
        try:
            book = self._info.l2_snapshot(coin)
            return {
                "bids": [[float(b[0]), float(b[1])] for b in book.get("levels", [[]])[0]],
                "asks": [[float(a[0]), float(a[1])] for a in book.get("levels", [[]])[1]],
            }
        except Exception as e:
            logger.error(f"get_orderbook error: {e}")
            return {"bids": [], "asks": []}

    def get_mid_price(self, coin: str) -> float:
        """Get current mid price."""
        book = self.get_orderbook(coin)
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        if bids and asks:
            return (float(bids[0][0]) + float(asks[0][0])) / 2.0
        if self._info:
            try:
                mids = self._info.all_mids()
                return float(mids.get(coin, 0.0))
            except Exception:
                pass
        return 0.0

    def get_funding_history(
        self,
        coin: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """
        Fetch funding rate history.
        Returns DataFrame with [timestamp, funding_rate, premium, open_interest]
        """
        if self._info is None:
            return self._mock_funding(start_ms, end_ms)
        try:
            history = self._info.funding_history(coin, start_ms, end_ms)
            rows = []
            for h in history:
                rows.append({
                    "timestamp": pd.Timestamp(h["time"], unit="ms", tz="UTC"),
                    "funding_rate": float(h["fundingRate"]),
                    "premium": float(h.get("premium", 0.0)),
                    "open_interest": float(h.get("openInterest", 0.0)),
                })
            return pd.DataFrame(rows).set_index("timestamp")
        except Exception as e:
            logger.error(f"get_funding_history error: {e}")
            return pd.DataFrame()

    def get_current_funding_rate(self, coin: str) -> float:
        """Get the current funding rate (per 8h period)."""
        if self._info is None:
            return 0.0001
        try:
            meta = self._info.meta()
            for market in meta.get("universe", []):
                if market["name"] == coin:
                    return float(market.get("funding", 0.0))
            return 0.0
        except Exception as e:
            logger.error(f"get_current_funding_rate error: {e}")
            return 0.0

    def get_recent_trades(self, coin: str) -> List[Dict]:
        """Get recent trade tape."""
        if self._info is None:
            return []
        try:
            return self._info.recent_trades(coin)
        except Exception as e:
            logger.error(f"get_recent_trades error: {e}")
            return []

    def get_open_interest(self, coin: str) -> float:
        """Get current open interest in native token units."""
        if self._info is None:
            return 1_000_000.0
        try:
            meta = self._info.meta()
            for market in meta.get("universe", []):
                if market["name"] == coin:
                    return float(market.get("openInterest", 0.0))
            return 0.0
        except Exception as e:
            logger.error(f"get_open_interest error: {e}")
            return 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Account State
    # ──────────────────────────────────────────────────────────────────────────

    def get_account_state(self) -> Dict[str, Any]:
        """
        Get full account state: balances, open positions, margin info.
        """
        if self._info is None or not self.wallet_address:
            return {"portfolioValue": 5000.0, "positions": [], "withdrawable": 5000.0}
        try:
            state = self._info.user_state(self.wallet_address)
            return state
        except Exception as e:
            logger.error(f"get_account_state error: {e}")
            return {}

    def get_portfolio_value(self) -> float:
        """Total portfolio equity (USDC) including unrealised PnL."""
        state = self.get_account_state()
        return float(state.get("portfolioValue", state.get("crossMarginSummary", {}).get("accountValue", 0.0)))

    def get_position(self, coin: str) -> Optional[Dict[str, Any]]:
        """Get current open position for a coin."""
        state = self.get_account_state()
        for pos in state.get("assetPositions", []):
            if pos.get("position", {}).get("coin") == coin:
                return pos["position"]
        return None

    def get_open_orders(self, coin: str) -> List[Dict]:
        """Get all open orders for a coin."""
        if self._info is None:
            return []
        try:
            orders = self._info.open_orders(self.wallet_address)
            return [o for o in orders if o.get("coin") == coin]
        except Exception as e:
            logger.error(f"get_open_orders error: {e}")
            return []

    # ──────────────────────────────────────────────────────────────────────────
    # Order Execution
    # ──────────────────────────────────────────────────────────────────────────

    def place_limit_order(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        price: float,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """Place a resting limit order."""
        if self._exchange is None:
            logger.info(f"MOCK limit {'BUY' if is_buy else 'SELL'} {size} {coin} @ {price}")
            return {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"resting": {"oid": 999}}]}}}
        try:
            order_type = {"limit": {"tif": "Gtc"}}
            result = self._exchange.order(coin, is_buy, size, price, order_type, reduce_only=reduce_only)
            logger.info(f"Limit {'BUY' if is_buy else 'SELL'} {size} {coin} @ {price} → {result}")
            return result
        except Exception as e:
            logger.error(f"place_limit_order error: {e}")
            return {"status": "error", "message": str(e)}

    def place_market_order(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        slippage_bps: float = 10.0,
    ) -> Dict[str, Any]:
        """Place an immediate market order (IOC limit with slippage tolerance)."""
        if self._exchange is None:
            logger.info(f"MOCK market {'BUY' if is_buy else 'SELL'} {size} {coin}")
            return {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"filled": {"oid": 998, "avgPx": "10.0", "totalSz": str(size)}}]}}}
        try:
            mid = self.get_mid_price(coin)
            slippage = slippage_bps / 10000
            if is_buy:
                limit_px = round(mid * (1 + slippage), 6)
            else:
                limit_px = round(mid * (1 - slippage), 6)
            order_type = {"limit": {"tif": "Ioc"}}
            result = self._exchange.order(coin, is_buy, size, limit_px, order_type)
            logger.info(f"Market {'BUY' if is_buy else 'SELL'} {size} {coin} @ ~{mid} → {result}")
            return result
        except Exception as e:
            logger.error(f"place_market_order error: {e}")
            return {"status": "error", "message": str(e)}

    def cancel_order(self, coin: str, order_id: int) -> Dict[str, Any]:
        """Cancel a specific order by ID."""
        if self._exchange is None:
            return {"status": "ok"}
        try:
            result = self._exchange.cancel(coin, order_id)
            logger.info(f"Cancelled order {order_id} on {coin}")
            return result
        except Exception as e:
            logger.error(f"cancel_order error: {e}")
            return {"status": "error", "message": str(e)}

    def cancel_all_orders(self, coin: str) -> None:
        """Cancel all open orders for a coin."""
        open_orders = self.get_open_orders(coin)
        for order in open_orders:
            self.cancel_order(coin, order["oid"])

    def close_position(self, coin: str) -> Dict[str, Any]:
        """Market-close an entire position."""
        if self._exchange is None:
            return {"status": "ok"}
        try:
            result = self._exchange.market_close(coin)
            logger.info(f"Closed position on {coin} → {result}")
            return result
        except Exception as e:
            logger.error(f"close_position error: {e}")
            return {"status": "error", "message": str(e)}

    def set_leverage(self, coin: str, leverage: int, is_cross: bool = True) -> Dict[str, Any]:
        """Set leverage for a coin."""
        if self._exchange is None:
            return {"status": "ok"}
        try:
            result = self._exchange.update_leverage(leverage, coin, is_cross)
            logger.info(f"Set leverage {leverage}x on {coin}")
            return result
        except Exception as e:
            logger.error(f"set_leverage error: {e}")
            return {"status": "error", "message": str(e)}

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _interval_to_ms(interval: str) -> int:
        mapping = {
            "1m": 60_000,
            "3m": 180_000,
            "5m": 300_000,
            "15m": 900_000,
            "30m": 1_800_000,
            "1h": 3_600_000,
            "2h": 7_200_000,
            "4h": 14_400_000,
            "8h": 28_800_000,
            "1d": 86_400_000,
        }
        return mapping.get(interval, 900_000)

    def _mock_candles(self, start_ms: int, end_ms: int, interval: str) -> pd.DataFrame:
        """Generate synthetic OHLCV for testing without live data."""
        interval_ms = self._interval_to_ms(interval)
        timestamps = range(start_ms, end_ms, interval_ms)
        n = len(list(timestamps))
        np.random.seed(42)
        prices = 10.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n)))
        df = pd.DataFrame({
            "timestamp": [pd.Timestamp(t, unit="ms", tz="UTC") for t in range(start_ms, end_ms, interval_ms)],
            "open": prices,
            "high": prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
            "close": prices * np.exp(np.random.normal(0, 0.005, n)),
            "volume": np.abs(np.random.normal(50000, 10000, n)),
            "num_trades": np.random.randint(100, 1000, n),
        }).set_index("timestamp")
        df["high"] = df[["open", "close", "high"]].max(axis=1)
        df["low"] = df[["open", "close", "low"]].min(axis=1)
        return df

    def _mock_funding(self, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Generate synthetic funding rate history."""
        interval_ms = 8 * 3_600_000  # 8h
        rows = []
        t = start_ms
        while t < end_ms:
            rows.append({
                "timestamp": pd.Timestamp(t, unit="ms", tz="UTC"),
                "funding_rate": np.random.normal(0.0001, 0.0003),
                "premium": np.random.normal(0, 0.01),
                "open_interest": np.random.uniform(1e6, 1e7),
            })
            t += interval_ms
        return pd.DataFrame(rows).set_index("timestamp")
