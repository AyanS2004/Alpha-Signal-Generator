"""
Alpaca real-time data feed wrapper.

Uses alpaca-py StockDataStream in a background thread and provides a simple
callback-based interface to receive trade updates for a symbol.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Deque, Dict
from collections import deque
from time import time

try:
    from alpaca.data.live import StockDataStream  # type: ignore
    _HAS_ALPACA = True
except Exception:
    _HAS_ALPACA = False


@dataclass
class LiveTick:
    timestamp_ns: int
    price: float
    size: int
    symbol: str


class AlpacaDataFeed:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        symbol: str,
        feed: str = "iex",
        max_buffer: int = 1000,
    ) -> None:
        if not _HAS_ALPACA:
            raise RuntimeError("alpaca-py is not installed. Please install 'alpaca-py'.")

        self.api_key = api_key
        self.secret_key = secret_key
        self.symbol = symbol.upper()
        self.feed = feed

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[StockDataStream] = None
        self._running = False

        self._buffer: Deque[LiveTick] = deque(maxlen=max_buffer)
        self._on_data: Optional[Callable[[Dict], None]] = None

    def start(self, on_data: Optional[Callable[[Dict], None]] = None) -> None:
        self._on_data = on_data
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        if self._thread:
            self._thread.join(timeout=3)

    def get_latest(self) -> Optional[Dict]:
        if not self._buffer:
            return None
        last = self._buffer[-1]
        return {
            "timestamp": last.timestamp_ns,
            "symbol": last.symbol,
            "price": last.price,
            "size": last.size,
        }

    # Internal
    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._run())

    async def _run(self) -> None:
        self._stream = StockDataStream(self.api_key, self.secret_key, feed=self.feed)

        async def on_trade(data) -> None:  # data is a Trade object
            # Collect trade
            tick = LiveTick(
                timestamp_ns=int(getattr(data, "timestamp", time() * 1e9)),
                price=float(getattr(data, "price", 0.0)),
                size=int(getattr(data, "size", 0)),
                symbol=str(getattr(data, "symbol", self.symbol)),
            )
            self._buffer.append(tick)
            if self._on_data is not None:
                try:
                    self._on_data(
                        {
                            "timestamp": tick.timestamp_ns,
                            "symbol": tick.symbol,
                            "price": tick.price,
                            "size": tick.size,
                        }
                    )
                except Exception:
                    pass

        self._stream.subscribe_trades(on_trade, self.symbol)

        try:
            await self._stream._run_forever()  # use protected run to block until stop
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        try:
            if self._stream is not None:
                await self._stream.stop()
        except Exception:
            pass
        try:
            if self._loop is not None:
                self._loop.stop()
        except Exception:
            pass


