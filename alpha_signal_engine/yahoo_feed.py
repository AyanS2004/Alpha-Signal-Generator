"""
Yahoo-based polling feed (no API key needed).

Note: Yahoo prices can be delayed and update cadence is ~seconds to minutes.
Suitable as a key-only fallback when a provider secret is unavailable.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Callable, Dict

import yfinance as yf


class YahooPollingFeed:
    def __init__(self, symbol: str, interval_sec: float = 2.0) -> None:
        self.symbol = symbol.upper()
        self.interval_sec = interval_sec
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest: Optional[Dict] = None
        self._on_data: Optional[Callable[[Dict], None]] = None
        self._ticker = yf.Ticker(self.symbol)

    def start(self, on_data: Optional[Callable[[Dict], None]] = None) -> None:
        self._on_data = on_data
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def get_latest(self) -> Optional[Dict]:
        return self._latest

    def _loop(self) -> None:
        while self._running:
            try:
                price = None
                try:
                    fi = self._ticker.fast_info
                    price = float(fi.last_price) if getattr(fi, 'last_price', None) is not None else None
                except Exception:
                    pass
                if price is None:
                    hist = self._ticker.history(period='1d', interval='1m')
                    if not hist.empty:
                        price = float(hist['Close'].iloc[-1])

                if price is not None:
                    tick = {
                        'timestamp': int(time.time() * 1e9),
                        'symbol': self.symbol,
                        'price': price,
                        'size': 0,
                    }
                    self._latest = tick
                    if self._on_data:
                        try:
                            self._on_data(tick)
                        except Exception:
                            pass
            except Exception:
                pass
            time.sleep(self.interval_sec)


