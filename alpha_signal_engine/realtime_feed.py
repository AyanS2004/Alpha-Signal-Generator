"""
Real-time Data Feed for Alpha Signal Engine.
Provides live market data streaming and real-time signal generation.
"""

import pandas as pd
import numpy as np
import asyncio
import websocket
import json
import threading
import time
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class RealTimeDataFeed:
    """
    Real-time data feed for live trading.
    
    Features:
    - WebSocket connection for live data
    - Real-time signal generation
    - Market data caching
    - Event-driven architecture
    """
    
    def __init__(self, config, symbol: str = "AAPL"):
        """
        Initialize Real-time Data Feed.
        
        Args:
            config: Configuration object
            symbol: Trading symbol
        """
        self.config = config
        self.symbol = symbol
        self.ws = None
        self.is_connected = False
        self.data_buffer = []
        self.max_buffer_size = 1000
        self.callbacks = []
        
        # Data storage
        self.current_price = None
        self.current_volume = None
        self.last_update = None
        
        # Signal tracking
        self.current_signal = 0
        self.signal_history = []
        
    def start_feed(self, on_data_callback: Optional[Callable] = None,
                   on_signal_callback: Optional[Callable] = None):
        """
        Start the real-time data feed.
        
        Args:
            on_data_callback: Callback function for new data
            on_signal_callback: Callback function for new signals
        """
        if on_data_callback:
            self.callbacks.append(('data', on_data_callback))
        if on_signal_callback:
            self.callbacks.append(('signal', on_signal_callback))
        
        # Start WebSocket connection
        self._connect_websocket()
        
        # Start data processing thread
        self.processing_thread = threading.Thread(target=self._process_data_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print(f"🚀 Real-time feed started for {self.symbol}")
    
    def stop_feed(self):
        """Stop the real-time data feed."""
        self.is_connected = False
        if self.ws:
            self.ws.close()
        print("🛑 Real-time feed stopped")
    
    def _connect_websocket(self):
        """Connect to WebSocket for live data."""
        try:
            # For demonstration, we'll simulate real-time data
            # In production, connect to actual market data provider
            self.is_connected = True
            print(f"📡 Connected to {self.symbol} data feed")
            
            # Start data simulation
            self._simulate_live_data()
            
        except Exception as e:
            print(f"❌ Failed to connect: {str(e)}")
            self.is_connected = False
    
    def _simulate_live_data(self):
        """Simulate live market data for testing."""
        def generate_simulated_data():
            base_price = 150.0  # Base price for simulation
            volatility = 0.02   # 2% volatility
            
            while self.is_connected:
                # Generate random price movement
                price_change = np.random.normal(0, volatility)
                new_price = base_price * (1 + price_change)
                
                # Generate volume
                volume = np.random.randint(1000, 10000)
                
                # Create data point
                data_point = {
                    'timestamp': datetime.now(),
                    'symbol': self.symbol,
                    'price': new_price,
                    'volume': volume,
                    'open': new_price * 0.999,
                    'high': new_price * 1.002,
                    'low': new_price * 0.998,
                    'close': new_price
                }
                
                # Update current data
                self.current_price = new_price
                self.current_volume = volume
                self.last_update = data_point['timestamp']
                
                # Add to buffer
                self.data_buffer.append(data_point)
                
                # Maintain buffer size
                if len(self.data_buffer) > self.max_buffer_size:
                    self.data_buffer.pop(0)
                
                # Trigger callbacks
                self._trigger_callbacks('data', data_point)
                
                # Generate signal if enough data
                if len(self.data_buffer) >= 20:
                    signal = self._generate_realtime_signal()
                    if signal != self.current_signal:
                        self.current_signal = signal
                        self.signal_history.append({
                            'timestamp': datetime.now(),
                            'signal': signal,
                            'price': new_price
                        })
                        self._trigger_callbacks('signal', {
                            'timestamp': datetime.now(),
                            'signal': signal,
                            'price': new_price
                        })
                
                time.sleep(1)  # 1-second intervals
        
        # Start simulation in separate thread
        simulation_thread = threading.Thread(target=generate_simulated_data)
        simulation_thread.daemon = True
        simulation_thread.start()
    
    def _generate_realtime_signal(self) -> int:
        """Generate real-time trading signal."""
        if len(self.data_buffer) < 20:
            return 0
        
        # Convert buffer to DataFrame
        df = pd.DataFrame(self.data_buffer)
        df.set_index('timestamp', inplace=True)
        
        # Calculate technical indicators
        returns = df['close'].pct_change()
        sma_10 = df['close'].rolling(window=10).mean()
        sma_20 = df['close'].rolling(window=20).mean()
        
        # Simple signal logic
        current_price = df['close'].iloc[-1]
        current_sma_10 = sma_10.iloc[-1]
        current_sma_20 = sma_20.iloc[-1]
        
        # Momentum signal
        if current_sma_10 > current_sma_20 and current_price > current_sma_10:
            return 1  # Buy signal
        elif current_sma_10 < current_sma_20 and current_price < current_sma_10:
            return -1  # Sell signal
        
        return 0  # Hold
    
    def _trigger_callbacks(self, callback_type: str, data: Dict):
        """Trigger registered callbacks."""
        for cb_type, callback in self.callbacks:
            if cb_type == callback_type:
                try:
                    callback(data)
                except Exception as e:
                    print(f"⚠️ Callback error: {str(e)}")
    
    def _process_data_loop(self):
        """Main data processing loop."""
        while self.is_connected:
            try:
                # Process any pending data
                if self.data_buffer:
                    # Update technical indicators
                    self._update_indicators()
                
                time.sleep(0.1)  # 100ms intervals
                
            except Exception as e:
                print(f"⚠️ Data processing error: {str(e)}")
                time.sleep(1)
    
    def _update_indicators(self):
        """Update technical indicators."""
        if len(self.data_buffer) < 20:
            return
        
        # Convert to DataFrame for calculations
        df = pd.DataFrame(self.data_buffer)
        df.set_index('timestamp', inplace=True)
        
        # Calculate indicators (this would be more sophisticated in production)
        # For now, just ensure we have enough data
        pass
    
    def get_current_data(self) -> Dict:
        """Get current market data."""
        return {
            'symbol': self.symbol,
            'price': self.current_price,
            'volume': self.current_volume,
            'last_update': self.last_update,
            'is_connected': self.is_connected
        }
    
    def get_signal_history(self) -> List[Dict]:
        """Get signal history."""
        return self.signal_history.copy()
    
    def get_data_buffer(self) -> List[Dict]:
        """Get current data buffer."""
        return self.data_buffer.copy()


class LiveSignalGenerator:
    """
    Live signal generator for real-time trading.
    """
    
    def __init__(self, config):
        """
        Initialize Live Signal Generator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.signal_generators = {}
        self.current_signals = {}
        
    def add_signal_generator(self, name: str, generator_func: Callable):
        """
        Add a signal generator function.
        
        Args:
            name: Name of the signal generator
            generator_func: Function that generates signals
        """
        self.signal_generators[name] = generator_func
    
    def generate_live_signals(self, data: Dict) -> Dict:
        """
        Generate live trading signals.
        
        Args:
            data: Current market data
            
        Returns:
            Dictionary with generated signals
        """
        signals = {}
        
        for name, generator_func in self.signal_generators.items():
            try:
                signal = generator_func(data)
                signals[name] = signal
                self.current_signals[name] = signal
            except Exception as e:
                print(f"⚠️ Signal generator {name} error: {str(e)}")
                signals[name] = 0
        
        return signals
    
    def get_combined_signal(self, signals: Dict) -> int:
        """
        Combine multiple signals into a single signal.
        
        Args:
            signals: Dictionary of signals
            
        Returns:
            Combined signal (-1, 0, 1)
        """
        if not signals:
            return 0
        
        # Simple majority voting
        positive_signals = sum(1 for s in signals.values() if s > 0)
        negative_signals = sum(1 for s in signals.values() if s < 0)
        
        if positive_signals > negative_signals:
            return 1
        elif negative_signals > positive_signals:
            return -1
        else:
            return 0


class MarketDataCache:
    """
    Market data cache for efficient data storage and retrieval.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize Market Data Cache.
        
        Args:
            max_size: Maximum number of data points to store
        """
        self.max_size = max_size
        self.data = []
        self.lock = threading.Lock()
    
    def add_data(self, data_point: Dict):
        """
        Add data point to cache.
        
        Args:
            data_point: Market data point
        """
        with self.lock:
            self.data.append(data_point)
            
            # Maintain cache size
            if len(self.data) > self.max_size:
                self.data.pop(0)
    
    def get_recent_data(self, n_points: int = 100) -> List[Dict]:
        """
        Get recent data points.
        
        Args:
            n_points: Number of recent points to retrieve
            
        Returns:
            List of recent data points
        """
        with self.lock:
            return self.data[-n_points:].copy()
    
    def get_data_range(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """
        Get data within a time range.
        
        Args:
            start_time: Start time
            end_time: End time
            
        Returns:
            List of data points in range
        """
        with self.lock:
            return [
                point for point in self.data
                if start_time <= point['timestamp'] <= end_time
            ]
    
    def clear_cache(self):
        """Clear the data cache."""
        with self.lock:
            self.data.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            return {
                'total_points': len(self.data),
                'max_size': self.max_size,
                'utilization': len(self.data) / self.max_size
            }
