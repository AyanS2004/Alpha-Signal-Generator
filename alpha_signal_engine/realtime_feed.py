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
import random
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

# Import the advanced live signal generator
try:
    from .live_signal_generator import LiveSignalGenerator
except ImportError:
    LiveSignalGenerator = None


class MarketState(Enum):
    """Market state enumeration for stateful simulation."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"


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
        
        # Stateful simulation properties
        self.current_market_state = MarketState.RANGING
        self.state_start_time = datetime.now()
        self.state_duration = random.randint(30, 90)  # 30-90 seconds per state
        self.state_transition_count = 0
        self.last_signal_time = datetime.now()
        self.signal_cooldown = 2  # Minimum seconds between signals
        
        # Advanced signal generation
        self.advanced_mode = False
        self.live_signal_generator = None
        self.model_path = None
        
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
    
    def enable_advanced_mode(self, model_path: Optional[str] = None):
        """
        Enable advanced signal generation with ML model and Kalman filtering.
        
        Args:
            model_path: Path to pre-trained ML model file
        """
        if LiveSignalGenerator is None:
            print("⚠️ Advanced mode not available - LiveSignalGenerator not found")
            return
        
        try:
            self.model_path = model_path
            self.live_signal_generator = LiveSignalGenerator(self.config, model_path)
            self.advanced_mode = True
            print("🧠 Advanced mode enabled with ML model and Kalman filtering")
        except Exception as e:
            print(f"❌ Failed to enable advanced mode: {e}")
            self.advanced_mode = False
    
    def disable_advanced_mode(self):
        """Disable advanced mode and return to basic simulation."""
        if self.live_signal_generator:
            self.live_signal_generator.stop()
            self.live_signal_generator = None
        self.advanced_mode = False
        print("📊 Switched to basic simulation mode")
    
    def stop_feed(self):
        """Stop the real-time data feed."""
        self.is_connected = False
        if self.ws:
            self.ws.close()
        
        # Stop advanced signal generator if active
        if self.live_signal_generator:
            self.live_signal_generator.stop()
        
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
                
                # Generate signal using appropriate method
                if self.advanced_mode and self.live_signal_generator:
                    # Use advanced ML-powered signal generation
                    signal_data = self.live_signal_generator.add_data_point(data_point)
                    if signal_data and signal_data['signal'] != self.current_signal:
                        self.current_signal = signal_data['signal']
                        self.signal_history.append(signal_data)
                        self._trigger_callbacks('signal', signal_data)
                else:
                    # Use basic stateful simulation
                    if len(self.data_buffer) >= 5:
                        signal = self._generate_realtime_signal()
                        if signal != self.current_signal:
                            self.current_signal = signal
                            confidence = self._get_confidence_score()
                            
                            signal_data = {
                                'timestamp': datetime.now(),
                                'signal': signal,
                                'price': new_price,
                                'confidence': confidence,
                                'market_state': self.current_market_state.value,
                                'state_duration': (datetime.now() - self.state_start_time).total_seconds()
                            }
                            
                            self.signal_history.append(signal_data)
                            self._trigger_callbacks('signal', signal_data)
                
                time.sleep(1)  # 1-second intervals
        
        # Start simulation in separate thread
        simulation_thread = threading.Thread(target=generate_simulated_data)
        simulation_thread.daemon = True
        simulation_thread.start()
    
    def _generate_realtime_signal(self) -> int:
        """Generate stateful real-time trading signal with market memory."""
        if len(self.data_buffer) < 5:  # Reduced requirement for faster response
            return 0
        
        # Check if we should transition to a new market state
        self._check_state_transition()
        
        # Check signal cooldown to avoid too frequent signals
        current_time = datetime.now()
        if (current_time - self.last_signal_time).total_seconds() < self.signal_cooldown:
            return 0
        
        # Generate signal based on current market state
        signal = self._generate_state_based_signal()
        
        # Update last signal time if we generated a non-zero signal
        if signal != 0:
            self.last_signal_time = current_time
        
        return signal
    
    def _check_state_transition(self):
        """Check if it's time to transition to a new market state."""
        current_time = datetime.now()
        time_in_state = (current_time - self.state_start_time).total_seconds()
        
        if time_in_state >= self.state_duration:
            self._transition_to_new_state()
    
    def _transition_to_new_state(self):
        """Transition to a new market state."""
        # Define transition probabilities
        transition_matrix = {
            MarketState.TRENDING_UP: {
                MarketState.TRENDING_UP: 0.3,    # Stay in trend
                MarketState.TRENDING_DOWN: 0.2,  # Reverse trend
                MarketState.RANGING: 0.5         # Go to ranging
            },
            MarketState.TRENDING_DOWN: {
                MarketState.TRENDING_UP: 0.2,    # Reverse trend
                MarketState.TRENDING_DOWN: 0.3,  # Stay in trend
                MarketState.RANGING: 0.5         # Go to ranging
            },
            MarketState.RANGING: {
                MarketState.TRENDING_UP: 0.4,    # Start uptrend
                MarketState.TRENDING_DOWN: 0.4,  # Start downtrend
                MarketState.RANGING: 0.2         # Stay ranging
            }
        }
        
        # Select new state based on probabilities
        current_state = self.current_market_state
        probabilities = transition_matrix[current_state]
        
        rand = random.random()
        cumulative = 0
        
        for new_state, prob in probabilities.items():
            cumulative += prob
            if rand <= cumulative:
                self.current_market_state = new_state
                break
        
        # Update state tracking
        self.state_start_time = datetime.now()
        self.state_duration = random.randint(30, 90)  # New random duration
        self.state_transition_count += 1
        
        print(f"🔄 Market state transitioned to: {self.current_market_state.value}")
    
    def _generate_state_based_signal(self) -> int:
        """Generate signal based on current market state."""
        rand = random.random()
        
        if self.current_market_state == MarketState.TRENDING_UP:
            # Trending up: 80% BUY, 20% HOLD
            if rand < 0.8:
                return 1  # Buy signal
            else:
                return 0  # Hold signal
                
        elif self.current_market_state == MarketState.TRENDING_DOWN:
            # Trending down: 80% SELL, 20% HOLD
            if rand < 0.8:
                return -1  # Sell signal
            else:
                return 0  # Hold signal
                
        else:  # RANGING
            # Ranging: 90% HOLD, 5% BUY, 5% SELL
            if rand < 0.9:
                return 0  # Hold signal
            elif rand < 0.95:
                return 1  # Buy signal
            else:
                return -1  # Sell signal
    
    def _get_confidence_score(self) -> float:
        """Get confidence score based on current market state."""
        if self.current_market_state in [MarketState.TRENDING_UP, MarketState.TRENDING_DOWN]:
            # Strong trend: high confidence (75-95%)
            return random.uniform(0.75, 0.95)
        else:  # RANGING
            # Ranging market: low confidence (40-60%)
            return random.uniform(0.40, 0.60)
    
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
        
        # Calculate basic indicators for real-time display/use
        try:
            df['return'] = df['close'].pct_change()
            df['sma_10'] = df['close'].rolling(window=10, min_periods=10).mean()
            df['sma_20'] = df['close'].rolling(window=20, min_periods=20).mean()
            # Simple RSI (Wilder's smoothing approximation)
            delta = df['close'].diff()
            gain = (delta.clip(lower=0)).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / (loss.replace(0, np.nan))
            df['rsi_14'] = 100 - (100 / (1 + rs))
        except Exception:
            return
        
        # Store latest indicators on the instance for quick access if needed
        self.latest_indicators = {
            'sma_10': float(df['sma_10'].iloc[-1]) if not np.isnan(df['sma_10'].iloc[-1]) else None,
            'sma_20': float(df['sma_20'].iloc[-1]) if not np.isnan(df['sma_20'].iloc[-1]) else None,
            'rsi_14': float(df['rsi_14'].iloc[-1]) if not np.isnan(df['rsi_14'].iloc[-1]) else None,
        }
    
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
    
    def get_current_market_state(self) -> Dict:
        """Get current market state information."""
        return {
            'state': self.current_market_state.value,
            'state_start_time': self.state_start_time,
            'state_duration': self.state_duration,
            'time_in_state': (datetime.now() - self.state_start_time).total_seconds(),
            'transition_count': self.state_transition_count,
            'current_signal': self.current_signal,
            'last_signal_time': self.last_signal_time
        }
    
    def get_simulation_stats(self) -> Dict:
        """Get simulation statistics."""
        if not self.signal_history:
            return {'total_signals': 0, 'state_distribution': {}}
        
        # Count signals by state
        state_counts = {}
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        for signal_data in self.signal_history:
            state = signal_data.get('market_state', 'unknown')
            signal = signal_data.get('signal', 0)
            
            state_counts[state] = state_counts.get(state, 0) + 1
            
            if signal > 0:
                signal_counts['BUY'] += 1
            elif signal < 0:
                signal_counts['SELL'] += 1
            else:
                signal_counts['HOLD'] += 1
        
        return {
            'total_signals': len(self.signal_history),
            'state_distribution': state_counts,
            'signal_distribution': signal_counts,
            'current_state': self.current_market_state.value,
            'state_transitions': self.state_transition_count
        }
    
    def get_advanced_mode_info(self) -> Dict:
        """Get information about advanced mode status."""
        if not self.advanced_mode or not self.live_signal_generator:
            return {
                'advanced_mode': False,
                'model_loaded': False,
                'kalman_filter_active': False
            }
        
        model_info = self.live_signal_generator.get_model_info()
        performance_metrics = self.live_signal_generator.get_performance_metrics()
        
        return {
            'advanced_mode': True,
            'model_loaded': model_info['model_loaded'],
            'model_last_updated': model_info['last_updated'],
            'retraining_active': model_info['retraining_active'],
            'retraining_data_points': model_info['retraining_data_points'],
            'total_signals': performance_metrics['total_signals'],
            'kalman_filter_active': True,
            'model_path': self.model_path
        }


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
