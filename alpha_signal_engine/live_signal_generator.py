"""
Advanced Live Signal Generator for Alpha Signal Engine.
Provides context-aware, ML-powered real-time signal generation with adaptive capabilities.
"""

import pandas as pd
import numpy as np
import pickle
import threading
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .advanced_signals import AdvancedSignalGenerator
from .config import Config

warnings.filterwarnings('ignore')


class KalmanFilter:
    """
    Kalman Filter for price smoothing and trend estimation.
    Reduces noise in real-time market data for more stable signals.
    """
    
    def __init__(self, process_variance: float = 0.01, measurement_variance: float = 0.1):
        """
        Initialize Kalman Filter.
        
        Args:
            process_variance: Variance in the process model (how much we expect price to change)
            measurement_variance: Variance in measurements (how noisy our price data is)
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        
        # State vector: [price, velocity]
        self.state = np.array([0.0, 0.0])  # [price, velocity]
        self.covariance = np.eye(2)  # Initial covariance matrix
        
        # State transition matrix (constant velocity model)
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        
        # Process noise covariance
        self.Q = np.array([[self.process_variance, 0.0], [0.0, self.process_variance]])
        
        # Measurement matrix (we only observe price)
        self.H = np.array([[1.0, 0.0]])
        
        # Measurement noise covariance
        self.R = np.array([[self.measurement_variance]])
        
        self.initialized = False
    
    def update(self, measurement: float) -> Tuple[float, float]:
        """
        Update the filter with a new price measurement.
        
        Args:
            measurement: New price measurement
            
        Returns:
            Tuple of (smoothed_price, velocity)
        """
        if not self.initialized:
            self.state[0] = measurement
            self.initialized = True
            return measurement, 0.0
        
        # Predict step
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        
        # Update step
        y = measurement - self.H @ self.state  # Innovation
        S = self.H @ self.covariance @ self.H.T + self.R  # Innovation covariance
        K = self.covariance @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.state = self.state + K @ y
        self.covariance = (np.eye(2) - K @ self.H) @ self.covariance
        
        return self.state[0], self.state[1]


class LiveSignalGenerator:
    """
    Advanced Live Signal Generator with context-aware ML predictions.
    
    Features:
    - Kalman Filter price smoothing
    - Full feature engineering pipeline
    - ML model integration
    - Market regime detection
    - Dynamic model retraining
    - Hot-swapping capabilities
    """
    
    def __init__(self, config: Config, model_path: Optional[str] = None):
        """
        Initialize Live Signal Generator.
        
        Args:
            config: Configuration object
            model_path: Path to pre-trained ML model
        """
        self.config = config
        self.model_path = model_path
        
        # Data storage
        self.data_buffer = []
        self.max_buffer_size = 1000
        self.feature_buffer = []
        
        # Kalman Filter
        self.kalman_filter = KalmanFilter()
        
        # Advanced signal generator
        self.advanced_generator = AdvancedSignalGenerator(config)
        
        # ML model and scaler
        self.ml_model = None
        self.scaler = StandardScaler()
        self.model_last_updated = None
        
        # Retraining parameters
        self.retraining_interval = 24 * 60 * 60  # 24 hours in seconds
        self.min_retraining_data = 500  # Minimum data points for retraining
        self.retraining_data = []
        
        # Performance tracking
        self.signal_history = []
        self.performance_metrics = {
            'total_signals': 0,
            'correct_predictions': 0,
            'accuracy': 0.0
        }
        
        # Threading
        self.lock = threading.Lock()
        self.retraining_thread = None
        self.retraining_active = False
        
        # Load initial model
        self._load_model()
        
        # Start retraining scheduler
        self._start_retraining_scheduler()
    
    def _load_model(self):
        """Load the ML model from file."""
        if self.model_path and os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.ml_model = model_data['model']
                    self.scaler = model_data['scaler']
                    self.model_last_updated = model_data.get('timestamp', datetime.now())
                print(f"✅ Loaded ML model from {self.model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}")
                self.ml_model = None
        else:
            print("ℹ️ No pre-trained model found, will train on live data")
    
    def _save_model(self):
        """Save the current ML model to file."""
        if self.ml_model is None:
            return
        
        if self.model_path:
            try:
                model_data = {
                    'model': self.ml_model,
                    'scaler': self.scaler,
                    'timestamp': datetime.now(),
                    'config': self.config
                }
                with open(self.model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                print(f"💾 Saved updated ML model to {self.model_path}")
            except Exception as e:
                print(f"⚠️ Failed to save model: {e}")
    
    def add_data_point(self, data_point: Dict) -> Optional[Dict]:
        """
        Add a new data point and generate a signal.
        
        Args:
            data_point: Dictionary with market data (price, volume, etc.)
            
        Returns:
            Dictionary with signal information or None if insufficient data
        """
        with self.lock:
            # Apply Kalman Filter to smooth the price
            raw_price = data_point.get('price', data_point.get('close', 0))
            smoothed_price, velocity = self.kalman_filter.update(raw_price)
            
            # Create enhanced data point with smoothed price
            enhanced_data = data_point.copy()
            enhanced_data['price'] = smoothed_price
            enhanced_data['velocity'] = velocity
            enhanced_data['timestamp'] = datetime.now()
            
            # Add to buffer
            self.data_buffer.append(enhanced_data)
            
            # Maintain buffer size
            if len(self.data_buffer) > self.max_buffer_size:
                self.data_buffer.pop(0)
            
            # Store for retraining
            self.retraining_data.append(enhanced_data)
            
            # Generate signal if we have enough data
            if len(self.data_buffer) >= 50:  # Need sufficient data for features
                return self._generate_advanced_signal()
            
            return None
    
    def _generate_advanced_signal(self) -> Dict:
        """
        Generate advanced signal using full feature engineering and ML.
        
        Returns:
            Dictionary with comprehensive signal information
        """
        # Convert buffer to DataFrame
        df = pd.DataFrame(self.data_buffer)
        df.set_index('timestamp', inplace=True)
        
        # Ensure we have required columns
        if 'close' not in df.columns:
            df['close'] = df['price']
        if 'volume' not in df.columns:
            df['volume'] = np.random.randint(1000, 10000, len(df))
        
        # Generate features using the same pipeline as backtesting
        features = self.advanced_generator._create_ml_features(df)
        
        # Get market regime
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        trend_strength = self._calculate_trend_strength(df)
        market_regime = self.advanced_generator._classify_market_regime(trend_strength, volatility)
        
        # Generate ML signal if model is available
        ml_signal = 0
        ml_confidence = 0.5
        
        if self.ml_model is not None and len(features) > 0:
            try:
                # Get latest features
                latest_features = features.iloc[-1:].fillna(0)
                
                # Scale features
                scaled_features = self.scaler.transform(latest_features)
                
                # Get prediction
                ml_prediction = self.ml_model.predict(scaled_features)[0]
                ml_probabilities = self.ml_model.predict_proba(scaled_features)[0]
                ml_confidence = np.max(ml_probabilities)
                
                ml_signal = ml_prediction
            except Exception as e:
                print(f"⚠️ ML prediction error: {e}")
        
        # Generate regime-based signals
        regime_signals = self.advanced_generator.generate_market_regime_signals(df)
        momentum_signals = self._generate_momentum_signals(df)
        mean_reversion_signals = self._generate_mean_reversion_signals(df)
        
        # Combine signals using the same logic as backtesting
        combined_signal = self._combine_signals(
            ml_signal, ml_confidence,
            momentum_signals.get('momentum_signal', 0),
            mean_reversion_signals.get('mean_reversion_signal', 0),
            market_regime.iloc[-1] if len(market_regime) > 0 else 'ranging'
        )
        
        # Create comprehensive signal data
        signal_data = {
            'timestamp': datetime.now(),
            'signal': combined_signal,
            'ml_signal': ml_signal,
            'ml_confidence': ml_confidence,
            'momentum_signal': momentum_signals.get('momentum_signal', 0),
            'mean_reversion_signal': mean_reversion_signals.get('mean_reversion_signal', 0),
            'market_regime': market_regime.iloc[-1] if len(market_regime) > 0 else 'ranging',
            'price': df['close'].iloc[-1],
            'smoothed_price': df['price'].iloc[-1],
            'velocity': df.get('velocity', 0).iloc[-1],
            'volatility': volatility.iloc[-1] if len(volatility) > 0 else 0,
            'trend_strength': trend_strength.iloc[-1] if len(trend_strength) > 0 else 0
        }
        
        # Update performance tracking
        self._update_performance_tracking(signal_data)
        
        return signal_data
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength for market regime classification."""
        sma_20 = df['close'].rolling(window=20).mean()
        sma_50 = df['close'].rolling(window=50).mean()
        trend_strength = (sma_20 - sma_50) / sma_50
        return trend_strength
    
    def _generate_momentum_signals(self, df: pd.DataFrame) -> Dict:
        """Generate momentum signals."""
        lookback = self.config.momentum_lookback
        threshold = self.config.momentum_threshold
        
        momentum = df['close'] / df['close'].shift(lookback) - 1
        signals = pd.Series(0, index=df.index)
        
        signals[momentum > threshold] = 1
        signals[momentum < -threshold] = -1
        
        return {'momentum_signal': signals.iloc[-1] if len(signals) > 0 else 0}
    
    def _generate_mean_reversion_signals(self, df: pd.DataFrame) -> Dict:
        """Generate mean reversion signals."""
        lookback = self.config.mean_reversion_lookback
        std_multiplier = self.config.mean_reversion_std_multiplier
        
        sma = df['close'].rolling(window=lookback).mean()
        std = df['close'].rolling(window=lookback).std()
        
        upper_band = sma + std_multiplier * std
        lower_band = sma - std_multiplier * std
        
        signals = pd.Series(0, index=df.index)
        signals[df['close'] < lower_band] = 1
        signals[df['close'] > upper_band] = -1
        
        return {'mean_reversion_signal': signals.iloc[-1] if len(signals) > 0 else 0}
    
    def _combine_signals(self, ml_signal: int, ml_confidence: float, 
                        momentum_signal: int, mean_reversion_signal: int, 
                        market_regime: str) -> int:
        """Combine signals using regime-aware logic."""
        # Regime-aware combination (same as backtesting)
        if market_regime == 'trending':
            # Trending: momentum + ML
            combined = 0.6 * momentum_signal + 0.4 * ml_signal
        elif market_regime == 'ranging':
            # Ranging: mean reversion + ML
            combined = 0.6 * mean_reversion_signal + 0.4 * ml_signal
        else:  # volatile
            # Volatile: cautious, blend all
            combined = 0.4 * momentum_signal + 0.4 * mean_reversion_signal + 0.2 * ml_signal
        
        # Apply confidence adjustment
        threshold = self.config.final_signal_threshold / ml_confidence if ml_confidence > 0 else self.config.final_signal_threshold
        
        # Convert to discrete signal
        if combined > threshold:
            return 1
        elif combined < -threshold:
            return -1
        else:
            return 0
    
    def _update_performance_tracking(self, signal_data: Dict):
        """Update performance tracking metrics."""
        self.signal_history.append(signal_data)
        self.performance_metrics['total_signals'] += 1
        
        # Keep only recent history
        if len(self.signal_history) > 1000:
            self.signal_history.pop(0)
    
    def _start_retraining_scheduler(self):
        """Start the dynamic retraining scheduler."""
        self.retraining_active = True
        self.retraining_thread = threading.Thread(target=self._retraining_loop, daemon=True)
        self.retraining_thread.start()
        print("🔄 Dynamic retraining scheduler started")
    
    def _retraining_loop(self):
        """Main retraining loop."""
        while self.retraining_active:
            try:
                time.sleep(self.retraining_interval)
                
                if len(self.retraining_data) >= self.min_retraining_data:
                    print("🔄 Starting dynamic model retraining...")
                    self._retrain_model()
                    
            except Exception as e:
                print(f"⚠️ Retraining error: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def _retrain_model(self):
        """Retrain the ML model with new data."""
        try:
            # Convert retraining data to DataFrame
            df = pd.DataFrame(self.retraining_data)
            df.set_index('timestamp', inplace=True)
            
            # Ensure we have required columns
            if 'close' not in df.columns:
                df['close'] = df['price']
            
            # Create features and targets
            features = self.advanced_generator._create_ml_features(df)
            targets = self.advanced_generator._create_target_variable(df)
            
            # Remove NaN values
            valid_idx = ~(features.isna().any(axis=1) | targets.isna())
            features_clean = features[valid_idx]
            targets_clean = targets[valid_idx]
            
            if len(features_clean) < 100:
                print("⚠️ Insufficient data for retraining")
                return
            
            # Train new model
            new_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            # Fit scaler and model
            features_scaled = self.scaler.fit_transform(features_clean)
            new_model.fit(features_scaled, targets_clean)
            
            # Hot-swap the model
            with self.lock:
                self.ml_model = new_model
                self.model_last_updated = datetime.now()
            
            # Save updated model
            self._save_model()
            
            print(f"✅ Model retrained successfully with {len(features_clean)} samples")
            
            # Clear old retraining data (keep recent data)
            self.retraining_data = self.retraining_data[-1000:]
            
        except Exception as e:
            print(f"❌ Model retraining failed: {e}")
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def get_model_info(self) -> Dict:
        """Get information about the current ML model."""
        return {
            'model_loaded': self.ml_model is not None,
            'last_updated': self.model_last_updated,
            'retraining_active': self.retraining_active,
            'retraining_data_points': len(self.retraining_data)
        }
    
    def stop(self):
        """Stop the live signal generator."""
        self.retraining_active = False
        if self.retraining_thread:
            self.retraining_thread.join(timeout=5)
        print("🛑 Live signal generator stopped")










