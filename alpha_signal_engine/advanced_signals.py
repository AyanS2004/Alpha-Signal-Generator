"""
Advanced Signal Generator for Alpha Signal Engine.
Provides sophisticated trading signals using machine learning, volatility analysis, and ensemble methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class AdvancedSignalGenerator:
    """
    Advanced signal generator with sophisticated trading strategies.
    
    Features:
    - Machine learning-based signals
    - Volatility regime detection
    - Ensemble signal combination
    - Market regime classification
    - Adaptive parameter optimization
    """
    
    def __init__(self, config):
        """
        Initialize Advanced Signal Generator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.scaler = StandardScaler()
        self.ml_model = None
        self.volatility_regime = None
        self.market_regime = None
        
    def generate_ml_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate machine learning-based trading signals.
        
        Args:
            data: DataFrame with price and technical indicators
            
        Returns:
            DataFrame with ML signals
        """
        # Create features for ML model
        features = self._create_ml_features(data)
        
        # Create target variable (next period return)
        target = self._create_target_variable(data)
        
        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        features_clean = features[valid_idx]
        target_clean = target[valid_idx]
        
        if len(features_clean) < 100:  # Need sufficient data
            return pd.DataFrame({'ml_signal': 0}, index=data.index)
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(
            features_clean, target_clean, test_size=0.3, random_state=42
        )
        
        # Train model
        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.ml_model.fit(X_train, y_train)
        
        # Generate predictions
        predictions = self.ml_model.predict(features)
        probabilities = self.ml_model.predict_proba(features)
        
        # Convert to signals
        ml_signals = pd.Series(0, index=data.index)
        ml_signals[valid_idx] = predictions
        
        # Add confidence scores
        confidence = pd.Series(0.0, index=data.index)
        confidence[valid_idx] = np.max(probabilities, axis=1)
        
        return pd.DataFrame({
            'ml_signal': ml_signals,
            'ml_confidence': confidence
        })
    
    def generate_volatility_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate volatility-based trading signals.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with volatility signals
        """
        # Calculate volatility measures
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=20).std()
        realized_vol = returns.rolling(window=5).std()
        
        # Volatility regime detection
        vol_regime = self._detect_volatility_regime(volatility)
        
        # Volatility-based signals
        vol_signals = pd.Series(0, index=data.index)
        
        # High volatility: mean reversion
        high_vol_mask = vol_regime == 'high'
        if high_vol_mask.any():
            vol_signals[high_vol_mask] = self._generate_mean_reversion_signals(
                data[high_vol_mask], lookback=10
            )
        
        # Low volatility: momentum
        low_vol_mask = vol_regime == 'low'
        if low_vol_mask.any():
            vol_signals[low_vol_mask] = self._generate_momentum_signals(
                data[low_vol_mask], lookback=20
            )
        
        return pd.DataFrame({
            'volatility_signal': vol_signals,
            'volatility_regime': vol_regime,
            'realized_volatility': realized_vol
        })
    
    def generate_ensemble_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ensemble signals by combining multiple signal sources.
        
        Args:
            signals: DataFrame with multiple signal columns
            
        Returns:
            DataFrame with ensemble signals
        """
        # Get all signal columns
        signal_columns = [col for col in signals.columns if 'signal' in col.lower()]
        
        if len(signal_columns) < 2:
            return pd.DataFrame({'ensemble_signal': signals.get(signal_columns[0], 0)})
        
        # Weighted ensemble
        weights = self._calculate_signal_weights(signals[signal_columns])
        
        # Calculate weighted signal
        weighted_signals = pd.DataFrame()
        for i, col in enumerate(signal_columns):
            weighted_signals[col] = signals[col] * weights[i]
        
        ensemble_signal = weighted_signals.sum(axis=1)
        
        # Normalize to [-1, 1] range
        ensemble_signal = np.clip(ensemble_signal, -1, 1)
        
        # Convert to discrete signals
        final_signal = pd.Series(0, index=signals.index)
        final_signal[ensemble_signal > 0.3] = 1
        final_signal[ensemble_signal < -0.3] = -1
        
        return pd.DataFrame({
            'ensemble_signal': final_signal,
            'ensemble_strength': ensemble_signal,
            'signal_agreement': self._calculate_signal_agreement(signals[signal_columns])
        })
    
    def generate_market_regime_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on market regime classification.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with regime-based signals
        """
        # Calculate market regime indicators
        returns = data['Close'].pct_change()
        
        # Trend strength
        sma_20 = data['Close'].rolling(window=20).mean()
        sma_50 = data['Close'].rolling(window=50).mean()
        trend_strength = (sma_20 - sma_50) / sma_50
        
        # Volatility regime
        volatility = returns.rolling(window=20).std()
        vol_regime = self._detect_volatility_regime(volatility)
        
        # Market regime classification
        market_regime = self._classify_market_regime(trend_strength, volatility)
        
        # Generate regime-specific signals
        regime_signals = pd.Series(0, index=data.index)
        
        for regime in ['trending', 'ranging', 'volatile']:
            regime_mask = market_regime == regime
            if regime_mask.any():
                regime_signals[regime_mask] = self._generate_regime_signals(
                    data[regime_mask], regime
                )
        
        return pd.DataFrame({
            'regime_signal': regime_signals,
            'market_regime': market_regime,
            'trend_strength': trend_strength,
            'volatility_regime': vol_regime
        })
    
    def _create_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for machine learning model."""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        features['price_momentum'] = data['Close'] / data['Close'].shift(5) - 1
        features['price_acceleration'] = features['price_momentum'].diff()
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(data['Close'])
        features['macd'] = self._calculate_macd(data['Close'])
        features['bb_position'] = self._calculate_bb_position(data['Close'])
        
        # Volatility features
        returns = data['Close'].pct_change()
        features['volatility'] = returns.rolling(window=20).std()
        features['volatility_change'] = features['volatility'].pct_change()
        
        # Volume features (if available)
        if 'Volume' in data.columns:
            features['volume_ratio'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
            features['volume_trend'] = data['Volume'].pct_change()
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'volatility_lag_{lag}'] = features['volatility'].shift(lag)
        
        return features
    
    def _create_target_variable(self, data: pd.DataFrame) -> pd.Series:
        """Create target variable for ML model."""
        future_returns = data['Close'].pct_change().shift(-1)
        
        # Create binary classification target
        target = pd.Series(0, index=data.index)
        target[future_returns > future_returns.rolling(window=20).mean()] = 1
        
        return target
    
    def _detect_volatility_regime(self, volatility: pd.Series) -> pd.Series:
        """Detect volatility regime (low, medium, high)."""
        vol_quantiles = volatility.rolling(window=60).quantile([0.33, 0.67])
        
        regime = pd.Series('medium', index=volatility.index)
        regime[volatility < vol_quantiles.iloc[:, 0]] = 'low'
        regime[volatility > vol_quantiles.iloc[:, 1]] = 'high'
        
        return regime
    
    def _classify_market_regime(self, trend_strength: pd.Series, volatility: pd.Series) -> pd.Series:
        """Classify market regime."""
        regime = pd.Series('ranging', index=trend_strength.index)
        
        # Trending regime
        strong_trend = abs(trend_strength) > trend_strength.rolling(window=60).std()
        regime[strong_trend] = 'trending'
        
        # Volatile regime
        high_vol = volatility > volatility.rolling(window=60).quantile(0.8)
        regime[high_vol] = 'volatile'
        
        return regime
    
    def _generate_mean_reversion_signals(self, data: pd.DataFrame, lookback: int = 10) -> pd.Series:
        """Generate mean reversion signals."""
        sma = data['Close'].rolling(window=lookback).mean()
        std = data['Close'].rolling(window=lookback).std()
        
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
        signals = pd.Series(0, index=data.index)
        signals[data['Close'] < lower_band] = 1  # Buy signal
        signals[data['Close'] > upper_band] = -1  # Sell signal
        
        return signals
    
    def _generate_momentum_signals(self, data: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Generate momentum signals."""
        momentum = data['Close'] / data['Close'].shift(lookback) - 1
        
        signals = pd.Series(0, index=data.index)
        signals[momentum > 0.02] = 1  # Buy signal
        signals[momentum < -0.02] = -1  # Sell signal
        
        return signals
    
    def _generate_regime_signals(self, data: pd.DataFrame, regime: str) -> pd.Series:
        """Generate signals based on market regime."""
        if regime == 'trending':
            return self._generate_momentum_signals(data, lookback=20)
        elif regime == 'ranging':
            return self._generate_mean_reversion_signals(data, lookback=10)
        elif regime == 'volatile':
            return self._generate_mean_reversion_signals(data, lookback=5)
        else:
            return pd.Series(0, index=data.index)
    
    def _calculate_signal_weights(self, signals: pd.DataFrame) -> List[float]:
        """Calculate weights for ensemble signals."""
        # Simple equal weights for now
        # Could be enhanced with signal performance history
        n_signals = len(signals.columns)
        return [1.0 / n_signals] * n_signals
    
    def _calculate_signal_agreement(self, signals: pd.DataFrame) -> pd.Series:
        """Calculate agreement between different signals."""
        # Count positive and negative signals
        positive_signals = (signals > 0).sum(axis=1)
        negative_signals = (signals < 0).sum(axis=1)
        total_signals = len(signals.columns)
        
        # Calculate agreement ratio
        agreement = np.maximum(positive_signals, negative_signals) / total_signals
        return agreement
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Bollinger Bands position."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        return bb_position
