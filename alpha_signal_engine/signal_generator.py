"""
Signal generator module for Alpha Signal Engine.
Implements momentum and mean-reversion trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from numba import jit
import warnings


class SignalGenerator:
    """Generates trading signals using momentum and mean-reversion strategies."""
    
    def __init__(self, config):
        """Initialize SignalGenerator with configuration."""
        self.config = config
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for the given data.
        
        Args:
            df: Preprocessed DataFrame with technical indicators
            
        Returns:
            DataFrame with signal columns added
        """
        df = df.copy()
        
        # Generate momentum signals
        df = self._generate_momentum_signals(df)
        
        # Generate mean reversion signals
        df = self._generate_mean_reversion_signals(df)
        
        # Combine signals
        df = self._combine_signals(df)
        
        return df
    
    def _generate_momentum_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum-based trading signals."""
        lookback = self.config.momentum_lookback
        threshold = self.config.momentum_threshold
        volume_threshold = self.config.momentum_volume_threshold
        
        # Price momentum signal
        df['momentum_signal'] = 0
        
        # Strong upward momentum with volume confirmation
        price_momentum = df['Close'] / df['Close'].shift(lookback) - 1
        volume_confirmation = df['volume_ratio'] > volume_threshold
        
        # Buy signal: strong positive momentum with volume
        buy_condition = (price_momentum > threshold) & volume_confirmation
        df.loc[buy_condition, 'momentum_signal'] = 1
        
        # Sell signal: strong negative momentum
        sell_condition = price_momentum < -threshold
        df.loc[sell_condition, 'momentum_signal'] = -1
        
        # EMA crossover signal
        df['ema_signal'] = 0
        ema_buy = (df['ema_12'] > df['ema_26']) & (df['ema_12'].shift(1) <= df['ema_26'].shift(1))
        ema_sell = (df['ema_12'] < df['ema_26']) & (df['ema_12'].shift(1) >= df['ema_26'].shift(1))
        
        df.loc[ema_buy, 'ema_signal'] = 1
        df.loc[ema_sell, 'ema_signal'] = -1
        
        return df
    
    def _generate_mean_reversion_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate mean-reversion trading signals."""
        lookback = self.config.mean_reversion_lookback
        std_multiplier = self.config.mean_reversion_std_multiplier
        threshold = self.config.mean_reversion_threshold
        
        df['mean_reversion_signal'] = 0
        
        # Bollinger Bands mean reversion
        bb_position = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Buy signal: price near lower band (oversold)
        bb_buy = bb_position < 0.2
        df.loc[bb_buy, 'mean_reversion_signal'] = 1
        
        # Sell signal: price near upper band (overbought)
        bb_sell = bb_position > 0.8
        df.loc[bb_sell, 'mean_reversion_signal'] = -1
        
        # RSI mean reversion
        df['rsi_signal'] = 0
        rsi_buy = df['rsi'] < 30  # Oversold
        rsi_sell = df['rsi'] > 70  # Overbought
        
        df.loc[rsi_buy, 'rsi_signal'] = 1
        df.loc[rsi_sell, 'rsi_signal'] = -1
        
        # Moving average mean reversion
        df['ma_signal'] = 0
        ma_deviation = (df['Close'] - df['sma_20']) / df['sma_20']
        
        # Buy when price is significantly below moving average
        ma_buy = ma_deviation < -threshold
        df.loc[ma_buy, 'ma_signal'] = 1
        
        # Sell when price is significantly above moving average
        ma_sell = ma_deviation > threshold
        df.loc[ma_sell, 'ma_signal'] = -1
        
        return df
    
    def _combine_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine different signals into final trading signal."""
        # Weighted combination of signals
        df['combined_signal'] = 0.0
        
        # Momentum signals (positive weight)
        df['combined_signal'] += 0.4 * df['momentum_signal']
        df['combined_signal'] += 0.3 * df['ema_signal']
        
        # Mean reversion signals (negative weight for contrarian)
        df['combined_signal'] += 0.2 * df['mean_reversion_signal']
        df['combined_signal'] += 0.1 * df['rsi_signal']
        
        # Final signal: threshold-based
        df['final_signal'] = 0
        df.loc[df['combined_signal'] > 0.5, 'final_signal'] = 1  # Buy
        df.loc[df['combined_signal'] < -0.5, 'final_signal'] = -1  # Sell
        
        return df
    
    @staticmethod
    @jit(nopython=True)
    def _optimized_signal_calculation(prices: np.ndarray, volumes: np.ndarray, 
                                    lookback: int, threshold: float) -> np.ndarray:
        """
        Numba-optimized signal calculation.
        
        Args:
            prices: Array of prices
            volumes: Array of volumes
            lookback: Lookback period
            threshold: Signal threshold
            
        Returns:
            Array of signals (1 for buy, -1 for sell, 0 for hold)
        """
        n = len(prices)
        signals = np.zeros(n)
        
        for i in range(lookback, n):
            # Calculate momentum
            momentum = (prices[i] / prices[i - lookback]) - 1
            
            # Calculate volume ratio
            avg_volume = np.mean(volumes[i - lookback:i])
            volume_ratio = volumes[i] / avg_volume if avg_volume > 0 else 1.0
            
            # Generate signal
            if momentum > threshold and volume_ratio > 1.5:
                signals[i] = 1
            elif momentum < -threshold:
                signals[i] = -1
                
        return signals
    
    def get_signal_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary of generated signals."""
        signals = df['final_signal']
        
        return {
            'total_signals': len(signals[signals != 0]),
            'buy_signals': len(signals[signals == 1]),
            'sell_signals': len(signals[signals == -1]),
            'signal_frequency': len(signals[signals != 0]) / len(signals),
            'avg_signal_strength': df['combined_signal'].abs().mean(),
            'momentum_signals': len(df[df['momentum_signal'] != 0]),
            'mean_reversion_signals': len(df[df['mean_reversion_signal'] != 0])
        }

