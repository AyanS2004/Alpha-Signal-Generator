"""
Multi-Timeframe Analysis for Alpha Signal Engine.
Hierarchical signal aggregation across different timeframes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
import yfinance as yf
from datetime import datetime, timedelta

from .config import Config
from .signal_generator import SignalGenerator
from .advanced_signals import AdvancedSignalGenerator

logger = logging.getLogger(__name__)

class Timeframe(Enum):
    """Supported timeframes for multi-timeframe analysis."""
    MINUTE_1 = '1m'
    MINUTE_5 = '5m'
    MINUTE_15 = '15m'
    HOUR_1 = '1h'
    HOUR_4 = '4h'
    DAY_1 = '1d'
    WEEK_1 = '1wk'
    MONTH_1 = '1mo'

@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe."""
    timeframe: Timeframe
    weight: float
    lookback_periods: int
    signal_threshold: float
    enabled: bool = True

@dataclass
class MultiTimeframeSignal:
    """Multi-timeframe signal result."""
    final_signal: float
    timeframe_signals: Dict[str, float]
    signal_strength: float
    consensus_level: float
    dominant_timeframe: str
    confidence_score: float

class MultiTimeframeStrategy:
    """
    Multi-timeframe analysis strategy with hierarchical signal aggregation.
    
    This strategy analyzes multiple timeframes simultaneously and combines
    their signals using weighted aggregation and consensus mechanisms.
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize multi-timeframe strategy.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        
        # Default timeframe configurations
        self.timeframe_configs = {
            '1m': TimeframeConfig(Timeframe.MINUTE_1, 0.05, 20, 0.1),
            '5m': TimeframeConfig(Timeframe.MINUTE_5, 0.1, 50, 0.15),
            '15m': TimeframeConfig(Timeframe.MINUTE_15, 0.15, 100, 0.2),
            '1h': TimeframeConfig(Timeframe.HOUR_1, 0.2, 200, 0.25),
            '4h': TimeframeConfig(Timeframe.HOUR_4, 0.25, 500, 0.3),
            '1d': TimeframeConfig(Timeframe.DAY_1, 0.25, 1000, 0.35)
        }
        
        # Signal generators for each timeframe
        self.signal_generators = {}
        self.advanced_generators = {}
        
        # Data cache
        self.data_cache = {}
        self.cache_expiry = {}
        
    def set_timeframe_config(self, timeframe: str, config: TimeframeConfig):
        """Set configuration for a specific timeframe."""
        self.timeframe_configs[timeframe] = config
    
    def get_multi_timeframe_data(self, symbol: str, period: str = '1y') -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes.
        
        Args:
            symbol: Stock symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            Dictionary with timeframe as key and DataFrame as value
        """
        try:
            ticker = yf.Ticker(symbol)
            data_dict = {}
            
            for tf_str, tf_config in self.timeframe_configs.items():
                if not tf_config.enabled:
                    continue
                
                try:
                    # Check cache first
                    cache_key = f"{symbol}_{tf_str}_{period}"
                    if (cache_key in self.data_cache and 
                        cache_key in self.cache_expiry and 
                        datetime.now() < self.cache_expiry[cache_key]):
                        data_dict[tf_str] = self.data_cache[cache_key]
                        continue
                    
                    # Fetch data
                    if tf_str in ['1m', '5m', '15m']:
                        # Intraday data - limit to 7 days for 1m, 60 days for others
                        if tf_str == '1m':
                            data = ticker.history(period='7d', interval=tf_str)
                        else:
                            data = ticker.history(period='60d', interval=tf_str)
                    else:
                        # Daily and higher timeframes
                        data = ticker.history(period=period, interval=tf_str)
                    
                    if not data.empty:
                        # Standardize column names
                        data = data.rename(columns={
                            'Open': 'Open', 'High': 'High', 'Low': 'Low', 
                            'Close': 'Close', 'Volume': 'Volume'
                        })
                        
                        # Cache the data for 5 minutes
                        self.data_cache[cache_key] = data.copy()
                        self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=5)
                        
                        data_dict[tf_str] = data
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch {tf_str} data for {symbol}: {str(e)}")
                    continue
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error fetching multi-timeframe data for {symbol}: {str(e)}")
            return {}
    
    def generate_single_timeframe_signal(self, data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """
        Generate signal for a single timeframe.
        
        Args:
            data: Price data for the timeframe
            timeframe: Timeframe string
            
        Returns:
            Dictionary with signal information
        """
        try:
            if data.empty or len(data) < 50:
                return {
                    'signal': 0.0,
                    'strength': 0.0,
                    'confidence': 0.0,
                    'indicators': {}
                }
            
            # Get or create signal generator for this timeframe
            if timeframe not in self.signal_generators:
                self.signal_generators[timeframe] = SignalGenerator(self.config)
            
            if timeframe not in self.advanced_generators:
                self.advanced_generators[timeframe] = AdvancedSignalGenerator(self.config)
            
            # Generate basic signals
            basic_signals = self.signal_generators[timeframe].generate_signals(data)
            
            # Generate advanced signals
            advanced_signals = self.advanced_generators[timeframe].generate_advanced_signals(data)
            
            # Combine signals
            final_signal = basic_signals['final_signal'].iloc[-1] if not basic_signals.empty else 0.0
            ml_signal = advanced_signals.get('ml_signal', 0.0)
            regime_signal = advanced_signals.get('regime_signal', 0.0)
            
            # Weighted combination
            combined_signal = (
                0.4 * final_signal + 
                0.4 * ml_signal + 
                0.2 * regime_signal
            )
            
            # Calculate signal strength and confidence
            signal_strength = abs(combined_signal)
            confidence = min(signal_strength * 2, 1.0)  # Scale to 0-1
            
            # Extract key indicators
            indicators = {
                'momentum': basic_signals.get('momentum_signal', pd.Series([0])).iloc[-1] if not basic_signals.empty else 0.0,
                'mean_reversion': basic_signals.get('mean_reversion_signal', pd.Series([0])).iloc[-1] if not basic_signals.empty else 0.0,
                'ml_confidence': advanced_signals.get('ml_confidence', 0.0),
                'market_regime': advanced_signals.get('market_regime', 'unknown'),
                'rsi': advanced_signals.get('rsi', 50.0),
                'volume_ratio': advanced_signals.get('volume_ratio', 1.0)
            }
            
            return {
                'signal': combined_signal,
                'strength': signal_strength,
                'confidence': confidence,
                'indicators': indicators
            }
            
        except Exception as e:
            logger.error(f"Error generating signal for timeframe {timeframe}: {str(e)}")
            return {
                'signal': 0.0,
                'strength': 0.0,
                'confidence': 0.0,
                'indicators': {}
            }
    
    def generate_multi_timeframe_signals(self, symbol: str, period: str = '1y') -> MultiTimeframeSignal:
        """
        Generate multi-timeframe signals with hierarchical aggregation.
        
        Args:
            symbol: Stock symbol
            period: Data period
            
        Returns:
            MultiTimeframeSignal object
        """
        try:
            # Fetch multi-timeframe data
            data_dict = self.get_multi_timeframe_data(symbol, period)
            
            if not data_dict:
                return MultiTimeframeSignal(
                    final_signal=0.0,
                    timeframe_signals={},
                    signal_strength=0.0,
                    consensus_level=0.0,
                    dominant_timeframe='none',
                    confidence_score=0.0
                )
            
            # Generate signals for each timeframe
            timeframe_signals = {}
            timeframe_weights = {}
            timeframe_confidences = {}
            
            for tf_str, data in data_dict.items():
                tf_config = self.timeframe_configs.get(tf_str)
                if not tf_config or not tf_config.enabled:
                    continue
                
                signal_info = self.generate_single_timeframe_signal(data, tf_str)
                
                timeframe_signals[tf_str] = signal_info['signal']
                timeframe_weights[tf_str] = tf_config.weight
                timeframe_confidences[tf_str] = signal_info['confidence']
            
            if not timeframe_signals:
                return MultiTimeframeSignal(
                    final_signal=0.0,
                    timeframe_signals={},
                    signal_strength=0.0,
                    consensus_level=0.0,
                    dominant_timeframe='none',
                    confidence_score=0.0
                )
            
            # Hierarchical signal aggregation
            final_signal = self._aggregate_signals(timeframe_signals, timeframe_weights)
            
            # Calculate signal strength
            signal_strength = abs(final_signal)
            
            # Calculate consensus level
            consensus_level = self._calculate_consensus(timeframe_signals)
            
            # Find dominant timeframe
            dominant_timeframe = self._find_dominant_timeframe(timeframe_signals, timeframe_weights)
            
            # Calculate overall confidence
            confidence_score = self._calculate_confidence(timeframe_confidences, timeframe_weights)
            
            return MultiTimeframeSignal(
                final_signal=final_signal,
                timeframe_signals=timeframe_signals,
                signal_strength=signal_strength,
                consensus_level=consensus_level,
                dominant_timeframe=dominant_timeframe,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error generating multi-timeframe signals: {str(e)}")
            return MultiTimeframeSignal(
                final_signal=0.0,
                timeframe_signals={},
                signal_strength=0.0,
                consensus_level=0.0,
                dominant_timeframe='none',
                confidence_score=0.0
            )
    
    def _aggregate_signals(self, signals: Dict[str, float], weights: Dict[str, float]) -> float:
        """Aggregate signals using weighted average with trend alignment."""
        if not signals:
            return 0.0
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0
        
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate weighted average
        weighted_sum = sum(signals[tf] * normalized_weights[tf] for tf in signals)
        
        # Apply trend alignment bonus
        trend_alignment = self._calculate_trend_alignment(signals)
        trend_bonus = 0.1 * trend_alignment  # 10% bonus for strong trend alignment
        
        return weighted_sum * (1 + trend_bonus)
    
    def _calculate_trend_alignment(self, signals: Dict[str, float]) -> float:
        """Calculate how aligned the signals are across timeframes."""
        if len(signals) < 2:
            return 0.0
        
        signal_values = list(signals.values())
        
        # Count positive and negative signals
        positive_count = sum(1 for s in signal_values if s > 0.1)
        negative_count = sum(1 for s in signal_values if s < -0.1)
        total_signals = len(signal_values)
        
        # Calculate alignment ratio
        max_alignment = max(positive_count, negative_count)
        alignment_ratio = max_alignment / total_signals if total_signals > 0 else 0.0
        
        return alignment_ratio
    
    def _calculate_consensus(self, signals: Dict[str, float]) -> float:
        """Calculate consensus level among timeframes."""
        if len(signals) < 2:
            return 1.0
        
        signal_values = list(signals.values())
        
        # Calculate standard deviation of signals
        signal_std = np.std(signal_values)
        
        # Convert to consensus (lower std = higher consensus)
        max_std = 2.0  # Maximum expected standard deviation
        consensus = max(0.0, 1.0 - (signal_std / max_std))
        
        return consensus
    
    def _find_dominant_timeframe(self, signals: Dict[str, float], weights: Dict[str, float]) -> str:
        """Find the timeframe with the strongest weighted signal."""
        if not signals:
            return 'none'
        
        # Calculate weighted signal strength for each timeframe
        weighted_strengths = {}
        for tf, signal in signals.items():
            weight = weights.get(tf, 0.0)
            weighted_strengths[tf] = abs(signal) * weight
        
        # Find timeframe with maximum weighted strength
        dominant_tf = max(weighted_strengths, key=weighted_strengths.get)
        
        return dominant_tf
    
    def _calculate_confidence(self, confidences: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate overall confidence score."""
        if not confidences:
            return 0.0
        
        # Weighted average of confidences
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0
        
        weighted_confidence = sum(
            confidences[tf] * weights[tf] for tf in confidences
        ) / total_weight
        
        return weighted_confidence
    
    def get_timeframe_analysis(self, symbol: str, period: str = '1y') -> Dict[str, Any]:
        """
        Get detailed analysis across all timeframes.
        
        Args:
            symbol: Stock symbol
            period: Data period
            
        Returns:
            Dictionary with detailed timeframe analysis
        """
        try:
            # Get multi-timeframe data
            data_dict = self.get_multi_timeframe_data(symbol, period)
            
            analysis = {
                'symbol': symbol,
                'period': period,
                'timeframes': {},
                'summary': {}
            }
            
            # Analyze each timeframe
            for tf_str, data in data_dict.items():
                tf_config = self.timeframe_configs.get(tf_str)
                if not tf_config or not tf_config.enabled:
                    continue
                
                signal_info = self.generate_single_timeframe_signal(data, tf_str)
                
                analysis['timeframes'][tf_str] = {
                    'data_points': len(data),
                    'signal': signal_info['signal'],
                    'strength': signal_info['strength'],
                    'confidence': signal_info['confidence'],
                    'weight': tf_config.weight,
                    'indicators': signal_info['indicators'],
                    'latest_price': float(data['Close'].iloc[-1]) if not data.empty else 0.0,
                    'price_change': float((data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100) if len(data) > 1 else 0.0
                }
            
            # Generate multi-timeframe signal
            mtf_signal = self.generate_multi_timeframe_signals(symbol, period)
            
            analysis['summary'] = {
                'final_signal': mtf_signal.final_signal,
                'signal_strength': mtf_signal.signal_strength,
                'consensus_level': mtf_signal.consensus_level,
                'dominant_timeframe': mtf_signal.dominant_timeframe,
                'confidence_score': mtf_signal.confidence_score,
                'signal_direction': 'BUY' if mtf_signal.final_signal > 0.1 else 'SELL' if mtf_signal.final_signal < -0.1 else 'HOLD'
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in timeframe analysis: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timeframes': {},
                'summary': {}
            }
    
    def optimize_timeframe_weights(self, symbol: str, lookback_periods: int = 252) -> Dict[str, float]:
        """
        Optimize timeframe weights based on historical performance.
        
        Args:
            symbol: Stock symbol
            lookback_periods: Number of periods to look back
            
        Returns:
            Dictionary with optimized weights
        """
        try:
            # This is a simplified optimization - in practice, you'd want more sophisticated methods
            data_dict = self.get_multi_timeframe_data(symbol, '1y')
            
            if not data_dict:
                return {tf: config.weight for tf, config in self.timeframe_configs.items()}
            
            # Calculate performance metrics for each timeframe
            timeframe_performance = {}
            
            for tf_str, data in data_dict.items():
                if len(data) < 100:  # Need sufficient data
                    continue
                
                # Calculate simple performance metric (Sharpe-like ratio)
                returns = data['Close'].pct_change().dropna()
                if len(returns) > 0:
                    sharpe_like = returns.mean() / returns.std() if returns.std() > 0 else 0
                    timeframe_performance[tf_str] = abs(sharpe_like)
                else:
                    timeframe_performance[tf_str] = 0.0
            
            # Normalize performance to weights
            total_performance = sum(timeframe_performance.values())
            if total_performance > 0:
                optimized_weights = {
                    tf: perf / total_performance 
                    for tf, perf in timeframe_performance.items()
                }
            else:
                # Fallback to original weights
                optimized_weights = {tf: config.weight for tf, config in self.timeframe_configs.items()}
            
            return optimized_weights
            
        except Exception as e:
            logger.error(f"Error optimizing timeframe weights: {str(e)}")
            return {tf: config.weight for tf, config in self.timeframe_configs.items()}


