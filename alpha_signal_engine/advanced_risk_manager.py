"""
Advanced Risk Management for Alpha Signal Engine.
Dynamic risk controls, Kelly criterion, and sophisticated position sizing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class RiskRegime(Enum):
    """Risk regime classifications."""
    LOW_VOLATILITY = 'low_vol'
    NORMAL = 'normal'
    HIGH_VOLATILITY = 'high_vol'
    CRISIS = 'crisis'

@dataclass
class RiskMetrics:
    """Risk metrics container."""
    portfolio_var: float
    portfolio_volatility: float
    max_drawdown: float
    sharpe_ratio: float
    beta: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float

@dataclass
class PositionSizingResult:
    """Result of position sizing calculation."""
    position_size: float
    risk_adjusted_size: float
    kelly_size: float
    max_position_size: float
    risk_metrics: RiskMetrics
    sizing_factors: Dict[str, float]

class AdvancedRiskManager:
    """
    Advanced risk management system with dynamic controls and sophisticated position sizing.
    
    Features:
    - Dynamic risk controls based on market conditions
    - Kelly criterion for optimal position sizing
    - Volatility targeting
    - Correlation filtering
    - VaR-based position limits
    - Regime-aware risk adjustments
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize advanced risk manager.
        
        Args:
            config: Risk management configuration
        """
        self.config = config or self._get_default_config()
        
        # Risk parameters
        self.max_portfolio_var = self.config.get('max_portfolio_var', 0.02)  # 2% daily VaR limit
        self.max_correlation = self.config.get('max_correlation', 0.7)
        self.volatility_target = self.config.get('volatility_target', 0.15)  # 15% annual volatility
        self.max_position_size = self.config.get('max_position_size', 0.2)  # 20% max position
        self.kelly_fraction = self.config.get('kelly_fraction', 0.25)  # Quarter-Kelly
        
        # Risk regime thresholds
        self.volatility_thresholds = {
            'low': 0.10,    # 10% annual volatility
            'normal': 0.20, # 20% annual volatility
            'high': 0.35,   # 35% annual volatility
            'crisis': 0.50  # 50% annual volatility
        }
        
        # Historical performance tracking
        self.performance_history = []
        self.risk_history = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default risk management configuration."""
        return {
            'max_portfolio_var': 0.02,
            'max_correlation': 0.7,
            'volatility_target': 0.15,
            'max_position_size': 0.2,
            'kelly_fraction': 0.25,
            'var_confidence': 0.95,
            'lookback_periods': 252,
            'rebalance_frequency': 21
        }
    
    def calculate_position_size(self, 
                              signal_strength: float,
                              current_portfolio: Dict[str, float],
                              market_data: Dict[str, Any],
                              trade_history: List[Dict[str, Any]] = None) -> PositionSizingResult:
        """
        Calculate optimal position size using multiple risk management techniques.
        
        Args:
            signal_strength: Signal strength (-1 to 1)
            current_portfolio: Current portfolio positions {symbol: weight}
            market_data: Market data including volatility, correlations, etc.
            trade_history: Historical trade performance
            
        Returns:
            PositionSizingResult with sizing recommendations
        """
        try:
            # Calculate base position size from signal strength
            base_size = abs(signal_strength) * self.max_position_size
            
            # 1. Volatility targeting
            vol_adjustment = self._calculate_volatility_adjustment(market_data)
            
            # 2. Correlation filtering
            correlation_penalty = self._calculate_correlation_penalty(
                signal_strength, current_portfolio, market_data
            )
            
            # 3. VaR constraint
            var_adjusted_size = self._apply_var_constraint(
                base_size * vol_adjustment * correlation_penalty,
                current_portfolio, market_data
            )
            
            # 4. Kelly criterion (if trade history available)
            kelly_size = 0.0
            if trade_history and len(trade_history) > 10:
                kelly_size = self._calculate_kelly_criterion_size(trade_history)
            
            # 5. Risk regime adjustment
            regime_adjustment = self._get_regime_adjustment(market_data)
            
            # Final position size
            final_size = var_adjusted_size * regime_adjustment
            
            # Ensure within limits
            final_size = min(final_size, self.max_position_size)
            final_size = max(final_size, 0.0)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(current_portfolio, market_data)
            
            # Sizing factors for transparency
            sizing_factors = {
                'base_size': base_size,
                'volatility_adjustment': vol_adjustment,
                'correlation_penalty': correlation_penalty,
                'var_adjustment': var_adjusted_size / (base_size * vol_adjustment * correlation_penalty) if base_size > 0 else 1.0,
                'regime_adjustment': regime_adjustment,
                'kelly_size': kelly_size
            }
            
            return PositionSizingResult(
                position_size=final_size,
                risk_adjusted_size=var_adjusted_size,
                kelly_size=kelly_size,
                max_position_size=self.max_position_size,
                risk_metrics=risk_metrics,
                sizing_factors=sizing_factors
            )
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return PositionSizingResult(
                position_size=0.0,
                risk_adjusted_size=0.0,
                kelly_size=0.0,
                max_position_size=self.max_position_size,
                risk_metrics=RiskMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                sizing_factors={}
            )
    
    def _calculate_volatility_adjustment(self, market_data: Dict[str, Any]) -> float:
        """Calculate volatility-based position size adjustment."""
        try:
            current_vol = market_data.get('volatility', 0.20)  # Default 20% annual
            
            # Volatility targeting: adjust position size inversely to volatility
            vol_adjustment = self.volatility_target / current_vol if current_vol > 0 else 1.0
            
            # Cap the adjustment to prevent extreme sizes
            vol_adjustment = np.clip(vol_adjustment, 0.5, 2.0)
            
            return vol_adjustment
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment: {str(e)}")
            return 1.0
    
    def _calculate_correlation_penalty(self, 
                                     signal_strength: float,
                                     current_portfolio: Dict[str, float],
                                     market_data: Dict[str, Any]) -> float:
        """Calculate correlation-based position size penalty."""
        try:
            if not current_portfolio:
                return 1.0
            
            # Get correlation matrix
            correlation_matrix = market_data.get('correlation_matrix', None)
            if correlation_matrix is None:
                return 1.0
            
            # Calculate average correlation with existing positions
            correlations = []
            for symbol in current_portfolio.keys():
                if symbol in correlation_matrix.columns:
                    # Get correlation with other positions
                    symbol_correlations = correlation_matrix[symbol].drop(symbol)
                    if not symbol_correlations.empty:
                        correlations.extend(symbol_correlations.values)
            
            if not correlations:
                return 1.0
            
            avg_correlation = np.mean(np.abs(correlations))
            
            # Apply penalty for high correlation
            if avg_correlation > self.max_correlation:
                penalty = 1.0 - (avg_correlation - self.max_correlation) / (1.0 - self.max_correlation)
                penalty = max(penalty, 0.1)  # Minimum 10% of original size
            else:
                penalty = 1.0
            
            return penalty
            
        except Exception as e:
            logger.error(f"Error calculating correlation penalty: {str(e)}")
            return 1.0
    
    def _apply_var_constraint(self, 
                            proposed_size: float,
                            current_portfolio: Dict[str, float],
                            market_data: Dict[str, Any]) -> float:
        """Apply Value at Risk constraint to position size."""
        try:
            # Calculate current portfolio VaR
            current_var = self._calculate_portfolio_var(current_portfolio, market_data)
            
            # Calculate proposed portfolio VaR with new position
            proposed_portfolio = current_portfolio.copy()
            # Add the new position (simplified - in practice you'd need the symbol)
            proposed_portfolio['new_position'] = proposed_size
            
            proposed_var = self._calculate_portfolio_var(proposed_portfolio, market_data)
            
            # If proposed VaR exceeds limit, scale down the position
            if proposed_var > self.max_portfolio_var:
                scaling_factor = self.max_portfolio_var / proposed_var
                adjusted_size = proposed_size * scaling_factor
            else:
                adjusted_size = proposed_size
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error applying VaR constraint: {str(e)}")
            return proposed_size
    
    def _calculate_portfolio_var(self, 
                               portfolio: Dict[str, float],
                               market_data: Dict[str, Any],
                               confidence: float = 0.95) -> float:
        """Calculate portfolio Value at Risk."""
        try:
            if not portfolio:
                return 0.0
            
            # Get portfolio weights
            weights = np.array(list(portfolio.values()))
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
            
            # Get covariance matrix
            cov_matrix = market_data.get('covariance_matrix', None)
            if cov_matrix is None:
                # Use simple volatility assumption
                portfolio_vol = np.sqrt(np.sum(weights**2 * 0.04))  # Assume 20% vol for each asset
                return portfolio_vol * stats.norm.ppf(confidence)
            
            # Calculate portfolio variance
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_variance)
            
            # Calculate VaR
            var = portfolio_vol * stats.norm.ppf(confidence)
            
            return var
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {str(e)}")
            return 0.0
    
    def _calculate_kelly_criterion_size(self, trade_history: List[Dict[str, Any]]) -> float:
        """Calculate Kelly criterion optimal position size."""
        try:
            if len(trade_history) < 10:
                return 0.0
            
            # Extract win/loss information
            returns = [trade.get('return', 0.0) for trade in trade_history if 'return' in trade]
            
            if not returns:
                return 0.0
            
            # Calculate win rate and average win/loss
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r < 0]
            
            if not wins or not losses:
                return 0.0
            
            win_rate = len(wins) / len(returns)
            avg_win = np.mean(wins)
            avg_loss = abs(np.mean(losses))
            
            if avg_loss == 0:
                return 0.0
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_f = (b * p - q) / b
            
            # Apply Kelly fraction and cap
            kelly_size = max(0, min(kelly_f * self.kelly_fraction, self.max_position_size))
            
            return kelly_size
            
        except Exception as e:
            logger.error(f"Error calculating Kelly criterion: {str(e)}")
            return 0.0
    
    def _get_regime_adjustment(self, market_data: Dict[str, Any]) -> float:
        """Get risk regime-based position size adjustment."""
        try:
            current_vol = market_data.get('volatility', 0.20)
            regime = self._classify_risk_regime(current_vol)
            
            # Regime-based adjustments
            regime_adjustments = {
                RiskRegime.LOW_VOLATILITY: 1.2,    # Increase size in low vol
                RiskRegime.NORMAL: 1.0,            # Normal size
                RiskRegime.HIGH_VOLATILITY: 0.7,   # Reduce size in high vol
                RiskRegime.CRISIS: 0.3             # Drastically reduce in crisis
            }
            
            return regime_adjustments.get(regime, 1.0)
            
        except Exception as e:
            logger.error(f"Error getting regime adjustment: {str(e)}")
            return 1.0
    
    def _classify_risk_regime(self, volatility: float) -> RiskRegime:
        """Classify current risk regime based on volatility."""
        if volatility <= self.volatility_thresholds['low']:
            return RiskRegime.LOW_VOLATILITY
        elif volatility <= self.volatility_thresholds['normal']:
            return RiskRegime.NORMAL
        elif volatility <= self.volatility_thresholds['high']:
            return RiskRegime.HIGH_VOLATILITY
        else:
            return RiskRegime.CRISIS
    
    def _calculate_risk_metrics(self, 
                              portfolio: Dict[str, float],
                              market_data: Dict[str, Any]) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        try:
            # Portfolio VaR
            portfolio_var = self._calculate_portfolio_var(portfolio, market_data)
            
            # Portfolio volatility
            portfolio_vol = self._estimate_portfolio_volatility(portfolio, market_data)
            
            # Other metrics (simplified calculations)
            max_drawdown = market_data.get('max_drawdown', 0.0)
            sharpe_ratio = market_data.get('sharpe_ratio', 0.0)
            beta = market_data.get('beta', 1.0)
            
            # Correlation risk
            correlation_risk = self._calculate_correlation_risk(portfolio, market_data)
            
            # Concentration risk
            concentration_risk = self._calculate_concentration_risk(portfolio)
            
            # Liquidity risk (simplified)
            liquidity_risk = market_data.get('liquidity_risk', 0.0)
            
            return RiskMetrics(
                portfolio_var=portfolio_var,
                portfolio_volatility=portfolio_vol,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                beta=beta,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return RiskMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def _estimate_portfolio_volatility(self, 
                                     portfolio: Dict[str, float],
                                     market_data: Dict[str, Any]) -> float:
        """Estimate portfolio volatility."""
        try:
            if not portfolio:
                return 0.0
            
            weights = np.array(list(portfolio.values()))
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
            
            # Get covariance matrix
            cov_matrix = market_data.get('covariance_matrix', None)
            if cov_matrix is None:
                # Simple estimate using individual volatilities
                individual_vols = market_data.get('individual_volatilities', [0.20] * len(weights))
                return np.sqrt(np.sum((weights * individual_vols)**2))
            
            # Calculate portfolio variance
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            return np.sqrt(portfolio_variance)
            
        except Exception as e:
            logger.error(f"Error estimating portfolio volatility: {str(e)}")
            return 0.0
    
    def _calculate_correlation_risk(self, 
                                  portfolio: Dict[str, float],
                                  market_data: Dict[str, Any]) -> float:
        """Calculate correlation risk in portfolio."""
        try:
            correlation_matrix = market_data.get('correlation_matrix', None)
            if correlation_matrix is None or len(portfolio) < 2:
                return 0.0
            
            # Calculate average correlation
            correlations = []
            symbols = list(portfolio.keys())
            
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    if symbol1 in correlation_matrix.columns and symbol2 in correlation_matrix.index:
                        corr = correlation_matrix.loc[symbol2, symbol1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {str(e)}")
            return 0.0
    
    def _calculate_concentration_risk(self, portfolio: Dict[str, float]) -> float:
        """Calculate concentration risk using Herfindahl index."""
        try:
            if not portfolio:
                return 0.0
            
            weights = np.array(list(portfolio.values()))
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
            
            # Herfindahl index (sum of squared weights)
            herfindahl = np.sum(weights**2)
            
            # Convert to concentration risk (0 = perfectly diversified, 1 = fully concentrated)
            n_assets = len(weights)
            max_herfindahl = 1.0  # Single asset
            min_herfindahl = 1.0 / n_assets  # Equal weights
            
            concentration_risk = (herfindahl - min_herfindahl) / (max_herfindahl - min_herfindahl)
            
            return concentration_risk
            
        except Exception as e:
            logger.error(f"Error calculating concentration risk: {str(e)}")
            return 0.0
    
    def update_performance_history(self, trade_result: Dict[str, Any]):
        """Update performance history for Kelly criterion calculation."""
        self.performance_history.append(trade_result)
        
        # Keep only recent history
        max_history = 1000
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
    
    def get_risk_report(self, portfolio: Dict[str, float], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        try:
            risk_metrics = self._calculate_risk_metrics(portfolio, market_data)
            regime = self._classify_risk_regime(market_data.get('volatility', 0.20))
            
            return {
                'risk_metrics': {
                    'portfolio_var': risk_metrics.portfolio_var,
                    'portfolio_volatility': risk_metrics.portfolio_volatility,
                    'max_drawdown': risk_metrics.max_drawdown,
                    'sharpe_ratio': risk_metrics.sharpe_ratio,
                    'beta': risk_metrics.beta,
                    'correlation_risk': risk_metrics.correlation_risk,
                    'concentration_risk': risk_metrics.concentration_risk,
                    'liquidity_risk': risk_metrics.liquidity_risk
                },
                'risk_regime': regime.value,
                'risk_limits': {
                    'max_portfolio_var': self.max_portfolio_var,
                    'max_correlation': self.max_correlation,
                    'volatility_target': self.volatility_target,
                    'max_position_size': self.max_position_size
                },
                'risk_alerts': self._generate_risk_alerts(risk_metrics, regime),
                'recommendations': self._generate_risk_recommendations(risk_metrics, regime)
            }
            
        except Exception as e:
            logger.error(f"Error generating risk report: {str(e)}")
            return {'error': str(e)}
    
    def _generate_risk_alerts(self, risk_metrics: RiskMetrics, regime: RiskRegime) -> List[str]:
        """Generate risk alerts based on current metrics."""
        alerts = []
        
        if risk_metrics.portfolio_var > self.max_portfolio_var:
            alerts.append(f"Portfolio VaR ({risk_metrics.portfolio_var:.2%}) exceeds limit ({self.max_portfolio_var:.2%})")
        
        if risk_metrics.correlation_risk > self.max_correlation:
            alerts.append(f"High correlation risk ({risk_metrics.correlation_risk:.2%})")
        
        if risk_metrics.concentration_risk > 0.5:
            alerts.append(f"High concentration risk ({risk_metrics.concentration_risk:.2%})")
        
        if regime == RiskRegime.CRISIS:
            alerts.append("Crisis regime detected - consider reducing risk exposure")
        
        return alerts
    
    def _generate_risk_recommendations(self, risk_metrics: RiskMetrics, regime: RiskRegime) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        if risk_metrics.portfolio_volatility > self.volatility_target * 1.2:
            recommendations.append("Consider reducing position sizes to target volatility")
        
        if risk_metrics.correlation_risk > 0.6:
            recommendations.append("Diversify portfolio to reduce correlation risk")
        
        if risk_metrics.concentration_risk > 0.4:
            recommendations.append("Reduce concentration in largest positions")
        
        if regime in [RiskRegime.HIGH_VOLATILITY, RiskRegime.CRISIS]:
            recommendations.append("Increase cash allocation during high volatility periods")
        
        return recommendations


