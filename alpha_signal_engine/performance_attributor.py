"""
Performance Attribution and Advanced Metrics for Alpha Signal Engine.
Comprehensive performance analysis with factor attribution and advanced risk metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class AttributionResult:
    """Performance attribution result."""
    total_return: float
    factor_attribution: Dict[str, float]
    alpha: float
    beta: float
    information_coefficient: float
    hit_rate: float
    factor_exposures: Dict[str, float]
    factor_returns: Dict[str, float]

@dataclass
class AdvancedMetrics:
    """Advanced performance metrics."""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    tail_ratio: float
    gain_to_pain_ratio: float
    sterling_ratio: float
    burke_ratio: float
    kappa_3: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    max_drawdown_duration: int
    recovery_factor: float
    ulcer_index: float

class PerformanceAttributor:
    """
    Performance attribution system for analyzing strategy returns.
    
    Features:
    - Factor-based attribution (Fama-French style)
    - Information coefficient calculation
    - Hit rate analysis
    - Factor exposure tracking
    - Alpha and beta calculation
    """
    
    def __init__(self):
        """Initialize performance attributor."""
        self.factors = ['momentum', 'mean_reversion', 'ml', 'regime', 'risk_management', 'market']
        self.factor_returns_history = []
        self.portfolio_returns_history = []
        
    def attribute_returns(self, 
                         portfolio_returns: pd.Series,
                         factor_exposures: Dict[str, pd.Series],
                         factor_returns: Dict[str, pd.Series],
                         benchmark_returns: pd.Series = None) -> AttributionResult:
        """
        Perform factor-based performance attribution.
        
        Args:
            portfolio_returns: Portfolio returns time series
            factor_exposures: Factor exposures over time
            factor_returns: Factor returns over time
            benchmark_returns: Benchmark returns (optional)
            
        Returns:
            AttributionResult with attribution analysis
        """
        try:
            # Align all series
            aligned_data = self._align_series(portfolio_returns, factor_exposures, factor_returns)
            portfolio_returns = aligned_data['portfolio']
            factor_exposures = aligned_data['exposures']
            factor_returns = aligned_data['factor_returns']
            
            # Calculate total return
            total_return = portfolio_returns.sum()
            
            # Calculate factor contributions
            factor_attribution = {}
            for factor in self.factors:
                if factor in factor_exposures and factor in factor_returns:
                    exposure = factor_exposures[factor]
                    factor_return = factor_returns[factor]
                    
                    # Factor contribution = exposure * factor_return
                    factor_contribution = (exposure * factor_return).sum()
                    factor_attribution[factor] = factor_contribution
            
            # Calculate alpha (unexplained returns)
            total_explained = sum(factor_attribution.values())
            alpha = total_return - total_explained
            
            # Calculate beta if benchmark provided
            beta = 0.0
            if benchmark_returns is not None:
                beta = self._calculate_beta(portfolio_returns, benchmark_returns)
            
            # Calculate information coefficient
            ic = self._calculate_information_coefficient(portfolio_returns, factor_exposures)
            
            # Calculate hit rate
            hit_rate = self._calculate_hit_rate(portfolio_returns)
            
            # Calculate average factor exposures and returns
            avg_exposures = {factor: factor_exposures[factor].mean() 
                           for factor in factor_exposures.keys()}
            avg_factor_returns = {factor: factor_returns[factor].mean() 
                                for factor in factor_returns.keys()}
            
            return AttributionResult(
                total_return=total_return,
                factor_attribution=factor_attribution,
                alpha=alpha,
                beta=beta,
                information_coefficient=ic,
                hit_rate=hit_rate,
                factor_exposures=avg_exposures,
                factor_returns=avg_factor_returns
            )
            
        except Exception as e:
            logger.error(f"Error in performance attribution: {str(e)}")
            return AttributionResult(
                total_return=0.0,
                factor_attribution={},
                alpha=0.0,
                beta=0.0,
                information_coefficient=0.0,
                hit_rate=0.0,
                factor_exposures={},
                factor_returns={}
            )
    
    def _align_series(self, 
                     portfolio_returns: pd.Series,
                     factor_exposures: Dict[str, pd.Series],
                     factor_returns: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Align all time series to common index."""
        try:
            # Find common index
            all_series = [portfolio_returns]
            all_series.extend(factor_exposures.values())
            all_series.extend(factor_returns.values())
            
            common_index = all_series[0].index
            for series in all_series[1:]:
                common_index = common_index.intersection(series.index)
            
            # Align all series
            aligned_portfolio = portfolio_returns.reindex(common_index).dropna()
            aligned_exposures = {k: v.reindex(common_index).dropna() 
                               for k, v in factor_exposures.items()}
            aligned_factor_returns = {k: v.reindex(common_index).dropna() 
                                    for k, v in factor_returns.items()}
            
            return {
                'portfolio': aligned_portfolio,
                'exposures': aligned_exposures,
                'factor_returns': aligned_factor_returns
            }
            
        except Exception as e:
            logger.error(f"Error aligning series: {str(e)}")
            return {
                'portfolio': portfolio_returns,
                'exposures': factor_exposures,
                'factor_returns': factor_returns
            }
    
    def _calculate_beta(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta relative to benchmark."""
        try:
            # Align series
            common_index = portfolio_returns.index.intersection(benchmark_returns.index)
            portfolio_aligned = portfolio_returns.reindex(common_index).dropna()
            benchmark_aligned = benchmark_returns.reindex(common_index).dropna()
            
            if len(portfolio_aligned) < 2 or len(benchmark_aligned) < 2:
                return 0.0
            
            # Calculate covariance and variance
            covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
            benchmark_variance = np.var(benchmark_aligned)
            
            if benchmark_variance == 0:
                return 0.0
            
            beta = covariance / benchmark_variance
            return beta
            
        except Exception as e:
            logger.error(f"Error calculating beta: {str(e)}")
            return 0.0
    
    def _calculate_information_coefficient(self, 
                                         portfolio_returns: pd.Series,
                                         factor_exposures: Dict[str, pd.Series]) -> float:
        """Calculate information coefficient (correlation between predictions and returns)."""
        try:
            # Use momentum factor as proxy for predictions
            if 'momentum' in factor_exposures:
                momentum_exposure = factor_exposures['momentum']
                
                # Align series
                common_index = portfolio_returns.index.intersection(momentum_exposure.index)
                returns_aligned = portfolio_returns.reindex(common_index).dropna()
                exposure_aligned = momentum_exposure.reindex(common_index).dropna()
                
                if len(returns_aligned) > 1 and len(exposure_aligned) > 1:
                    # Calculate Spearman rank correlation
                    ic = stats.spearmanr(exposure_aligned, returns_aligned)[0]
                    return ic if not np.isnan(ic) else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating information coefficient: {str(e)}")
            return 0.0
    
    def _calculate_hit_rate(self, returns: pd.Series) -> float:
        """Calculate hit rate (percentage of positive returns)."""
        try:
            if len(returns) == 0:
                return 0.0
            
            positive_returns = (returns > 0).sum()
            total_returns = len(returns)
            
            return positive_returns / total_returns if total_returns > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating hit rate: {str(e)}")
            return 0.0

class AdvancedMetricsCalculator:
    """
    Calculator for advanced performance metrics.
    
    Features:
    - Omega ratio, Tail ratio, Gain-to-pain ratio
    - Sterling ratio, Burke ratio, Kappa-3
    - VaR, CVaR, Ulcer index
    - Drawdown analysis
    """
    
    @staticmethod
    def calculate_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio."""
        try:
            if len(returns) == 0:
                return 0.0
            
            upside = returns[returns > threshold]
            downside = returns[returns <= threshold]
            
            if len(downside) == 0:
                return np.inf if len(upside) > 0 else 0.0
            
            upside_expectation = np.mean(upside - threshold) if len(upside) > 0 else 0.0
            downside_expectation = np.mean(threshold - downside)
            
            return upside_expectation / downside_expectation if downside_expectation != 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Omega ratio: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_tail_ratio(returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        try:
            if len(returns) < 20:
                return 0.0
            
            percentile_95 = np.percentile(returns, 95)
            percentile_5 = np.percentile(returns, 5)
            
            return percentile_95 / abs(percentile_5) if percentile_5 != 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating tail ratio: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_gain_to_pain_ratio(returns: pd.Series) -> float:
        """Calculate gain-to-pain ratio."""
        try:
            if len(returns) == 0:
                return 0.0
            
            gains = returns[returns > 0].sum()
            pains = abs(returns[returns < 0].sum())
            
            return gains / pains if pains != 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating gain-to-pain ratio: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_sterling_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Sterling ratio."""
        try:
            if len(returns) == 0:
                return 0.0
            
            annual_return = returns.mean() * periods_per_year
            max_dd = AdvancedMetricsCalculator.calculate_max_drawdown(returns)
            
            return annual_return / abs(max_dd) if max_dd != 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Sterling ratio: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_burke_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Burke ratio."""
        try:
            if len(returns) == 0:
                return 0.0
            
            annual_return = returns.mean() * periods_per_year
            drawdowns = AdvancedMetricsCalculator.calculate_drawdowns(returns)
            
            # Sum of squared drawdowns
            sum_squared_dd = np.sum(drawdowns**2)
            
            return annual_return / np.sqrt(sum_squared_dd) if sum_squared_dd != 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Burke ratio: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_kappa_3(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Kappa-3 ratio (third moment)."""
        try:
            if len(returns) < 3:
                return 0.0
            
            annual_return = returns.mean() * periods_per_year
            downside_returns = returns[returns < 0]
            
            if len(downside_returns) == 0:
                return np.inf if annual_return > 0 else 0.0
            
            # Third moment of downside returns
            downside_moment_3 = np.mean(downside_returns**3)
            
            return annual_return / (downside_moment_3**(1/3)) if downside_moment_3 != 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Kappa-3: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        try:
            if len(returns) == 0:
                return 0.0
            
            return np.percentile(returns, (1 - confidence) * 100)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        try:
            if len(returns) == 0:
                return 0.0
            
            var = AdvancedMetricsCalculator.calculate_var(returns, confidence)
            tail_returns = returns[returns <= var]
            
            return tail_returns.mean() if len(tail_returns) > 0 else var
            
        except Exception as e:
            logger.error(f"Error calculating CVaR: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            if len(returns) == 0:
                return 0.0
            
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            return drawdown.min()
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_drawdowns(returns: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        try:
            if len(returns) == 0:
                return pd.Series(dtype=float)
            
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            
            return drawdowns
            
        except Exception as e:
            logger.error(f"Error calculating drawdowns: {str(e)}")
            return pd.Series(dtype=float)
    
    @staticmethod
    def calculate_max_drawdown_duration(returns: pd.Series) -> int:
        """Calculate maximum drawdown duration in periods."""
        try:
            if len(returns) == 0:
                return 0
            
            drawdowns = AdvancedMetricsCalculator.calculate_drawdowns(returns)
            
            # Find periods in drawdown
            in_drawdown = drawdowns < 0
            
            # Calculate consecutive periods in drawdown
            max_duration = 0
            current_duration = 0
            
            for is_dd in in_drawdown:
                if is_dd:
                    current_duration += 1
                    max_duration = max(max_duration, current_duration)
                else:
                    current_duration = 0
            
            return max_duration
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown duration: {str(e)}")
            return 0
    
    @staticmethod
    def calculate_recovery_factor(returns: pd.Series) -> float:
        """Calculate recovery factor (total return / max drawdown)."""
        try:
            if len(returns) == 0:
                return 0.0
            
            total_return = (1 + returns).prod() - 1
            max_dd = abs(AdvancedMetricsCalculator.calculate_max_drawdown(returns))
            
            return total_return / max_dd if max_dd != 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating recovery factor: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_ulcer_index(returns: pd.Series) -> float:
        """Calculate Ulcer Index."""
        try:
            if len(returns) == 0:
                return 0.0
            
            drawdowns = AdvancedMetricsCalculator.calculate_drawdowns(returns)
            
            # Ulcer Index = sqrt(mean(squared drawdowns))
            squared_drawdowns = drawdowns**2
            ulcer_index = np.sqrt(squared_drawdowns.mean())
            
            return ulcer_index
            
        except Exception as e:
            logger.error(f"Error calculating Ulcer Index: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_all_metrics(returns: pd.Series, 
                            benchmark_returns: pd.Series = None,
                            periods_per_year: int = 252) -> AdvancedMetrics:
        """Calculate all advanced metrics."""
        try:
            # Basic metrics
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0.0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
            sortino_ratio = returns.mean() / downside_std * np.sqrt(periods_per_year) if downside_std > 0 else 0.0
            
            # Calmar ratio
            annual_return = returns.mean() * periods_per_year
            max_dd = abs(AdvancedMetricsCalculator.calculate_max_drawdown(returns))
            calmar_ratio = annual_return / max_dd if max_dd != 0 else 0.0
            
            # Advanced ratios
            omega_ratio = AdvancedMetricsCalculator.calculate_omega_ratio(returns)
            tail_ratio = AdvancedMetricsCalculator.calculate_tail_ratio(returns)
            gain_to_pain = AdvancedMetricsCalculator.calculate_gain_to_pain_ratio(returns)
            sterling_ratio = AdvancedMetricsCalculator.calculate_sterling_ratio(returns, periods_per_year)
            burke_ratio = AdvancedMetricsCalculator.calculate_burke_ratio(returns, periods_per_year)
            kappa_3 = AdvancedMetricsCalculator.calculate_kappa_3(returns, periods_per_year)
            
            # Risk metrics
            var_95 = AdvancedMetricsCalculator.calculate_var(returns, 0.95)
            cvar_95 = AdvancedMetricsCalculator.calculate_cvar(returns, 0.95)
            max_drawdown = AdvancedMetricsCalculator.calculate_max_drawdown(returns)
            max_dd_duration = AdvancedMetricsCalculator.calculate_max_drawdown_duration(returns)
            recovery_factor = AdvancedMetricsCalculator.calculate_recovery_factor(returns)
            ulcer_index = AdvancedMetricsCalculator.calculate_ulcer_index(returns)
            
            return AdvancedMetrics(
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                omega_ratio=omega_ratio,
                tail_ratio=tail_ratio,
                gain_to_pain_ratio=gain_to_pain,
                sterling_ratio=sterling_ratio,
                burke_ratio=burke_ratio,
                kappa_3=kappa_3,
                var_95=var_95,
                cvar_95=cvar_95,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_dd_duration,
                recovery_factor=recovery_factor,
                ulcer_index=ulcer_index
            )
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {str(e)}")
            return AdvancedMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis system.
    Combines attribution and advanced metrics.
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.attributor = PerformanceAttributor()
        self.metrics_calculator = AdvancedMetricsCalculator()
    
    def analyze_performance(self, 
                          portfolio_returns: pd.Series,
                          factor_exposures: Dict[str, pd.Series] = None,
                          factor_returns: Dict[str, pd.Series] = None,
                          benchmark_returns: pd.Series = None,
                          periods_per_year: int = 252) -> Dict[str, Any]:
        """
        Perform comprehensive performance analysis.
        
        Args:
            portfolio_returns: Portfolio returns
            factor_exposures: Factor exposures over time
            factor_returns: Factor returns over time
            benchmark_returns: Benchmark returns
            periods_per_year: Periods per year for annualization
            
        Returns:
            Dictionary with comprehensive analysis
        """
        try:
            # Calculate advanced metrics
            advanced_metrics = self.metrics_calculator.calculate_all_metrics(
                portfolio_returns, benchmark_returns, periods_per_year
            )
            
            # Perform attribution if factor data available
            attribution = None
            if factor_exposures and factor_returns:
                attribution = self.attributor.attribute_returns(
                    portfolio_returns, factor_exposures, factor_returns, benchmark_returns
                )
            
            # Create comprehensive analysis
            analysis = {
                'advanced_metrics': {
                    'sharpe_ratio': advanced_metrics.sharpe_ratio,
                    'sortino_ratio': advanced_metrics.sortino_ratio,
                    'calmar_ratio': advanced_metrics.calmar_ratio,
                    'omega_ratio': advanced_metrics.omega_ratio,
                    'tail_ratio': advanced_metrics.tail_ratio,
                    'gain_to_pain_ratio': advanced_metrics.gain_to_pain_ratio,
                    'sterling_ratio': advanced_metrics.sterling_ratio,
                    'burke_ratio': advanced_metrics.burke_ratio,
                    'kappa_3': advanced_metrics.kappa_3,
                    'var_95': advanced_metrics.var_95,
                    'cvar_95': advanced_metrics.cvar_95,
                    'max_drawdown': advanced_metrics.max_drawdown,
                    'max_drawdown_duration': advanced_metrics.max_drawdown_duration,
                    'recovery_factor': advanced_metrics.recovery_factor,
                    'ulcer_index': advanced_metrics.ulcer_index
                },
                'attribution': attribution.__dict__ if attribution else None,
                'summary': {
                    'total_return': portfolio_returns.sum(),
                    'annualized_return': portfolio_returns.mean() * periods_per_year,
                    'annualized_volatility': portfolio_returns.std() * np.sqrt(periods_per_year),
                    'total_trades': len(portfolio_returns),
                    'positive_trades': (portfolio_returns > 0).sum(),
                    'negative_trades': (portfolio_returns < 0).sum(),
                    'win_rate': (portfolio_returns > 0).mean() if len(portfolio_returns) > 0 else 0.0
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {str(e)}")
            return {'error': str(e)}


