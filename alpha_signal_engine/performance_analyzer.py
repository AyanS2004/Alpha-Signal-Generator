"""
Performance analyzer module for Alpha Signal Engine.
Calculates key performance metrics and risk statistics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings


class PerformanceAnalyzer:
    """Analyzes trading performance and calculates risk metrics."""
    
    def __init__(self, config):
        """Initialize PerformanceAnalyzer with configuration."""
        self.config = config
        
    def calculate_performance_metrics(self, results: Dict, df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            results: Backtest results dictionary
            df: Original DataFrame with price data
            
        Returns:
            Dictionary containing all performance metrics
        """
        equity = results['equity']
        pnl = results['pnl']
        trades = results['trades']
        drawdown = results['drawdown']
        
        # Basic return metrics
        return_metrics = self._calculate_return_metrics(equity, df)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(equity, drawdown)
        
        # Trade metrics
        trade_metrics = self._calculate_trade_metrics(pnl, trades)
        
        # Risk-adjusted metrics
        risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(equity, df)
        
        # Drawdown analysis
        drawdown_metrics = self._calculate_drawdown_metrics(drawdown)
        
        # Combine all metrics
        all_metrics = {
            **return_metrics,
            **risk_metrics,
            **trade_metrics,
            **risk_adjusted_metrics,
            **drawdown_metrics
        }
        
        return all_metrics
    
    def _calculate_return_metrics(self, equity: np.ndarray, df: pd.DataFrame) -> Dict:
        """Calculate return-based metrics."""
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        # Calculate period returns
        returns = np.diff(equity) / equity[:-1]
        returns = np.insert(returns, 0, 0)
        
        # Annualized metrics
        periods_per_year = self.config.trading_days_per_year  # Assuming daily data
        if len(df) > 1:
            actual_periods = len(df)
            annualized_return = total_return * (periods_per_year / actual_periods)
        else:
            annualized_return = 0
        
        # Calculate cumulative returns
        cumulative_returns = (equity - equity[0]) / equity[0]
        
        # Best and worst periods
        if len(returns) > 1:
            best_period = np.max(returns)
            worst_period = np.min(returns)
        else:
            best_period = worst_period = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'best_period_return': best_period,
            'worst_period_return': worst_period,
            'final_equity': equity[-1],
            'initial_equity': equity[0]
        }
    
    def _calculate_risk_metrics(self, equity: np.ndarray, drawdown: np.ndarray) -> Dict:
        """Calculate risk-based metrics."""
        # Calculate returns for volatility
        returns = np.diff(equity) / equity[:-1]
        returns = np.insert(returns, 0, 0)
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(self.config.trading_days_per_year)
        
        # Maximum drawdown
        max_drawdown = np.max(drawdown)
        
        # Value at Risk (VaR)
        if len(returns) > 1:
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
        else:
            var_95 = var_99 = 0
        
        # Expected Shortfall (Conditional VaR)
        if len(returns) > 1:
            es_95 = np.mean(returns[returns <= var_95])
            es_99 = np.mean(returns[returns <= var_99])
        else:
            es_95 = es_99 = 0
        
        return {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'expected_shortfall_99': es_99
        }
    
    def _calculate_trade_metrics(self, pnl: np.ndarray, trades: np.ndarray) -> Dict:
        """Calculate trade-based metrics."""
        # Trade statistics
        total_trades = np.sum(np.abs(trades))
        buy_trades = np.sum(trades == 1)
        sell_trades = np.sum(trades == -1)
        
        # PnL analysis
        trade_pnl = pnl[pnl != 0]
        winning_trades = np.sum(trade_pnl > 0)
        losing_trades = np.sum(trade_pnl < 0)
        
        # Win rate and average trade metrics
        win_rate = winning_trades / len(trade_pnl) if len(trade_pnl) > 0 else 0
        
        if len(trade_pnl) > 0:
            avg_trade_pnl = np.mean(trade_pnl)
            avg_winning_trade = np.mean(trade_pnl[trade_pnl > 0]) if winning_trades > 0 else 0
            avg_losing_trade = np.mean(trade_pnl[trade_pnl < 0]) if losing_trades > 0 else 0
            profit_factor = abs(avg_winning_trade * winning_trades / (avg_losing_trade * losing_trades)) if losing_trades > 0 and avg_losing_trade != 0 else float('inf')
        else:
            avg_trade_pnl = avg_winning_trade = avg_losing_trade = 0
            profit_factor = 0
        
        # Largest winning and losing trades
        if len(trade_pnl) > 0:
            largest_win = np.max(trade_pnl) if winning_trades > 0 else 0
            largest_loss = np.min(trade_pnl) if losing_trades > 0 else 0
        else:
            largest_win = largest_loss = 0
        
        return {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_trade_pnl': avg_trade_pnl,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'profit_factor': profit_factor,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }
    
    def _calculate_risk_adjusted_metrics(self, equity: np.ndarray, df: pd.DataFrame) -> Dict:
        """Calculate risk-adjusted performance metrics."""
        # Calculate returns
        returns = np.diff(equity) / equity[:-1]
        returns = np.insert(returns, 0, 0)
        
        # Sharpe ratio
        risk_free_rate = self.config.risk_free_rate
        excess_returns = returns - (risk_free_rate / self.config.trading_days_per_year)
        sharpe_ratio = (
            np.mean(excess_returns) / np.std(returns) * np.sqrt(self.config.trading_days_per_year)
            if np.std(returns) > 0
            else 0
        )
        
        # Sortino ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio (annualized return / max drawdown)
        total_return = (equity[-1] - equity[0]) / equity[0]
        periods_per_year = self.config.trading_days_per_year
        if len(df) > 1:
            annualized_return = total_return * (periods_per_year / len(df))
        else:
            annualized_return = 0
        
        max_drawdown = np.max((np.maximum.accumulate(equity) - equity) / np.maximum.accumulate(equity))
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Information ratio (assuming benchmark is risk-free rate)
        tracking_error = np.std(returns) * np.sqrt(self.config.trading_days_per_year)
        information_ratio = np.mean(excess_returns) / tracking_error if tracking_error > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio
        }
    
    def _calculate_drawdown_metrics(self, drawdown: np.ndarray) -> Dict:
        """Calculate drawdown-related metrics."""
        # Maximum drawdown
        max_drawdown = np.max(drawdown)
        
        # Average drawdown
        avg_drawdown = np.mean(drawdown)
        
        # Drawdown duration analysis
        drawdown_periods = drawdown > 0
        if np.any(drawdown_periods):
            # Find drawdown periods
            drawdown_starts = np.where(np.diff(drawdown_periods.astype(int)) == 1)[0]
            drawdown_ends = np.where(np.diff(drawdown_periods.astype(int)) == -1)[0]
            
            if len(drawdown_starts) > 0 and len(drawdown_ends) > 0:
                # Ensure we have complete drawdown periods
                if drawdown_starts[0] > drawdown_ends[0]:
                    drawdown_ends = drawdown_ends[1:]
                if len(drawdown_starts) > len(drawdown_ends):
                    drawdown_starts = drawdown_starts[:-1]
                
                if len(drawdown_starts) > 0 and len(drawdown_ends) > 0:
                    drawdown_durations = drawdown_ends - drawdown_starts
                    avg_drawdown_duration = np.mean(drawdown_durations)
                    max_drawdown_duration = np.max(drawdown_durations)
                else:
                    avg_drawdown_duration = max_drawdown_duration = 0
            else:
                avg_drawdown_duration = max_drawdown_duration = 0
        else:
            avg_drawdown_duration = max_drawdown_duration = 0
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_drawdown_duration': max_drawdown_duration
        }
    
    def generate_performance_report(self, metrics: Dict) -> str:
        """Generate a formatted performance report."""
        report = []
        report.append("=" * 60)
        report.append("ALPHA SIGNAL ENGINE - PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Return Metrics
        report.append("RETURN METRICS:")
        report.append("-" * 20)
        report.append(f"Total Return: {metrics['total_return']:.2%}")
        report.append(f"Annualized Return: {metrics['annualized_return']:.2%}")
        report.append(f"Best Period Return: {metrics['best_period_return']:.2%}")
        report.append(f"Worst Period Return: {metrics['worst_period_return']:.2%}")
        report.append("")
        
        # Risk Metrics
        report.append("RISK METRICS:")
        report.append("-" * 20)
        report.append(f"Volatility: {metrics['volatility']:.2%}")
        report.append(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        report.append(f"VaR (95%): {metrics['var_95']:.2%}")
        report.append(f"VaR (99%): {metrics['var_99']:.2%}")
        report.append("")
        
        # Risk-Adjusted Metrics
        report.append("RISK-ADJUSTED METRICS:")
        report.append("-" * 25)
        report.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        report.append(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        report.append(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")
        report.append("")
        
        # Trade Metrics
        report.append("TRADE METRICS:")
        report.append("-" * 20)
        report.append(f"Total Trades: {metrics['total_trades']}")
        report.append(f"Win Rate: {metrics['win_rate']:.2%}")
        report.append(f"Profit Factor: {metrics['profit_factor']:.3f}")
        report.append(f"Average Trade PnL: ${metrics['avg_trade_pnl']:.2f}")
        report.append("")
        
        # Drawdown Metrics
        report.append("DRAWDOWN METRICS:")
        report.append("-" * 20)
        report.append(f"Average Drawdown: {metrics['avg_drawdown']:.2%}")
        report.append(f"Average Drawdown Duration: {metrics['avg_drawdown_duration']:.1f} periods")
        report.append(f"Max Drawdown Duration: {metrics['max_drawdown_duration']:.1f} periods")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)

