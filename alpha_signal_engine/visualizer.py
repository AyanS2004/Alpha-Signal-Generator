"""
Visualization module for Alpha Signal Engine.
Provides comprehensive plotting capabilities for trading signals, performance metrics, and analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple
import warnings

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class Visualizer:
    """
    Visualization class for Alpha Signal Engine.
    
    Provides methods for:
    - Performance plotting (equity curve, drawdown, PnL)
    - Signal analysis visualization
    - Risk metrics visualization
    - Trading activity visualization
    """
    
    def __init__(self, config):
        """
        Initialize Visualizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Set figure size and DPI for better quality
        self.figsize = (12, 8)
        self.dpi = 100
        
        # Color scheme
        self.colors = {
            'buy': '#2E8B57',      # Sea green
            'sell': '#DC143C',      # Crimson
            'hold': '#808080',      # Gray
            'equity': '#1f77b4',    # Blue
            'drawdown': '#ff7f0e',  # Orange
            'pnl': '#2ca02c',       # Green
            'volume': '#d62728',    # Red
            'price': '#9467bd'      # Purple
        }
    
    def plot_performance(self, signals: pd.DataFrame, backtest_results: Dict,
                        save_path: Optional[str] = None, show_plot: bool = True) -> None:
        """
        Create comprehensive performance visualization.
        
        Args:
            signals: DataFrame with trading signals
            backtest_results: Dictionary with backtest results
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Alpha Signal Engine - Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Equity Curve
        axes[0, 0].plot(backtest_results['equity'], color=self.colors['equity'], linewidth=2)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        axes[0, 1].fill_between(range(len(backtest_results['drawdown'])), 
                               backtest_results['drawdown'], 
                               color=self.colors['drawdown'], alpha=0.7)
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. PnL Distribution
        pnl = backtest_results['pnl']
        axes[1, 0].hist(pnl, bins=50, color=self.colors['pnl'], alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('PnL Distribution')
        axes[1, 0].set_xlabel('PnL ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Cumulative PnL
        cumulative_pnl = np.cumsum(pnl)
        axes[1, 1].plot(cumulative_pnl, color=self.colors['pnl'], linewidth=2)
        axes[1, 1].set_title('Cumulative PnL')
        axes[1, 1].set_ylabel('Cumulative PnL ($)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Price and Signals
        price_data = signals['Close']
        buy_signals = signals[signals['final_signal'] == 1]
        sell_signals = signals[signals['final_signal'] == -1]
        
        axes[2, 0].plot(price_data, color=self.colors['price'], linewidth=1, alpha=0.8)
        axes[2, 0].scatter(buy_signals.index, buy_signals['Close'], 
                          color=self.colors['buy'], s=50, marker='^', label='Buy')
        axes[2, 0].scatter(sell_signals.index, sell_signals['Close'], 
                          color=self.colors['sell'], s=50, marker='v', label='Sell')
        axes[2, 0].set_title('Price and Trading Signals')
        axes[2, 0].set_ylabel('Price ($)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Monthly Returns Heatmap
        if len(pnl) > 30:  # Only if we have enough data
            monthly_returns = self._calculate_monthly_returns(signals, backtest_results)
            if monthly_returns is not None:
                sns.heatmap(monthly_returns, annot=True, fmt='.2%', cmap='RdYlGn', 
                           center=0, ax=axes[2, 1])
                axes[2, 1].set_title('Monthly Returns Heatmap')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Performance plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_signal_analysis(self, signals: pd.DataFrame, save_path: Optional[str] = None,
                           show_plot: bool = True) -> None:
        """
        Create signal analysis visualization.
        
        Args:
            signals: DataFrame with trading signals
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Alpha Signal Engine - Signal Analysis', fontsize=16, fontweight='bold')
        
        # 1. Signal Distribution
        signal_counts = signals['final_signal'].value_counts()
        colors = [self.colors['sell'], self.colors['hold'], self.colors['buy']]
        axes[0, 0].pie(signal_counts.values, labels=['Sell', 'Hold', 'Buy'], 
                      colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Signal Distribution')
        
        # 2. Signal Frequency Over Time
        signal_frequency = signals['final_signal'].rolling(window=50).apply(
            lambda x: (x != 0).sum() / len(x)
        )
        axes[0, 1].plot(signal_frequency, color=self.colors['equity'], linewidth=2)
        axes[0, 1].set_title('Signal Frequency (50-period rolling)')
        axes[0, 1].set_ylabel('Signal Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Individual Signal Components
        signal_components = ['momentum_signal', 'mean_reversion_signal', 'ema_signal', 'rsi_signal']
        signal_data = signals[signal_components].sum()
        axes[0, 2].bar(signal_data.index, signal_data.values, 
                       color=[self.colors['buy'], self.colors['sell'], 
                              self.colors['hold'], self.colors['price']])
        axes[0, 2].set_title('Signal Component Analysis')
        axes[0, 2].set_ylabel('Net Signal Strength')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Signal Correlation Matrix
        signal_corr = signals[signal_components].corr()
        sns.heatmap(signal_corr, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[1, 0], cbar_kws={'shrink': 0.8})
        axes[1, 0].set_title('Signal Correlation Matrix')
        
        # 5. Signal Strength Distribution
        signal_strength = signals['final_signal'].abs()
        axes[1, 1].hist(signal_strength, bins=20, color=self.colors['equity'], 
                       alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Signal Strength Distribution')
        axes[1, 1].set_xlabel('Signal Strength')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Signal Timing Analysis
        if len(signals) > 100:
            # Calculate signal timing metrics
            signal_timing = self._analyze_signal_timing(signals)
            timing_metrics = ['Avg Hold Time', 'Win Rate', 'Avg Win', 'Avg Loss']
            timing_values = [signal_timing.get(metric, 0) for metric in timing_metrics]
            
            bars = axes[1, 2].bar(timing_metrics, timing_values, 
                                 color=[self.colors['buy'], self.colors['sell'], 
                                        self.colors['hold'], self.colors['price']])
            axes[1, 2].set_title('Signal Timing Metrics')
            axes[1, 2].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, timing_values):
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Signal analysis plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_risk_metrics(self, performance_metrics: Dict, save_path: Optional[str] = None,
                         show_plot: bool = True) -> None:
        """
        Create risk metrics visualization.
        
        Args:
            performance_metrics: Dictionary with performance metrics
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Alpha Signal Engine - Risk Analysis', fontsize=16, fontweight='bold')
        
        # 1. Risk-Return Scatter
        risk_metrics = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'information_ratio']
        risk_values = [performance_metrics.get(metric, 0) for metric in risk_metrics]
        
        axes[0, 0].bar(risk_metrics, risk_values, 
                       color=[self.colors['buy'], self.colors['sell'], 
                              self.colors['hold'], self.colors['price']])
        axes[0, 0].set_title('Risk-Adjusted Return Metrics')
        axes[0, 0].set_ylabel('Ratio')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drawdown Analysis
        if 'drawdown_periods' in performance_metrics:
            drawdown_data = performance_metrics['drawdown_periods']
            axes[0, 1].hist(drawdown_data, bins=20, color=self.colors['drawdown'], 
                           alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Drawdown Duration Distribution')
            axes[0, 1].set_xlabel('Drawdown Duration (periods)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Win/Loss Analysis
        win_rate = performance_metrics.get('win_rate', 0)
        loss_rate = 1 - win_rate
        
        axes[1, 0].pie([win_rate, loss_rate], labels=['Wins', 'Losses'], 
                      colors=[self.colors['buy'], self.colors['sell']], 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Win/Loss Distribution')
        
        # 4. Profit Factor and Recovery Factor
        profit_factor = performance_metrics.get('profit_factor', 0)
        recovery_factor = performance_metrics.get('recovery_factor', 0)
        
        factors = ['Profit Factor', 'Recovery Factor']
        factor_values = [profit_factor, recovery_factor]
        
        bars = axes[1, 1].bar(factors, factor_values, 
                             color=[self.colors['buy'], self.colors['sell']])
        axes[1, 1].set_title('Risk Management Metrics')
        axes[1, 1].set_ylabel('Factor')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, factor_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Risk metrics plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_trading_activity(self, signals: pd.DataFrame, backtest_results: Dict,
                             save_path: Optional[str] = None, show_plot: bool = True) -> None:
        """
        Create trading activity visualization.
        
        Args:
            signals: DataFrame with trading signals
            backtest_results: Dictionary with backtest results
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Alpha Signal Engine - Trading Activity Analysis', fontsize=16, fontweight='bold')
        
        # 1. Trading Volume Analysis
        if 'Volume' in signals.columns:
            volume_data = signals['Volume']
            axes[0, 0].plot(volume_data, color=self.colors['volume'], alpha=0.7)
            axes[0, 0].set_title('Trading Volume')
            axes[0, 0].set_ylabel('Volume')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Position Size Over Time
        if 'position_size' in backtest_results:
            position_sizes = backtest_results['position_size']
            axes[0, 1].plot(position_sizes, color=self.colors['equity'], linewidth=2)
            axes[0, 1].set_title('Position Size Over Time')
            axes[0, 1].set_ylabel('Position Size')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Trade Duration Analysis
        trade_durations = self._calculate_trade_durations(signals)
        if trade_durations:
            axes[1, 0].hist(trade_durations, bins=20, color=self.colors['buy'], 
                           alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Trade Duration Distribution')
            axes[1, 0].set_xlabel('Duration (periods)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Monthly Trading Activity
        if len(signals) > 30:
            monthly_activity = self._calculate_monthly_activity(signals)
            if monthly_activity is not None:
                months = list(monthly_activity.keys())
                trades = list(monthly_activity.values())
                
                bars = axes[1, 1].bar(months, trades, color=self.colors['equity'])
                axes[1, 1].set_title('Monthly Trading Activity')
                axes[1, 1].set_ylabel('Number of Trades')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Trading activity plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def _calculate_monthly_returns(self, signals: pd.DataFrame, backtest_results: Dict) -> Optional[pd.DataFrame]:
        """Calculate monthly returns for heatmap."""
        try:
            # Group by month and calculate returns
            signals_copy = signals.copy()
            signals_copy['month'] = signals_copy.index.to_period('M')
            signals_copy['returns'] = backtest_results['pnl']
            
            monthly_returns = signals_copy.groupby('month')['returns'].sum()
            
            if len(monthly_returns) > 1:
                # Reshape for heatmap (year x month)
                monthly_returns.index = monthly_returns.index.astype(str)
                return monthly_returns.to_frame().T
            return None
        except Exception:
            return None
    
    def _analyze_signal_timing(self, signals: pd.DataFrame) -> Dict:
        """Analyze signal timing metrics."""
        try:
            # Calculate basic timing metrics
            total_signals = len(signals[signals['final_signal'] != 0])
            if total_signals == 0:
                return {}
            
            # Calculate average hold time (simplified)
            signal_changes = signals['final_signal'].diff().abs()
            avg_hold_time = len(signals) / total_signals if total_signals > 0 else 0
            
            # Calculate win rate (simplified)
            positive_signals = len(signals[signals['final_signal'] > 0])
            win_rate = positive_signals / total_signals if total_signals > 0 else 0
            
            return {
                'Avg Hold Time': avg_hold_time,
                'Win Rate': win_rate,
                'Avg Win': 1.0,  # Placeholder
                'Avg Loss': -0.5  # Placeholder
            }
        except Exception:
            return {}
    
    def _calculate_trade_durations(self, signals: pd.DataFrame) -> list:
        """Calculate trade durations."""
        try:
            durations = []
            current_duration = 0
            
            for signal in signals['final_signal']:
                if signal != 0:
                    if current_duration > 0:
                        durations.append(current_duration)
                    current_duration = 0
                else:
                    current_duration += 1
            
            return durations
        except Exception:
            return []
    
    def _calculate_monthly_activity(self, signals: pd.DataFrame) -> Optional[Dict]:
        """Calculate monthly trading activity."""
        try:
            signals_copy = signals.copy()
            signals_copy['month'] = signals_copy.index.to_period('M')
            
            monthly_trades = signals_copy[signals_copy['final_signal'] != 0].groupby('month').size()
            
            if len(monthly_trades) > 0:
                return monthly_trades.to_dict()
            return None
        except Exception:
            return None
