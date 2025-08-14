"""
Backtester module for Alpha Signal Engine.
Implements Numba-optimized trading simulation with transaction costs.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from numba import jit
import warnings


class Backtester:
    """Numba-optimized backtesting engine with transaction costs."""
    
    def __init__(self, config):
        """Initialize Backtester with configuration."""
        self.config = config
        
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """
        Run backtest simulation on the given data with signals.
        
        Args:
            df: DataFrame with signals and OHLCV data
            
        Returns:
            Dictionary containing backtest results
        """
        if 'final_signal' not in df.columns:
            raise ValueError("DataFrame must contain 'final_signal' column")
        
        # Extract arrays for Numba optimization
        prices = df['Close'].values
        signals = df['final_signal'].values
        volumes = df['Volume'].values
        
        # Run backtest (Numba or Python) and normalize to dict
        if self.config.use_numba:
            capital, positions, trades, pnl, drawdown, equity = self._run_optimized_backtest(
                prices,
                signals,
                volumes,
                self.config.initial_capital,
                self.config.position_size,
                self.config.transaction_cost_bps,
                self.config.max_position_size,
                self.config.stop_loss_bps,
                self.config.take_profit_bps,
            )
            results: Dict = {
                'capital': capital,
                'positions': positions,
                'trades': trades,
                'pnl': pnl,
                'drawdown': drawdown,
                'equity': equity,
            }
        else:
            results = self._run_python_backtest(df)

        return results
    
    @staticmethod
    @jit(nopython=True)
    def _run_optimized_backtest(prices: np.ndarray, signals: np.ndarray, 
                               volumes: np.ndarray, initial_capital: float,
                               position_size: float, transaction_cost_bps: float,
                               max_position_size: float, stop_loss_bps: float,
                               take_profit_bps: float) -> Tuple[np.ndarray, np.ndarray, 
                                                              np.ndarray, np.ndarray, 
                                                              np.ndarray, np.ndarray]:
        """
        Numba-optimized backtest simulation.
        
        Args:
            prices: Array of closing prices
            signals: Array of trading signals (1=buy, -1=sell, 0=hold)
            volumes: Array of trading volumes
            initial_capital: Starting capital
            position_size: Position size as fraction of capital
            transaction_cost_bps: Transaction cost in basis points
            max_position_size: Maximum position size
            stop_loss_bps: Stop loss in basis points
            take_profit_bps: Take profit in basis points
            
        Returns:
            Tuple of arrays: (capital, positions, trades, pnl, drawdown, equity)
        """
        n = len(prices)
        
        # Initialize arrays
        capital = np.full(n, initial_capital, dtype=np.float64)
        positions = np.zeros(n, dtype=np.float64)
        trades = np.zeros(n, dtype=np.int32)
        pnl = np.zeros(n, dtype=np.float64)
        drawdown = np.zeros(n, dtype=np.float64)
        equity = np.full(n, initial_capital, dtype=np.float64)
        
        # Trading state
        current_position = 0.0
        entry_price = 0.0
        entry_capital = initial_capital
        max_equity = initial_capital
        
        for i in range(1, n):
            # Update capital and equity
            capital[i] = capital[i-1]
            equity[i] = equity[i-1]
            
            # Calculate position value change
            if current_position != 0:
                price_change = (prices[i] - prices[i-1]) / prices[i-1]
                position_pnl = current_position * price_change
                pnl[i] = position_pnl
                equity[i] += position_pnl
                
                # Check stop loss and take profit
                if current_position > 0:  # Long position
                    loss_pct = (prices[i] - entry_price) / entry_price
                    if loss_pct < -stop_loss_bps / 10000 or loss_pct > take_profit_bps / 10000:
                        # Close position
                        trade_value = current_position * prices[i]
                        transaction_cost = trade_value * transaction_cost_bps / 10000
                        capital[i] = capital[i-1] + trade_value - transaction_cost
                        current_position = 0.0
                        trades[i] = -1  # Sell signal
                        pnl[i] = trade_value - entry_capital - transaction_cost
                        equity[i] = capital[i]
                
            # Process new signals
            if signals[i] != 0 and current_position == 0:
                if signals[i] == 1:  # Buy signal
                    trade_value = capital[i] * position_size
                    trade_value = min(trade_value, capital[i] * max_position_size)
                    
                    if trade_value > 0:
                        transaction_cost = trade_value * transaction_cost_bps / 10000
                        current_position = trade_value / prices[i]
                        capital[i] -= transaction_cost
                        entry_price = prices[i]
                        entry_capital = trade_value
                        trades[i] = 1  # Buy signal
                        
            elif signals[i] == -1 and current_position > 0:  # Sell signal
                trade_value = current_position * prices[i]
                transaction_cost = trade_value * transaction_cost_bps / 10000
                capital[i] = capital[i-1] + trade_value - transaction_cost
                current_position = 0.0
                trades[i] = -1  # Sell signal
                pnl[i] = trade_value - entry_capital - transaction_cost
                equity[i] = capital[i]
            
            # Update drawdown
            if equity[i] > max_equity:
                max_equity = equity[i]
            drawdown[i] = (max_equity - equity[i]) / max_equity if max_equity > 0 else 0
        
        return capital, positions, trades, pnl, drawdown, equity
    
    def _run_python_backtest(self, df: pd.DataFrame) -> Dict:
        """Python-based backtest (fallback when Numba is disabled)."""
        capital = [self.config.initial_capital]
        positions = [0.0]
        trades = [0]
        pnl = [0.0]
        drawdown = [0.0]
        equity = [self.config.initial_capital]
        
        current_position = 0.0
        entry_price = 0.0
        entry_capital = self.config.initial_capital
        max_equity = self.config.initial_capital
        
        for i in range(1, len(df)):
            # Update capital and equity
            capital.append(capital[-1])
            equity.append(equity[-1])
            
            # Calculate position value change
            if current_position != 0:
                price_change = (df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]
                position_pnl = current_position * price_change
                pnl.append(position_pnl)
                equity[-1] += position_pnl
                
                # Check stop loss and take profit
                if current_position > 0:  # Long position
                    loss_pct = (df['Close'].iloc[i] - entry_price) / entry_price
                    if loss_pct < -self.config.stop_loss_bps / 10000 or loss_pct > self.config.take_profit_bps / 10000:
                        # Close position
                        trade_value = current_position * df['Close'].iloc[i]
                        transaction_cost = trade_value * self.config.transaction_cost_bps / 10000
                        capital[-1] = capital[-2] + trade_value - transaction_cost
                        current_position = 0.0
                        trades.append(-1)
                        pnl[-1] = trade_value - entry_capital - transaction_cost
                        equity[-1] = capital[-1]
                    else:
                        trades.append(0)
                else:
                    trades.append(0)
            else:
                trades.append(0)
                pnl.append(0.0)
            
            # Process new signals
            signal = df['final_signal'].iloc[i]
            if signal != 0 and current_position == 0:
                if signal == 1:  # Buy signal
                    trade_value = capital[-1] * self.config.position_size
                    trade_value = min(trade_value, capital[-1] * self.config.max_position_size)
                    
                    if trade_value > 0:
                        transaction_cost = trade_value * self.config.transaction_cost_bps / 10000
                        current_position = trade_value / df['Close'].iloc[i]
                        capital[-1] -= transaction_cost
                        entry_price = df['Close'].iloc[i]
                        entry_capital = trade_value
                        trades[-1] = 1
                        
            elif signal == -1 and current_position > 0:  # Sell signal
                trade_value = current_position * df['Close'].iloc[i]
                transaction_cost = trade_value * self.config.transaction_cost_bps / 10000
                capital[-1] = capital[-2] + trade_value - transaction_cost
                current_position = 0.0
                trades[-1] = -1
                pnl[-1] = trade_value - entry_capital - transaction_cost
                equity[-1] = capital[-1]
            
            # Update drawdown
            if equity[-1] > max_equity:
                max_equity = equity[-1]
            drawdown.append((max_equity - equity[-1]) / max_equity if max_equity > 0 else 0)
        
        return {
            'capital': np.array(capital),
            'positions': np.array(positions),
            'trades': np.array(trades),
            'pnl': np.array(pnl),
            'drawdown': np.array(drawdown),
            'equity': np.array(equity)
        }
    
    def get_backtest_summary(self, results: Dict, df: pd.DataFrame) -> Dict:
        """Get summary statistics from backtest results."""
        equity = results['equity']
        pnl = results['pnl']
        trades = results['trades']
        drawdown = results['drawdown']
        
        # Calculate returns
        returns = np.diff(equity) / equity[:-1]
        returns = np.insert(returns, 0, 0)
        
        # Performance metrics
        total_return = (equity[-1] - equity[0]) / equity[0]
        annualized_return = total_return * (252 / len(df))
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        risk_free_rate = self.config.risk_free_rate
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Maximum drawdown
        max_drawdown = np.max(drawdown)
        
        # Trade statistics
        total_trades = np.sum(np.abs(trades))
        winning_trades = np.sum(pnl > 0)
        losing_trades = np.sum(pnl < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Average trade PnL
        avg_trade_pnl = np.mean(pnl[pnl != 0]) if np.any(pnl != 0) else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_trade_pnl': avg_trade_pnl,
            'final_capital': equity[-1],
            'initial_capital': equity[0]
        }

