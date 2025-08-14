"""
Configuration module for Alpha Signal Engine.
Contains all trading parameters, thresholds, and settings.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration class for Alpha Signal Engine parameters."""
    
    # Data parameters
    csv_file_path: str = "AAPL_minute.csv"
    
    # Trading parameters
    initial_capital: float = 100000.0
    transaction_cost_bps: float = 1.0  # 1 basis point = 0.01%
    position_size: float = 0.1  # 10% of capital per trade
    
    # Momentum strategy parameters
    momentum_lookback: int = 20
    momentum_threshold: float = 0.02  # 2% price change
    momentum_volume_threshold: float = 1.5  # 1.5x average volume
    
    # Mean reversion strategy parameters
    mean_reversion_lookback: int = 50
    mean_reversion_std_multiplier: float = 2.0
    mean_reversion_threshold: float = 0.01  # 1% deviation
    
    # Risk management
    max_position_size: float = 0.3  # 30% max position
    stop_loss_bps: float = 50.0  # 0.5% stop loss
    take_profit_bps: float = 100.0  # 1% take profit
    
    # Performance calculation
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    trading_days_per_year: int = 252
    
    # Visualization
    plot_pnl: bool = True
    plot_drawdown: bool = True
    plot_signals: bool = True
    
    # Numba optimization
    use_numba: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.transaction_cost_bps < 0:
            raise ValueError("Transaction cost must be non-negative")
        if self.position_size <= 0 or self.position_size > 1:
            raise ValueError("Position size must be between 0 and 1")
        if self.max_position_size <= 0 or self.max_position_size > 1:
            raise ValueError("Max position size must be between 0 and 1")
        if self.momentum_lookback <= 0:
            raise ValueError("Momentum lookback must be positive")
        if self.mean_reversion_lookback <= 0:
            raise ValueError("Mean reversion lookback must be positive")

