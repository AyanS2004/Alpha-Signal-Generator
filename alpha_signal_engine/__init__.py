"""
Alpha Signal Engine - A modular trading signal generation and backtesting system.

This package provides:
- Data ingestion from CSV files
- Momentum and mean-reversion signal generation
- Numba-optimized backtesting simulation
- Performance metrics calculation
- Visualization capabilities
- Advanced ML-based signals
- Real-time data feeds
- Market regime detection
"""

from .data_loader import DataLoader
from .signal_generator import SignalGenerator
from .backtester import Backtester
from .performance_analyzer import PerformanceAnalyzer
from .visualizer import Visualizer
from .config import Config
from .engine import AlphaSignalEngine
from .advanced_signals import AdvancedSignalGenerator

# Advanced features
from .bayesian_optimizer import BayesianOptimizer, AdvancedParameterOptimizer
from .multi_timeframe_strategy import MultiTimeframeStrategy, MultiTimeframeSignal
from .advanced_risk_manager import AdvancedRiskManager, PositionSizingResult
from .performance_attributor import PerformanceAttributor, AdvancedMetricsCalculator
from .ensemble_signal_generator import EnsembleSignalGenerator, EnsembleResult

# Optional realtime import (websocket-client may not be installed)
try:
    from .realtime_feed import RealTimeDataFeed, LiveSignalGenerator, MarketDataCache
    _HAS_REALTIME = True
except Exception:  # ImportError or other optional deps errors
    _HAS_REALTIME = False

__version__ = "1.1.0"
__author__ = "Alpha Signal Engine"

__all__ = [
    "DataLoader",
    "SignalGenerator",
    "Backtester",
    "PerformanceAnalyzer",
    "Visualizer",
    "Config",
    "AlphaSignalEngine",
    "AdvancedSignalGenerator",
    # Advanced features
    "BayesianOptimizer",
    "AdvancedParameterOptimizer",
    "MultiTimeframeStrategy",
    "MultiTimeframeSignal",
    "AdvancedRiskManager",
    "PositionSizingResult",
    "PerformanceAttributor",
    "AdvancedMetricsCalculator",
    "EnsembleSignalGenerator",
    "EnsembleResult",
]

if _HAS_REALTIME:
    __all__.extend(["RealTimeDataFeed", "LiveSignalGenerator", "MarketDataCache"])

