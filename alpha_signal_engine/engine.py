"""
Main Alpha Signal Engine class.
Orchestrates all components for complete trading signal generation and backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings

from .config import Config
from .data_loader import DataLoader
from .signal_generator import SignalGenerator
from .backtester import Backtester
from .performance_analyzer import PerformanceAnalyzer
from .visualizer import Visualizer


class AlphaSignalEngine:
    """
    Main Alpha Signal Engine for trading signal generation and backtesting.
    
    This class orchestrates all components:
    - Data loading and preprocessing
    - Signal generation (momentum and mean-reversion)
    - Numba-optimized backtesting
    - Performance analysis
    - Visualization
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize Alpha Signal Engine.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config if config is not None else Config()
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.backtester = Backtester(self.config)
        self.performance_analyzer = PerformanceAnalyzer(self.config)
        self.visualizer = Visualizer(self.config)
        
        # Results storage
        self.data = None
        self.signals = None
        self.backtest_results = None
        self.performance_metrics = None
        
    def run_complete_analysis(self, csv_file_path: Optional[str] = None,
                             plot_results: bool = True,
                             save_plots: bool = False) -> Dict:
        """
        Run complete analysis pipeline.
        
        Args:
            csv_file_path: Path to CSV file. If None, uses config default.
            plot_results: Whether to display plots
            save_plots: Whether to save plots to files
            
        Returns:
            Dictionary containing all results and metrics
        """
        print("🚀 Starting Alpha Signal Engine Analysis...")
        print("=" * 60)
        
        # Step 1: Load and preprocess data
        print("📊 Loading and preprocessing data...")
        self.data = self.data_loader.load_data(csv_file_path)
        self.data = self.data_loader.preprocess_data(self.data)
        
        data_summary = self.data_loader.get_data_summary(self.data)
        print(f"   Data loaded: {data_summary['total_periods']} periods")
        print(f"   Date range: {data_summary['start_date']} to {data_summary['end_date']}")
        print(f"   Price range: ${data_summary['price_range'][0]:.2f} - ${data_summary['price_range'][1]:.2f}")
        
        # Step 2: Generate trading signals
        print("\n📈 Generating trading signals...")
        self.signals = self.signal_generator.generate_signals(self.data)
        
        signal_summary = self.signal_generator.get_signal_summary(self.signals)
        print(f"   Total signals: {signal_summary['total_signals']}")
        print(f"   Buy signals: {signal_summary['buy_signals']}")
        print(f"   Sell signals: {signal_summary['sell_signals']}")
        print(f"   Signal frequency: {signal_summary['signal_frequency']:.2%}")
        
        # Step 3: Run backtest
        print("\n💰 Running backtest simulation...")
        self.backtest_results = self.backtester.run_backtest(self.signals)
        
        backtest_summary = self.backtester.get_backtest_summary(self.backtest_results, self.signals)
        print(f"   Total return: {backtest_summary['total_return']:.2%}")
        print(f"   Sharpe ratio: {backtest_summary['sharpe_ratio']:.3f}")
        print(f"   Max drawdown: {backtest_summary['max_drawdown']:.2%}")
        print(f"   Total trades: {backtest_summary['total_trades']}")
        
        # Step 4: Calculate performance metrics
        print("\n📊 Calculating performance metrics...")
        self.performance_metrics = self.performance_analyzer.calculate_performance_metrics(
            self.backtest_results, self.signals
        )
        
        # Step 5: Generate performance report
        print("\n📋 Generating performance report...")
        performance_report = self.performance_analyzer.generate_performance_report(self.performance_metrics)
        print(performance_report)
        
        # Step 6: Create visualizations
        if plot_results:
            print("\n📊 Creating visualizations...")
            try:
                self.visualizer.plot_performance(self.signals, self.backtest_results)
                
                if save_plots:
                    self.visualizer.plot_performance(
                        self.signals, self.backtest_results, 
                        save_path="performance_analysis.png"
                    )
                    self.visualizer.plot_signal_analysis(
                        self.signals, save_path="signal_analysis.png"
                    )
                    print("   Plots saved as 'performance_analysis.png' and 'signal_analysis.png'")
                    
            except Exception as e:
                print(f"   Warning: Could not create plots: {str(e)}")
        
        # Compile results
        results = {
            'data_summary': data_summary,
            'signal_summary': signal_summary,
            'backtest_summary': backtest_summary,
            'performance_metrics': self.performance_metrics,
            'performance_report': performance_report,
            'data': self.data,
            'signals': self.signals,
            'backtest_results': self.backtest_results
        }
        
        print("\n✅ Analysis complete!")
        print("=" * 60)
        
        return results
    
    def optimize_parameters(self, param_ranges: Dict, csv_file_path: Optional[str] = None) -> Dict:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            param_ranges: Dictionary of parameter ranges to test
            csv_file_path: Path to CSV file
            
        Returns:
            Dictionary with optimization results
        """
        print("🔍 Starting parameter optimization...")
        
        # Load data once
        data = self.data_loader.load_data(csv_file_path)
        data = self.data_loader.preprocess_data(data)
        
        best_sharpe = -np.inf
        best_params = None
        best_results = None
        all_results = []
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_ranges)
        total_combinations = len(param_combinations)
        
        print(f"Testing {total_combinations} parameter combinations...")
        
        for i, params in enumerate(param_combinations):
            # Update config with new parameters
            self._update_config_params(params)
            
            # Generate signals and run backtest
            signals = self.signal_generator.generate_signals(data)
            backtest_results = self.backtester.run_backtest(signals)
            backtest_summary = self.backtester.get_backtest_summary(backtest_results, signals)
            
            # Store results
            result = {
                'params': params.copy(),
                'sharpe_ratio': backtest_summary['sharpe_ratio'],
                'total_return': backtest_summary['total_return'],
                'max_drawdown': backtest_summary['max_drawdown'],
                'total_trades': backtest_summary['total_trades']
            }
            all_results.append(result)
            
            # Update best if better
            if backtest_summary['sharpe_ratio'] > best_sharpe:
                best_sharpe = backtest_summary['sharpe_ratio']
                best_params = params.copy()
                best_results = backtest_summary.copy()
            
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{total_combinations} combinations tested")
        
        # Sort results by Sharpe ratio
        all_results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        
        optimization_results = {
            'best_params': best_params,
            'best_sharpe': best_sharpe,
            'best_results': best_results,
            'all_results': all_results,
            'top_10_results': all_results[:10]
        }
        
        print(f"\n🏆 Best Sharpe Ratio: {best_sharpe:.3f}")
        print(f"📊 Best Parameters: {best_params}")
        
        return optimization_results
    
    def _generate_param_combinations(self, param_ranges: Dict) -> list:
        """Generate all parameter combinations for grid search."""
        import itertools
        
        # Get all parameter values
        param_values = list(param_ranges.values())
        param_names = list(param_ranges.keys())
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        param_combinations = []
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            param_combinations.append(param_dict)
        
        return param_combinations
    
    def _update_config_params(self, params: Dict) -> None:
        """Update config with new parameters."""
        for param_name, value in params.items():
            if hasattr(self.config, param_name):
                setattr(self.config, param_name, value)
    
    def get_data(self) -> pd.DataFrame:
        """Get the loaded and preprocessed data."""
        return self.data
    
    def get_signals(self) -> pd.DataFrame:
        """Get the data with generated signals."""
        return self.signals
    
    def get_backtest_results(self) -> Dict:
        """Get the backtest results."""
        return self.backtest_results
    
    def get_performance_metrics(self) -> Dict:
        """Get the performance metrics."""
        return self.performance_metrics



