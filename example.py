#!/usr/bin/env python3
"""
Alpha Signal Engine - Example Usage

This script demonstrates how to use the Alpha Signal Engine for:
- Loading minute-level stock data
- Generating momentum and mean-reversion signals
- Running Numba-optimized backtests
- Analyzing performance metrics
- Creating visualizations
- Parameter optimization
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict

# Add the alpha_signal_engine directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'alpha_signal_engine'))

from alpha_signal_engine import AlphaSignalEngine, Config


def main():
    """Main example function."""
    print("🎯 Alpha Signal Engine - Example Usage")
    print("=" * 60)
    
    # Example 1: Basic usage with default parameters
    print("\n📊 Example 1: Basic Analysis with Default Parameters")
    print("-" * 50)
    
    # Initialize engine
    engine = AlphaSignalEngine()
    
    # Run complete analysis
    results = engine.run_complete_analysis(
        csv_file_path="AAPL_minute.csv",
        plot_results=True,
        save_plots=True
    )
    
    # Example 2: Custom configuration
    print("\n\n⚙️ Example 2: Custom Configuration")
    print("-" * 50)
    
    # Create custom config
    custom_config = Config(
        initial_capital=50000.0,
        position_size=0.15,  # 15% position size
        momentum_lookback=15,
        momentum_threshold=0.015,  # 1.5% threshold
        mean_reversion_lookback=30,
        transaction_cost_bps=2.0,  # 2 basis points
        stop_loss_bps=30.0,  # 0.3% stop loss
        take_profit_bps=60.0  # 0.6% take profit
    )
    
    # Initialize engine with custom config
    custom_engine = AlphaSignalEngine(custom_config)
    
    # Run analysis with custom parameters
    custom_results = custom_engine.run_complete_analysis(
        csv_file_path="AAPL_minute.csv",
        plot_results=False  # Don't show plots for this example
    )
    
    # Compare results
    print(f"\nDefault Config Results:")
    print(f"  Total Return: {results['backtest_summary']['total_return']:.2%}")
    print(f"  Sharpe Ratio: {results['backtest_summary']['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {results['backtest_summary']['max_drawdown']:.2%}")
    
    print(f"\nCustom Config Results:")
    print(f"  Total Return: {custom_results['backtest_summary']['total_return']:.2%}")
    print(f"  Sharpe Ratio: {custom_results['backtest_summary']['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {custom_results['backtest_summary']['max_drawdown']:.2%}")
    
    # Example 3: Parameter optimization
    print("\n\n🔍 Example 3: Parameter Optimization")
    print("-" * 50)
    
    # Define parameter ranges to test
    param_ranges = {
        'momentum_lookback': [10, 15, 20, 25],
        'momentum_threshold': [0.01, 0.015, 0.02, 0.025],
        'position_size': [0.05, 0.1, 0.15],
        'transaction_cost_bps': [1.0, 2.0, 3.0]
    }
    
    # Run optimization
    optimization_results = engine.optimize_parameters(
        param_ranges=param_ranges,
        csv_file_path="AAPL_minute.csv"
    )
    
    # Display top 5 results
    print("\n🏆 Top 5 Parameter Combinations:")
    print("-" * 40)
    for i, result in enumerate(optimization_results['top_10_results'][:5]):
        print(f"{i+1}. Sharpe: {result['sharpe_ratio']:.3f}, "
              f"Return: {result['total_return']:.2%}, "
              f"Params: {result['params']}")
    
    # Example 4: Detailed analysis
    print("\n\n📈 Example 4: Detailed Analysis")
    print("-" * 50)
    
    # Get detailed data
    data = engine.get_data()
    signals = engine.get_signals()
    backtest_results = engine.get_backtest_results()
    metrics = engine.get_performance_metrics()
    
    # Display detailed statistics
    print(f"\nData Summary:")
    print(f"  Total periods: {len(data)}")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")
    print(f"  Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    print(f"\nSignal Analysis:")
    print(f"  Total signals: {len(signals[signals['final_signal'] != 0])}")
    print(f"  Buy signals: {len(signals[signals['final_signal'] == 1])}")
    print(f"  Sell signals: {len(signals[signals['final_signal'] == -1])}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    print(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Profit Factor: {metrics['profit_factor']:.3f}")
    
    # Example 5: Export results
    print("\n\n💾 Example 5: Export Results")
    print("-" * 50)
    
    # Save signals to CSV
    signals.to_csv("trading_signals.csv")
    print("  Trading signals saved to 'trading_signals.csv'")
    
    # Save performance metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("performance_metrics.csv", index=False)
    print("  Performance metrics saved to 'performance_metrics.csv'")
    
    # Save backtest results
    backtest_df = pd.DataFrame({
        'equity': backtest_results['equity'],
        'pnl': backtest_results['pnl'],
        'drawdown': backtest_results['drawdown']
    })
    backtest_df.to_csv("backtest_results.csv")
    print("  Backtest results saved to 'backtest_results.csv'")
    
    print("\n✅ All examples completed successfully!")
    print("=" * 60)


def demonstrate_advanced_features():
    """Demonstrate advanced features of the Alpha Signal Engine."""
    print("\n🚀 Advanced Features Demonstration")
    print("=" * 60)
    
    # Create engine with advanced configuration
    advanced_config = Config(
        initial_capital=100000.0,
        position_size=0.2,
        momentum_lookback=25,
        momentum_threshold=0.025,
        mean_reversion_lookback=40,
        mean_reversion_std_multiplier=2.5,
        transaction_cost_bps=1.5,
        stop_loss_bps=40.0,
        take_profit_bps=80.0,
        risk_free_rate=0.03,
        use_numba=True
    )
    
    engine = AlphaSignalEngine(advanced_config)
    
    # Run analysis
    results = engine.run_complete_analysis(
        csv_file_path="AAPL_minute.csv",
        plot_results=True,
        save_plots=True
    )
    
    # Demonstrate signal analysis
    signals = engine.get_signals()
    
    print(f"\nAdvanced Signal Analysis:")
    print(f"  Momentum signals: {len(signals[signals['momentum_signal'] != 0])}")
    print(f"  Mean reversion signals: {len(signals[signals['mean_reversion_signal'] != 0])}")
    print(f"  EMA crossover signals: {len(signals[signals['ema_signal'] != 0])}")
    print(f"  RSI signals: {len(signals[signals['rsi_signal'] != 0])}")
    
    # Show correlation between signals
    signal_columns = ['momentum_signal', 'mean_reversion_signal', 'ema_signal', 'rsi_signal']
    signal_corr = signals[signal_columns].corr()
    print(f"\nSignal Correlation Matrix:")
    print(signal_corr.round(3))
    
    # Risk analysis
    metrics = engine.get_performance_metrics()
    print(f"\nAdvanced Risk Metrics:")
    print(f"  VaR (95%): {metrics['var_95']:.2%}")
    print(f"  VaR (99%): {metrics['var_99']:.2%}")
    print(f"  Expected Shortfall (95%): {metrics['expected_shortfall_95']:.2%}")
    print(f"  Average Drawdown Duration: {metrics['avg_drawdown_duration']:.1f} periods")
    print(f"  Largest Win: ${metrics['largest_win']:.2f}")
    print(f"  Largest Loss: ${metrics['largest_loss']:.2f}")


if __name__ == "__main__":
    try:
        # Check if data file exists
        if not os.path.exists("AAPL_minute.csv"):
            print("❌ Error: AAPL_minute.csv not found in current directory")
            print("Please ensure the CSV file is in the same directory as this script.")
            sys.exit(1)
        
        # Run main examples
        main()
        
        # Run advanced features demonstration
        demonstrate_advanced_features()
        
    except Exception as e:
        print(f"❌ Error running examples: {str(e)}")
        print("Please check that all required dependencies are installed.")
        sys.exit(1)



