#!/usr/bin/env python3
"""
Custom Backtesting Script for Alpha Signal Engine

This script allows you to run backtests on custom stocks with custom time periods
and parameters. It's similar to example.py but with more flexibility for different
stocks and time ranges.

Usage:
    python custom_backtest.py --symbol AAPL --start-date 2023-01-01 --end-date 2024-01-01
    python custom_backtest.py --symbol TSLA --start-date 2022-01-01 --end-date 2023-12-31 --config custom_config.json
"""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional

# Add the alpha_signal_engine directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'alpha_signal_engine'))

from alpha_signal_engine import AlphaSignalEngine, Config
import yfinance as yf


def download_stock_data(symbol: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
    """
    Download stock data using yfinance.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval ('1d', '1h', '5m', etc.)
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"📊 Downloading {symbol} data from {start_date} to {end_date}...")
    
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data found for {symbol} from {start_date} to {end_date}")
        
        print(f"✅ Downloaded {len(data)} data points")
        print(f"   Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error downloading data for {symbol}: {str(e)}")
        raise


def load_config(config_file: Optional[str] = None) -> Dict:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_file: Path to JSON config file
    
    Returns:
        Configuration dictionary
    """
    if config_file and os.path.exists(config_file):
        print(f"📋 Loading configuration from {config_file}")
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        print("📋 Using default configuration")
        return {
            "initialCapital": 100000.0,
            "positionSize": 0.1,
            "momentumLookback": 20,
            "momentumThreshold": 0.02,
            "meanReversionLookback": 50,
            "meanReversionThreshold": 0.01,
            "transactionCost": 1.0,
            "stopLoss": 50.0,
            "takeProfit": 100.0,
            "riskFreeRate": 0.03
        }


def create_engine_config(config_dict: Dict) -> Config:
    """
    Create AlphaSignalEngine Config from dictionary.
    
    Args:
        config_dict: Configuration dictionary
    
    Returns:
        Config object
    """
    return Config(
        initial_capital=config_dict.get('initialCapital', 100000.0),
        position_size=config_dict.get('positionSize', 0.1),
        momentum_lookback=config_dict.get('momentumLookback', 20),
        momentum_threshold=config_dict.get('momentumThreshold', 0.02),
        mean_reversion_lookback=config_dict.get('meanReversionLookback', 50),
        mean_reversion_std_multiplier=config_dict.get('meanReversionThreshold', 0.01),
        transaction_cost_bps=config_dict.get('transactionCost', 1.0),
        stop_loss_bps=config_dict.get('stopLoss', 50.0),
        take_profit_bps=config_dict.get('takeProfit', 100.0),
        risk_free_rate=config_dict.get('riskFreeRate', 0.03)
    )


def run_backtest(symbol: str, start_date: str, end_date: str, interval: str = '1d', 
                config_file: Optional[str] = None, save_results: bool = True) -> Dict:
    """
    Run a complete backtest for the specified stock and time period.
    
    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        interval: Data interval
        config_file: Optional config file path
        save_results: Whether to save results to files
    
    Returns:
        Backtest results dictionary
    """
    print(f"\n🎯 Running Custom Backtest")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Interval: {interval}")
    
    # Load configuration
    config_dict = load_config(config_file)
    
    # Download data
    data = download_stock_data(symbol, start_date, end_date, interval)
    
    # Save data to CSV
    csv_filename = f"{symbol}_{start_date}_{end_date}.csv"
    data.to_csv(csv_filename)
    print(f"💾 Data saved to {csv_filename}")
    
    # Create engine with custom config
    engine_config = create_engine_config(config_dict)
    engine = AlphaSignalEngine(engine_config)
    
    # Run backtest
    print("\n🚀 Running backtest analysis...")
    results = engine.run_complete_analysis(
        csv_file_path=csv_filename,
        plot_results=True,
        save_plots=save_results
    )
    
    # Display results
    print("\n📈 Backtest Results")
    print("-" * 40)
    print(f"Total Return: {results['backtest_summary']['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['backtest_summary']['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {results['backtest_summary']['max_drawdown']:.2%}")
    print(f"Total Trades: {results['backtest_summary']['total_trades']}")
    print(f"Win Rate: {results['performance_metrics']['win_rate']:.2%}")
    print(f"Profit Factor: {results['performance_metrics']['profit_factor']:.3f}")
    
    if 'sortino_ratio' in results['performance_metrics']:
        print(f"Sortino Ratio: {results['performance_metrics']['sortino_ratio']:.3f}")
    if 'calmar_ratio' in results['performance_metrics']:
        print(f"Calmar Ratio: {results['performance_metrics']['calmar_ratio']:.3f}")
    
    # Save detailed results
    if save_results:
        results_filename = f"{symbol}_backtest_results.json"
        # Convert numpy types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, (np.integer, np.floating)):
                        json_results[key][k] = float(v)
                    elif isinstance(v, np.ndarray):
                        json_results[key][k] = v.tolist()
                    else:
                        json_results[key][k] = v
            else:
                json_results[key] = value
        
        with open(results_filename, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        print(f"💾 Detailed results saved to {results_filename}")
    
    return results


def main():
    """Main function to handle command line arguments and run backtest."""
    parser = argparse.ArgumentParser(description='Custom Backtesting for Alpha Signal Engine')
    parser.add_argument('--symbol', '-s', help='Stock symbol (e.g., AAPL, TSLA)')
    parser.add_argument('--start-date', '-start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', '-end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', '-i', default='1d', help='Data interval (1d, 1h, 5m, etc.)')
    parser.add_argument('--config', '-c', help='Path to JSON configuration file')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to files')
    parser.add_argument('--list-symbols', action='store_true', help='List available stock symbols')
    
    args = parser.parse_args()
    
    if args.list_symbols:
        print("📋 Available Stock Symbols:")
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'UBER', 'LYFT',
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'BND'
        ]
        for i, symbol in enumerate(symbols, 1):
            print(f"  {i:2d}. {symbol}")
        return
    
    # Check required arguments for backtest
    if not args.symbol or not args.start_date or not args.end_date:
        print("❌ Error: --symbol, --start-date, and --end-date are required for backtesting")
        print("Use --list-symbols to see available symbols")
        sys.exit(1)
    
    try:
        # Validate dates
        datetime.strptime(args.start_date, '%Y-%m-%d')
        datetime.strptime(args.end_date, '%Y-%m-%d')
        
        # Run backtest
        results = run_backtest(
            symbol=args.symbol.upper(),
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval,
            config_file=args.config,
            save_results=not args.no_save
        )
        
        print("\n✅ Backtest completed successfully!")
        
    except ValueError as e:
        print(f"❌ Invalid date format: {e}")
        print("Please use YYYY-MM-DD format (e.g., 2023-01-01)")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running backtest: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
