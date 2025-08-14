#!/usr/bin/env python3
"""
Test script for Alpha Signal Engine.
Verifies that all components work correctly.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the alpha_signal_engine directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'alpha_signal_engine'))

from alpha_signal_engine import AlphaSignalEngine, Config


def create_test_data():
    """Create synthetic test data."""
    print("📊 Creating test data...")
    
    # Generate synthetic price data
    np.random.seed(42)
    n_periods = 1000
    
    # Base price with trend and noise
    base_price = 150.0
    trend = np.linspace(0, 0.1, n_periods)  # 10% upward trend
    noise = np.random.normal(0, 0.02, n_periods)  # 2% volatility
    
    prices = base_price * (1 + trend + noise)
    
    # Create OHLCV data
    data = []
    for i in range(n_periods):
        price = prices[i]
        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = price * (1 + np.random.normal(0, 0.002))
        volume = np.random.randint(100000, 1000000)
        
        data.append({
            'Date': datetime.now() - timedelta(minutes=n_periods-i),
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': price,
            'Volume': volume
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    # Save test data
    df.to_csv('test_data.csv')
    print(f"✅ Test data created: {len(df)} periods")
    print(f"   Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    return df


def test_basic_functionality():
    """Test basic engine functionality."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Create test data
        test_data = create_test_data()
        
        # Initialize engine
        engine = AlphaSignalEngine()
        
        # Run analysis
        results = engine.run_complete_analysis(
            csv_file_path="test_data.csv",
            plot_results=False,
            save_plots=False
        )
        
        # Verify results
        assert 'backtest_summary' in results
        assert 'signal_summary' in results
        assert 'performance_metrics' in results
        
        print("✅ Basic functionality test passed")
        print(f"   Total return: {results['backtest_summary']['total_return']:.2%}")
        print(f"   Sharpe ratio: {results['backtest_summary']['sharpe_ratio']:.3f}")
        print(f"   Total signals: {results['signal_summary']['total_signals']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {str(e)}")
        return False


def test_custom_configuration():
    """Test custom configuration."""
    print("\n⚙️ Testing custom configuration...")
    
    try:
        # Create custom config
        custom_config = Config(
            initial_capital=50000.0,
            position_size=0.15,
            momentum_lookback=15,
            momentum_threshold=0.015,
            transaction_cost_bps=2.0,
            stop_loss_bps=30.0,
            take_profit_bps=60.0
        )
        
        # Initialize engine with custom config
        engine = AlphaSignalEngine(custom_config)
        
        # Run analysis
        results = engine.run_complete_analysis(
            csv_file_path="test_data.csv",
            plot_results=False,
            save_plots=False
        )
        
        print("✅ Custom configuration test passed")
        print(f"   Initial capital: ${custom_config.initial_capital:,.0f}")
        print(f"   Position size: {custom_config.position_size:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom configuration test failed: {str(e)}")
        return False


def test_parameter_optimization():
    """Test parameter optimization."""
    print("\n🔍 Testing parameter optimization...")
    
    try:
        # Initialize engine
        engine = AlphaSignalEngine()
        
        # Define parameter ranges
        param_ranges = {
            'momentum_lookback': [10, 15, 20],
            'momentum_threshold': [0.01, 0.015, 0.02],
            'position_size': [0.05, 0.1, 0.15]
        }
        
        # Run optimization
        optimization_results = engine.optimize_parameters(
            param_ranges=param_ranges,
            csv_file_path="test_data.csv"
        )
        
        # Verify results
        assert 'best_params' in optimization_results
        assert 'best_sharpe' in optimization_results
        assert 'top_10_results' in optimization_results
        
        print("✅ Parameter optimization test passed")
        print(f"   Best Sharpe ratio: {optimization_results['best_sharpe']:.3f}")
        print(f"   Best parameters: {optimization_results['best_params']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter optimization test failed: {str(e)}")
        return False


def test_data_access():
    """Test data access methods."""
    print("\n📈 Testing data access...")
    
    try:
        # Initialize engine
        engine = AlphaSignalEngine()
        
        # Run analysis to populate data
        engine.run_complete_analysis(
            csv_file_path="test_data.csv",
            plot_results=False,
            save_plots=False
        )
        
        # Test data access methods
        data = engine.get_data()
        signals = engine.get_signals()
        backtest_results = engine.get_backtest_results()
        performance_metrics = engine.get_performance_metrics()
        
        # Verify data
        assert len(data) > 0
        assert len(signals) > 0
        assert 'equity' in backtest_results
        assert 'sharpe_ratio' in performance_metrics
        
        print("✅ Data access test passed")
        print(f"   Data points: {len(data)}")
        print(f"   Signal points: {len(signals)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data access test failed: {str(e)}")
        return False


def test_visualization():
    """Test visualization capabilities."""
    print("\n🎨 Testing visualization...")
    
    try:
        # Initialize engine
        engine = AlphaSignalEngine()
        
        # Run analysis
        results = engine.run_complete_analysis(
            csv_file_path="test_data.csv",
            plot_results=False,
            save_plots=True
        )
        
        # Check if plots were saved
        import os
        if os.path.exists("performance_analysis.png"):
            print("✅ Performance plot saved")
        if os.path.exists("signal_analysis.png"):
            print("✅ Signal analysis plot saved")
        
        print("✅ Visualization test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("🚀 Alpha Signal Engine - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Custom Configuration", test_custom_configuration),
        ("Parameter Optimization", test_parameter_optimization),
        ("Data Access", test_data_access),
        ("Visualization", test_visualization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Alpha Signal Engine is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)
