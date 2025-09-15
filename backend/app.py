"""
Flask Backend API for Alpha Signal Engine Frontend.
Provides REST API endpoints for the React frontend.
"""

from flask import Flask, request, jsonify, send_file
import logging
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys
import tempfile
from datetime import datetime, timedelta
import json
import yfinance as yf

# Ensure project root is on sys.path so we can import the alpha_signal_engine package
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from alpha_signal_engine import (
    AlphaSignalEngine, Config, BayesianOptimizer, AdvancedParameterOptimizer,
    MultiTimeframeStrategy, AdvancedRiskManager, PerformanceAttributor,
    AdvancedMetricsCalculator, EnsembleSignalGenerator
)
from alpha_signal_engine.visualizer import Visualizer
from alpha_signal_engine.walk_forward import WalkForwardOptimizer
from alpha_signal_engine.factor_strategy import FactorStrategy
from alpha_signal_engine.rl_position_sizer import RLPositionSizer
from alpha_signal_engine.transaction_costs import SmartTransactionCostModel
import os

app = Flask(__name__)
CORS(app)

# Basic structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger("alpha-signal-backend")

# Global engine instance
engine = None
# Realtime globals
ALPACA_FEED = None
ALPACA_SYMBOL = None

# Advanced features instances
bayesian_optimizer = None
multi_timeframe_strategy = None
advanced_risk_manager = None
performance_attributor = None
ensemble_generator = None
walk_forward_optimizer = WalkForwardOptimizer()
factor_strategy = None
rl_position_sizer = RLPositionSizer()
tx_cost_model = SmartTransactionCostModel()

# Settings file
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'settings.json')
PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), 'portfolio.json')

def _load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    # defaults
    return {
        'initialCapital': 100000,
        'positionSize': 0.1,
        'transactionCost': 1.0,
        'stopLoss': 50,
        'takeProfit': 100,
        'momentumLookback': 20,
        'momentumThreshold': 0.02,
        'meanReversionLookback': 50,
        'meanReversionThreshold': 0.01,
        'autoTrading': False,
        'notifications': True,
        'dataRetention': 30,
        'riskManagement': True,
        'apiKey': '',
        'apiSecret': '',
        'dataProvider': 'alpha_vantage',
    }

def _save_settings(settings: dict):
    tmp = SETTINGS_FILE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2)
    os.replace(tmp, SETTINGS_FILE)

def _load_portfolio() -> dict:
    try:
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {'positions': []}

def _save_portfolio(portfolio: dict):
    tmp = PORTFOLIO_FILE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(portfolio, f, indent=2)
    os.replace(tmp, PORTFOLIO_FILE)

def _compute_trades_list(results: dict, signals_df: pd.DataFrame) -> list:
    try:
        trades_arr = results.get('trades')
        equity = results.get('equity')
        pnl = results.get('pnl')
        prices = signals_df['Close'].values
        index = signals_df.index
        trades = []
        entry_idx = None
        entry_price = None
        for i, t in enumerate(trades_arr):
            if t == 1 and entry_idx is None:
                entry_idx = i
                entry_price = float(prices[i])
            elif t == -1 and entry_idx is not None:
                exit_price = float(prices[i])
                realized_pnl = float(pnl[i]) if i < len(pnl) else float((exit_price - entry_price))
                ret = (exit_price - entry_price) / entry_price if entry_price else 0.0
                trades.append({
                    'entryDate': str(index[entry_idx]) if entry_idx < len(index) else str(entry_idx),
                    'exitDate': str(index[i]) if i < len(index) else str(i),
                    'entryPrice': entry_price if entry_price is not None else 0.0,
                    'exitPrice': exit_price,
                    'pnl': realized_pnl,
                    'return': ret,
                })
                entry_idx = None
                entry_price = None
        return trades
    except Exception:
        return []

def convert_numpy_types(obj):
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def download_stock_data(symbol, start_date, end_date, interval='1d'):
    """Download stock data using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data found for {symbol} from {start_date} to {end_date}")
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Reset index to make datetime a column
        data = data.reset_index()
        data['Datetime'] = data['Date']
        data = data.set_index('Datetime')
        
        return data
        
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {str(e)}")
        raise

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    logger.info("health_check")
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.1.0'
    })

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    """Get dashboard data."""
    try:
        global engine
        if engine is None or engine.get_backtest_results() is None:
            return jsonify({'error': 'No backtest results yet'}), 400

        bt = engine.get_backtest_results()
        perf = engine.get_performance_metrics() or {}
        sig = engine.get_signals()

        # KPIs
        total_return = float(((bt['equity'][-1] - bt['equity'][0]) / bt['equity'][0]) * 100)
        sharpe = float(perf.get('sharpe_ratio', 0))
        max_dd = float(np.max(bt.get('drawdown', np.array([0]))) * -100.0) * -1  # keep negative percentage
        trades = int(np.sum(np.array(bt.get('trades', [])) == -1))
        win_rate = float(perf.get('win_rate', 0) * 100)

        # Current signal
        current_signal = 'HOLD'
        try:
            last_sig = int(sig['final_signal'].iloc[-1])
            current_signal = 'BUY' if last_sig > 0 else 'SELL' if last_sig < 0 else 'HOLD'
        except Exception:
            pass

        # Equity curve
        equityData = [{'date': str(i), 'value': float(v)} for i, v in enumerate(bt.get('equity', []))]

        # Benchmark data (SPY) aligned with backtest period
        benchmarkData = []
        try:
            import yfinance as yf
            spy = yf.Ticker("SPY")
            
            # Get date range from signals
            if sig is not None and not sig.empty:
                start_date = sig.index[0].strftime('%Y-%m-%d')
                end_date = sig.index[-1].strftime('%Y-%m-%d')
                
                # Fetch SPY data for the same period
                hist = spy.history(start=start_date, end=end_date)
                if not hist.empty:
                    # Align with equity curve dates
                    initial_price = hist['Close'].iloc[0]
                    initial_equity = bt['equity'][0] if bt['equity'] else 10000
                    
                    for i, (date, row) in enumerate(hist.iterrows()):
                        if i < len(equityData):
                            benchmark_return = (row['Close'] / initial_price - 1) * 100
                            benchmarkData.append({
                                'date': str(i),
                                'value': benchmark_return
                            })
                else:
                    # Fallback if no SPY data
                    for i in range(len(equityData)):
                        benchmarkData.append({
                            'date': str(i),
                            'value': i * 0.1  # Simple linear benchmark
                        })
            else:
                # Fallback to simple benchmark
                for i in range(len(equityData)):
                    benchmarkData.append({
                        'date': str(i),
                        'value': i * 0.1  # Simple linear benchmark
                    })
        except Exception as e:
            # Fallback to simple benchmark
            for i in range(len(equityData)):
                benchmarkData.append({
                    'date': str(i),
                    'value': i * 0.1  # Simple linear benchmark
                })

        # Calculate beta against SPY benchmark
        beta = 0.0
        try:
            if sig is not None and not sig.empty and len(bt.get('equity', [])) > 1:
                # Get SPY data for the same period
                spy = yf.Ticker("SPY")
                start_date = sig.index[0].strftime('%Y-%m-%d')
                end_date = sig.index[-1].strftime('%Y-%m-%d')
                spy_hist = spy.history(start=start_date, end=end_date)
                
                if not spy_hist.empty and len(spy_hist) > 1:
                    # Calculate strategy returns
                    equity_series = pd.Series(bt['equity'])
                    strategy_returns = equity_series.pct_change().dropna()
                    
                    # Calculate SPY returns
                    spy_returns = spy_hist['Close'].pct_change().dropna()
                    
                    # Align the series by date
                    min_length = min(len(strategy_returns), len(spy_returns))
                    if min_length > 1:
                        strategy_returns = strategy_returns.iloc[:min_length]
                        spy_returns = spy_returns.iloc[:min_length]
                        
                        # Calculate beta: covariance(strategy, market) / variance(market)
                        covariance = np.cov(strategy_returns, spy_returns)[0, 1]
                        market_variance = np.var(spy_returns)
                        
                        if market_variance > 0:
                            beta = float(covariance / market_variance)
        except Exception as e:
            logger.warning(f"Could not calculate beta: {str(e)}")
            beta = 0.0

        # Risk metrics
        risk = {
            'var': float(perf.get('var_95', 0) * 100),
            'beta': beta,
            'alpha': float(perf.get('annualized_return', 0) * 100),
            'volatility': float(perf.get('volatility', 0) * 100),
        }

        return jsonify({
            'totalReturn': total_return,
            'sharpeRatio': sharpe,
            'maxDrawdown': float(perf.get('max_drawdown', 0) * 100),
            'totalTrades': trades,
            'winRate': win_rate,
            'currentSignal': current_signal,
            'equityData': equityData,
            'benchmarkData': benchmarkData,
            'portfolioValue': float(bt['equity'][-1]),
            'dailyPnL': float(bt['pnl'][-1]) if len(bt.get('pnl', [])) else 0.0,
            'openPositions': int(bt.get('positions', np.array([0]))[-1] != 0),
            'riskMetrics': risk,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Run backtest with uploaded file."""
    try:
        logger.info("backtest:start")
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Initialize engine
            global engine
            engine = AlphaSignalEngine()
            
            # Run backtest
            results = engine.run_complete_analysis(
                csv_file_path=tmp_path,
                plot_results=False,
                save_plots=False
            )
            
            # Convert numpy types and format results for frontend
            trades_list = _compute_trades_list(results['backtest_results'], results['signals'])
            backtest_data = {
                'totalReturn': float(results['backtest_summary']['total_return'] * 100),
                'sharpeRatio': float(results['backtest_summary']['sharpe_ratio']),
                'maxDrawdown': float(results['backtest_summary']['max_drawdown'] * 100),
                'totalTrades': int(results['backtest_summary']['total_trades']),
                'winRate': float(results['performance_metrics']['win_rate'] * 100),
                'profitFactor': float(results['performance_metrics']['profit_factor']),
                'equityData': [
                    {'date': str(i), 'value': float(val)}
                    for i, val in enumerate(results['backtest_results']['equity'])
                ],
                'signals': [
                    {
                        'date': str(idx),
                        'signal': 'BUY' if signal == 1 else 'SELL' if signal == -1 else 'HOLD',
                        'price': float(price)
                    }
                    for idx, (signal, price) in enumerate(zip(
                        results['signals']['final_signal'],
                        results['signals']['Close']
                    ))
                    if signal != 0
                ][:10],  # Limit to first 10 signals
                'trades': trades_list[:10]
            }
            
            logger.info("backtest:success", extra={"total_trades": backtest_data['totalTrades']})
            return jsonify(backtest_data)
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        logger.exception("backtest:error")
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest/custom', methods=['POST'])
def run_custom_backtest():
    """Run backtest with custom stock symbol and time period."""
    try:
        logger.info("custom_backtest:start")
        data = request.json or {}
        
        # Extract parameters
        symbol = data.get('symbol', 'AAPL').upper()
        start_date = data.get('startDate', '2023-01-01')
        end_date = data.get('endDate', '2024-01-01')
        interval = data.get('interval', '1d')
        
        # Optional custom configuration
        custom_config = data.get('config', {})
        
        logger.info(f"Custom backtest for {symbol} from {start_date} to {end_date}")
        
        # Download stock data
        stock_data = download_stock_data(symbol, start_date, end_date, interval)
        
        # Save data temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            stock_data.to_csv(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Initialize engine with custom config if provided
            global engine
            if custom_config:
                config = Config(
                    initial_capital=custom_config.get('initialCapital', 100000.0),
                    position_size=custom_config.get('positionSize', 0.1),
                    momentum_lookback=custom_config.get('momentumLookback', 20),
                    momentum_threshold=custom_config.get('momentumThreshold', 0.02),
                    mean_reversion_lookback=custom_config.get('meanReversionLookback', 50),
                    mean_reversion_threshold=custom_config.get('meanReversionThreshold', 0.01),
                    mean_reversion_std_multiplier=custom_config.get('meanReversionStdMultiplier', 2.0),
                    transaction_cost_bps=custom_config.get('transactionCost', 1.0),
                    stop_loss_bps=custom_config.get('stopLoss', 50.0),
                    take_profit_bps=custom_config.get('takeProfit', 100.0)
                )
                engine = AlphaSignalEngine(config)
            else:
                engine = AlphaSignalEngine()
            
            # Run backtest
            results = engine.run_complete_analysis(
                csv_file_path=tmp_path,
                plot_results=False,
                save_plots=False
            )
            
            # Convert all numpy types and format results
            trades_list = _compute_trades_list(results['backtest_results'], results['signals'])
            backtest_data = {
                'symbol': symbol,
                'startDate': start_date,
                'endDate': end_date,
                'interval': interval,
                'dataPoints': len(stock_data),
                'totalReturn': float(results['backtest_summary']['total_return'] * 100),
                'sharpeRatio': float(results['backtest_summary']['sharpe_ratio']),
                'maxDrawdown': float(results['backtest_summary']['max_drawdown'] * 100),
                'totalTrades': int(results['backtest_summary']['total_trades']),
                'winRate': float(results['performance_metrics']['win_rate'] * 100),
                'profitFactor': float(results['performance_metrics']['profit_factor']),
                'sortinoRatio': float(results['performance_metrics'].get('sortino_ratio', 0)),
                'calmarRatio': float(results['performance_metrics'].get('calmar_ratio', 0)),
                'equityData': [
                    {
                        'date': stock_data.index[i].strftime('%Y-%m-%d') if hasattr(stock_data.index[i], 'strftime') else str(i),
                        'value': float(val)
                    }
                    for i, val in enumerate(results['backtest_results']['equity'])
                ],
                'signals': [
                    {
                        'date': stock_data.index[idx].strftime('%Y-%m-%d') if hasattr(stock_data.index[idx], 'strftime') else str(idx),
                        'signal': 'BUY' if signal == 1 else 'SELL' if signal == -1 else 'HOLD',
                        'price': float(price)
                    }
                    for idx, (signal, price) in enumerate(zip(
                        results['signals']['final_signal'],
                        results['signals']['Close']
                    ))
                    if signal != 0
                ][:20],  # Limit to first 20 signals
                'trades': trades_list[:10]
            }
            
            logger.info("custom_backtest:success", extra={
                "symbol": symbol,
                "total_trades": backtest_data['totalTrades'],
                "total_return": backtest_data['totalReturn']
            })
            return jsonify(backtest_data)
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        logger.exception("custom_backtest:error")
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimization', methods=['POST'])
def run_optimization():
    """Run parameter optimization."""
    try:
        logger.info("optimization:start")
        data = request.json
        param_ranges = data.get('paramRanges', {})
        
        # Initialize engine
        global engine
        engine = AlphaSignalEngine()
        
        # Run optimization
        csv_path = data.get('csvPath')
        temp_csv = None
        if not csv_path or not os.path.exists(csv_path):
            # Synthesize a small dataset for optimization if none provided
            try:
                synth_len = 500
                dates = pd.date_range(end=pd.Timestamp.today(), periods=synth_len, freq='D')
                prices = 100 + np.cumsum(np.random.normal(0, 1, synth_len))
                highs = prices * (1 + np.abs(np.random.normal(0, 0.005, synth_len)))
                lows = prices * (1 - np.abs(np.random.normal(0, 0.005, synth_len)))
                opens = prices * (1 + np.random.normal(0, 0.002, synth_len))
                vols = np.random.randint(100000, 500000, synth_len)
                df = pd.DataFrame({
                    'Datetime': dates,
                    'Open': opens,
                    'High': highs,
                    'Low': lows,
                    'Close': prices,
                    'Volume': vols,
                })
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmpf:
                    df.to_csv(tmpf.name, index=False)
                    temp_csv = tmpf.name
                csv_path = temp_csv
            except Exception:
                pass

        results = engine.optimize_parameters(
            param_ranges=param_ranges,
            csv_file_path=csv_path
        )
        
        # Format results for frontend
        optimization_data = {
            'bestParams': results['best_params'],
            'bestSharpe': results['best_sharpe'],
            'topResults': results['top_10_results'][:5]  # Top 5 results
        }
        
        logger.info("optimization:success", extra={"best_sharpe": optimization_data['bestSharpe']})
        return jsonify(optimization_data)
        
    except Exception as e:
        logger.exception("optimization:error")
        return jsonify({'error': str(e)}), 500

@app.route('/api/realtime/start', methods=['POST'])
def start_realtime_feed():
    """Start real-time data feed (Alpaca or simulated)."""
    try:
        data = request.json or {}
        symbol = data.get('symbol', 'AAPL').upper()
        provider = data.get('provider', 'yahoo')
        logger.info("realtime:start", extra={"symbol": symbol, "provider": provider})

        global ALPACA_FEED, ALPACA_SYMBOL
        ALPACA_SYMBOL = symbol

        if provider == 'alpaca':
            api_key = os.environ.get('ALPACA_API_KEY')
            secret_key = os.environ.get('ALPACA_SECRET_KEY')
            if not api_key or not secret_key:
                # Fall back to Yahoo polling if secret is unavailable
                provider = 'yahoo'

            try:
                from alpha_signal_engine.alpaca_feed import AlpacaDataFeed
            except Exception as e:
                logger.exception("realtime:alpaca_import:error")
                return jsonify({'error': f'Alpaca feed unavailable: {e}'}), 500

            if provider == 'alpaca':
                ALPACA_FEED = AlpacaDataFeed(api_key, secret_key, symbol)
                ALPACA_FEED.start()
        if provider == 'yahoo':
            try:
                from alpha_signal_engine.yahoo_feed import YahooPollingFeed
            except Exception as e:
                return jsonify({'error': f'Yahoo feed unavailable: {e}'}), 500
            ALPACA_FEED = YahooPollingFeed(symbol)
            ALPACA_FEED.start()
        elif provider == 'simulate':
            from alpha_signal_engine.realtime_feed import RealTimeDataFeed
            ALPACA_FEED = RealTimeDataFeed(Config(), symbol)
            ALPACA_FEED.start_feed()

        return jsonify({'status': 'started', 'symbol': symbol, 'provider': provider})

    except Exception as e:
        logger.exception("realtime:start:error")
        return jsonify({'error': str(e)}), 500

@app.route('/api/realtime/stop', methods=['POST'])
def stop_realtime_feed():
    """Stop real-time data feed."""
    try:
        logger.info("realtime:stop")
        global ALPACA_FEED, ALPACA_SYMBOL
        if ALPACA_FEED is not None:
            try:
                if hasattr(ALPACA_FEED, 'stop'):
                    ALPACA_FEED.stop()
                elif hasattr(ALPACA_FEED, 'stop_feed'):
                    ALPACA_FEED.stop_feed()
            finally:
                ALPACA_FEED = None
                ALPACA_SYMBOL = None
        return jsonify({
            'status': 'stopped',
            'message': 'Real-time feed stopped'
        })
        
    except Exception as e:
        logger.exception("realtime:stop:error")
        return jsonify({'error': str(e)}), 500

@app.route('/api/realtime/latest', methods=['GET'])
def realtime_latest():
    """Get the latest real-time tick from the active feed."""
    try:
        global ALPACA_FEED
        if ALPACA_FEED is None:
            return jsonify({'connected': False})

        # Build a richer payload that may include a simple live signal and confidence
        latest = None
        if hasattr(ALPACA_FEED, 'get_latest'):
            latest = ALPACA_FEED.get_latest()
        elif hasattr(ALPACA_FEED, 'get_current_data'):
            latest = ALPACA_FEED.get_current_data()

        if not latest:
            return jsonify({'connected': False})

        # Check if advanced mode is available
        advanced_info = {}
        if hasattr(ALPACA_FEED, 'get_advanced_mode_info'):
            advanced_info = ALPACA_FEED.get_advanced_mode_info()

        # Try to access indicators if available from simulated feed
        signal = None
        confidence = None
        try:
            # Prefer simulated RealTimeDataFeed indicators
            if hasattr(ALPACA_FEED, 'latest_indicators'):
                ind = getattr(ALPACA_FEED, 'latest_indicators') or {}
                sma_10 = ind.get('sma_10')
                sma_20 = ind.get('sma_20')
                price = (latest.get('price') if isinstance(latest, dict) else None) or ind.get('sma_10')
                if sma_10 is not None and sma_20 is not None:
                    if sma_10 > sma_20:
                        signal = 'BUY'
                    elif sma_10 < sma_20:
                        signal = 'SELL'
                    else:
                        signal = 'HOLD'
                    # Confidence as normalized SMA spread
                    spread = abs(sma_10 - sma_20)
                    denom = (sma_10 + sma_20) / 2 or 1.0
                    confidence = max(0.0, min(0.99, spread / denom * 10))
        except Exception:
            pass

        resp = {
            'connected': True,
            'tick': latest,
            'advanced_mode': advanced_info
        }
        if signal is not None:
            resp['signal'] = signal
        if confidence is not None:
            resp['confidence'] = confidence
        return jsonify(resp)

        return jsonify({'connected': False})
    except Exception as e:
        logger.exception("realtime:latest:error")
        return jsonify({'error': str(e)}), 500

@app.route('/api/realtime/advanced/enable', methods=['POST'])
def enable_advanced_mode():
    """Enable advanced ML-powered signal generation."""
    try:
        global ALPACA_FEED
        if ALPACA_FEED is None:
            return jsonify({'error': 'No active feed'}), 400
        
        data = request.json or {}
        model_path = data.get('model_path')
        
        if hasattr(ALPACA_FEED, 'enable_advanced_mode'):
            ALPACA_FEED.enable_advanced_mode(model_path)
            return jsonify({'status': 'success', 'message': 'Advanced mode enabled'})
        else:
            return jsonify({'error': 'Advanced mode not supported'}), 400
            
    except Exception as e:
        logger.exception("realtime:advanced:enable:error")
        return jsonify({'error': str(e)}), 500

@app.route('/api/realtime/advanced/disable', methods=['POST'])
def disable_advanced_mode():
    """Disable advanced mode and return to basic simulation."""
    try:
        global ALPACA_FEED
        if ALPACA_FEED is None:
            return jsonify({'error': 'No active feed'}), 400
        
        if hasattr(ALPACA_FEED, 'disable_advanced_mode'):
            ALPACA_FEED.disable_advanced_mode()
            return jsonify({'status': 'success', 'message': 'Advanced mode disabled'})
        else:
            return jsonify({'error': 'Advanced mode not supported'}), 400
            
    except Exception as e:
        logger.exception("realtime:advanced:disable:error")
        return jsonify({'error': str(e)}), 500

@app.route('/api/realtime/advanced/status', methods=['GET'])
def get_advanced_status():
    """Get advanced mode status and statistics."""
    try:
        global ALPACA_FEED
        if ALPACA_FEED is None:
            return jsonify({'error': 'No active feed'}), 400
        
        if hasattr(ALPACA_FEED, 'get_advanced_mode_info'):
            info = ALPACA_FEED.get_advanced_mode_info()
            return jsonify(info)
        else:
            return jsonify({'error': 'Advanced mode not supported'}), 400
            
    except Exception as e:
        logger.exception("realtime:advanced:status:error")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stocks/search', methods=['GET'])
def search_stocks():
    """Search for stock symbols."""
    try:
        query = request.args.get('q', '').upper()
        if not query or len(query) < 1:
            return jsonify({'stocks': []})
        
        # Common stock symbols for demo
        common_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'UBER', 'LYFT',
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'BND'
        ]
        
        # Filter stocks that match the query
        matching_stocks = [stock for stock in common_stocks if query in stock]
        
        return jsonify({'stocks': matching_stocks[:10], 'demo': True})
        
    except Exception as e:
        logger.exception("search_stocks:error")
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current settings."""
    try:
        settings = _load_settings()
        return jsonify(settings)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['POST'])
def save_settings():
    """Save settings."""
    try:
        settings = request.json or {}
        _save_settings(settings)
        return jsonify({'status': 'saved'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/download', methods=['POST'])
def download_results():
    """Download backtest results."""
    try:
        logger.info("download_results:start")
        data = request.json
        results_type = data.get('type', 'backtest')
        
        # Create temporary file with results
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            # If we have real results from last backtest, export them; otherwise, sample
            global engine
            df = None
            try:
                if engine is not None:
                    bt = engine.get_backtest_results()
                    sig = engine.get_signals()
                    if bt is not None:
                        df = pd.DataFrame({
                            'Equity': bt.get('equity', []),
                            'PnL': bt.get('pnl', []),
                            'Drawdown': bt.get('drawdown', []),
                        })
                        if isinstance(sig, pd.DataFrame) and 'Close' in sig.columns and 'final_signal' in sig.columns:
                            # Align sizes safely
                            min_len = min(len(df), len(sig))
                            df = df.iloc[:min_len]
                            df.insert(0, 'Close', sig['Close'].iloc[:min_len].values)
                            df.insert(1, 'Signal', sig['final_signal'].iloc[:min_len].values)
            except Exception:
                df = None

            if df is None or df.empty:
                df = pd.DataFrame({
                    'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
                    'Price': np.random.randn(100).cumsum() + 100,
                    'Signal': np.random.choice(['BUY', 'SELL', 'HOLD'], 100),
                    'PnL': np.random.randn(100)
                })

            df.to_csv(tmp_file.name, index=False)

            return send_file(
                tmp_file.name,
                as_attachment=True,
                download_name=f'{results_type}_results.csv',
                mimetype='text/csv'
            )
            
    except Exception as e:
        logger.exception("download_results:error")
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    try:
        pf = _load_portfolio()
        # Compute market values using latest Close from engine if possible
        prices = {}
        try:
            if engine is not None and isinstance(engine.get_signals(), pd.DataFrame):
                last_close = float(engine.get_signals()['Close'].iloc[-1])
                # Single-symbol engines won't map; leave empty
        except Exception:
            pass
        return jsonify(pf)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/add', methods=['POST'])
def add_position():
    try:
        data = request.json or {}
        symbol = (data.get('symbol') or '').upper()
        shares = float(data.get('shares') or 0)
        if not symbol or shares <= 0:
            return jsonify({'error': 'symbol and positive shares required'}), 400

        price = None
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period='1d')
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
        except Exception:
            pass
        if price is None:
            price = 0.0

        pf = _load_portfolio()
        # If exists, increase shares with new avg price (simple)
        for p in pf['positions']:
            if p['symbol'] == symbol:
                total_shares = p['shares'] + shares
                p['avgPrice'] = (p['avgPrice'] * p['shares'] + price * shares) / total_shares if total_shares > 0 else price
                p['shares'] = total_shares
                break
        else:
            pf['positions'].append({'symbol': symbol, 'shares': shares, 'avgPrice': price})
        _save_portfolio(pf)
        return jsonify({'status': 'ok', 'portfolio': pf})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/remove', methods=['POST'])
def remove_position():
    try:
        data = request.json or {}
        symbol = (data.get('symbol') or '').upper()
        if not symbol:
            return jsonify({'error': 'symbol required'}), 400
        pf = _load_portfolio()
        pf['positions'] = [p for p in pf['positions'] if p['symbol'] != symbol]
        _save_portfolio(pf)
        return jsonify({'status': 'ok', 'portfolio': pf})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/signals', methods=['GET'])
def get_signals():
    try:
        global engine
        if engine is None or engine.get_signals() is None:
            # Return empty signals data instead of error
            return jsonify({
                'current_signal': None,
                'recent_signals': [],
                'total_signals': 0
            })

        signals = engine.get_signals()
        if signals.empty:
            # Return empty signals data instead of error
            return jsonify({
                'current_signal': None,
                'recent_signals': [],
                'total_signals': 0
            })

        # Get recent signals (last 50)
        recent_signals = signals.tail(50)
        
        signals_data = []
        for idx, row in recent_signals.iterrows():
            signal_type = 'BUY' if row['final_signal'] > 0 else 'SELL' if row['final_signal'] < 0 else 'HOLD'
            confidence = row.get('ml_confidence', 0.5) if 'ml_confidence' in row else 0.5
            
            signals_data.append({
                'date': idx.strftime('%Y-%m-%d %H:%M:%S') if hasattr(idx, 'strftime') else str(idx),
                'signal': signal_type,
                'confidence': float(confidence),
                'price': float(row['Close']),
                'momentum_signal': float(row.get('momentum_signal', 0)),
                'mean_reversion_signal': float(row.get('mean_reversion_signal', 0)),
                'ml_signal': float(row.get('ml_signal', 0)),
                'market_regime': row.get('market_regime', 'unknown'),
                'rsi': float(row.get('rsi', 50)),
                'volume_ratio': float(row.get('volume_ratio', 1.0))
            })

        # Current signal analysis
        latest = recent_signals.iloc[-1]
        current_signal = {
            'signal': 'BUY' if latest['final_signal'] > 0 else 'SELL' if latest['final_signal'] < 0 else 'HOLD',
            'confidence': float(latest.get('ml_confidence', 0.5)),
            'price': float(latest['Close']),
            'momentum_score': float(latest.get('momentum_signal', 0)),
            'mean_reversion_score': float(latest.get('mean_reversion_signal', 0)),
            'ml_score': float(latest.get('ml_signal', 0)),
            'market_regime': latest.get('market_regime', 'unknown'),
            'rsi': float(latest.get('rsi', 50)),
            'volume_ratio': float(latest.get('volume_ratio', 1.0))
        }

        return jsonify({
            'current_signal': current_signal,
            'recent_signals': signals_data,
            'total_signals': len(signals_data)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/risk/metrics', methods=['GET'])
def risk_metrics():
    """Return last computed risk/performance metrics."""
    try:
        global engine
        if engine is None or engine.get_performance_metrics() is None:
            return jsonify({'error': 'No metrics available. Run a backtest first.'}), 400
        return jsonify(convert_numpy_types(engine.get_performance_metrics()))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/risk/plots', methods=['GET'])
def risk_plots():
    """Generate and send risk and trading plots as a zip."""
    try:
        global engine
        if engine is None or engine.get_backtest_results() is None:
            return jsonify({'error': 'No results available. Run a backtest first.'}), 400
        # Generate plots to temp files
        vis = Visualizer(engine.config)
        tmpdir = tempfile.mkdtemp()
        perf_png = os.path.join(tmpdir, 'performance.png')
        sig_png = os.path.join(tmpdir, 'signals.png')
        risk_png = os.path.join(tmpdir, 'risk.png')
        trade_png = os.path.join(tmpdir, 'trading.png')
        vis.plot_performance(engine.get_signals(), engine.get_backtest_results(), save_path=perf_png, show_plot=False)
        vis.plot_signal_analysis(engine.get_signals(), save_path=sig_png, show_plot=False)
        vis.plot_risk_metrics(engine.get_performance_metrics(), save_path=risk_png, show_plot=False)
        vis.plot_trading_activity(engine.get_signals(), engine.get_backtest_results(), save_path=trade_png, show_plot=False)
        # Zip them
        import zipfile
        zip_path = os.path.join(tmpdir, 'risk_plots.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(perf_png, arcname='performance.png')
            zf.write(sig_png, arcname='signals.png')
            zf.write(risk_png, arcname='risk.png')
            zf.write(trade_png, arcname='trading.png')
        return send_file(zip_path, as_attachment=True, download_name='risk_plots.zip')
    except Exception as e:
        logger.exception('risk_plots:error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get ML model feature importances from the last trained model."""
    try:
        global engine
        if engine is None or engine.get_signals() is None:
            return jsonify({'error': 'No ML model available. Run a backtest first.'}), 400
        
        # Get the advanced signal generator from the engine
        advanced_generator = engine.signal_generator.advanced_generator
        
        if advanced_generator.ml_model is None:
            return jsonify({'error': 'No trained ML model found.'}), 400
        
        # Get feature importances from the trained model
        feature_importances = advanced_generator.ml_model.feature_importances_
        
        # Get feature names from the last feature creation
        # We need to recreate the features to get the names
        signals = engine.get_signals()
        if signals is None or signals.empty:
            return jsonify({'error': 'No signals data available.'}), 400
        
        # Create features to get feature names
        features = advanced_generator._create_ml_features(signals)
        feature_names = features.columns.tolist()
        
        # Combine feature names with importances
        feature_importance_data = []
        for name, importance in zip(feature_names, feature_importances):
            feature_importance_data.append({
                'name': name,
                'importance': float(importance)
            })
        
        # Sort by importance (descending)
        feature_importance_data.sort(key=lambda x: x['importance'], reverse=True)
        
        # Return top 10 most important features
        return jsonify({
            'feature_importances': feature_importance_data[:10],
            'total_features': len(feature_names)
        })
        
    except Exception as e:
        logger.exception('feature_importance:error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml/evaluation', methods=['GET'])
def get_ml_evaluation():
    """Get ML model evaluation metrics including confusion matrix and market regime distribution."""
    try:
        global engine
        if engine is None or engine.get_signals() is None:
            return jsonify({'error': 'No ML data available. Run a backtest first.'}), 400
        
        signals = engine.get_signals()
        if signals is None or signals.empty:
            return jsonify({'error': 'No signals data available.'}), 400
        
        # Get market regime distribution
        regime_counts = signals['market_regime'].value_counts().to_dict()
        regime_distribution = []
        total_regimes = len(signals)
        
        for regime, count in regime_counts.items():
            regime_distribution.append({
                'regime': regime,
                'count': int(count),
                'percentage': float(count / total_regimes * 100)
            })
        
        # Calculate confusion matrix if we have ML predictions and actual outcomes
        confusion_matrix = None
        if 'ml_signal' in signals.columns and 'final_signal' in signals.columns:
            # Create binary predictions and actuals
            ml_predictions = (signals['ml_signal'] > 0).astype(int)
            actual_outcomes = (signals['final_signal'] > 0).astype(int)
            
            # Calculate confusion matrix components
            true_positives = int(((ml_predictions == 1) & (actual_outcomes == 1)).sum())
            false_positives = int(((ml_predictions == 1) & (actual_outcomes == 0)).sum())
            true_negatives = int(((ml_predictions == 0) & (actual_outcomes == 0)).sum())
            false_negatives = int(((ml_predictions == 0) & (actual_outcomes == 1)).sum())
            
            confusion_matrix = {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives,
                'accuracy': float((true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)) if (true_positives + false_positives + true_negatives + false_negatives) > 0 else 0.0,
                'precision': float(true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0.0,
                'recall': float(true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0.0
            }
        
        # Get ML confidence distribution
        confidence_distribution = []
        if 'ml_confidence' in signals.columns:
            confidences = signals['ml_confidence'].dropna()
            if len(confidences) > 0:
                # Create histogram bins
                bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
                hist, bin_edges = np.histogram(confidences, bins=bins)
                
                for i in range(len(hist)):
                    confidence_distribution.append({
                        'range': f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
                        'count': int(hist[i]),
                        'percentage': float(hist[i] / len(confidences) * 100)
                    })
        
        return jsonify({
            'market_regime_distribution': regime_distribution,
            'confusion_matrix': confusion_matrix,
            'confidence_distribution': confidence_distribution,
            'total_predictions': len(signals)
        })
        
    except Exception as e:
        logger.exception('ml_evaluation:error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/walk-forward', methods=['POST'])
def walk_forward_api():
    """Run walk-forward analysis on provided CSV path or last signals."""
    try:
        global walk_forward_optimizer
        payload = request.json or {}
        csv_path = payload.get('csvPath')
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            # Fallback to engine signals
            global engine
            if engine is None or engine.get_signals() is None:
                return jsonify({'error': 'No data available'}), 400
            df = engine.get_signals().reset_index()
        result = walk_forward_optimizer.walk_forward_backtest(df)
        return jsonify({
            'aggregated_sharpe': result.aggregated_sharpe,
            'aggregated_return': result.aggregated_return,
            'segments': [
                {
                    'start_index': s.start_index,
                    'end_index': s.end_index,
                    'params': s.params,
                    'sharpe_ratio': s.sharpe_ratio,
                    'total_return': s.total_return
                } for s in result.segments
            ]
        })
    except Exception as e:
        logger.exception('walk_forward:error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/factors/pca', methods=['POST'])
def factors_pca_api():
    """Run PCA factor strategy on a price matrix (symbols as columns)."""
    try:
        global factor_strategy
        if factor_strategy is None:
            factor_strategy = FactorStrategy(n_components=5)
        payload = request.json or {}
        price_matrix = pd.DataFrame(payload.get('price_matrix') or {})
        if price_matrix.empty:
            return jsonify({'error': 'price_matrix required'}), 400
        signals = factor_strategy.generate(price_matrix)
        return jsonify({
            'factor_momentum': signals.factor_momentum,
            'factor_mean_reversion': signals.factor_mean_reversion,
            'combined_signal': signals.combined_signal
        })
    except Exception as e:
        logger.exception('factors_pca:error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/rl/position-size', methods=['POST'])
def rl_position_size_api():
    """Get position size via RL agent (stubbed)."""
    try:
        global rl_position_sizer
        payload = request.json or {}
        market_state = np.array(payload.get('market_state', [0, 0, 0]), dtype=float)
        portfolio_state = np.array(payload.get('portfolio_state', [0, 0]), dtype=float)
        signal_strength = float(payload.get('signal_strength', 0.0))
        size = rl_position_sizer.get_position_size(market_state, portfolio_state, signal_strength)
        return jsonify({'position_size': size})
    except Exception as e:
        logger.exception('rl_position_size:error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/costs/estimate', methods=['POST'])
def costs_estimate_api():
    """Estimate transaction cost with regime-aware model."""
    try:
        global tx_cost_model
        payload = request.json or {}
        trade_size = float(payload.get('trade_size', 10000))
        market_conditions = payload.get('market_conditions', {})
        cost = tx_cost_model.calculate_transaction_cost(trade_size, market_conditions)
        return jsonify({'estimated_cost': cost})
    except Exception as e:
        logger.exception('costs_estimate:error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimize/bayesian', methods=['POST'])
def bayesian_optimize():
    """Run Bayesian Optimization for engine parameters."""
    try:
        global engine, bayesian_optimizer
        data = request.json or {}
        csv_path = data.get('csvPath')
        n_calls = int(data.get('nCalls', 50))
        n_initial = int(data.get('nInitialPoints', 10))

        if engine is None:
            engine = AlphaSignalEngine()

        bayesian_optimizer = BayesianOptimizer(engine, n_calls=n_calls, n_initial_points=n_initial)
        result = bayesian_optimizer.optimize(csv_file_path=csv_path)

        return jsonify({
            'bestParams': result.best_params,
            'bestSharpe': result.best_score,
            'history': result.optimization_history[:50],
            'convergence': result.convergence_plot_data,
            'paramImportance': result.parameter_importance
        })
    except Exception as e:
        logger.exception('bayesian_optimize:error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/mtf/<symbol>', methods=['GET'])
def multi_timeframe(symbol: str):
    """Multi-timeframe analysis for a symbol."""
    try:
        global multi_timeframe_strategy
        period = request.args.get('period', '1y')
        if multi_timeframe_strategy is None:
            multi_timeframe_strategy = MultiTimeframeStrategy()
        analysis = multi_timeframe_strategy.get_timeframe_analysis(symbol.upper(), period)
        return jsonify(analysis)
    except Exception as e:
        logger.exception('multi_timeframe:error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/risk/position-size', methods=['POST'])
def risk_position_size():
    """Compute risk-adjusted position size given inputs."""
    try:
        global advanced_risk_manager
        payload = request.json or {}
        signal_strength = float(payload.get('signal_strength', 0.0))
        current_portfolio = payload.get('current_portfolio', {})
        market_data = payload.get('market_data', {})
        trade_history = payload.get('trade_history', [])

        if advanced_risk_manager is None:
            advanced_risk_manager = AdvancedRiskManager()

        res = advanced_risk_manager.calculate_position_size(
            signal_strength=signal_strength,
            current_portfolio=current_portfolio,
            market_data=market_data,
            trade_history=trade_history
        )

        return jsonify({
            'position_size': res.position_size,
            'risk_adjusted_size': res.risk_adjusted_size,
            'kelly_size': res.kelly_size,
            'max_position_size': res.max_position_size,
            'risk_metrics': res.risk_metrics.__dict__,
            'sizing_factors': res.sizing_factors
        })
    except Exception as e:
        logger.exception('risk_position_size:error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/ensemble/fit', methods=['POST'])
def ensemble_fit():
    """Fit ensemble models using the engine's current signals."""
    try:
        global engine, ensemble_generator
        if engine is None or engine.get_signals() is None:
            return jsonify({'error': 'No signals available. Run a backtest first.'}), 400

        signals = engine.get_signals()
        if signals is None or signals.empty:
            return jsonify({'error': 'Signals are empty'}), 400

        if ensemble_generator is None:
            ensemble_generator = EnsembleSignalGenerator(engine.config)

        lookforward = int((request.json or {}).get('lookforward', 5))
        ensemble_generator.fit(signals, lookforward=lookforward)
        summary = ensemble_generator.get_ensemble_summary()
        return jsonify({'status': 'fitted', 'summary': summary})
    except Exception as e:
        logger.exception('ensemble_fit:error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/ensemble/predict', methods=['GET'])
def ensemble_predict():
    """Get current ensemble prediction based on latest signals."""
    try:
        global engine, ensemble_generator
        if ensemble_generator is None:
            return jsonify({'error': 'Ensemble not fitted yet'}), 400

        signals = engine.get_signals() if engine is not None else None
        if signals is None or signals.empty:
            return jsonify({'error': 'No signals available'}), 400

        result = ensemble_generator.predict(signals)
        return jsonify({
            'final_prediction': result.final_prediction,
            'base_predictions': result.base_predictions,
            'meta_prediction': result.meta_prediction,
            'confidence': result.confidence,
            'model_weights': result.model_weights,
            'feature_importance': result.feature_importance
        })
    except Exception as e:
        logger.exception('ensemble_predict:error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance/attribution', methods=['POST'])
def performance_attribution_api():
    """Compute performance attribution and advanced metrics."""
    try:
        global performance_attributor
        payload = request.json or {}
        portfolio_returns = pd.Series(payload.get('portfolio_returns', []))
        factor_exposures = {k: pd.Series(v) for k, v in (payload.get('factor_exposures', {}) or {}).items()}
        factor_returns = {k: pd.Series(v) for k, v in (payload.get('factor_returns', {}) or {}).items()}
        benchmark_returns = pd.Series(payload.get('benchmark_returns', [])) if payload.get('benchmark_returns') else None

        analyzer = PerformanceAttributor()
        result = analyzer.attribute_returns(
            portfolio_returns=portfolio_returns,
            factor_exposures=factor_exposures,
            factor_returns=factor_returns,
            benchmark_returns=benchmark_returns
        )

        metrics = AdvancedMetricsCalculator.calculate_all_metrics(portfolio_returns)

        return jsonify({
            'attribution': {
                'total_return': result.total_return,
                'factor_attribution': result.factor_attribution,
                'alpha': result.alpha,
                'beta': result.beta,
                'information_coefficient': result.information_coefficient,
                'hit_rate': result.hit_rate,
                'factor_exposures': result.factor_exposures,
                'factor_returns': result.factor_returns
            },
            'advanced_metrics': metrics.__dict__
        })
    except Exception as e:
        logger.exception('performance_attribution:error')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

