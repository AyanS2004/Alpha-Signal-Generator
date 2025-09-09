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

from alpha_signal_engine import AlphaSignalEngine, Config
from alpha_signal_engine.visualizer import Visualizer
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

# Settings file
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'settings.json')

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
        # Simulate dashboard data
        data = {
            'totalReturn': 15.7,
            'sharpeRatio': 1.23,
            'maxDrawdown': -8.5,
            'totalTrades': 156,
            'winRate': 62.5,
            'currentSignal': 'BUY',
            'equityData': [
                {'date': 'Jan', 'value': 100},
                {'date': 'Feb', 'value': 105},
                {'date': 'Mar', 'value': 110},
                {'date': 'Apr', 'value': 108},
                {'date': 'May', 'value': 115},
                {'date': 'Jun', 'value': 120},
            ],
            'demo': True
        }
        return jsonify(data)
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

