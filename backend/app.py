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
        'version': '1.0.0'
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
            ]
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
                ][:10]  # Limit to first 10 signals
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
                'trades': [
                    {
                        'entryDate': str(trade.get('entry_date', '')),
                        'exitDate': str(trade.get('exit_date', '')),
                        'entryPrice': float(trade.get('entry_price', 0)),
                        'exitPrice': float(trade.get('exit_price', 0)),
                        'pnl': float(trade.get('pnl', 0)),
                        'return': float(trade.get('return', 0))
                    }
                    for trade in results.get('trades', [])
                ][:10]  # Limit to first 10 trades
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
        results = engine.optimize_parameters(
            param_ranges=param_ranges,
            csv_file_path=data.get('csvPath', 'test_data.csv')
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

        if hasattr(ALPACA_FEED, 'get_latest'):
            tick = ALPACA_FEED.get_latest()
            return jsonify({'connected': tick is not None, 'tick': tick})
        elif hasattr(ALPACA_FEED, 'get_current_data'):
            data = ALPACA_FEED.get_current_data()
            return jsonify({'connected': data.get('is_connected', False), 'tick': data})

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
        
        return jsonify({'stocks': matching_stocks[:10]})
        
    except Exception as e:
        logger.exception("search_stocks:error")
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current settings."""
    try:
        # Return default settings
        settings = {
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
            'dataProvider': 'alpha_vantage'
        }
        
        return jsonify(settings)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['POST'])
def save_settings():
    """Save settings."""
    try:
        settings = request.json
        
        # Here you would typically save to database or config file
        # For now, just return success
        
        return jsonify({
            'status': 'saved',
            'message': 'Settings saved successfully'
        })
        
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
            # Generate sample data
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

