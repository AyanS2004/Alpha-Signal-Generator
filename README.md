# Alpha Signal Engine

Production-ready signal research and backtesting stack with web dashboard, CLI, API, and Alpaca real-time integration.

## Highlights
- Data ingest and preprocessing (OHLCV)
- Signal generation: momentum + mean-reversion
- Numba-accelerated backtesting with realistic costs
- Performance analytics (Sharpe, drawdown, trade stats)
- Real-time quotes via Alpaca (or simulation fallback)
- Web dashboard (React + MUI) and REST API (Flask)
- Dockerized with CI

---

## Quickstart

### A) Docker (recommended)
1) Optionally set Alpaca credentials for live quotes (both required):
   - PowerShell:
     - `$env:ALPACA_API_KEY="YOUR_KEY"`
     - `$env:ALPACA_SECRET_KEY="YOUR_SECRET"`
2) Start
```bash
docker compose up --build
```
3) Open
- Frontend: http://localhost:3000
- Backend:  http://localhost:5000

Without both Alpaca vars, Real-Time will remain disconnected (no fake ticks).

### B) Local dev
```bash
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\pip install -r backend\requirements.txt

# Optional: set Alpaca creds for live quotes
$env:ALPACA_API_KEY="YOUR_KEY"
$env:ALPACA_SECRET_KEY="YOUR_SECRET"

python start_app.py
```

---

## Dashboard
- Backtest CSV via `POST /api/backtest`, view KPIs and equity curve
- Optimization via `POST /api/optimization`
- Real-Time: enter symbol and Start Feed → backend connects to Alpaca, UI polls `GET /api/realtime/latest`

CSV columns: `Datetime,Open,High,Low,Close,Volume`. See `AAPL_minute.csv`.

---

## Real-time (Alpaca)
If you only have an API key and no Alpaca secret, the backend will automatically fall back to Yahoo polling (no keys required). This provides delayed quotes suitable for demos.

Endpoints
- `POST /api/realtime/start` body `{ symbol: "AAPL", provider: "yahoo" }` (no key needed)
- `POST /api/realtime/start` body `{ symbol: "AAPL", provider: "alpaca" }` (requires API + SECRET)
- `GET /api/realtime/latest`
- `POST /api/realtime/stop`

Python
```python
import os
from alpha_signal_engine.alpaca_feed import AlpacaDataFeed

os.environ['ALPACA_API_KEY'] = 'YOUR_KEY'
os.environ['ALPACA_SECRET_KEY'] = 'YOUR_SECRET'

feed = AlpacaDataFeed(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_SECRET_KEY'], symbol='AAPL')
feed.start(on_data=lambda d: print('trade', d))
feed.stop()
```

---

## CLI
```bash
alpha-signal run --csv AAPL_minute.csv --no-plot --save-plots
alpha-signal optimize --csv AAPL_minute.csv --momentum-lookback 10 20 30 --position-size 0.05 0.1
```

---

## API
- `GET /api/health`
- `POST /api/backtest` (multipart file `file`)
- `POST /api/backtest/custom` (JSON: `{symbol, startDate, endDate, interval?, config?}`)
- `GET /api/stocks/search?q=<query>`
- `POST /api/optimization`
- `POST /api/realtime/start`
- `GET /api/realtime/latest`
- `POST /api/realtime/stop`
- `GET /api/settings`, `POST /api/settings`
- `POST /api/data/download`

---

## Python

### Basic Usage
```python
from alpha_signal_engine import AlphaSignalEngine, Config

engine = AlphaSignalEngine()
results = engine.run_complete_analysis(csv_file_path='AAPL_minute.csv', plot_results=False)
print(results['backtest_summary']['sharpe_ratio'])
```

### Custom Backtesting
```python
# Using the custom backtesting script
python custom_backtest.py --symbol AAPL --start-date 2023-01-01 --end-date 2024-01-01

# With custom configuration
python custom_backtest.py --symbol TSLA --start-date 2022-01-01 --end-date 2023-12-31 --config sample_config.json

# List available symbols
python custom_backtest.py --list-symbols
```

### API Usage
```python
import requests

# Custom backtest via API
response = requests.post('http://localhost:5000/api/backtest/custom', json={
    'symbol': 'AAPL',
    'startDate': '2023-01-01',
    'endDate': '2024-01-01',
    'config': {
        'initialCapital': 100000,
        'positionSize': 0.15,
        'momentumLookback': 25
    }
})
results = response.json()
print(f"Total Return: {results['totalReturn']:.2f}%")
```

---

## Tests & CI
```bash
pip install -e .[dev]
pytest -q
```
CI workflow runs lint and tests on push/PR.

---

## Structure
```
alpha_signal_engine/  # Core library
backend/              # Flask API
frontend/             # React dashboard
docker-compose.yml    # Orchestration
pyproject.toml        # Packaging, CLI
tests/                # Smoke tests
```

---

## Troubleshooting
- Real-time not connected: set BOTH `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`.
- Ports 3000/5000 busy: stop other services or change ports.
- First backtest slow: Numba JIT compiles on first call.

---

