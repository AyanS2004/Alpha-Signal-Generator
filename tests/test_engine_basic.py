import os
from alpha_signal_engine import AlphaSignalEngine


def test_run_analysis_smoke(tmp_path):
    # Use sample CSV if available, otherwise synthesize a tiny one
    csv_path = os.path.join(os.path.dirname(__file__), "..", "AAPL_minute.csv")
    if not os.path.exists(csv_path):
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w") as f:
            f.write("Datetime,Open,High,Low,Close,Volume\n")
            for i in range(100):
                f.write(f"2024-01-01 09:{i:02d}:00,100,101,99,{100+i*0.01},100000\n")

    engine = AlphaSignalEngine()
    results = engine.run_complete_analysis(csv_file_path=str(csv_path), plot_results=False)

    assert "backtest_summary" in results
    assert "signal_summary" in results
    assert results["backtest_summary"]["total_trades"] >= 0


