#!/usr/bin/env python3
"""
Command Line Interface for Alpha Signal Engine.

Usage examples:
  alpha-signal run --csv AAPL_minute.csv --no-plot --save-plots
  alpha-signal optimize --csv AAPL_minute.csv --momentum-lookback 10 20 30 --position-size 0.05 0.1

Outputs JSON summaries to stdout for easy piping.
"""

from __future__ import annotations

import argparse
import json
from typing import List, Dict

from .engine import AlphaSignalEngine
from .config import Config


def _parse_float_list(values: List[str]) -> List[float]:
    return [float(v) for v in values]


def _parse_int_list(values: List[str]) -> List[int]:
    return [int(v) for v in values]


def cmd_run(args: argparse.Namespace) -> int:
    config = Config(
        initial_capital=args.initial_capital,
        position_size=args.position_size,
        transaction_cost_bps=args.transaction_cost_bps,
        momentum_lookback=args.momentum_lookback,
        momentum_threshold=args.momentum_threshold,
        mean_reversion_lookback=args.mean_reversion_lookback,
        mean_reversion_std_multiplier=args.mean_reversion_std_multiplier,
        stop_loss_bps=args.stop_loss_bps,
        take_profit_bps=args.take_profit_bps,
        use_numba=not args.no_numba,
    )

    engine = AlphaSignalEngine(config)
    results = engine.run_complete_analysis(
        csv_file_path=args.csv,
        plot_results=not args.no_plot,
        save_plots=args.save_plots,
    )

    # Produce compact JSON summary
    summary: Dict = {
        "data_summary": {
            "total_periods": int(results["data_summary"]["total_periods"]),
            "start_date": str(results["data_summary"]["start_date"]),
            "end_date": str(results["data_summary"]["end_date"]),
        },
        "signal_summary": results["signal_summary"],
        "backtest_summary": results["backtest_summary"],
        "performance_metrics": results["performance_metrics"],
    }
    print(json.dumps(summary, default=float))
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    config = Config(use_numba=not args.no_numba)
    engine = AlphaSignalEngine(config)

    param_ranges: Dict[str, List] = {}
    if args.momentum_lookback:
        param_ranges["momentum_lookback"] = _parse_int_list(args.momentum_lookback)
    if args.momentum_threshold:
        param_ranges["momentum_threshold"] = _parse_float_list(args.momentum_threshold)
    if args.position_size:
        param_ranges["position_size"] = _parse_float_list(args.position_size)

    results = engine.optimize_parameters(param_ranges=param_ranges, csv_file_path=args.csv)
    print(json.dumps(results, default=float))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alpha-signal", description="Alpha Signal Engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    run_p = subparsers.add_parser("run", help="Run full analysis on a CSV")
    run_p.add_argument("--csv", required=True, help="Path to OHLCV CSV")
    run_p.add_argument("--no-plot", action="store_true", help="Disable plotting")
    run_p.add_argument("--save-plots", action="store_true", help="Save plots to files")
    run_p.add_argument("--no-numba", action="store_true", help="Disable Numba acceleration")
    run_p.add_argument("--initial-capital", type=float, default=100000.0)
    run_p.add_argument("--position-size", type=float, default=0.1)
    run_p.add_argument("--transaction-cost-bps", type=float, default=1.0)
    run_p.add_argument("--momentum-lookback", type=int, default=20)
    run_p.add_argument("--momentum-threshold", type=float, default=0.02)
    run_p.add_argument("--mean-reversion-lookback", type=int, default=50)
    run_p.add_argument("--mean-reversion-std-multiplier", type=float, default=2.0)
    run_p.add_argument("--stop-loss-bps", type=float, default=50.0)
    run_p.add_argument("--take-profit-bps", type=float, default=100.0)
    run_p.set_defaults(func=cmd_run)

    # optimize command
    opt_p = subparsers.add_parser("optimize", help="Grid search over parameter ranges")
    opt_p.add_argument("--csv", required=True, help="Path to OHLCV CSV")
    opt_p.add_argument("--momentum-lookback", nargs="*", help="List of ints, e.g. 10 20 30")
    opt_p.add_argument("--momentum-threshold", nargs="*", help="List of floats, e.g. 0.01 0.02")
    opt_p.add_argument("--position-size", nargs="*", help="List of floats, e.g. 0.05 0.1")
    opt_p.add_argument("--no-numba", action="store_true", help="Disable Numba acceleration")
    opt_p.set_defaults(func=cmd_optimize)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())


