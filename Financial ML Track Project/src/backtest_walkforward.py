"""
Backtest the walk-forward model predictions.

This script turns the saved walk-forward predictions into a simple long/flat
strategy on NIFTY 50 close-to-close returns and applies transaction costs on
position changes.

Outputs:
    data/processed/walkforward_backtest.csv
    data/processed/walkforward_backtest_summary.json
    data/processed/walkforward_threshold_sweep.csv
    data/processed/walkforward_threshold_sweep.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


BASE_DIR = Path(".")
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "intern_data"


def load_predictions() -> pd.DataFrame:
    predictions_path = PROCESSED_DIR / "walkforward_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

    predictions = pd.read_csv(predictions_path, parse_dates=["date"])
    if predictions.empty:
        raise ValueError("walkforward_predictions.csv is empty")

    # Keep the latest prediction per date if a date ever appears in multiple folds.
    predictions = predictions.sort_values(["date", "fold", "prob_up"]).drop_duplicates(subset=["date"], keep="last")
    return predictions.sort_values("date").reset_index(drop=True)


def load_market_returns() -> pd.DataFrame:
    nifty_path = RAW_DIR / "nifty50.csv"
    if not nifty_path.exists():
        raise FileNotFoundError(f"Missing market data file: {nifty_path}")

    market = pd.read_csv(nifty_path, parse_dates=["date"]) 
    if "Close" not in market.columns:
        raise ValueError("nifty50.csv must contain a Close column")

    market = market.sort_values("date").reset_index(drop=True)
    market["next_day_return"] = market["Close"].shift(-1) / market["Close"] - 1.0
    return market[["date", "next_day_return"]]


def compute_backtest(predictions: pd.DataFrame, market_returns: pd.DataFrame, threshold: float = 0.5, transaction_cost_bps: float = 10.0) -> pd.DataFrame:
    backtest = predictions.merge(market_returns, on="date", how="left")
    backtest = backtest.dropna(subset=["next_day_return"]).copy()

    backtest["signal"] = (backtest["prob_up"] >= threshold).astype(int)
    backtest["position_change"] = backtest["signal"].diff().abs().fillna(backtest["signal"])
    backtest["transaction_cost"] = backtest["position_change"] * (transaction_cost_bps / 10000.0)
    backtest["strategy_return"] = backtest["signal"] * backtest["next_day_return"] - backtest["transaction_cost"]
    backtest["equity_curve"] = (1.0 + backtest["strategy_return"]).cumprod()
    backtest["buy_hold_curve"] = (1.0 + backtest["next_day_return"]).cumprod()
    backtest["benchmark_return"] = backtest["next_day_return"]

    rolling_max = backtest["equity_curve"].cummax()
    backtest["drawdown"] = backtest["equity_curve"] / rolling_max - 1.0

    return backtest


def summarize_backtest(backtest: pd.DataFrame, threshold: float, transaction_cost_bps: float) -> dict:
    total_return = backtest["equity_curve"].iloc[-1] - 1.0
    benchmark_return = backtest["buy_hold_curve"].iloc[-1] - 1.0
    avg_daily_return = backtest["strategy_return"].mean()
    daily_vol = backtest["strategy_return"].std()
    sharpe = (avg_daily_return / daily_vol) * (252 ** 0.5) if daily_vol and daily_vol > 0 else float("nan")
    win_rate = (backtest["strategy_return"] > 0).mean()
    max_drawdown = backtest["drawdown"].min()
    traded_days = int((backtest["signal"] > 0).sum())
    turnover = float(backtest["position_change"].sum())

    return {
        "start_date": backtest["date"].min().date().isoformat(),
        "end_date": backtest["date"].max().date().isoformat(),
        "observations": int(len(backtest)),
        "threshold": float(threshold),
        "transaction_cost_bps": float(transaction_cost_bps),
        "total_return": float(total_return),
        "benchmark_return": float(benchmark_return),
        "excess_return": float(total_return - benchmark_return),
        "sharpe_ratio": float(sharpe),
        "win_rate": float(win_rate),
        "max_drawdown": float(max_drawdown),
        "traded_days": traded_days,
        "turnover": turnover,
    }


def sweep_thresholds(predictions: pd.DataFrame, market_returns: pd.DataFrame, thresholds: list[float], transaction_cost_bps: float = 10.0) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        backtest = compute_backtest(predictions, market_returns, threshold=threshold, transaction_cost_bps=transaction_cost_bps)
        if backtest.empty:
            continue

        summary = summarize_backtest(backtest, threshold=threshold, transaction_cost_bps=transaction_cost_bps)
        rows.append(summary)

    return pd.DataFrame(rows).sort_values(["sharpe_ratio", "total_return"], ascending=False).reset_index(drop=True)


def main() -> None:
    predictions = load_predictions()
    market_returns = load_market_returns()
    threshold = 0.5
    transaction_cost_bps = 10.0

    backtest = compute_backtest(predictions, market_returns, threshold=threshold, transaction_cost_bps=transaction_cost_bps)

    if backtest.empty:
        raise ValueError("No overlapping prediction and return dates available for backtesting")

    summary = summarize_backtest(backtest, threshold=threshold, transaction_cost_bps=transaction_cost_bps)

    threshold_grid = [round(x / 100, 2) for x in range(30, 71, 5)]
    sweep_df = sweep_thresholds(predictions, market_returns, thresholds=threshold_grid, transaction_cost_bps=transaction_cost_bps)
    if sweep_df.empty:
        raise ValueError("Threshold sweep produced no results")

    best_row = sweep_df.iloc[0].to_dict()

    backtest_path = PROCESSED_DIR / "walkforward_backtest.csv"
    summary_path = PROCESSED_DIR / "walkforward_backtest_summary.json"
    sweep_path = PROCESSED_DIR / "walkforward_threshold_sweep.csv"
    sweep_summary_path = PROCESSED_DIR / "walkforward_threshold_sweep.json"

    backtest.to_csv(backtest_path, index=False)
    sweep_df.to_csv(sweep_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(sweep_summary_path, "w", encoding="utf-8") as handle:
        json.dump(best_row, handle, indent=2)

    logger.info("Saved backtest curve to %s", backtest_path)
    logger.info("Saved backtest summary to %s", summary_path)
    logger.info("Saved threshold sweep to %s", sweep_path)
    logger.info("Saved best-threshold summary to %s", sweep_summary_path)
    logger.info(
        "Total return=%.3f benchmark=%.3f sharpe=%.3f max_drawdown=%.3f win_rate=%.3f",
        summary["total_return"],
        summary["benchmark_return"],
        summary["sharpe_ratio"],
        summary["max_drawdown"],
        summary["win_rate"],
    )
    logger.info(
        "Best threshold=%.2f total_return=%.3f sharpe=%.3f win_rate=%.3f",
        float(best_row["threshold"]),
        float(best_row["total_return"]),
        float(best_row["sharpe_ratio"]),
        float(best_row["win_rate"]),
    )


if __name__ == "__main__":
    main()
