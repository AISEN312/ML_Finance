# NIFTY 50 Direction Prediction — Data Bundle

## Files

- `nifty50.csv` — NIFTY 50 daily OHLCV, 2022-01-01 to 2025-12-31
- `banknifty.csv` — Bank Nifty daily OHLCV, same window
- `indiavix.csv` — India VIX daily close (OHLC), same window
- `starter_features.csv` — approximately 30 candidate features, aligned to NIFTY 50 trading dates

## Source

All data sourced from Yahoo Finance via the yfinance library. Dates are NSE trading days only.

## Notes

The `starter_features.csv` file is provided as a convenience. It has not been rigorously audited. You are free to use it as-is, modify it, drop columns, or ignore it entirely and build your own features from the raw OHLCV files. Your submission should make clear which features you used and why.

Column names in `starter_features.csv` are descriptive but not exhaustively documented. Assume standard conventions (rolling windows are trailing, returns are simple percentage changes, cross-asset columns are forward-filled over missing days) unless the behaviour of a column suggests otherwise.

## Target

The prediction target is NIFTY 50 next-day close direction: 1 if `close[T+1] > close[T]`, 0 otherwise. Design choices around flat or near-zero returns are yours to make and defend in the report.
