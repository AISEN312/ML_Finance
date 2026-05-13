"""
Walk-forward split generation for time-series cross-validation.

Generates train/test date ranges and saves splits to
`data/processed/splits/` as CSV files listing train and test dates.

Usage:
    python src/walkforward.py

Defaults: 4-month train, 1-month test, 1-month step
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WalkForwardSplitter:
    def __init__(self, dates: pd.Series):
        self.dates = pd.to_datetime(dates).sort_values().reset_index(drop=True)

    def generate_splits(self, train_months: int = 4, test_months: int = 1, step_months: int = 1) -> List[Dict]:
        """Generate walk-forward splits.

        Returns a list of dicts with keys: train_start, train_end, test_start, test_end,
        train_idx, test_idx
        """
        splits = []
        min_date = self.dates.min()
        max_date = self.dates.max()

        cur_train_start = min_date
        fold = 0
        while True:
            train_end = cur_train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
            test_start = train_end + pd.Timedelta(days=1)
            test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

            if test_end > max_date:
                break

            # get indices
            train_idx = self.dates[(self.dates >= cur_train_start) & (self.dates <= train_end)].index.tolist()
            test_idx = self.dates[(self.dates >= test_start) & (self.dates <= test_end)].index.tolist()

            if len(train_idx) == 0 or len(test_idx) == 0:
                # advance window
                cur_train_start = cur_train_start + pd.DateOffset(months=step_months)
                continue

            splits.append({
                'fold': fold,
                'train_start': cur_train_start.date(),
                'train_end': train_end.date(),
                'test_start': test_start.date(),
                'test_end': test_end.date(),
                'train_idx': train_idx,
                'test_idx': test_idx
            })

            fold += 1
            # advance window by step_months
            cur_train_start = cur_train_start + pd.DateOffset(months=step_months)

        return splits

    def save_splits(self, splits: List[Dict], output_dir: str):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for s in splits:
            fold = s['fold']
            df_train = pd.DataFrame({'date': self.dates.iloc[s['train_idx']].dt.strftime('%Y-%m-%d')})
            df_test = pd.DataFrame({'date': self.dates.iloc[s['test_idx']].dt.strftime('%Y-%m-%d')})

            df_train.to_csv(out / f'fold_{fold}_train_dates.csv', index=False)
            df_test.to_csv(out / f'fold_{fold}_test_dates.csv', index=False)

        logger.info(f"Saved {len(splits)} splits to {out}")


def get_index_splits(
    dates: pd.Series,
    train_months: int = 4,
    test_months: int = 1,
    step_months: int = 1,
) -> List[Dict]:
    splitter = WalkForwardSplitter(dates)
    splits = splitter.generate_splits(
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
    )
    return [(split['train_idx'], split['test_idx']) for split in splits]


def main():
    processed_dates = Path('data/processed/dates.csv')
    if not processed_dates.exists():
        logger.error("data/processed/dates.csv not found. Run data processor first.")
        return

    dates = pd.read_csv(processed_dates, parse_dates=['date'])['date']

    splitter = WalkForwardSplitter(dates)
    splits = splitter.generate_splits(train_months=4, test_months=1, step_months=1)

    if not splits:
        logger.warning("No splits generated with the given parameters")
        return

    # Print summary
    logger.info(f"Generated {len(splits)} folds")
    for s in splits:
        logger.info(f"Fold {s['fold']}: Train {s['train_start']} -> {s['train_end']}, Test {s['test_start']} -> {s['test_end']} (train {len(s['train_idx'])} days, test {len(s['test_idx'])} days)")

    # Save splits
    splitter.save_splits(splits, 'data/processed/splits')


if __name__ == '__main__':
    main()
