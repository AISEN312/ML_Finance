"""
Walk-forward model training for the processed financial dataset.

Loads the processed feature matrix, target, and dates, then evaluates a
lightweight classification model across walk-forward splits.

Outputs:
    data/processed/walkforward_metrics.csv
    data/processed/walkforward_predictions.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.walkforward import get_index_splits
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from walkforward import get_index_splits

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


PROCESSED_DIR = Path("data/processed")
MLFLOW_EXPERIMENT_NAME = "nifty50_walkforward_training"


def load_processed_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    features = pd.read_csv(PROCESSED_DIR / "X_features.csv")
    target = pd.read_csv(PROCESSED_DIR / "y_target.csv").iloc[:, 0]
    dates = pd.read_csv(PROCESSED_DIR / "dates.csv", parse_dates=[0]).iloc[:, 0]

    if len(features) != len(target) or len(features) != len(dates):
        raise ValueError("Processed feature, target, and date files must have the same number of rows")

    return features, target, dates


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def run_walk_forward(train_months: int = 4, test_months: int = 1, step_months: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    features, target, dates = load_processed_data()
    splits = get_index_splits(dates, train_months=train_months, test_months=test_months, step_months=step_months)

    if not splits:
        raise ValueError("No walk-forward splits were generated")

    metrics_rows = []
    prediction_rows = []

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    parent_run_ctx = mlflow.start_run(run_name="walkforward_training") if MLFLOW_AVAILABLE else None

    try:
        for fold, (train_idx, test_idx) in enumerate(splits):
            X_train = features.iloc[train_idx]
            y_train = target.iloc[train_idx]
            X_test = features.iloc[test_idx]
            y_test = target.iloc[test_idx]
            test_dates = dates.iloc[test_idx]

            model = build_model()

            if MLFLOW_AVAILABLE:
                with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                    mlflow.log_params(
                        {
                            "fold": fold,
                            "train_months": train_months,
                            "test_months": test_months,
                            "step_months": step_months,
                            "train_samples": len(train_idx),
                            "test_samples": len(test_idx),
                            "model": "LogisticRegression",
                            "max_iter": 1000,
                            "scaler": "StandardScaler",
                        }
                    )
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(y_test, y_prob) if len(pd.unique(y_test)) > 1 else float("nan")
            fold_metrics = {
                "fold": fold,
                "train_samples": len(train_idx),
                "test_samples": len(test_idx),
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc,
            }
            metrics_rows.append(fold_metrics)

            if MLFLOW_AVAILABLE:
                mlflow.log_metrics({k: v for k, v in fold_metrics.items() if k not in {"fold", "train_samples", "test_samples"} and pd.notna(v)})
                mlflow.sklearn.log_model(model, artifact_path="model")

            for row_idx, actual, pred, prob, date in zip(test_idx, y_test, y_pred, y_prob, test_dates):
                prediction_rows.append(
                    {
                        "fold": fold,
                        "row_index": row_idx,
                        "date": date,
                        "actual": int(actual),
                        "predicted": int(pred),
                        "prob_up": float(prob),
                    }
                )

            logger.info(
                "Fold %s: accuracy=%.3f precision=%.3f recall=%.3f f1=%.3f roc_auc=%s",
                fold,
                fold_metrics["accuracy"],
                fold_metrics["precision"],
                fold_metrics["recall"],
                fold_metrics["f1"],
                "nan" if pd.isna(fold_metrics["roc_auc"]) else f"{fold_metrics['roc_auc']:.3f}",
            )

        metrics_df = pd.DataFrame(metrics_rows)
        predictions_df = pd.DataFrame(prediction_rows)

        if MLFLOW_AVAILABLE:
            mlflow.log_params(
                {
                    "fold_count": len(metrics_df),
                    "average_train_samples": float(metrics_df["train_samples"].mean()),
                    "average_test_samples": float(metrics_df["test_samples"].mean()),
                }
            )
            mlflow.log_metrics(
                {
                    "avg_accuracy": float(metrics_df["accuracy"].mean()),
                    "avg_precision": float(metrics_df["precision"].mean()),
                    "avg_recall": float(metrics_df["recall"].mean()),
                    "avg_f1": float(metrics_df["f1"].mean()),
                    "avg_roc_auc": float(metrics_df["roc_auc"].mean(skipna=True)),
                }
            )
            metrics_df.to_csv(PROCESSED_DIR / "walkforward_metrics.csv", index=False)
            predictions_df.to_csv(PROCESSED_DIR / "walkforward_predictions.csv", index=False)
            mlflow.log_artifact(str(PROCESSED_DIR / "walkforward_metrics.csv"))
            mlflow.log_artifact(str(PROCESSED_DIR / "walkforward_predictions.csv"))

    finally:
        if parent_run_ctx is not None:
            parent_run_ctx.__exit__(None, None, None)
    return metrics_df, predictions_df


def main() -> None:
    metrics_df, predictions_df = run_walk_forward()

    metrics_path = PROCESSED_DIR / "walkforward_metrics.csv"
    predictions_path = PROCESSED_DIR / "walkforward_predictions.csv"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved predictions to %s", predictions_path)
    logger.info(
        "Average fold accuracy=%.3f, f1=%.3f, roc_auc=%.3f",
        metrics_df["accuracy"].mean(),
        metrics_df["f1"].mean(),
        metrics_df["roc_auc"].mean(skipna=True),
    )


if __name__ == "__main__":
    main()
