"""Naive and seasonal-naive baselines for daily revenue forecasting.

These use no model at all — just yesterday's value (naive) or the same
weekday last week (seasonal-naive) — as the point of comparison the
gradient boosting model in train_model.py has to beat.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
TEST_DAYS = 56


def compute_metrics(actual: pd.Series, predicted: pd.Series, name: str) -> dict:
    rmse = float(np.sqrt(((actual - predicted) ** 2).mean()))
    mae = float((actual - predicted).abs().mean())
    nonzero = actual != 0
    mape = float((((actual[nonzero] - predicted[nonzero]).abs()) / actual[nonzero]).mean() * 100)
    return {"model": name, "rmse": round(rmse, 2), "mae": round(mae, 2), "mape_pct": round(mape, 2)}


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "daily_sales_features.csv", parse_dates=["date"]).set_index("date")
    test = df.iloc[-TEST_DAYS:]

    naive_metrics = compute_metrics(test["revenue"], test["revenue_lag_1"], "naive (t-1)")
    seasonal_naive_metrics = compute_metrics(test["revenue"], test["revenue_lag_7"], "seasonal_naive (t-7)")

    print(naive_metrics)
    print(seasonal_naive_metrics)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUTS_DIR / "metrics.json"
    all_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    all_metrics["baseline"] = {
        "test_days": TEST_DAYS,
        "naive": naive_metrics,
        "seasonal_naive": seasonal_naive_metrics,
    }
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"Wrote baseline metrics to {metrics_path}")


if __name__ == "__main__":
    main()
