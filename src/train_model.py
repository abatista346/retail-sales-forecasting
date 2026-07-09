"""Train an XGBoost regressor on the daily revenue series and compare
it against the naive/seasonal-naive baselines from train_baseline.py.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
TEST_DAYS = 56

COLOR_ACCENT = "#1D5FD6"
COLOR_TEXT_SECONDARY = "#4A5B6E"
COLOR_SUCCESS = "#187A4F"

FEATURE_COLS = [
    "day_of_week", "is_weekend", "month", "day_of_month", "week_of_year",
    "revenue_lag_1", "revenue_lag_7", "revenue_lag_14", "revenue_lag_28",
    "revenue_rolling_mean_7", "revenue_rolling_std_7",
    "revenue_rolling_mean_28", "revenue_rolling_std_28",
]


def compute_metrics(actual: pd.Series, predicted: np.ndarray, name: str) -> dict:
    predicted = pd.Series(predicted, index=actual.index)
    rmse = float(np.sqrt(((actual - predicted) ** 2).mean()))
    mae = float((actual - predicted).abs().mean())
    nonzero = actual != 0
    mape = float((((actual[nonzero] - predicted[nonzero]).abs()) / actual[nonzero]).mean() * 100)
    return {"model": name, "rmse": round(rmse, 2), "mae": round(mae, 2), "mape_pct": round(mape, 2)}


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "daily_sales_features.csv", parse_dates=["date"]).set_index("date")

    train, test = df.iloc[:-TEST_DAYS], df.iloc[-TEST_DAYS:]
    X_train, y_train = train[FEATURE_COLS], train["revenue"]
    X_test, y_test = test[FEATURE_COLS], test["revenue"]

    # Hyperparameters picked via a time-based validation split (last 56 days
    # of the training period, held out from the test set) — see git history
    # for the small grid search this came from.
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    xgb_metrics = compute_metrics(y_test, preds, "xgboost")
    print(xgb_metrics)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUTS_DIR / "metrics.json"
    all_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    all_metrics["xgboost"] = xgb_metrics
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"Wrote model metrics to {metrics_path}")

    plot_comparison(y_test, preds, all_metrics)
    plot_feature_importance(model)


def plot_comparison(y_test: pd.Series, preds: np.ndarray, all_metrics: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(y_test.index, y_test.values, color=COLOR_TEXT_SECONDARY, linewidth=1.6, label="Actual")
    ax.plot(y_test.index, preds, color=COLOR_ACCENT, linewidth=1.6, label="XGBoost forecast")
    ax.set_ylabel("Daily revenue (R$)")
    ax.set_title(f"XGBoost forecast vs. actual daily revenue — last {len(y_test)} days (test set)", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_forecast_vs_actual.png", dpi=150)
    plt.close(fig)

    models = [all_metrics["baseline"]["naive"], all_metrics["baseline"]["seasonal_naive"], all_metrics["xgboost"]]
    names = [m["model"] for m in models]
    rmses = [m["rmse"] for m in models]
    best_idx = rmses.index(min(rmses))
    colors = [COLOR_SUCCESS if i == best_idx else COLOR_TEXT_SECONDARY for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, rmses, color=colors)
    ax.set_ylabel("RMSE (R$)")
    ax.set_title(f"Lowest test RMSE: {names[best_idx]}", loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_rmse_comparison.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(model: XGBRegressor) -> None:
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance.index, importance.values, color=COLOR_ACCENT)
    ax.set_xlabel("Feature importance")
    ax.set_title("What drives the XGBoost forecast", loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_feature_importance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
