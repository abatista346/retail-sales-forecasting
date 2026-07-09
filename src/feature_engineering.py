"""Aggregate order-item data into a daily national sales series with
date features, lags, and rolling stats for forecasting.

The first four months of the dataset (Sep 2016 - early Jan 2017) are a
sparse marketplace ramp-up with many zero-order days, which would distort
lag/rolling features — the series starts from 2017-01-05, the first date
with continuous daily activity.
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SERIES_START = "2017-01-05"
SERIES_END = "2018-08-31"
LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 28]


def load_daily_series() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "orders_merged.csv", parse_dates=["order_purchase_timestamp"])
    daily = (
        df.groupby(df["order_purchase_timestamp"].dt.floor("D"))
        .agg(revenue=("price", "sum"), n_orders=("order_id", "nunique"))
        .rename_axis("date")
    )
    full_range = pd.date_range(SERIES_START, SERIES_END, freq="D")
    daily = daily.reindex(full_range).fillna(0.0)
    daily.index.name = "date"
    return daily


def add_date_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["day_of_week"] = daily.index.dayofweek
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
    daily["month"] = daily.index.month
    daily["day_of_month"] = daily.index.day
    daily["week_of_year"] = daily.index.isocalendar().week.astype(int)
    return daily


def add_lag_and_rolling_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    for lag in LAGS:
        daily[f"revenue_lag_{lag}"] = daily["revenue"].shift(lag)
    for window in ROLLING_WINDOWS:
        daily[f"revenue_rolling_mean_{window}"] = daily["revenue"].shift(1).rolling(window).mean()
        daily[f"revenue_rolling_std_{window}"] = daily["revenue"].shift(1).rolling(window).std()
    return daily


def main() -> None:
    daily = load_daily_series()
    daily = add_date_features(daily)
    daily = add_lag_and_rolling_features(daily)

    before = len(daily)
    daily = daily.dropna()
    out_path = PROCESSED_DIR / "daily_sales_features.csv"
    daily.to_csv(out_path, index_label="date")
    print(f"Wrote {len(daily)} rows to {out_path} (dropped {before - len(daily)} warm-up rows for lag/rolling features)")
    print(daily.columns.tolist())


if __name__ == "__main__":
    main()
