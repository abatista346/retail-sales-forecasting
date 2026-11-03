"""Tests for the forecasting features and metrics.

The property that matters most in time-series features is that nothing
from day t leaks into the features used to predict day t.
"""

import numpy as np
import pandas as pd
import pytest

from feature_engineering import LAGS, ROLLING_WINDOWS, add_date_features, add_lag_and_rolling_features
from train_baseline import compute_metrics


@pytest.fixture
def daily() -> pd.DataFrame:
    idx = pd.date_range("2018-01-01", periods=60, freq="D", name="date")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"revenue": rng.uniform(1_000, 5_000, len(idx)), "n_orders": 10}, index=idx)


def test_lag_features_equal_past_values(daily):
    out = add_lag_and_rolling_features(daily)
    for lag in LAGS:
        expected = daily["revenue"].shift(lag)
        pd.testing.assert_series_equal(out[f"revenue_lag_{lag}"], expected, check_names=False)


def test_rolling_features_exclude_current_day(daily):
    out = add_lag_and_rolling_features(daily)
    day = 40
    for window in ROLLING_WINDOWS:
        past = daily["revenue"].iloc[day - window:day]
        assert out[f"revenue_rolling_mean_{window}"].iloc[day] == pytest.approx(past.mean())
        assert out[f"revenue_rolling_std_{window}"].iloc[day] == pytest.approx(past.std())


def test_changing_today_does_not_change_todays_features(daily):
    shocked = daily.copy()
    shocked.iloc[45, shocked.columns.get_loc("revenue")] *= 100
    a = add_lag_and_rolling_features(daily).iloc[45].drop(["revenue", "n_orders"])
    b = add_lag_and_rolling_features(shocked).iloc[45].drop(["revenue", "n_orders"])
    pd.testing.assert_series_equal(a, b)


def test_date_features(daily):
    out = add_date_features(daily)
    assert out.loc["2018-01-06", "is_weekend"] == 1  # Saturday
    assert out.loc["2018-01-08", "is_weekend"] == 0  # Monday
    assert out.loc["2018-01-08", "day_of_week"] == 0
    assert out.loc["2018-02-15", "month"] == 2


def test_compute_metrics_known_values():
    actual = pd.Series([100.0, 200.0, 0.0])
    predicted = pd.Series([110.0, 180.0, 5.0])
    m = compute_metrics(actual, predicted, "toy")
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 400 + 25) / 3), abs=0.01)
    assert m["mae"] == pytest.approx(35 / 3, abs=0.01)
    # Zero-revenue days are excluded from MAPE instead of dividing by zero.
    assert m["mape_pct"] == pytest.approx((10 / 100 + 20 / 200) / 2 * 100, abs=0.01)


def test_perfect_forecast_has_zero_error(daily):
    m = compute_metrics(daily["revenue"], daily["revenue"], "perfect")
    assert m["rmse"] == 0 and m["mae"] == 0 and m["mape_pct"] == 0
