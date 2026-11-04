import warnings

import numpy as np
import pandas as pd
import pytest

from neuralforecast.utils import (
    DayOfWeek,
    DayOfYear,
    PredictionIntervals,
    WeekOfYear,
    augment_calendar_df,
    level_to_quantiles,
    quantiles_to_level,
)


# Test level_to_quantiles
def test_level_to_quantiles():
    level_base = [80, 90]
    quantiles_base = [0.05, 0.1, 0.9, 0.95]
    quantiles = level_to_quantiles(level_base)
    level = quantiles_to_level(quantiles_base)

    assert quantiles == quantiles_base
    assert level == level_base


@pytest.mark.parametrize("step_size", [0, -1])
def test_prediction_intervals_step_size_validation(step_size):
    with pytest.raises(ValueError, match="step_size must be at least 1"):
        PredictionIntervals(step_size=step_size)


def test_week_of_year_uses_iso_week():
    # 2024-12-30 starts ISO week 1 of 2025; 2026-12-31 and 2027-01-01 are in ISO week 53
    index = pd.DatetimeIndex(["2024-12-29", "2024-12-30", "2026-12-31", "2027-01-01"])
    expected = (np.array([52, 1, 53, 53]) - 1) / 52.0 - 0.5
    np.testing.assert_allclose(WeekOfYear()(index), expected)


def test_day_of_week_and_day_of_year():
    # 2024-01-01 is a Monday
    index = pd.date_range("2024-01-01", periods=7, freq="D")
    np.testing.assert_allclose(DayOfWeek()(index), np.arange(7) / 6.0 - 0.5)
    np.testing.assert_allclose(DayOfYear()(index), np.arange(7) / 365.0 - 0.5)


def test_augment_calendar_df_weekly():
    df = pd.DataFrame(
        {
            "unique_id": "a",
            "ds": pd.date_range("2024-01-07", periods=4, freq="W"),
            "y": range(4),
        }
    )
    out, cols = augment_calendar_df(df, freq="W")
    assert cols == ["monthday", "yearweek"]
    np.testing.assert_allclose(out["yearweek"], np.arange(4) / 52.0 - 0.5)


@pytest.mark.skipif(
    not hasattr(pd.errors, "Pandas4Warning"),
    reason="Pandas4Warning only exists in pandas>=3.1",
)
@pytest.mark.parametrize(
    "freq, pd_freq",
    [
        ("Q", "QS"),
        ("M", "MS"),
        ("MS", "MS"),
        ("W", "W"),
        ("D", "D"),
        ("B", "B"),
        ("H", "h"),
        ("T", "min"),
        ("S", "s"),
    ],
)
def test_augment_calendar_df_no_pandas4_warnings(freq, pd_freq):
    df = pd.DataFrame(
        {
            "unique_id": "a",
            "ds": pd.date_range("2024-01-01", periods=10, freq=pd_freq),
            "y": range(10),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.Pandas4Warning)
        augment_calendar_df(df, freq=freq)
