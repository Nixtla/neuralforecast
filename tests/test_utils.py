import pandas as pd
import pytest

from neuralforecast.utils import (
    PredictionIntervals,
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


@pytest.mark.parametrize("index", [[10, 11, 12], [2, 0, 1], [5, 5, 7]])
def test_augment_calendar_df_keeps_input_index(index):
    df = pd.DataFrame(
        {
            "unique_id": "a",
            "ds": pd.date_range("2024-01-01", periods=3, freq="D"),
            "y": range(3),
        },
        index=index,
    )
    out, cols = augment_calendar_df(df, freq="D")
    expected, _ = augment_calendar_df(df.reset_index(drop=True), freq="D")

    assert len(out) == len(df)
    assert out.index.tolist() == index
    assert not out[cols].isna().any().any()
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), expected, check_dtype=False
    )
