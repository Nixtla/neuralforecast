import numpy as np
import pandas as pd
import pytest

from neuralforecast.utils import (
    PredictionIntervals,
    _conformal_rank,
    add_conformal_distribution_intervals,
    add_conformal_error_intervals,
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


@pytest.mark.parametrize(
    "n, coverage, expected",
    [(2, 0.8, 2), (9, 0.9, 9), (10, 0.9, 10), (19, 0.9, 18), (20, 0.8, 17), (99, 0.95, 95)],
)
def test_conformal_rank(n, coverage, expected):
    assert _conformal_rank(n, coverage) == expected


def _conformal_intervals(method, **kwargs):
    # one series, one step, 10 windows with absolute errors 1..10
    cs_df = pd.DataFrame({"m": np.random.default_rng(0).permutation(10) + 1.0})
    out, cols = method(np.zeros((1, 1)), cs_df, "m", 10, 1, 1, **kwargs)
    return dict(zip(cols, out[0, 1:]))


@pytest.mark.parametrize(
    "method", [add_conformal_distribution_intervals, add_conformal_error_intervals]
)
def test_conformal_intervals_finite_sample_rank(method):
    # the bounds are at least the ceil((n + 1) * lv / 100)-th smallest score:
    # ceil(11 * 0.7) = 8 and ceil(11 * 0.8) = 9
    fcst = _conformal_intervals(method, level=[70, 80])
    assert fcst["m-hi-70"] >= 8 and fcst["m-lo-70"] <= -8
    assert fcst["m-hi-80"] >= 9 and fcst["m-lo-80"] <= -9
    if method is add_conformal_distribution_intervals:
        # the pooled cuts land exactly on the k-th smallest score
        assert fcst["m-hi-70"] == pytest.approx(8)
        assert fcst["m-lo-80"] == pytest.approx(-9)


def test_conformal_error_intervals_asymmetric_quantiles():
    # each bound uses the coverage of its own quantile: 0.1 -> 80%, 0.8 -> 60%
    fcst = _conformal_intervals(add_conformal_error_intervals, quantiles=[0.1, 0.5, 0.8])
    assert fcst["m-ql0.1"] <= -9  # ceil(11 * 0.8) = 9
    assert 7 <= fcst["m-ql0.8"] < 8  # ceil(11 * 0.6) = 7
    assert fcst["m-ql0.5"] == 0
