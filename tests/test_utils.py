import numpy as np
import pandas as pd
import pytest

from neuralforecast.utils import (
    PredictionIntervals,
    _conformal_quantile,
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
    "n, q, upper, expected",
    [
        # upper bound: ceil((n + 1) * q), capped at n
        (2, 0.8, True, 2),
        (9, 0.9, True, 9),
        (19, 0.9, True, 18),
        (20, 0.8, True, 17),
        (99, 0.95, True, 95),
        # an upper bound below the median still rounds up
        (10, 0.4, True, 5),
        (10, 0.3, True, 4),
        # lower bound: floor((n + 1) * q), floored at 1
        (100, 0.025, False, 2),
        (10, 0.1, False, 1),
    ],
)
def test_conformal_rank(n, q, upper, expected):
    assert _conformal_rank(n, q, upper) == expected


def test_conformal_quantile_is_an_exact_order_statistic():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((10, 3))
    srt = np.sort(scores, axis=0)
    np.testing.assert_array_equal(_conformal_quantile(scores, 0.8, upper=True), srt[8])
    out = _conformal_quantile(scores, [0.1, 0.9], [False, True])
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, srt[[0, 9]])
    with pytest.raises(ValueError, match="zero scores"):
        _conformal_quantile(np.empty((0, 2)), 0.9, upper=True)


@pytest.mark.parametrize("n, level", [(10, 80), (20, 90), (50, 95)])
def test_conformal_quantile_tails_cover(n, level):
    # each tail should miss at most (1 - level) / 2 of new exchangeable draws
    rng = np.random.default_rng(1)
    reps = 20_000
    alpha = 1 - level / 100
    draws = rng.standard_normal((reps, n + 1))
    lo, hi = _conformal_quantile(
        draws[:, :n], [alpha / 2, 1 - alpha / 2], [False, True], axis=1
    )
    se = np.sqrt(alpha / 2 * (1 - alpha / 2) / reps)
    assert np.mean(draws[:, n] < lo) <= alpha / 2 + 3 * se
    assert np.mean(draws[:, n] > hi) <= alpha / 2 + 3 * se


def _conformal_intervals(method, **kwargs):
    # one series, one step, 10 windows with absolute errors 1..10
    cs_df = pd.DataFrame({"m": np.random.default_rng(0).permutation(10) + 1.0})
    out, cols = method(np.zeros((1, 1)), cs_df, "m", 10, 1, 1, **kwargs)
    return dict(zip(cols, out[0, 1:]))


@pytest.mark.parametrize(
    "method", [add_conformal_distribution_intervals, add_conformal_error_intervals]
)
def test_conformal_intervals_finite_sample_rank(method):
    # both methods put the bounds exactly at -/+ the ceil((n + 1) * lv / 100)-th
    # smallest score: ceil(11 * 0.7) = 8 and ceil(11 * 0.8) = 9
    fcst = _conformal_intervals(method, level=[70, 80])
    for lv, rank in [(70, 8), (80, 9)]:
        assert fcst[f"m-hi-{lv}"] == pytest.approx(rank)
        assert fcst[f"m-lo-{lv}"] == pytest.approx(-rank)


@pytest.mark.parametrize(
    "method", [add_conformal_distribution_intervals, add_conformal_error_intervals]
)
def test_conformal_intervals_levels_below_50(method):
    # below 50% the margin is still an upper bound, so the rank rounds up:
    # ceil(11 * 0.3) = 4, ceil(11 * 0.4) = 5 and ceil(11 * 0.49) = 6
    fcst = _conformal_intervals(method, level=[30, 40, 49])
    for lv, rank in [(30, 4), (40, 5), (49, 6)]:
        assert fcst[f"m-hi-{lv}"] == pytest.approx(rank)
        assert fcst[f"m-lo-{lv}"] == pytest.approx(-rank)


def test_conformal_error_intervals_asymmetric_quantiles():
    # each bound uses the coverage of its own quantile: 0.1 -> 80%, 0.8 -> 60%
    fcst = _conformal_intervals(add_conformal_error_intervals, quantiles=[0.1, 0.5, 0.8])
    assert fcst["m-ql0.1"] == pytest.approx(-9)  # ceil(11 * 0.8) = 9
    assert fcst["m-ql0.8"] == pytest.approx(7)  # ceil(11 * 0.6) = 7
    assert fcst["m-ql0.5"] == 0
