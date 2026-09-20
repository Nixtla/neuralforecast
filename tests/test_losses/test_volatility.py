import numpy as np
import pandas as pd
import polars as pl
import pytest

from neuralforecast.losses.numpy import (
    _reshape_windows_by_date,
    cross_validation_to_windows,
    excess_volatility,
    forecast_percentage_change,
)


QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MEDIAN_IDX = QUANTILES.index(0.5)


def _pair(y_value, before, update, quantiles=(0.5,)):
    """Build the smallest setup with exactly one comparable forecast pair.

    With T=2, H=2 and stride=1 only date 1 is predicted twice: by window 0 at h=1
    (the earlier, "before" forecast) and by window 1 at h=0 (the "update").
    """
    y = np.full((1, 2, 2, 1), y_value, dtype=float)
    y_hat = np.zeros((1, 2, 2, 1, len(quantiles)))
    y_hat[0, 0, 1, 0, :] = before
    y_hat[0, 1, 0, 0, :] = update
    return y, y_hat


# ---------------------------------------------------------------------------
# A perfectly stable forecaster scores zero on both metrics
# ---------------------------------------------------------------------------


def test_metrics_are_zero_for_self_consistent_rolling_forecast():
    b, t, h, c = 1, 10, 5, 2
    pattern = np.array([1, 2, 3, 4, 5], dtype=float)
    rolled = np.stack([np.roll(pattern, -i) for i in range(t)])  # [T, H]
    point = np.stack([rolled, rolled], axis=-1)[None]  # [B, T, H, C]

    y = point.copy()
    y_hat = point[..., np.newaxis].repeat(len(QUANTILES), axis=-1)
    mask = np.ones((b, t, h, c))

    assert forecast_percentage_change(y_hat=y_hat[..., MEDIAN_IDX], mask=mask) == 0.0
    assert excess_volatility(y=y, y_hat=y_hat, quantiles=QUANTILES, mask=mask) == 0.0


def test_metrics_are_zero_for_constant_forecast():
    b, t, h, c = 1, 10, 5, 2
    y = np.ones((b, t, h, c))
    y_hat = np.ones((b, t, h, c, len(QUANTILES)))
    mask = np.ones((b, t, h, c))

    assert forecast_percentage_change(y_hat=y_hat[..., MEDIAN_IDX], mask=mask) == 0.0
    assert excess_volatility(y=y, y_hat=y_hat, quantiles=QUANTILES, mask=mask) == 0.0


# ---------------------------------------------------------------------------
# forecast_percentage_change
# ---------------------------------------------------------------------------


def test_sfpc_matches_hand_computed_value():
    # single pair, before=10, update=20 -> 200 * 10 / 30
    y_hat = np.zeros((1, 2, 2, 1))
    y_hat[0, 0, :, 0] = 10.0
    y_hat[0, 1, :, 0] = 20.0
    assert forecast_percentage_change(y_hat=y_hat) == pytest.approx(200 / 3, rel=1e-6)


def test_fpc_is_relative_to_the_earlier_forecast():
    y_hat = np.zeros((1, 2, 2, 1))
    y_hat[0, 0, :, 0] = 10.0
    y_hat[0, 1, :, 0] = 20.0
    # |20 - 10| / |10|, with no 200x scaling
    assert forecast_percentage_change(y_hat=y_hat, symmetric=False) == pytest.approx(
        1.0, rel=1e-5
    )


def test_fpc_is_asymmetric_but_sfpc_is_not():
    up = np.zeros((1, 2, 2, 1))
    up[0, 0, :, 0], up[0, 1, :, 0] = 10.0, 20.0
    down = np.zeros((1, 2, 2, 1))
    down[0, 0, :, 0], down[0, 1, :, 0] = 20.0, 10.0

    assert forecast_percentage_change(y_hat=up) == pytest.approx(
        forecast_percentage_change(y_hat=down), rel=1e-6
    )
    assert forecast_percentage_change(y_hat=up, symmetric=False) != pytest.approx(
        forecast_percentage_change(y_hat=down, symmetric=False), rel=1e-6
    )


def test_fpc_rejects_non_4d_input():
    with pytest.raises(ValueError, match=r"\[B, T, H, C\]"):
        forecast_percentage_change(y_hat=np.ones((2, 2, 2)))


def test_fpc_mask_excludes_revisions():
    y_hat = np.zeros((1, 2, 2, 1))
    y_hat[0, 0, :, 0] = 10.0
    y_hat[0, 1, :, 0] = 20.0

    mask = np.ones((1, 2, 2, 1))
    mask[0, 1, 0, 0] = 0  # drop the update half of the only pair
    assert np.isnan(forecast_percentage_change(y_hat=y_hat, mask=mask))


# ---------------------------------------------------------------------------
# excess_volatility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "y_value, expected",
    [
        (10.0, 10.0),  # churn: the old forecast was right, the revision was pure cost
        (20.0, 0.0),  # fully justified: the revision moved exactly onto the truth
        (15.0, 5.0),  # revision helped halfway, so half of its cost is excess
        (12.0, 8.0),  # revision overshot the truth
    ],
)
def test_ev_matches_hand_computed_values(y_value, expected):
    y, y_hat = _pair(y_value, before=10.0, update=20.0)
    got = excess_volatility(y=y, y_hat=y_hat, quantiles=[0.5], stride=1, scaling=False)
    assert got == pytest.approx(expected, rel=1e-9)


def test_ev_scaling_divides_by_summed_absolute_target():
    y, y_hat = _pair(10.0, before=10.0, update=20.0)
    unscaled = excess_volatility(
        y=y, y_hat=y_hat, quantiles=[0.5], stride=1, scaling=False
    )
    scaled = excess_volatility(
        y=y, y_hat=y_hat, quantiles=[0.5], stride=1, scaling=True
    )
    # one comparable pair with |y| = 10
    assert scaled == pytest.approx(unscaled / 10.0, rel=1e-6)


def test_ev_is_never_negative():
    """The pinball loss is subadditive, so by the triangle inequality the accuracy a
    revision buys can never exceed what the revision costs."""
    rng = np.random.default_rng(0)
    quantiles = [0.1, 0.5, 0.9]
    for _ in range(50):
        y = rng.normal(0, 5, (2, 4, 3, 2))
        y_hat = np.sort(rng.normal(0, 5, (2, 4, 3, 2, len(quantiles))), axis=-1)
        assert (
            excess_volatility(
                y=y, y_hat=y_hat, quantiles=quantiles, stride=1, scaling=False
            )
            >= 0.0
        )


def test_ev_mask_excludes_pairs():
    y, y_hat = _pair(10.0, before=10.0, update=20.0)
    mask = np.ones((1, 2, 2, 1))
    mask[0, 0, 1, 0] = 0  # drop the "before" half of the only pair
    assert excess_volatility(
        y=y, y_hat=y_hat, quantiles=[0.5], stride=1, scaling=False, mask=mask
    ) == pytest.approx(0.0)


def test_ev_rejects_mismatched_quantiles():
    y, y_hat = _pair(10.0, 10.0, 20.0, quantiles=(0.1, 0.5, 0.9))
    with pytest.raises(ValueError, match="does not match the last axis"):
        excess_volatility(y=y, y_hat=y_hat, quantiles=[0.5])


def test_ev_rejects_mismatched_target_shape():
    _, y_hat = _pair(10.0, 10.0, 20.0)
    with pytest.raises(ValueError, match="incompatible"):
        excess_volatility(y=np.ones((1, 2, 3, 1)), y_hat=y_hat, quantiles=[0.5])


# ---------------------------------------------------------------------------
# stride validation, shared by both metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stride, match", [(2, "non-overlapping"), (3, "no forecast")])
def test_metrics_reject_non_overlapping_windows(stride, match):
    y, y_hat = _pair(10.0, 10.0, 20.0)
    with pytest.raises(ValueError, match=match):
        forecast_percentage_change(y_hat=np.ones((1, 2, 2, 1)), stride=stride)
    with pytest.raises(ValueError, match=match):
        excess_volatility(y=y, y_hat=y_hat, quantiles=[0.5], stride=stride)


def test_metrics_reject_non_positive_stride():
    with pytest.raises(ValueError, match="positive integer"):
        forecast_percentage_change(y_hat=np.ones((1, 2, 2, 1)), stride=0)


# ---------------------------------------------------------------------------
# _reshape_windows_by_date
# ---------------------------------------------------------------------------


def test_reshape_groups_forecasts_by_target_date():
    # window t predicts dates t..t+2, tagging each value with its target date
    t, h, stride = 3, 3, 1
    x = np.array([[[float(w + i) for i in range(h)] for w in range(t)]])[..., None]

    out = _reshape_windows_by_date(x, stride=stride)
    assert out.shape == (1, (t - 1) * stride + h, h, 1)

    # date 2 is reachable from window 0 at h=2, window 1 at h=1 and window 2 at h=0,
    # and every one of them carries the value 2
    np.testing.assert_allclose(out[0, 2, :, 0], [2.0, 2.0, 2.0])
    # date 0 is only reachable at h=0
    assert not np.isnan(out[0, 0, 0, 0])
    assert np.isnan(out[0, 0, 1:, 0]).all()


def test_reshape_nans_out_masked_positions():
    x = np.ones((1, 2, 2, 1))
    mask = np.ones((1, 2, 2, 1))
    mask[0, 0, 0, 0] = 0
    out = _reshape_windows_by_date(x, stride=1, mask=mask)
    assert np.isnan(out[0, 0, 0, 0])


# ---------------------------------------------------------------------------
# cross_validation_to_windows
# ---------------------------------------------------------------------------

N_SERIES, N_WINDOWS, HORIZON = 2, 4, 3


def _cv_frame(step_size=1, model="NHITS", n_windows=N_WINDOWS):
    """Build a frame shaped like NeuralForecast.cross_validation output."""
    rng = np.random.default_rng(1)
    rows = []
    for uid in range(N_SERIES):
        for w in range(n_windows):
            cutoff = pd.Timestamp("2020-01-01") + pd.Timedelta(days=w * step_size - 1)
            for h in range(HORIZON):
                rows.append(
                    {
                        "unique_id": f"s{uid}",
                        "ds": pd.Timestamp("2020-01-01")
                        + pd.Timedelta(days=w * step_size + h),
                        "cutoff": cutoff,
                        "y": float(rng.integers(1, 10)),
                        f"{model}-lo-80": 1.0 + h,
                        f"{model}-median": 2.0 + h,
                        f"{model}-hi-80": 3.0 + h,
                    }
                )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("frame", ["pandas", "polars"])
@pytest.mark.parametrize("step_size", [1, 2])
def test_converter_recovers_dense_windows(frame, step_size):
    df = _cv_frame(step_size=step_size)
    expected_y = df.sort_values(["unique_id", "cutoff", "ds"])["y"].to_numpy()
    if frame == "polars":
        df = pl.from_pandas(df)

    windows = cross_validation_to_windows(df, model="NHITS")

    assert windows.quantiles == [0.1, 0.5, 0.9]
    assert windows.stride == step_size
    assert windows.y.shape == (N_SERIES, N_WINDOWS, HORIZON, 1)
    assert windows.y_hat.shape == (N_SERIES, N_WINDOWS, HORIZON, 1, 3)
    assert (windows.mask == 1).all()
    np.testing.assert_allclose(windows.y[..., 0].reshape(-1), expected_y)
    # -lo-80 < -median < -hi-80 must land in ascending quantile order
    np.testing.assert_allclose(
        windows.y_hat[0, 0, :, 0, :],
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
    )


def test_converter_is_insensitive_to_row_order():
    df = _cv_frame()
    shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
    ordered = cross_validation_to_windows(df, model="NHITS")
    scrambled = cross_validation_to_windows(shuffled, model="NHITS")
    np.testing.assert_allclose(ordered.y, scrambled.y)
    np.testing.assert_allclose(ordered.y_hat, scrambled.y_hat)


def test_converter_masks_missing_targets():
    df = _cv_frame()
    df.loc[0, "y"] = np.nan
    windows = cross_validation_to_windows(df, model="NHITS")
    assert windows.mask.sum() == windows.mask.size - 1


def test_converter_rejects_missing_columns():
    df = _cv_frame().drop(columns=["cutoff"])
    with pytest.raises(ValueError, match="Missing required column"):
        cross_validation_to_windows(df, model="NHITS")


def test_converter_rejects_point_forecast_only():
    df = _cv_frame().drop(columns=["NHITS-lo-80", "NHITS-median", "NHITS-hi-80"])
    df["NHITS"] = 1.0
    with pytest.raises(ValueError, match="No quantile columns"):
        cross_validation_to_windows(df, model="NHITS")


def test_converter_rejects_conflicting_step_size():
    with pytest.raises(ValueError, match="disagrees with the cutoffs"):
        cross_validation_to_windows(_cv_frame(step_size=1), model="NHITS", step_size=2)


def test_converter_rejects_unevenly_spaced_windows():
    df = _cv_frame()
    last_cutoff = df["cutoff"].max()
    df = df[df["cutoff"] != last_cutoff]  # leaves cutoffs spaced 1, 1, then a gap
    df = pd.concat(
        [df, _cv_frame(n_windows=6).query("cutoff == cutoff.max()")], ignore_index=True
    )
    with pytest.raises(ValueError, match="not evenly spaced"):
        cross_validation_to_windows(df, model="NHITS")


def test_converter_needs_more_than_one_window_to_infer_step_size():
    df = _cv_frame(n_windows=1)
    with pytest.raises(ValueError, match="Cannot infer step_size"):
        cross_validation_to_windows(df, model="NHITS")
    # ...but an explicit step_size is enough
    assert cross_validation_to_windows(df, model="NHITS", step_size=1).stride == 1


def test_converter_handles_series_with_fewer_windows():
    """Short series get fewer cross validation windows, leaving trailing empty cells."""
    df = _cv_frame()
    kept_cutoffs = sorted(df["cutoff"].unique())[-2:]
    # series s1 only has its last two windows
    df = df[(df["unique_id"] == "s0") | (df["cutoff"].isin(kept_cutoffs))]

    windows = cross_validation_to_windows(df, model="NHITS")

    assert windows.stride == 1
    assert windows.y.shape == (N_SERIES, N_WINDOWS, HORIZON, 1)
    # s0 keeps all four windows, s1 only the two it has
    assert windows.mask[0].sum() == N_WINDOWS * HORIZON
    assert windows.mask[1].sum() == 2 * HORIZON
    assert (windows.mask[1, 2:] == 0).all()
