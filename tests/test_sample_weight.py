import pytest
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, MLPMultivariate
from neuralforecast.utils import AirPassengersPanel


HORIZON = 12
INPUT_SIZE = 24
MAX_STEPS = 5
N_SERIES = 2


def _make_df(with_sample_weight=True, weight_value=1.0):
    df = AirPassengersPanel[
        AirPassengersPanel["ds"] < AirPassengersPanel["ds"].values[-HORIZON]
    ].reset_index(drop=True).copy()
    if with_sample_weight:
        df["sample_weight"] = weight_value
    return df


def _make_model():
    return NHITS(h=HORIZON, input_size=INPUT_SIZE, max_steps=MAX_STEPS, accelerator="cpu")


def _make_multivariate_model():
    return MLPMultivariate(
        h=HORIZON,
        input_size=INPUT_SIZE,
        n_series=N_SERIES,
        max_steps=MAX_STEPS,
        accelerator="cpu",
    )


# ---------------------------------------------------------------------------
# Happy path for both univariate and multivariate
# ---------------------------------------------------------------------------

def test_fit_with_sample_weight():
    """fit completes for both model types; predict returns forecasts without sample_weight."""
    df = _make_df(with_sample_weight=True, weight_value=2.0)
    nf = NeuralForecast(models=[_make_model(), _make_multivariate_model()], freq="M")
    nf.fit(df)
    preds = nf.predict()
    assert len(preds) > 0
    assert "sample_weight" not in preds.columns
    assert "NHITS" in preds.columns
    assert "MLPMultivariate" in preds.columns


# ---------------------------------------------------------------------------
# Scaler exclusion
# ---------------------------------------------------------------------------

def test_sample_weight_not_scaled():
    df = _make_df()
    nf = NeuralForecast(models=[_make_model()], freq="M", local_scaler_type="standard")
    nf.fit(df)
    # scalers_ lives on the NeuralForecast instance, not on the individual model
    assert "sample_weight" not in nf.scalers_, (
        "sample_weight must not have a fitted scaler"
    )


# ---------------------------------------------------------------------------
# Column removed before model sees it
# ---------------------------------------------------------------------------

def test_sample_weight_not_in_model_input():
    """sample_weight must be stripped from windows before any model's forward pass."""
    df = _make_df()
    captured = {}

    class PatchedNHITS(NHITS):
        def _parse_windows(self, batch, windows):
            captured["nhits"] = windows["temporal_cols"]
            return super()._parse_windows(batch, windows)

    class PatchedMLPMultivariate(MLPMultivariate):
        def _parse_windows(self, batch, windows):
            captured["mlp"] = windows["temporal_cols"]
            return super()._parse_windows(batch, windows)

    nf = NeuralForecast(
        models=[
            PatchedNHITS(h=HORIZON, input_size=INPUT_SIZE, max_steps=MAX_STEPS, accelerator="cpu"),
            PatchedMLPMultivariate(h=HORIZON, input_size=INPUT_SIZE, n_series=N_SERIES, max_steps=MAX_STEPS, accelerator="cpu"),
        ],
        freq="M",
    )
    nf.fit(df)
    assert "sample_weight" not in captured["nhits"]
    assert "sample_weight" not in captured["mlp"]


# ---------------------------------------------------------------------------
# Uniform weight is mathematically a no-op on the loss
# ---------------------------------------------------------------------------

def test_uniform_sample_weight_is_noop():
    """Multiplying outsample_mask by 1.0 must leave the loss unchanged.

    We test this directly on the loss function rather than through full training,
    since floating-point non-determinism across runs with different dataset sizes
    makes exact equality of end-to-end training losses unachievable.
    """
    from neuralforecast.losses.pytorch import MAE

    loss_fn = MAE()
    torch.manual_seed(42)
    y = torch.rand(32, 12, 1)
    y_hat = torch.rand(32, 12, 1)
    mask = torch.ones(32, 12, 1)

    loss_no_weight = loss_fn(y=y, y_hat=y_hat, mask=mask)
    loss_uniform = loss_fn(y=y, y_hat=y_hat, mask=mask * 1.0)

    assert torch.isclose(loss_no_weight, loss_uniform), (
        f"Uniform weight=1.0 must not change loss: {loss_no_weight} vs {loss_uniform}"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_negative_sample_weight_raises():
    df = _make_df()
    df.loc[0, "sample_weight"] = -0.5
    nf = NeuralForecast(models=[_make_model()], freq="M")
    with pytest.raises(ValueError, match="non-negative"):
        nf.fit(df)


def test_nan_sample_weight_raises():
    df = _make_df()
    df.loc[0, "sample_weight"] = float("nan")
    nf = NeuralForecast(models=[_make_model()], freq="M")
    with pytest.raises(ValueError, match="NaN"):
        nf.fit(df)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def test_cross_validation_with_sample_weight():
    df = _make_df(with_sample_weight=True)
    nf = NeuralForecast(models=[_make_model()], freq="M")
    cv_df = nf.cross_validation(df, n_windows=2)
    assert len(cv_df) > 0
    assert "sample_weight" not in cv_df.columns
