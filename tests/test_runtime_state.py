import numpy as np
import pytest
import torch

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import MLP
from neuralforecast.utils import AirPassengersDF, PredictionIntervals


def _model(**kwargs):
    return MLP(
        h=3,
        input_size=6,
        hidden_size=8,
        num_layers=2,
        max_steps=1,
        random_seed=1658,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **kwargs,
    )


@pytest.fixture
def fitted_forecaster():
    nf = NeuralForecast([_model()], freq="ME")
    nf.fit(AirPassengersDF)
    return nf


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_insample_restores_horizon_after_error(fitted_forecaster, monkeypatch, error):
    nf = fitted_forecaster
    model = nf.models[0]
    expected = nf.predict()
    initial = (nf.h, model.get_test_size())

    def interrupted(*args, **kwargs):
        raise error("interrupted prediction")

    with monkeypatch.context() as patch:
        patch.setattr(model, "predict", interrupted)
        with pytest.raises(error, match="interrupted prediction"):
            nf.predict_insample(step_size=3)

    assert (nf.h, model.get_test_size()) == initial
    np.testing.assert_array_equal(nf.predict()["MLP"], expected["MLP"])


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_forecast_restores_model_test_size(fitted_forecaster, monkeypatch, error):
    nf = fitted_forecaster
    model = nf.models[0]
    initial = model.get_test_size()

    def interrupted(*args, **kwargs):
        raise error("interrupted prediction")

    with monkeypatch.context() as patch:
        patch.setattr(model, "predict", interrupted)
        with pytest.raises(error, match="interrupted prediction"):
            nf.predict()

    assert model.get_test_size() == initial


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_recursive_prediction_restores_temporary_state(monkeypatch, error):
    model = _model()
    model.set_test_size(6)
    model.n_predicts = 2
    model.explain = True
    batch = {
        "temporal": torch.ones(1, 2, 12),
        "temporal_cols": ["y", "available_mask"],
        "y_idx": 0,
    }

    def interrupted(*args, **kwargs):
        raise error("interrupted recursive prediction")

    monkeypatch.setattr(model, "_create_windows", interrupted)
    with pytest.raises(error, match="interrupted recursive prediction"):
        model._predict_step_direct(batch, 0, recursive=True)

    assert model.get_test_size() == 6
    assert model.explain is True


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_calibration_interrupt_keeps_quantile_models(monkeypatch, error):
    nf = NeuralForecast([_model(), _model(loss=MQLoss(level=[80]), alias="MQ")], freq="ME")

    def interrupted(*args, **kwargs):
        raise error("interrupted calibration")

    with monkeypatch.context() as patch:
        patch.setattr(nf, "_no_refit_cross_validation", interrupted)
        with pytest.raises(error, match="interrupted calibration"):
            nf.fit(AirPassengersDF, prediction_intervals=PredictionIntervals())

    assert nf._add_level is False
    result = nf.cross_validation(AirPassengersDF, n_windows=1)
    assert len([column for column in result if column.startswith("MQ")]) == 3


def test_forecast_validation_error_restores_model_test_size(fitted_forecaster):
    nf = fitted_forecaster
    model = nf.models[0]
    initial = model.get_test_size()
    with pytest.raises(AttributeError, match="prediction_intervals"):
        nf.predict(level=[80])
    assert model.get_test_size() == initial
