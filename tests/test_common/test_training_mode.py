import pytest

from neuralforecast import NeuralForecast
from neuralforecast.models import MLP, NBEATS
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengersPanel

DF = AirPassengersPanel[["unique_id", "ds", "y"]]


def _raise(*args, **kwargs):
    raise RuntimeError("interrupted")


def _fit(model):
    nf = NeuralForecast([model], freq="ME")
    nf.fit(DF)
    return nf


def _record_training_steps(model):
    seen = []
    step = model.training_step

    def training_step(*args, **kwargs):
        seen.append(model.training)
        return step(*args, **kwargs)

    model.training_step = training_step
    return seen


def test_failed_predict_restores_training_mode_for_next_fit():
    nf = _fit(MLP(h=12, input_size=24, max_steps=2, enable_progress_bar=False, logger=False))
    model = nf.models[0]
    was_training = model.training

    forward = model.forward
    model.forward = _raise
    with pytest.raises(RuntimeError, match="interrupted"):
        nf.predict()
    model.forward = forward
    assert model.training == was_training

    seen = _record_training_steps(model)
    nf.fit(DF, use_init_models=False)
    assert seen and all(seen)


def test_failed_decompose_restores_training_mode_and_flag():
    nf = _fit(NBEATS(h=12, input_size=24, max_steps=2, enable_progress_bar=False, logger=False))
    model = nf.models[0]
    was_training = model.training
    dataset, *_ = TimeSeriesDataset.from_df(DF)

    forward = model.forward
    model.forward = _raise
    with pytest.raises(RuntimeError, match="interrupted"):
        model.decompose(dataset=dataset)
    model.forward = forward
    assert model.training == was_training
    assert model.decompose_forecast is False
