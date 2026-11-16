"""decompose() must build its Trainer the same way predict() does.

Lightning cannot gather predictions across devices. predict() already forces
devices=[0] and strategy="auto" when accelerator is gpu and the host has more
than one CUDA device. decompose() used to pass trainer_kwargs through unchanged,
so the default devices=-1 launched DDP and torch.vstack only saw this rank's
windows (or the call failed).
"""

import pytest
import torch

from neuralforecast.models import NBEATS, NHITS, NLinear
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengersDF


class _DataModule:
    def __init__(self, dataset, valid_batch_size, **kwargs):
        self.dataset = dataset


class _RecordingTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def predict(self, model, datamodule=None):
        assert model.decompose_forecast is True
        return [torch.zeros(2, 1)]


class _FailingTrainer:
    def __init__(self, **kwargs):
        pass

    def predict(self, model, datamodule=None):
        raise RuntimeError("predict failed")


def _model(**trainer_kwargs):
    model = NLinear(h=2, input_size=4, max_steps=1, early_stop_patience_steps=-1)
    model.trainer_kwargs = {
        "max_steps": 1,
        "enable_checkpointing": False,
        **trainer_kwargs,
    }
    return model


def _patch(monkeypatch, trainer_cls, device_count):
    monkeypatch.setattr(
        "neuralforecast.common._base_model.TimeSeriesDataModule", _DataModule
    )
    monkeypatch.setattr("neuralforecast.common._base_model.pl.Trainer", trainer_cls)
    monkeypatch.setattr(
        "neuralforecast.common._base_model.torch.cuda.device_count",
        lambda: device_count,
    )


@pytest.mark.parametrize(
    "accelerator, device_count, expected_devices",
    [
        ("gpu", 2, [0]),
        ("cuda", 2, [0]),
        ("auto", 2, [0]),
        ("gpu", 1, -1),
        ("cuda", 1, -1),
        ("cpu", 4, -1),
    ],
)
def test_decompose_pins_one_device(
    monkeypatch, accelerator, device_count, expected_devices
):
    captured = {}

    class Trainer(_RecordingTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured["kwargs"] = kwargs

    _patch(monkeypatch, Trainer, device_count=device_count)
    # "auto" counts as GPU only when CUDA is available.
    monkeypatch.setattr(
        "pytorch_lightning.accelerators.cuda.CUDAAccelerator.is_available",
        staticmethod(lambda: True),
    )
    model = _model(accelerator=accelerator, devices=-1)

    out = model.decompose(dataset=object())

    assert captured["kwargs"]["accelerator"] == accelerator
    assert captured["kwargs"]["devices"] == expected_devices
    if expected_devices == [0]:
        assert captured["kwargs"]["strategy"] == "auto"
    else:
        assert "strategy" not in captured["kwargs"]
    # The guard copies the kwargs. Training must keep devices=-1.
    assert model.trainer_kwargs["devices"] == -1
    assert "strategy" not in model.trainer_kwargs
    assert model.decompose_forecast is False
    assert out.shape == (2, 1)


def test_decompose_resets_flag_when_predict_fails(monkeypatch):
    _patch(monkeypatch, _FailingTrainer, device_count=2)
    model = _model(accelerator="gpu", devices=-1)

    with pytest.raises(RuntimeError, match="predict failed"):
        model.decompose(dataset=object())

    assert model.decompose_forecast is False


def _air_passengers_dataset():
    dataset, *_ = TimeSeriesDataset.from_df(AirPassengersDF)
    return dataset


def _real_model(model_cls, **kwargs):
    return model_cls(
        h=12,
        input_size=24,
        max_steps=1,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        **kwargs,
    )


@pytest.mark.parametrize("model_cls", [NHITS, NBEATS])
def test_decompose_real_model_matches_predict(model_cls):
    dataset = _air_passengers_dataset()
    n_stacks = 3
    model = _real_model(model_cls)

    decomposed = model.decompose(dataset=dataset)
    predicted = model.predict(dataset=dataset)

    assert decomposed.shape == (1, n_stacks + 1, 12)
    assert decomposed.sum(axis=1) == pytest.approx(
        predicted.reshape(1, 12), rel=1e-4, abs=1e-4
    )
    assert model.decompose_forecast is False


def test_decompose_real_model_resets_flag_when_forward_fails(monkeypatch):
    dataset = _air_passengers_dataset()
    model = _real_model(NHITS)

    def boom(*args, **kwargs):
        raise RuntimeError("forward failed")

    monkeypatch.setattr(model, "forward", boom)
    with pytest.raises(RuntimeError, match="forward failed"):
        model.decompose(dataset=dataset)

    assert model.decompose_forecast is False
