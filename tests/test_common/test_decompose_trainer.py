"""decompose() must build its Trainer the same way predict() does.

Lightning cannot gather predictions across devices. predict() already forces
devices=[0] and strategy="auto" when accelerator is gpu and the host has more
than one CUDA device. decompose() used to pass trainer_kwargs through unchanged,
so the default devices=-1 launched DDP and torch.vstack only saw this rank's
windows (or the call failed).
"""

import pytest
import torch

from neuralforecast.models import NLinear


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


def test_decompose_pins_one_device_on_multi_gpu(monkeypatch):
    captured = {}

    class Trainer(_RecordingTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured["kwargs"] = kwargs

    _patch(monkeypatch, Trainer, device_count=2)
    model = _model(accelerator="gpu", devices=-1)

    out = model.decompose(dataset=object())

    assert captured["kwargs"]["devices"] == [0]
    assert captured["kwargs"]["strategy"] == "auto"
    assert captured["kwargs"]["accelerator"] == "gpu"
    # The guard copies the kwargs. Training must keep devices=-1.
    assert model.trainer_kwargs["devices"] == -1
    assert "strategy" not in model.trainer_kwargs
    assert model.decompose_forecast is False
    assert out.shape == (2, 1)


def test_decompose_leaves_single_gpu_and_cpu_unchanged(monkeypatch):
    captured = {}

    class Trainer(_RecordingTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured["kwargs"] = kwargs

    _patch(monkeypatch, Trainer, device_count=1)
    model = _model(accelerator="gpu", devices=-1)
    model.decompose(dataset=object())
    assert captured["kwargs"]["devices"] == -1
    assert "strategy" not in captured["kwargs"]

    _patch(monkeypatch, Trainer, device_count=4)
    cpu = _model(accelerator="cpu", devices=-1)
    cpu.decompose(dataset=object())
    assert captured["kwargs"]["accelerator"] == "cpu"
    assert captured["kwargs"]["devices"] == -1
    assert "strategy" not in captured["kwargs"]


def test_decompose_resets_flag_when_predict_fails(monkeypatch):
    _patch(monkeypatch, _FailingTrainer, device_count=2)
    model = _model(accelerator="gpu", devices=-1)

    with pytest.raises(RuntimeError, match="predict failed"):
        model.decompose(dataset=object())

    assert model.decompose_forecast is False
