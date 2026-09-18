"""Native numerical tests, official-API contract tests, and NF integration tests.

Contract doubles do NOT test pretrained checkpoint quality. Tests marked
optional execute the installed official TTM implementation without a download.
"""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from neuralforecast.models.crosslinear import CrossLinear
from neuralforecast.models.timerxl import TimerXL
from neuralforecast.models.tinytimemixer import TinyTimeMixer
from neuralforecast.models.foundation import Chronos2, Moirai, MoiraiMoE, TimesFM, Toto


def batch(length=16, horizon=4, hist_size=2, futr_size=1, size=2):
    torch.manual_seed(17)
    return {
        "insample_y": torch.randn(size, length, 1),
        "insample_mask": torch.ones(size, length, 1),
        "hist_exog": torch.randn(size, length, hist_size) if hist_size else None,
        "futr_exog": torch.randn(size, length + horizon, futr_size) if futr_size else None,
        "stat_exog": None,
    }


def native(cls, hist_size=2, **kwargs):
    options = dict(h=4, input_size=16, patch_len=4, hidden_size=16,
                   hist_exog_list=[f"x{i}" for i in range(hist_size)],
                   max_steps=2, val_check_steps=2, logger=False,
                   enable_progress_bar=False, accelerator="cpu", devices=1)
    if cls is TimerXL:
        options.update(n_heads=2, n_layers=1, d_ff=32, dropout=0.0)
    else:
        options.update(d_ff=32)
    options.update(kwargs)
    return cls(**options)


@pytest.mark.parametrize("cls", [CrossLinear, TimerXL])
@pytest.mark.parametrize("hist_size", [0, 2])
def test_native_shape_gradient_and_no_future_target_leak(cls, hist_size):
    model = native(cls, hist_size).eval()
    windows = batch(hist_size=hist_size, futr_size=0)
    output = model(windows)
    assert output.shape == (2, 4, 1) and torch.isfinite(output).all()
    output.square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    windows["outsample_y"] = torch.full((2, 4, 1), float("nan"))
    torch.testing.assert_close(output, model(windows))


@pytest.mark.parametrize("cls", [CrossLinear, TimerXL])
def test_native_historical_covariates_affect_prediction(cls):
    model = native(cls).eval()
    windows = batch(futr_size=0)
    windows["hist_exog"].requires_grad_()
    prediction = model(windows)
    prediction.sum().backward()
    assert windows["hist_exog"].grad.abs().sum() > 0
    changed = dict(windows, hist_exog=windows["hist_exog"].detach().flip(1))
    assert not torch.allclose(prediction, model(changed))


@pytest.mark.parametrize("cls", [CrossLinear, TimerXL])
def test_native_constant_input_and_state_dict_roundtrip(cls):
    model = native(cls).eval()
    windows = batch(futr_size=0)
    windows["insample_y"].fill_(3)
    windows["hist_exog"].fill_(2)
    result = model(windows)
    assert torch.isfinite(result).all()
    restored = native(cls).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(result, restored(windows))


@pytest.mark.parametrize("cls", [CrossLinear, TimerXL])
def test_native_rejects_missing_history_and_future_covariates(cls):
    windows = batch(futr_size=0)
    windows["insample_mask"][:, 0] = 0
    with pytest.raises(ValueError, match="complete history"):
        native(cls)(windows)
    with pytest.raises(Exception, match="future"):
        native(cls, futr_exog_list=["known"])


def test_timer_covariate_mask_matches_official_structure():
    model = native(TimerXL)
    assert model.allowed.shape == (12, 12)
    assert model.allowed[-4:, :8].all()
    assert not model.allowed[:8, -4:].any()
    torch.testing.assert_close(model.allowed[:4, :4], torch.ones(4, 4, dtype=torch.bool).tril())
    assert not model.allowed[:4, 4:8].any()


@pytest.mark.parametrize("cls", [CrossLinear, TimerXL, Chronos2, Moirai, MoiraiMoE, TimesFM, Toto])
def test_rejects_invalid_horizon(cls):
    with pytest.raises(ValueError, match="h must"):
        cls(h=0, input_size=16)


def test_duplicate_covariates_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="distinct"):
        Chronos2(h=4, input_size=16, hist_exog_list=["a"], futr_exog_list=["a"])
    model = Chronos2(h=4, input_size=16, futr_exog_list=["known"])
    windows = batch(hist_size=0)
    windows["futr_exog"][0, -1, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(windows)


@pytest.mark.parametrize("cls", [Chronos2, Moirai, MoiraiMoE, TimesFM, Toto])
def test_pretrained_refuses_training_request(cls):
    with pytest.raises(ValueError, match="inference-only"):
        cls(h=4, input_size=16, max_steps=10)


class ChronosContract:
    quantiles = [0.1, 0.5, 0.9]

    def __init__(self):
        self.calls = []

    def predict(self, inputs, prediction_length, context_length, cross_learning):
        assert cross_learning is False
        self.calls.append(inputs)
        results = []
        for item in inputs:
            assert len(item["target"]) == context_length
            assert set(item["past_covariates"]) == {"x0", "known"}
            assert set(item["future_covariates"]) == {"known"}
            value = torch.as_tensor(item["future_covariates"]["known"])
            assert value.shape == (prediction_length,)
            results.append(value.expand(1, 3, prediction_length).clone())
        return results


def test_chronos_routes_covariates_masks_and_batch_independence():
    model = Chronos2(h=4, input_size=16, hist_exog_list=["x0"], futr_exog_list=["known"])
    backend = ChronosContract()
    model.__dict__["_backend"] = backend
    windows = batch(hist_size=1)
    windows["insample_mask"][0, :2] = 0
    result = model(windows)
    torch.testing.assert_close(result, windows["futr_exog"][:, -4:])
    assert np.isnan(backend.calls[0][0]["target"][:2]).all()
    assert np.isnan(backend.calls[0][0]["past_covariates"]["x0"][:2]).all()
    np.testing.assert_allclose(backend.calls[0][1]["past_covariates"]["known"], windows["futr_exog"][1, :16, 0])
    windows["outsample_y"] = torch.randn(2, 4, 1) * 1e9
    torch.testing.assert_close(result, model(windows))
    assert not any("backend" in key for key in model.state_dict())


def test_timesfm_fits_xreg_per_window_without_future_target_pooling():
    class Backend:
        def __init__(self):
            self.calls = []

        def forecast_with_covariates(self, inputs, dynamic_numerical_covariates, **kwargs):
            assert len(inputs) == 1
            assert kwargs["xreg_mode"] == "xreg + timesfm"
            self.calls.append(inputs[0].copy())
            return [dynamic_numerical_covariates["known"][0][-4:]], None

    model = TimesFM(h=4, input_size=16, futr_exog_list=["known"])
    backend = Backend()
    model.__dict__["_backend"] = backend
    windows = batch(hist_size=0)
    torch.testing.assert_close(model(windows), windows["futr_exog"][:, -4:])
    assert len(backend.calls) == 2
    for i, history in enumerate(backend.calls):
        np.testing.assert_allclose(history, windows["insample_y"][i, :, 0])
    with pytest.raises(Exception, match="historical"):
        TimesFM(h=4, input_size=16, hist_exog_list=["unknown_future"])


def test_toto_uses_last_channels_for_known_future_inputs():
    class Forecaster:
        def forecast(self, inputs, prediction_length, future_exogenous_variables, **kwargs):
            assert inputs.series.shape == (2, 4, 16)
            assert inputs.num_exogenous_variables == 1
            assert inputs.padding_mask.dtype == torch.bool
            assert prediction_length == 4 and kwargs["num_samples"] == 3
            torch.testing.assert_close(inputs.series[:, -1], windows["futr_exog"][:, :16, 0])
            value = future_exogenous_variables.expand(-1, 4, -1)
            return SimpleNamespace(mean=value)

    model = Toto(h=4, input_size=16, num_samples=3,
                 hist_exog_list=["x0", "x1"], futr_exog_list=["known"])
    windows = batch()
    model.__dict__["_backend"] = (Forecaster(), SimpleNamespace)
    torch.testing.assert_close(model(windows), windows["futr_exog"][:, -4:])


def test_ttm_configuration_and_future_target_placeholder(monkeypatch):
    class OfficialContract(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            assert config.mode == config.decoder_mode == "mix_channel"
            assert config.enable_forecast_channel_mixing is True
            assert config.prediction_channel_indices == [0]
            assert config.exogenous_channel_indices == [3]
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, past_values, past_observed_mask, future_values, return_loss, return_dict):
            assert past_values.shape == past_observed_mask.shape == (2, 16, 4)
            assert past_observed_mask.dtype == torch.bool
            assert future_values.shape == (2, 4, 4)
            assert not future_values[:, :, :3].any()
            assert return_loss is False and return_dict is True
            return SimpleNamespace(prediction_outputs=future_values[:, :, -1:] * self.weight)

    fake = ModuleType("tsfm_public.models.tinytimemixer")
    fake.TinyTimeMixerConfig = SimpleNamespace
    fake.TinyTimeMixerForPrediction = OfficialContract
    monkeypatch.setitem(sys.modules, "tsfm_public.models.tinytimemixer", fake)
    model = TinyTimeMixer(h=4, input_size=16, patch_len=4,
                         hist_exog_list=["x0", "x1"], futr_exog_list=["known"])
    windows = batch()
    windows["outsample_y"] = torch.full((2, 4, 1), 123456.0)
    prediction = model(windows)
    torch.testing.assert_close(prediction, windows["futr_exog"][:, -4:])
    prediction.sum().backward()
    assert model.model.weight.grad is not None


@pytest.mark.parametrize("cls,prefix", [(Moirai, "Moirai"), (MoiraiMoE, "MoiraiMoE")])
def test_moirai_isolated_worker_contract(cls, prefix, tmp_path, monkeypatch):
    package = tmp_path / "uni2ts" / "model"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / (cls.BACKEND_KIND + ".py")).write_text(f'''
import torch
class {prefix}Module:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        assert model_id == {cls.DEFAULT_MODEL_ID!r}
        return cls()
class {prefix}Forecast(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        assert kwargs['feat_dynamic_real_dim'] == 1
        assert kwargs['past_feat_dynamic_real_dim'] == 2
        assert kwargs['patch_size'] == 16
        self.samples = kwargs['num_samples']
    def forward(self, **kwargs):
        assert kwargs['past_target'].shape == (2, 16, 1)
        assert kwargs['past_feat_dynamic_real'].shape == (2, 16, 2)
        assert kwargs['feat_dynamic_real'].shape == (2, 20, 1)
        assert kwargs['past_observed_target'].dtype == torch.bool
        assert not kwargs['past_is_pad'][:, -1].any()
        future = kwargs['feat_dynamic_real'][:, -4:]
        return future.unsqueeze(1).expand(-1, self.samples, -1, -1)
''')
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    model = cls(h=4, input_size=16, backend_python=sys.executable, num_samples=2,
                hist_exog_list=["x0", "x1"], futr_exog_list=["known"])
    windows = batch()
    torch.testing.assert_close(model(windows), windows["futr_exog"][:, -4:])
    with pytest.raises(ValueError, match="backend_python"):
        cls(h=4, input_size=16)(batch(hist_size=0, futr_size=0))


@pytest.mark.optional
def test_real_official_ttm_forward_and_covariate_gradients():
    pytest.importorskip("tsfm_public.models.tinytimemixer")
    model = TinyTimeMixer(h=4, input_size=16, patch_len=4, hidden_size=8,
                         n_layers=1, dropout=0.0, hist_exog_list=["x0", "x1"],
                         futr_exog_list=["known"], max_steps=2).eval()
    windows = batch()
    windows["hist_exog"].requires_grad_()
    windows["futr_exog"].requires_grad_()
    prediction = model(windows)
    assert prediction.shape == (2, 4, 1)
    prediction.square().sum().backward()
    assert windows["hist_exog"].grad.abs().sum() > 0
    assert windows["futr_exog"].grad[:, -4:].abs().sum() > 0


@pytest.mark.integration
def test_native_neuralforecast_fit_predict_save_load(tmp_path):
    import pandas as pd
    from neuralforecast import NeuralForecast
    from neuralforecast.core import MODEL_FILENAME_DICT
    from neuralforecast import models

    names = ["CrossLinear", "TimerXL", "TinyTimeMixer", "Chronos2", "Moirai", "MoiraiMoE", "TimesFM", "Toto"]
    for name in names:
        assert getattr(models, name) is MODEL_FILENAME_DICT[name.lower()]
        assert name in models.__all__
    t = np.arange(48)
    frame = pd.DataFrame({
        "unique_id": np.repeat(["a", "b"], 48),
        "ds": np.tile(pd.date_range("2024-01-01", periods=48, freq="D"), 2),
        "y": np.tile(np.sin(t / 5), 2),
        "x0": np.tile(np.cos(t / 4), 2),
        "x1": np.tile(t / 48, 2),
    })
    nf = NeuralForecast(models=[native(CrossLinear), native(TimerXL)], freq="D")
    nf.fit(df=frame)
    result = nf.predict()
    assert len(result) == 8
    assert np.isfinite(result[["CrossLinear", "TimerXL"]].to_numpy()).all()
    nf.save(path=str(tmp_path / "checkpoint"), overwrite=True, save_dataset=True)
    restored = NeuralForecast.load(path=str(tmp_path / "checkpoint"))
    again = restored.predict()
    np.testing.assert_allclose(result[["CrossLinear", "TimerXL"]], again[["CrossLinear", "TimerXL"]], rtol=1e-5, atol=1e-5)


@pytest.mark.integration
def test_chronos_neuralforecast_future_frame_and_checkpoint(tmp_path, monkeypatch):
    import pandas as pd
    from neuralforecast import NeuralForecast

    monkeypatch.setattr(Chronos2, "_load_backend", lambda self: ChronosContract())
    frame = pd.DataFrame({
        "unique_id": ["a"] * 32, "ds": pd.date_range("2024-01-01", periods=32),
        "y": np.arange(32, dtype=float), "x0": np.arange(32, dtype=float),
        "known": np.arange(32, dtype=float),
    })
    nf = NeuralForecast(models=[Chronos2(h=4, input_size=16,
        hist_exog_list=["x0"], futr_exog_list=["known"], logger=False,
        enable_progress_bar=False, accelerator="cpu", devices=1)], freq="D")
    nf.fit(df=frame)
    future = pd.DataFrame({"unique_id": ["a"] * 4,
        "ds": pd.date_range("2024-02-02", periods=4), "known": [7., 8., 9., 10.]})
    result = nf.predict(futr_df=future)
    np.testing.assert_allclose(result["Chronos2"], future["known"])
    nf.save(path=str(tmp_path / "checkpoint"), overwrite=True, save_dataset=True)
    loaded = NeuralForecast.load(path=str(tmp_path / "checkpoint"))
    np.testing.assert_allclose(loaded.predict(futr_df=future)["Chronos2"], future["known"])
