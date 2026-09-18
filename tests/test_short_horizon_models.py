"""NF integration against pinned upstream source, plus pretrained API contracts.

The contract doubles do not download or evaluate TimesFM/Chronos checkpoints.
The source-backed tests run real SeesawNet, Dualformer and SearchCast code when
SEESAWNET_SOURCE, DUALFORMER_SOURCE and SEARCHCAST_SOURCE are set (as in CI).
"""

from copy import deepcopy
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from neuralforecast import NeuralForecast
from neuralforecast.core import MODEL_FILENAME_DICT
from neuralforecast.losses.pytorch import MAE, MSE, MQLoss
from neuralforecast.models import Chronos2, TimesFM, TimesFM3, SeesawNet, Dualformer, SearchCast
from neuralforecast.models._forecast_source import forecast_source
from neuralforecast.tsdataset import TimeSeriesDataset


def source(kind):
    path = os.environ.get(kind.upper() + "_SOURCE")
    if path is None:
        pytest.skip(f"Set {kind.upper()}_SOURCE to the documented official checkout.")
    assert Path(path).is_dir(), f"Invalid configured {kind} source directory"
    return path


def windows(h=4, length=32, hist_size=0, future_size=0):
    generator = torch.Generator().manual_seed(17)
    return dict(
        insample_y=torch.randn(2, length, 1, generator=generator),
        insample_mask=torch.ones(2, length, 1),
        hist_exog=torch.randn(2, length, hist_size, generator=generator) if hist_size else None,
        futr_exog=torch.randn(2, length + h, future_size, generator=generator) if future_size else None,
        outsample_y=torch.full((2, h, 1), float("nan")),
    )


def frame(length=520, hist=False):
    t = np.arange(length)
    pieces = []
    for uid in range(2):
        part = pd.DataFrame({
            "unique_id": str(uid), "ds": pd.date_range("2020-01-01", periods=length, freq="h"),
            "y": np.sin(t / 6 + uid) + .002 * t,
        })
        if hist:
            part["x"] = np.cos(t / 9 + uid)
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def runtime():
    return dict(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
                enable_model_summary=False, batch_size=2, windows_batch_size=4,
                inference_windows_batch_size=4, val_check_steps=2)


@pytest.mark.parametrize("cls", [TimesFM3, Chronos2, SeesawNet, Dualformer, SearchCast])
def test_registry_and_invalid_horizons(cls):
    assert MODEL_FILENAME_DICT[cls.__name__.lower()] is cls
    for h in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            cls(h=h, input_size=32)
    assert TimesFM.DEFAULT_MODEL_ID == "google/timesfm-2.5-200m-pytorch"
    assert Chronos2.DEFAULT_MODEL_ID == "amazon/chronos-2"


class TimesContract:
    def __init__(self):
        self.calls = []

    def predict_batch(self, contexts, horizon, past_only_covariates=None,
                      past_future_covariates=None, **kwargs):
        self.calls.append((contexts, past_only_covariates, past_future_covariates, kwargs))
        assert kwargs["make_positive"] is False
        for i, context in enumerate(contexts):
            prediction = np.repeat(context[-1], horizon)
            if past_only_covariates is not None:
                prediction = prediction + past_only_covariates[i][0, -1]
            if past_future_covariates is not None:
                prediction = prediction + past_future_covariates[i][0, -horizon:]
            yield SimpleNamespace(
                forecast=prediction,
                quantiles=prediction[:, None] + np.arange(1, 10)[None, :] / 10,
            )


@pytest.mark.parametrize("h", range(1, 73))
def test_timesfm3_native_horizons_covariates_and_no_future_target_leak(h):
    model = TimesFM3(h=h, input_size=32, hist_exog_list=["x"], futr_exog_list=["known"])
    backend = TimesContract()
    model.__dict__["_backend"] = backend
    batch = windows(h=h, hist_size=1, future_size=1)
    prediction = model(batch)
    assert prediction.shape == (2, h, 1)
    expected = batch["insample_y"][:, -1:] + batch["hist_exog"][:, -1:] + batch["futr_exog"][:, -h:]
    torch.testing.assert_close(prediction, expected)
    contexts, past, future, _ = backend.calls[-1]
    assert contexts[0].shape == (32,) and past[0].shape == (1, 32)
    assert future[0].shape == (1, 32 + h)
    batch["outsample_y"].fill_(1e9)
    torch.testing.assert_close(model(batch), prediction)
    assert not any("backend" in name for name in model.state_dict())


def test_timesfm3_native_quantile_order_and_rejections():
    model = TimesFM3(h=3, input_size=32, loss=MQLoss(quantiles=[.9, .1, .5]))
    model.__dict__["_backend"] = TimesContract()
    batch = windows(h=3)
    expected = batch["insample_y"][:, -1:] + torch.tensor([.9, .1, .5])[None, None]
    torch.testing.assert_close(model(batch), expected.expand(2, 3, 3))
    with pytest.raises(ValueError, match="native quantiles"):
        TimesFM3(h=3, input_size=32, loss=MQLoss(quantiles=[.05, .5, .95]))
    with pytest.raises(ValueError, match="32"):
        TimesFM3(h=3, input_size=32, hist_exog_list=[f"v{i}" for i in range(32)])
    with pytest.raises(ValueError, match="15360"):
        TimesFM3(h=3, input_size=15361)
    batch["insample_mask"][:, 0] = 0
    with pytest.raises(ValueError, match="complete history"):
        model(batch)


def test_timesfm3_backend_loader_uses_v3_config(monkeypatch):
    config = {}
    def construct(value):
        config.update(vars(value))
        return TimesContract()
    module = ModuleType("timesfm3")
    module.ModelConfig = SimpleNamespace
    module.TimesFM3Forecaster = construct
    monkeypatch.setitem(sys.modules, "timesfm3", module)
    model = TimesFM3(h=1, input_size=32, revision="abc", backend_batch_size=3)
    with pytest.warns(UserWarning, match="non-commercial"):
        model(windows(h=1))
    assert config == dict(checkpoint_path="google/timesfm-3.0-pytorch", revision="abc",
                          per_core_batch_size=3, device="cpu")
    assert model.max_steps == 0


@pytest.mark.parametrize("bad", [None, np.zeros((2, 4)), np.full(4, np.nan)])
def test_timesfm3_rejects_invalid_backend_output(bad):
    model = TimesFM3(h=4, input_size=32)
    model.__dict__["_backend"] = SimpleNamespace(
        predict_batch=lambda **kwargs: [SimpleNamespace(forecast=bad)] * 2,
    )
    with pytest.raises(ValueError):
        model(windows())


class ChronosContract:
    quantiles = [.1, .5, .9]

    def predict(self, inputs, prediction_length, context_length, cross_learning):
        assert not cross_learning
        assert context_length == 32
        result = []
        for item in inputs:
            assert set(item["past_covariates"]) == {"x", "known"}
            assert set(item["future_covariates"]) == {"known"}
            value = torch.tensor(item["future_covariates"]["known"])
            assert value.shape == (prediction_length,)
            result.append(value.expand(1, 3, prediction_length))
        return result


@pytest.mark.parametrize("h", range(1, 73))
def test_existing_chronos2_is_reused_and_routes_horizon(h):
    model = Chronos2(h=h, input_size=32, hist_exog_list=["x"], futr_exog_list=["known"])
    model.__dict__["_backend"] = ChronosContract()
    batch = windows(h=h, hist_size=1, future_size=1)
    torch.testing.assert_close(model(batch), batch["futr_exog"][:, -h:])


def trainable(cls, h=4):
    return cls(h=h, input_size=32, source_dir=source(cls.__name__),
               hidden_size=16, d_ff=32, n_heads=2, dropout=0.,
               hist_exog_list=["x"], max_steps=2, **runtime())


@pytest.mark.parametrize("cls", [SeesawNet, Dualformer])
@pytest.mark.parametrize("h", [1, 24, 72])
def test_real_architecture_shapes_gradients_covariates_and_state(cls, h):
    model = trainable(cls, h).eval()
    batch = windows(h=h, hist_size=1)
    batch["hist_exog"].requires_grad_()
    prediction = model(batch)
    assert prediction.shape == (2, h, 1) and torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    assert batch["hist_exog"].grad.abs().sum() > 0
    batch["outsample_y"].fill_(123456)
    torch.testing.assert_close(prediction, model(batch))
    changed = dict(batch, hist_exog=batch["hist_exog"].detach().flip(1))
    assert not torch.allclose(prediction, model(changed))
    restored = trainable(cls, h).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(prediction, restored(batch))


def test_real_sources_coexist_without_global_namespace_pollution():
    prior = {k: v for k, v in sys.modules.copy().items()
             if k == "layers" or k.startswith("layers.") or k == "utils" or k.startswith("utils.")}
    a, b = trainable(SeesawNet), trainable(Dualformer)
    assert a.network.__class__.__module__ != b.network.__class__.__module__
    after = {k: v for k, v in sys.modules.copy().items()
             if k == "layers" or k.startswith("layers.") or k == "utils" or k.startswith("utils.")}
    assert prior == after
    assert torch.isfinite(a(windows(hist_size=1))).all()
    assert torch.isfinite(b(windows(hist_size=1))).all()
    with pytest.raises(ValueError, match="requires source_dir"):
        forecast_source(None, "SeesawNet")


@pytest.mark.parametrize("cls", [SeesawNet, Dualformer])
def test_real_trainables_nf_fit_predict_save_load(cls, tmp_path):
    nf = NeuralForecast(models=[trainable(cls)], freq="h")
    df = frame(80, hist=True)
    nf.fit(df=df, val_size=4)
    pred = nf.predict()
    assert len(pred) == 8 and np.isfinite(pred[cls.__name__]).all()
    nf.save(path=str(tmp_path / cls.__name__), overwrite=True, save_dataset=True)
    restored = NeuralForecast.load(path=str(tmp_path / cls.__name__))
    np.testing.assert_allclose(restored.predict()[cls.__name__], pred[cls.__name__], rtol=1e-5, atol=1e-5)
    with pytest.raises(Exception, match="future"):
        cls(h=4, input_size=32, source_dir=source(cls.__name__), futr_exog_list=["future"])


def ridge(h=4, **kwargs):
    return SearchCast(h=h, input_size=16, source_dir=source("SearchCast"),
                      n_trials=2, n_folds=2, cv_val_size=max(12, h),
                      ridge_alphas=[.01, .1], **runtime(), **kwargs)


@pytest.mark.parametrize("h", [1, 24, 25, 72])
def test_real_searchcast_nf_fit_predict_save_load_and_cv(h, tmp_path):
    nf = NeuralForecast(models=[ridge(h)], freq="h")
    nf.fit(df=frame(), val_size=h)
    pred = nf.predict()
    assert len(pred) == 2 * h and np.isfinite(pred.SearchCast).all()
    nf.save(path=str(tmp_path / "ridge"), overwrite=True, save_dataset=True)
    restored = NeuralForecast.load(path=str(tmp_path / "ridge"))
    np.testing.assert_allclose(restored.predict().SearchCast, pred.SearchCast, rtol=1e-5, atol=1e-5)
    trained = nf.models[0]
    copied = deepcopy(trained)
    torch.testing.assert_close(copied.ridge_weights, trained.ridge_weights)
    assert int(trained.ridge_config[:, -1].sum()) == h
    if h == 1:
        fresh = NeuralForecast(models=[ridge(h)], freq="h")
        cv = fresh.cross_validation(df=frame(), n_windows=2, step_size=1, refit=True)
        assert len(cv) == 4 and np.isfinite(cv.SearchCast).all()


def test_real_searchcast_excludes_outer_holdouts():
    df = frame(180)
    dataset, *_ = TimeSeriesDataset.from_df(df)
    altered = df.copy()
    for _, group in altered.groupby("unique_id"):
        altered.loc[group.index[-20:], "y"] = 999999.
    different, *_ = TimeSeriesDataset.from_df(altered)
    a, b = ridge(), ridge()
    a._fit(dataset, batch_size=2, val_size=10, test_size=10)
    b._fit(different, batch_size=2, val_size=10, test_size=10)
    torch.testing.assert_close(a.ridge_weights, b.ridge_weights, rtol=0, atol=0)
    torch.testing.assert_close(a.ridge_config, b.ridge_config)
    torch.testing.assert_close(a.ridge_stats, b.ridge_stats, rtol=0, atol=0)
    series = a._training_series(dataset, 10, 10)
    assert all(len(s) == 160 for s in series)
    # At cut=100, training targets stop at 99 and validation targets start at 100.
    sequence = [torch.arange(160).double()]
    _, train_y = a._windows(sequence, 16, [100])
    _, val_y = a._windows(sequence, 16, [100], 12)
    assert train_y.max() == 99 and val_y.min() == 100 and val_y.max() == 111


def test_real_searchcast_import_does_not_change_process_settings():
    import optuna
    before = (os.environ.get("OMP_NUM_THREADS"), os.environ.get("MKL_NUM_THREADS"),
              torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
              optuna.logging.get_verbosity())
    forecast_source(source("SearchCast"), "SearchCast")
    after = (os.environ.get("OMP_NUM_THREADS"), os.environ.get("MKL_NUM_THREADS"),
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             optuna.logging.get_verbosity())
    assert before == after


def test_searchcast_rejects_unsupported_training_options():
    for options in (dict(max_steps=1), dict(loss=MAE()), dict(scaler_type="standard"),
                    dict(cv_val_size=3), dict(ridge_alphas=[0]), dict(n_trials=0)):
        with pytest.raises(ValueError):
            SearchCast(h=4, input_size=16, **options)
    for options in (dict(hist_exog_list=["x"]), dict(futr_exog_list=["x"])):
        with pytest.raises(Exception):
            SearchCast(h=4, input_size=16, **options)
    model = SearchCast(h=4, input_size=16, loss=MSE())
    with pytest.raises(RuntimeError, match="fitted"):
        model(windows(length=16))


def test_dualformer_device_patch_preserves_delay_aggregation():
    source_module = forecast_source(source("Dualformer"), "Dualformer")
    layer = source_module.AutoCorrelation(mask_flag=False, factor=1).eval()
    generator = torch.Generator().manual_seed(31)
    values = torch.randn(2, 2, 3, 16, generator=generator)
    correlation = torch.randn(2, 2, 3, 16, generator=generator)
    scores, delays = correlation.mean((1, 2)).topk(2, dim=-1)
    weights = scores.softmax(-1)
    expected = torch.stack([
        sum(weights[b, k] * torch.roll(values[b], -int(delays[b, k]), -1)
            for k in range(2)) for b in range(2)
    ])
    torch.testing.assert_close(layer.time_delay_agg_inference(values, correlation), expected)
    # Exercise the second device-only replacement too, with per-channel lags.
    scores, delays = correlation.topk(2, dim=-1)
    weights = scores.softmax(-1)
    index = (torch.arange(16)[None, None, None, :, None] + delays[..., None, :]) % 16
    gathered = torch.gather(values[..., None].expand(-1, -1, -1, -1, 2), 3, index)
    expected_full = (gathered * weights[..., None, :]).sum(-1)
    torch.testing.assert_close(layer.time_delay_agg_full(values, correlation), expected_full)
