"""Batch-two contracts and real NF/source integration; no pretrained downloads."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from neuralforecast import NeuralForecast
from neuralforecast.core import MODEL_FILENAME_DICT
from neuralforecast.models import APT, DAG, GLAFF, KITE, BaguanTS, ChronosX, Moirai2, RAG4CTS
from neuralforecast.models._research_source import source_module
import neuralforecast.models._isolated_research_forecast as isolated_forecast
import neuralforecast.models.research as research

MODELS = [DAG, KITE, GLAFF, APT, Moirai2, ChronosX, BaguanTS, RAG4CTS]
QUIET = dict(accelerator="cpu", devices=1, logger=False,
             enable_progress_bar=False, enable_model_summary=False)


def source(name):
    directory = os.environ.get("NF_RESEARCH_SOURCES")
    if not directory:
        pytest.skip("Set NF_RESEARCH_SOURCES to run actual official-source tests.")
    path = Path(directory) / name
    assert path.is_dir(), f"Missing pinned checkout: {path}"
    return str(path)


def make_model(name, **extra):
    args = dict(h=4, input_size=16, source_dir=source(name), max_steps=3,
                val_check_steps=1, windows_batch_size=4, batch_size=2,
                inference_windows_batch_size=4, **QUIET)
    if name == "DAG":
        args.update(futr_exog_list=["x"], hidden_size=32, n_heads=4,
                    encoder_layers=1, patch_len=4, stride=2, d_ff=64, dropout=0)
    elif name == "KITE":
        args.update(futr_exog_list=["x"], hidden_size=32, n_heads=4, depth=1,
                    num_sampling_steps=2, num_samples=2, p_uncond=0.0)
    elif name == "GLAFF":
        args.update(futr_exog_list=[f"c{i}" for i in range(6)], hidden_size=16,
                    n_heads=4, encoder_layers=1, d_ff=32, dropout=0, moving_avg_window=3)
    else:
        args.update(futr_exog_list=["tod", "dow"], warmup_steps=1,
                    timestamp_dim=8, timestamp_hidden=16, num_prototypes=4,
                    top_k=2, moving_avg_window=3)
    args.update(extra)
    return {"DAG": DAG, "KITE": KITE, "GLAFF": GLAFF, "APT": APT}[name](**args)


def frame():
    t = np.arange(52, dtype=float)
    data = pd.DataFrame(dict(ds=pd.date_range("2020-01-01", periods=len(t), freq="h"),
                             y=np.sin(t / 4) + t / 30, x=np.cos(t / 5),
                             tod=(t % 24) / 24 - 0.5, dow=((t // 24) % 7) / 7 - 0.5))
    for i in range(6):
        data[f"c{i}"] = np.sin(t / (i + 2))
    return pd.concat([data.assign(unique_id=uid, y=data.y + shift)
                      for uid, shift in (("a", 0), ("b", 2))], ignore_index=True)


def window(model, batch=2):
    generator = torch.Generator().manual_seed(7)
    y = torch.randn(batch, model.input_size, 1, generator=generator)
    values = dict(insample_y=y, insample_mask=torch.ones_like(y),
                  hist_exog=None, futr_exog=None, stat_exog=None)
    if model.hist_exog_size:
        values["hist_exog"] = torch.randn(batch, model.input_size, model.hist_exog_size, generator=generator)
    if model.futr_exog_size:
        values["futr_exog"] = torch.randn(batch, model.input_size + model.h, model.futr_exog_size, generator=generator)
    if isinstance(model, APT):
        t = torch.arange(model.input_size + model.h)
        values["futr_exog"] = torch.stack(((t % 24) / 24 - .5, ((t // 24) % 7) / 7 - .5), -1).repeat(batch, 1, 1)
    return values


@pytest.mark.parametrize("cls", MODELS)
def test_registration(cls):
    assert MODEL_FILENAME_DICT[cls.__name__.lower()] is cls


@pytest.mark.parametrize("name", ["DAG", "KITE", "GLAFF", "APT"])
def test_real_nf_fit_predict_save_load(name, tmp_path):
    torch.set_num_threads(1)
    model = make_model(name)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    data = frame()
    train = data.groupby("unique_id", sort=False).head(48)
    future = data.groupby("unique_id", sort=False).tail(4).drop(columns="y")
    nf = NeuralForecast(models=[model], freq="h")
    nf.fit(df=train, val_size=4)
    fitted = nf.models[0]
    assert any(not torch.equal(before[k], v) for k, v in fitted.named_parameters())
    predicted = nf.predict(futr_df=future)
    assert len(predicted) == 8 and np.isfinite(predicted[name]).all()
    path = str(tmp_path / name)
    nf.save(path=path, overwrite=False, save_dataset=True)
    loaded = NeuralForecast.load(path=path)
    restored = loaded.predict(futr_df=future)
    np.testing.assert_allclose(predicted[name], restored[name], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", ["DAG", "GLAFF", "APT"])
def test_source_forward_covariate_gradient_and_no_future_target(name):
    model = make_model(name, warmup_steps=0) if name == "APT" else make_model(name)
    model.eval()
    batch = window(model)
    batch["futr_exog"].requires_grad_()
    output = model(batch)
    assert output.shape == (2, 4, 1) and torch.isfinite(output).all()
    output.square().sum().backward()
    if name != "APT":  # APT intentionally discretizes calendar bins.
        assert batch["futr_exog"].grad is not None
        assert batch["futr_exog"].grad.abs().sum() > 0
    batch["outsample_y"] = torch.full_like(output, float("nan"))
    torch.testing.assert_close(model(batch), output.detach())


def test_dag_adds_official_auxiliary(monkeypatch):
    model = make_model("DAG")
    def parent(self, batch, index):
        self.__dict__["_auxiliary"] = torch.tensor(2., requires_grad=True)
        return torch.tensor(3., requires_grad=True)
    monkeypatch.setattr(research.ExogenousModel, "training_step", parent)
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    assert model.training_step({"temporal_cols": []}, 0).item() == 5
    with pytest.raises(ValueError, match="sample_weight"):
        model.training_step({"temporal_cols": ["sample_weight"]}, 0)


@pytest.mark.parametrize("past_only", [False, True])
def test_kite_original_flow_objective(past_only):
    kwargs = dict(hist_exog_list=["x"], futr_exog_list=None) if past_only else {}
    model = make_model("KITE", **kwargs)
    y, past, future = model._flow_inputs(window(model))
    model.train()
    objective = model.network.train_function(y, past, torch.randn(2, 4, 1), future)
    assert torch.isfinite(objective) and objective.requires_grad
    objective.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.network.parameters())
    model.eval()
    assert torch.isfinite(model(window(model))).all()


def test_apt_stages_and_calendar_validation(monkeypatch):
    model = make_model("APT")
    model.__dict__["_trainer"] = SimpleNamespace(global_step=0)
    def parent(self, batch, index):
        self(window(self))
        return sum(p.sum() * 0 for p in self.affine.parameters()) + 123
    monkeypatch.setattr(research.ExogenousModel, "training_step", parent)
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    first = model.training_step({"temporal_cols": []}, 0)
    assert first.requires_grad and first.item() != 123
    assert not any(p.requires_grad for p in model.backbone.parameters())
    model._trainer.global_step = 1
    assert model.training_step({"temporal_cols": []}, 0).item() == 123
    assert not any(p.requires_grad for p in model.backbone.parameters())
    model._trainer.global_step = 2
    model.training_step({"temporal_cols": []}, 0)
    assert all(p.requires_grad for p in model.backbone.parameters())
    batch = window(model)
    batch["futr_exog"][0, 0, 0] = 99
    with pytest.raises(ValueError, match="timestamps"):
        model(batch)
    model.__dict__.pop("_trainer", None)


def test_source_rejects_modified_entry(tmp_path):
    file = tmp_path / "plugin/Plugin/model.py"
    file.parent.mkdir(parents=True)
    file.write_text("raise RuntimeError('must never execute')")
    with pytest.raises(ValueError, match="reviewed blob"):
        source_module(tmp_path, "GLAFF")


def test_rag_uses_real_retriever_and_no_query_future_labels():
    model = RAG4CTS(h=4, input_size=24, source_dir=source("RAG4CTS"), query_size=4,
                    futr_exog_list=["x"], **QUIET)
    calls = []
    class Backend:
        def predict_df(self, df, *, future_df, prediction_length, **kwargs):
            assert "y" not in future_df
            assert prediction_length == 4
            calls.append((df.copy(), future_df.copy()))
            return pd.DataFrame({"0.5": future_df["x"].to_numpy()})
    official = source_module(source("RAG4CTS"), "RAG4CTS")
    config = dict(feature_mapping={"target": "y", "covariates": ["x"]},
                  model_window={"left_padding": 4, "core_window": 4, "right_padding": 0, "total_size": 8})
    model.__dict__["_backend"] = official.RAGPipeline(config, Backend(), ["x"], use_hint=False)
    batch = window(model)
    target = torch.arange(24.).repeat(2, 1).unsqueeze(-1)
    batch["insample_y"] = target
    query, bank = model._retrieval_inputs(target[0, :, 0].numpy(), batch["futr_exog"][0].numpy())
    assert all(item["df"].y.max() < 20 for item in bank)
    assert query.y.iloc[-4:].eq(0).all()
    prediction = model(batch)
    torch.testing.assert_close(prediction, batch["futr_exog"][:, -4:])
    assert len(calls) == 2


@pytest.mark.parametrize("cls", [ChronosX, BaguanTS])
def test_isolated_forward_transports_only_permitted_inputs(cls, monkeypatch):
    args = dict(h=4, input_size=16, futr_exog_list=["x"], model_id="/local/checkpoint",
                backend_python=sys.executable, **QUIET)
    if cls is BaguanTS:
        args.update(source_dir="/source", config_path="/source/config.yml")
    model = cls(**args)
    def worker(executable, config, arrays, timeout):
        assert set(arrays) == {"y", "futr"}
        assert arrays["y"].shape == (2, 16, 1)
        return arrays["futr"][:, -4:, 0]
    monkeypatch.setattr(isolated_forecast, "run_worker", worker)
    batch = window(model)
    batch["outsample_y"] = torch.full((2, 4, 1), float("nan"))
    torch.testing.assert_close(model(batch), batch["futr_exog"][:, -4:])


def test_rejected_configuration():
    with pytest.raises(ValueError, match="inference-only"):
        Moirai2(h=4, input_size=16, max_steps=1)
    with pytest.raises(ValueError, match="fixed"):
        Moirai2(h=4, input_size=16, patch_size=8)
    with pytest.raises(ValueError, match="checkpoint"):
        ChronosX(h=4, input_size=16, futr_exog_list=["x"])
    with pytest.raises(ValueError, match="reserves"):
        RAG4CTS(h=4, input_size=24, source_dir="/source", futr_exog_list=["time"])
    with pytest.raises(ValueError, match="exactly one"):
        KITE(h=4, input_size=16, hist_exog_list=["a"], futr_exog_list=["b"])
    with pytest.raises(ValueError, match="six"):
        GLAFF(h=4, input_size=16, futr_exog_list=["a"])
