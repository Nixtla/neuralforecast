"""Run official architectures with tiny RANDOM checkpoints, never accuracy tests.

Standalone: pytest in each backend's environment; neuralforecast is not imported.
Set NF_BACKEND_SMOKE=moirai2|chronosx|baguants to select an installed backend.
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
WORKERS = ROOT / "neuralforecast/models"


def worker_module(name):
    sys.path.insert(0, str(WORKERS))
    spec = importlib.util.spec_from_file_location(name, WORKERS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload():
    rng = np.random.default_rng(12)
    return dict(y=rng.normal(size=(2, 32, 1)).astype("float32"),
                hist=rng.normal(size=(2, 32, 1)).astype("float32"),
                futr=rng.normal(size=(2, 36, 1)).astype("float32"))


def run_worker(tmp_path, filename, config, arrays):
    directory = tmp_path / "request"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config))
    np.savez(directory / "inputs.npz", **arrays)
    completed = subprocess.run([sys.executable, str(WORKERS / filename), str(directory)],
                               capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    with np.load(directory / "outputs.npz", allow_pickle=False) as data:
        result = data["prediction"].copy()
    assert np.isfinite(result).all()
    return result


def test_chronosx_missing_indicators():
    arrays = payload()
    result = worker_module("_research_worker").chronosx_covariates(arrays, 32, 4)
    first = result[0]
    np.testing.assert_array_equal(first["past_feat_dynamic_real"][:, 2:], 0)
    np.testing.assert_array_equal(first["future_feat_dynamic_real"][:, 0], -1)
    np.testing.assert_array_equal(first["future_feat_dynamic_real"][:, 2], 1)
    np.testing.assert_array_equal(first["future_feat_dynamic_real"][:, 3], 0)
    np.testing.assert_array_equal(first["future_feat_dynamic_real"][:, 1], arrays["futr"][0, -4:, 0])


@pytest.mark.parametrize("kind,prefix", [("moirai", "Moirai"), ("moirai_moe", "MoiraiMoE"), ("moirai2", "Moirai2")])
def test_uni2ts_worker_aggregation_contract(kind, prefix, tmp_path, monkeypatch):
    # Doubles test routing/backward compatibility only; the actual Moirai2 test is below.
    from types import SimpleNamespace
    module = worker_module("_uni2ts_worker")
    backend = SimpleNamespace(quantile_levels=(0.1, 0.5, 0.9))
    class Module:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return backend
    class Forecast:
        def __init__(self, **kwargs):
            if kind == "moirai2":
                assert "patch_size" not in kwargs and "num_samples" not in kwargs
            else:
                assert kwargs["patch_size"] == 16 and kwargs["num_samples"] == 3
        def to(self, device):
            return self
        def eval(self):
            return self
        def __call__(self, **kwargs):
            return torch.tensor([1., 7., 30.]).reshape(1, 3, 1, 1).repeat(2, 1, 4, 1)
    package = SimpleNamespace(**{prefix + "Module": Module, prefix + "Forecast": Forecast})
    monkeypatch.setattr(module.importlib, "import_module", lambda name: package)
    config = dict(kind=kind, model_id="local", revision=None, h=4, input_size=32,
                  futr_size=1, hist_size=1, patch_size=16, num_samples=3, device="cpu", random_seed=1)
    (tmp_path / "config.json").write_text(json.dumps(config))
    np.savez(tmp_path / "inputs.npz", past_target=np.ones((2, 32, 1), dtype="float32"))
    module.main(tmp_path)
    with np.load(tmp_path / "outputs.npz", allow_pickle=False) as data:
        np.testing.assert_allclose(data["prediction"], 7 if kind == "moirai2" else 38/3)


def test_actual_moirai2_checkpoint(tmp_path):
    if os.environ.get("NF_BACKEND_SMOKE") != "moirai2":
        pytest.skip("Requires isolated official uni2ts environment.")
    from uni2ts.model.moirai2 import Moirai2Module, Moirai2Forecast
    torch.set_num_threads(1)
    torch.manual_seed(1)
    module = Moirai2Module(d_model=64, d_ff=128, num_layers=1, patch_size=16,
                          max_seq_len=128, attn_dropout_p=0, dropout_p=0)
    checkpoint = tmp_path / "tiny-moirai2"
    module.save_pretrained(checkpoint)
    raw = payload()
    arrays = dict(past_target=raw["y"], past_observed_target=np.ones_like(raw["y"], dtype=bool),
                  past_is_pad=np.zeros((2, 32), dtype=bool), feat_dynamic_real=raw["futr"],
                  observed_feat_dynamic_real=np.ones_like(raw["futr"], dtype=bool),
                  past_feat_dynamic_real=raw["hist"],
                  past_observed_feat_dynamic_real=np.ones_like(raw["hist"], dtype=bool))
    config = dict(kind="moirai2", model_id=str(checkpoint), revision=None, h=4, input_size=32,
                  futr_size=1, hist_size=1, device="cpu", random_seed=1)
    result = run_worker(tmp_path, "_uni2ts_worker.py", config, arrays)
    assert result.shape == (2, 4)
    forecast = Moirai2Forecast(module=module, prediction_length=4, context_length=32,
                              target_dim=1, feat_dynamic_real_dim=1, past_feat_dynamic_real_dim=1).eval()
    with torch.no_grad():
        direct = forecast(**{k: torch.from_numpy(v) for k, v in arrays.items()})[:, 4].numpy()
    np.testing.assert_allclose(result, direct, rtol=1e-5, atol=1e-5)


def test_actual_chronosx_checkpoint(tmp_path):
    if os.environ.get("NF_BACKEND_SMOKE") != "chronosx":
        pytest.skip("Requires isolated official ChronosX environment.")
    from chronosx.chronosx import ChronosX
    from transformers import T5Config
    config = T5Config(d_model=32, d_ff=64, num_layers=1, num_decoder_layers=1,
                      num_heads=4, d_kv=8, vocab_size=64, pad_token_id=0,
                      eos_token_id=1, decoder_start_token_id=0)
    config.chronos_config = dict(tokenizer_class="MeanScaleUniformBins",
                                tokenizer_kwargs={"low_limit": -15., "high_limit": 15.},
                                context_length=32, prediction_length=4, n_tokens=64,
                                n_special_tokens=2, pad_token_id=0, eos_token_id=1,
                                use_eos_token=True, model_type="seq2seq", num_samples=2,
                                temperature=1., top_k=20, top_p=1.)
    torch.manual_seed(1)
    model = ChronosX.set_state(num_covariates=4, covariate_injection="IIB+OIB",
                              hidden_dim=16, num_layers=1, vocab_size=64, model_dim=32)(config)
    checkpoint = tmp_path / "tiny-chronosx"
    model.save_pretrained(checkpoint, safe_serialization=True)
    request = dict(kind="chronosx", model_id=str(checkpoint), h=4, input_size=32,
                   num_samples=2, hidden_dim=16, num_layers=1, device="cpu", random_seed=1)
    result = run_worker(tmp_path, "_research_worker.py", request, payload())
    assert result.shape == (2, 4)


def test_actual_baguants_checkpoint(tmp_path):
    if os.environ.get("NF_BACKEND_SMOKE") != "baguants":
        pytest.skip("Requires isolated BaguanTS source environment.")
    import yaml
    source = Path(os.environ["NF_RESEARCH_SOURCES"]) / "BaguanTS"
    sys.path.insert(0, str(source))
    from src.pipeline.factory import ModelFactory
    config = yaml.safe_load((source / "configs/model_config.yml").read_text())
    config["nlayers"] = 1
    config["input_encoder"]["params"].update(ninp=32, patch_size=4)
    config["positional_embedder"]["params"].update(ninp=32)
    params = config["transformers"]["params"]
    params.update(d_model=32, dim_feedforward=64)
    for key in ("attention_between_features", "attention_between_ts", "attention_between_samples"):
        params[key]["params"].update(d_model=32, nheads=4)
    config["prediction_head"]["params"].update(input_dim=32, hidden_size=64, patch_size=4)
    config_path = tmp_path / "tiny.yml"
    config_path.write_text(yaml.safe_dump(config))
    torch.manual_seed(1)
    model = ModelFactory.from_config(str(config_path))
    checkpoint = tmp_path / "tiny.ckpt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    request = dict(kind="baguants", source_dir=str(source), config_path=str(config_path),
                   model_id=str(checkpoint), h=4, input_size=32, context_size=16,
                   neighbors=2, num_samples=1, device="cpu", random_seed=1)
    result = run_worker(tmp_path, "_research_worker.py", request, payload())
    assert result.shape == (2, 4)
