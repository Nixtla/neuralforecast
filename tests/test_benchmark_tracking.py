import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from neuralforecast.benchmark_tracking import Tracking, run_id, safe_config


def test_disabled_tracking_does_not_import_or_initialize_wandb(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    tracker = Tracking()
    tracker.log({"x": 1})
    tracker.finish()
    assert tracker.run is None


def test_stable_ids_isolate_phases_folds_and_experiments():
    base = ("group", "phase1", "GRU", 1, 2)
    assert run_id(*base) == run_id(*base)
    assert (
        len(
            {
                run_id(*base),
                run_id("group", "phase2", "GRU", 1, 2),
                run_id("group", "phase1", "GRU", 1, 3),
                run_id("next-group", "phase1", "GRU", 1, 2),
            }
        )
        == 4
    )


def test_credentials_are_removed(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-private-value")
    assert safe_config(
        {
            "nested": {"api_key": "hidden", "lr": 0.1},
            "error": "test-private-value",
            "password": "hidden",
        }
    ) == {"nested": {"lr": 0.1}, "error": "[REDACTED]"}


def test_sh_tracking_resumes_explicit_owned_run(monkeypatch, tmp_path):
    calls = []
    run = SimpleNamespace(define_metric=lambda *a, **k: None)

    def init(**kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=init, Settings=lambda **k: k)
    )
    options = {
        "entity": "team",
        "project": "project",
        "group": "exp",
        "directory": str(tmp_path),
    }
    Tracking(options, phase="phase1", candidate="GRU", fold=3)
    Tracking(options, phase="phase1", candidate="GRU", fold=3)
    assert calls[0]["id"] == calls[1]["id"]
    assert calls[0]["resume"] == "allow"
    assert calls[0]["save_code"] is False


@pytest.fixture
def runner():
    path = Path(__file__).parents[1] / "experiments/commodity_sota/run.py"
    spec = importlib.util.spec_from_file_location("commodity_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["--preflight", "--smoke-test"])
def test_non_experiment_cannot_enable_wandb(runner, monkeypatch, mode):
    monkeypatch.setattr(
        sys, "argv", ["run.py", "--data", "unused", "--target", "y", "--wandb", mode]
    )
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert runner.TRACKING is None


def test_lora_checkpoint_can_differ_from_inference(runner):
    config = {
        "TimesFM": {"backend_device": "cuda"},
        "TimesFM-LoRA": {"model_id": "transformers-weights"},
    }
    assert "model_id" not in runner._fixed_kwargs(config, "TimesFM")
    assert (
        runner._fixed_kwargs(config, "TimesFM", "LoRA")["model_id"]
        == "transformers-weights"
    )


def test_toto_legacy_hub_uses_official_local_loader(monkeypatch, tmp_path):
    from neuralforecast.models.toto import Toto

    calls = []

    class Official:
        @classmethod
        def _from_pretrained(cls, *, proxies, resume_download):
            raise AssertionError("Legacy Hub route must not be called")

        @classmethod
        def load_from_checkpoint(cls, path, strict):
            calls.append((path, strict))
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        model = "official-model"

    monkeypatch.setitem(sys.modules, "toto.model.toto", SimpleNamespace(Toto=Official))
    monkeypatch.setitem(
        sys.modules,
        "toto.inference.forecaster",
        SimpleNamespace(TotoForecaster=lambda model: model),
    )
    monkeypatch.setitem(
        sys.modules,
        "toto.data.util.dataset",
        SimpleNamespace(MaskedTimeseries="inputs"),
    )
    model = Toto(h=16, input_size=32, model_id=str(tmp_path), backend_device="cpu")
    assert model._load_backend() == ("official-model", "inputs")
    assert calls == [(str(tmp_path), False)]
