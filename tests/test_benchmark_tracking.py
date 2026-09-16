import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from neuralforecast.benchmark_tracking import (
    Tracking,
    ensure_open_project,
    run_id,
    safe_config,
)


@pytest.mark.parametrize("access", ["USER_WRITE", "USER_READ", "PRIVATE"])
def test_project_visibility_requires_open(monkeypatch, access):
    calls = []
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_BASE_URL", "https://wandb.example/")

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"data": {"upsertModel": {"project": {"access": access}}}},
        )

    monkeypatch.setattr("requests.post", post)
    if access == "USER_WRITE":
        ensure_open_project("team", "benchmark")
    else:
        with pytest.raises(RuntimeError, match="is not Open"):
            ensure_open_project("team", "benchmark")
    url, kwargs = calls[0]
    assert url == "https://wandb.example/graphql"
    assert kwargs["json"]["variables"] == {"entity": "team", "project": "benchmark"}
    assert 'access: "USER_WRITE"' in kwargs["json"]["query"]


def test_project_visibility_rejects_api_errors(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"errors": [{"message": "Permission denied"}]},
        ),
    )
    with pytest.raises(RuntimeError, match="Could not set W&B project"):
        ensure_open_project("team", "benchmark")


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


def test_evaluation_table_publishes_metrics_and_definitions(monkeypatch):
    import pandas as pd

    tracker = Tracking()
    tracker.run = SimpleNamespace(summary={})
    tables = []
    monkeypatch.setattr(tracker, "table", lambda name, frame: tables.append(name))
    frame = pd.DataFrame(
        [
            dict(
                candidate="Naive",
                mae=1.0,
                mape_pct=float("nan"),
                mse=1.0,
                rmse=1.0,
                da_pct=0.0,
                mape_n=0,
            )
        ]
    )
    tracker.evaluation_table("phase2", frame, {"da": "fixed-origin signs"})
    assert tables == ["phase2/leaderboard_with_naive"]
    assert tracker.run.summary["phase2_with_naive/Naive/rmse"] == 1.0
    assert tracker.run.summary["phase2_with_naive/Naive/mape_pct"] is None
    assert (
        tracker.run.summary["phase2/leaderboard_with_naive_definitions"]["da"]
        == "fixed-origin signs"
    )


def test_forecast_publishes_arrays_and_ordered_table(monkeypatch):
    tracker = Tracking()
    tracker.run = SimpleNamespace(summary={})
    tables = []
    monkeypatch.setattr(
        tracker,
        "table",
        lambda name, frame: tables.append((name, frame.to_dict(orient="list"))),
    )
    tracker.forecast([2, 4], [3, 3.5], 1)
    assert tracker.run.summary["forecast/actual"] == [2.0, 4.0]
    assert tracker.run.summary["forecast/prediction"] == [3.0, 3.5]
    assert tables == [
        (
            "forecast/series",
            {
                "horizon": [1, 2],
                "actual": [2.0, 4.0],
                "prediction": [3.0, 3.5],
                "error": [1.0, -0.5],
                "forecast_origin": [1.0, 1.0],
            },
        )
    ]


def test_forecast_rejects_unaligned_arrays():
    tracker = Tracking()
    tracker.run = SimpleNamespace(summary={})
    with pytest.raises(ValueError, match="aligned nonempty"):
        tracker.forecast([1], [], 0)


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
