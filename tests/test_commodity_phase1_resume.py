"""Resume interrupted successive halving without repeating completed fold jobs."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from neuralforecast.benchmark import Fold, SHPlan


@pytest.fixture
def runner(monkeypatch):
    path = Path(__file__).parents[1] / "experiments/commodity_sota/run.py"
    spec = importlib.util.spec_from_file_location("phase1_resume_runner", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "mode", ["partial", "complete", "early_stop", "missing", "incompatible", "truncated"]
)
def test_phase1_replays_scores_and_continues_saved_checkpoints(
    runner, tmp_path, monkeypatch, mode
):
    candidate = runner.Candidate(
        "Model",
        "scratch_hpo",
        "Model",
        [{"cid": 0}, {"cid": 1}],
        SHPlan(budgets=(100, 250), survivors=(2, 1)),
    )
    folds = [Fold(0, 4, 6), Fold(1, 5, 7)]
    root = tmp_path / "checkpoints_phase1"

    def result(cid, fold, budget):
        return dict(
            ok=True,
            candidate="Model",
            config_id=cid,
            fold=fold,
            budget=budget,
            protocol_version=runner.PROTOCOL_VERSION,
            actual=[1.0, 2.0],
            prediction=[1.0 + cid, 2.0 + cid],
            rmse=float(cid),
            forecast_origin=0.0,
            stop_reason="early_stopping" if mode == "early_stop" else "max_steps",
        )

    for cid in range(2):
        for fold in folds:
            folder = root / "Model" / str(cid) / str(fold.index)
            folder.mkdir(parents=True)
            checkpoint = folder / "resume.ckpt"
            checkpoint.write_text("saved optimizer state")
            row = dict(result(cid, fold.index, 100), checkpoint=str(checkpoint))
            if mode == "incompatible" and cid == 1 and fold.index == 1:
                row["protocol_version"] = "wrong"
            if not (mode == "missing" and cid == 1 and fold.index == 1):
                (folder / "result-100.json").write_text(json.dumps(row))
            if mode == "truncated" and cid == 1 and fold.index == 1:
                (folder / "result-100.json").write_text("")
            if mode == "complete" and cid == 0:
                (folder / "result-250.json").write_text(
                    json.dumps(result(cid, fold.index, 250))
                )

    submitted = []

    def submit(*args):
        submitted.append(args)
        return len(submitted) - 1

    def get(ref):
        args = submitted[ref]
        return result(args[2], args[4].index, args[5])

    monkeypatch.setattr(runner, "_evaluate_job", SimpleNamespace(remote=submit))
    monkeypatch.setattr(
        runner,
        "ray",
        SimpleNamespace(
            wait=lambda refs, **kw: ([refs[0]], refs[1:]),
            get=get,
        ),
    )
    trials, failures = runner._run_phase1("data", [candidate], folds, root, resume=True)
    assert not failures
    assert candidate.best_config == {"cid": 0}
    assert len(trials) == 3
    assert len(pd.read_csv(tmp_path / "phase1_trials.csv")) == 3
    if mode in {"complete", "early_stop"}:
        assert not submitted
    else:
        higher = [args for args in submitted if args[5] == 250]
        assert len(higher) == 2
        assert all(args[2] == 0 and args[6].endswith("resume.ckpt") for args in higher)
        lower = [args for args in submitted if args[5] == 100]
        assert len(lower) == (1 if mode in {"missing", "incompatible", "truncated"} else 0)


def test_resume_missing_ranking_returns_to_phase1(runner, tmp_path, monkeypatch):
    source = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-05", periods=112, freq="W-SUN"),
            "y": [float(i % 7 + 1) for i in range(112)],
        }
    ).to_csv(source, index=False)
    output = tmp_path / "output"
    candidate = runner.Candidate(
        "Model",
        "scratch_hpo",
        "Model",
        [{}],
        None,
        best_score=100.0,
        best_config={},
        best_fold_scores=(100.0,) * 3,
    )
    monkeypatch.setattr(runner, "_minimum_train", lambda *a: 32)
    monkeypatch.setattr(runner, "_build_candidates", lambda *a: ([candidate], []))
    calls = []

    def phase1(*args, **kwargs):
        calls.append(kwargs.get("resume", False))
        return [], []

    monkeypatch.setattr(runner, "_run_phase1", phase1)
    monkeypatch.setattr(
        runner, "_restore_phase1", lambda *a: pytest.fail("Ranking is not ready")
    )
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.ray, "init", lambda **kw: None)
    monkeypatch.setattr(runner.ray, "shutdown", lambda: None)
    arguments = [
        "run",
        "--data",
        str(source),
        "--target",
        "y",
        "--output",
        str(output),
        "--scheduler",
        "ray",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    runner.main()
    path = output / "run_config.json"
    config = json.loads(path.read_text())
    config["status"] = "phase1"
    path.write_text(json.dumps(config))
    (output / "phase1_ranking.csv").unlink()
    monkeypatch.setattr(sys, "argv", arguments + ["--resume"])
    runner.main()
    assert calls == [False, True]
    assert (output / "phase1_ranking.csv").is_file()
    assert json.loads(path.read_text())["status"] == "no_models_above_naive"
