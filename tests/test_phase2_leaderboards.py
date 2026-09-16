"""Regression coverage for live ranking, backfill, and independent publication."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from neuralforecast.benchmark import Fold
from neuralforecast.benchmark_leaderboard import (
    Phase2Publisher,
    TABLE_KEY,
    leaderboard,
    metric_definitions,
    phase2_publication,
    publisher_lock,
)


def load_script(name):
    path = Path(__file__).parents[1] / "experiments/commodity_sota" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"leaderboard_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def experiment(tmp_path):
    config = dict(
        horizon=2,
        phase2_first_fold=10,
        phase2_last_fold=11,
        phase2_start_cutoff=3,
        phase2_folds=2,
        step_size=1,
        phase2_policy={"max_steps": 500},
        protocol_version="test-v1",
        status="phase1",
        wandb=dict(entity="team", project="project", group="exp"),
    )
    (tmp_path / "run_config.json").write_text(json.dumps(config))
    pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}).to_pickle(
        tmp_path / "weekly.pkl"
    )
    pd.DataFrame([dict(candidate="Model", protocol="zero_shot", selected=True)]).to_csv(
        tmp_path / "phase1_ranking.csv",
        index=False,
    )
    for fold, actual in [(10, [4.0, 5.0]), (11, [5.0, 6.0])]:
        folder = tmp_path / "checkpoints_phase2" / "Model" / "0" / str(fold)
        folder.mkdir(parents=True)
        (folder / "result-0.json").write_text(
            json.dumps(
                dict(
                    ok=True,
                    candidate="Model",
                    config_id=0,
                    budget=0,
                    fold=fold,
                    actual=actual,
                    prediction=actual,
                    protocol_version="test-v1",
                )
            )
        )
    return tmp_path, config


def test_backfill_requires_every_retained_fold_and_ignores_extra_folds(experiment):
    output, config = experiment
    script = load_script("backfill_phase2_leaderboards")
    path = output / "checkpoints_phase2/Model/0/11/result-0.json"
    original = path.read_text()
    path.write_text("{")  # A worker is in the middle of writing.
    _, snapshot = script.collect_snapshot(output)
    assert snapshot[0].candidate.tolist() == ["Naive"]
    path.write_text(original)
    extra = output / "checkpoints_phase2/Model/0/9"
    extra.mkdir()
    (extra / "result-0.json").write_text("not an in-window result")
    _, snapshot = script.collect_snapshot(output)
    assert snapshot[0].candidate.tolist() == ["Model", "Naive"]
    assert snapshot[0].iloc[0].rmse == 0
    assert snapshot[2:] == (1, "phase2")


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"ok": False}, "incomplete"),
        ({"prediction": [float("nan"), 6]}, "incomplete"),
        ({"protocol_version": "wrong"}, "incompatible"),
        ({"actual": [99, 99]}, "actual"),
    ],
)
def test_backfill_rejects_failed_or_incompatible_results(experiment, change, expected):
    output, _ = experiment
    script = load_script("backfill_phase2_leaderboards")
    path = output / "checkpoints_phase2/Model/0/11/result-0.json"
    result = json.loads(path.read_text())
    path.write_text(json.dumps({**result, **change}))
    if expected == "incomplete":
        assert script.collect_snapshot(output)[1][0].candidate.tolist() == ["Naive"]
    else:
        with pytest.raises(ValueError):
            script.collect_snapshot(output)


def test_final_csv_is_validated_against_predictions(experiment):
    output, config = experiment
    script = load_script("backfill_phase2_leaderboards")
    reference = script.evaluation_reference(output, config)
    predictions = script.saved_predictions(output, config, {"Model": "zero_shot"})
    pd.DataFrame(predictions).to_csv(output / "phase2_predictions.csv", index=False)
    board = pd.DataFrame(leaderboard(predictions, reference))
    board.to_csv(output / "leaderboard.csv", index=False)
    (output / "metric_definitions.json").write_text(
        json.dumps({"phase2": metric_definitions(reference)})
    )
    (output / "run_config.json").write_text(
        json.dumps({**config, "status": "completed"})
    )
    pd.testing.assert_frame_equal(script.collect_snapshot(output)[1][0], board)
    board.loc[0, "rmse"] = 99
    board.to_csv(output / "leaderboard.csv", index=False)
    with pytest.raises(AssertionError):
        script.collect_snapshot(output)


def test_waiting_phase1_and_naive_only_experiment(experiment):
    output, config = experiment
    script = load_script("backfill_phase2_leaderboards")
    ranking = output / "phase1_ranking.csv"
    original = pd.read_csv(ranking)
    ranking.unlink()
    assert script.collect_snapshot(output)[1] is None
    original["selected"] = False
    original.to_csv(ranking, index=False)
    _, snapshot = script.collect_snapshot(output)
    assert snapshot[0].candidate.tolist() == ["Naive"]
    assert snapshot[2] == 0


def test_publisher_resumes_separate_run_and_deduplicates(experiment, monkeypatch):
    output, config = experiment
    script = load_script("backfill_phase2_leaderboards")
    _, (board, definitions, total, status) = script.collect_snapshot(output)
    logs, inits, finishes = [], [], []
    run = SimpleNamespace(
        summary={}, log=logs.append, finish=lambda: finishes.append(1)
    )

    def init(**kwargs):
        inits.append(kwargs)
        return run

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=init,
            Table=lambda **kw: kw,
            Settings=lambda **kw: kw,
        ),
    )
    for _ in range(2):
        publisher = Phase2Publisher(output, config["wandb"], config)
        publisher.publish(board, definitions, total_models=total, status=status)
        assert not publisher.publish(
            board, definitions, total_models=total, status=status
        )
        publisher.finish()
    assert len(logs) == 1
    assert logs[0][TABLE_KEY]["log_mode"] == "MUTABLE"
    assert inits[0]["id"] == inits[1]["id"]
    assert inits[0]["reinit"] == "create_new"
    assert inits[0]["job_type"] == "phase2-leaderboard"
    assert len(finishes) == 2
    assert (
        json.loads((output / "phase2_leaderboard.json").read_text())["status"] == status
    )


def test_publisher_retries_transient_init_failure(experiment, monkeypatch):
    import neuralforecast.benchmark_leaderboard as module

    output, config = experiment
    _, (board, definitions, total, status) = load_script(
        "backfill_phase2_leaderboards"
    ).collect_snapshot(output)
    calls = []

    def init(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ConnectionError("temporary")
        return SimpleNamespace(summary={}, log=lambda x: None, finish=lambda: None)

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=init,
            Table=lambda **kw: kw,
            Settings=lambda **kw: kw,
        ),
    )
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    publisher = Phase2Publisher(output, config["wandb"], config)
    publisher.publish(board, definitions, total_models=total, status=status)
    publisher.finish()
    assert len(calls) == 2


def test_lock_prevents_concurrent_writer_and_disabled_tracking(experiment, monkeypatch):
    output, config = experiment
    monkeypatch.setitem(sys.modules, "wandb", None)
    with phase2_publication(output, None, config) as publisher:
        assert publisher is None
    with publisher_lock(output):
        with pytest.raises(BlockingIOError):
            with publisher_lock(output):
                pytest.fail("Second writer obtained lock")
        with phase2_publication(output, config["wandb"], config) as publisher:
            assert publisher is None


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("model_count", [1, 2])
def test_runner_publishes_only_when_model_finishes_all_folds(
    tmp_path, monkeypatch, resume, model_count
):
    runner = load_script("run")
    candidates = [
        SimpleNamespace(name=f"Model{i}", protocol="zero_shot", best_config={})
        for i in range(model_count)
    ]
    folds = [Fold(0, 3, 5), Fold(1, 4, 6)]
    results = {
        i: dict(ok=True, actual=[i + 4.0, i + 5.0], prediction=[i + 4.0, i + 5.0])
        for i in range(2)
    }
    monkeypatch.setattr(runner, "_payload", lambda c: {"name": c.name})
    monkeypatch.setattr(
        runner, "_evaluate_job",
        SimpleNamespace(remote=lambda *args: (args[1]["name"], args[4].index)),
    )
    monkeypatch.setattr(
        runner,
        "ray",
        SimpleNamespace(
            wait=lambda refs, **kw: ([refs[0]], refs[1:]),
            get=lambda ref: results[ref[1]],
        ),
    )
    monkeypatch.setattr(
        runner, "_completed_phase2_result", lambda p, c, f, b, **kw: results[f.index]
    )
    snapshots = []
    runner._phase2(
        "data",
        candidates,
        folds,
        tmp_path / "checkpoints_phase2",
        resume=resume,
        on_model_complete=lambda rows: snapshots.append(list(rows)),
    )
    assert len(snapshots) == (1 if resume else model_count)
    assert len(snapshots[-1]) == 4 * model_count
