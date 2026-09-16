"""Actual small-model regression checks, with a CPU Trainer for CI."""

import importlib.util
from pathlib import Path
import sys
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import pytest
from ray import tune


@pytest.fixture
def runner(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "experiments/commodity_sota/run.py"
    spec = importlib.util.spec_from_file_location("commodity_run_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    original = module.pl.Trainer

    def cpu_trainer(**kwargs):
        kwargs.update(accelerator="cpu", devices=1, enable_model_summary=False)
        return original(**kwargs)

    monkeypatch.setattr(module.pl, "Trainer", cpu_trainer)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield module
    torch.set_num_threads(old_threads)


def test_all_phase1_candidate_types_use_sobol(runner, monkeypatch):
    class Model:
        MULTIVARIATE = False

    space = {"learning_rate": tune.loguniform(1e-4, 1e-2)}

    class Auto:
        def __init__(self, **kwargs):
            self.config = space
            self.cls_model = Model

    monkeypatch.setattr(
        runner, "auto_module",
        SimpleNamespace(__all__=["AutoExample"], AutoExample=Auto),
    )
    monkeypatch.setattr(runner, "model_module", SimpleNamespace(Example=Model))
    monkeypatch.setattr(runner, "INFERENCE_TUNING_MODELS", ["Example"])
    monkeypatch.setattr(runner, "get_inference_tuning_config", lambda *a, **kw: space)
    monkeypatch.setattr(
        runner, "_lora_api", lambda: (["Example"], None, lambda *a, **kw: space)
    )
    candidates, eligibility = runner._build_candidates(16, {}, 128)
    assert len(candidates) == 3, eligibility
    expected = runner.sample_ray_configs(space)
    for candidate in candidates:
        assert [c["learning_rate"] for c in candidate.configs] == [
            c["learning_rate"] for c in expected
        ]
        if candidate.protocol != "zero_shot":
            assert candidate.plan.survivors == (10, 5, 1)


def data():
    return pd.DataFrame(
        {
            "unique_id": "gasoline",
            "ds": pd.date_range("2020-01-05", periods=112, freq="W-SUN"),
            "y": 3 + np.sin(np.arange(112) / 7),
        }
    )


def params():
    return dict(
        h=16,
        input_size=16,
        encoder_hidden_size=8,
        decoder_hidden_size=8,
        encoder_n_layers=1,
        windows_batch_size=16,
        batch_size=1,
    )


def test_resume_accepts_only_complete_matching_fold_result(runner, tmp_path):
    from neuralforecast.benchmark import Fold

    candidate = runner.Candidate("GRU", "scratch_hpo", "GRU", [{}], None)
    fold = Fold(index=2, train_end=80, valid_end=96)
    path = tmp_path / "result-500.json"
    result = {
        "ok": True,
        "candidate": "GRU",
        "config_id": 0,
        "fold": 2,
        "budget": 500,
        "protocol_version": runner.PROTOCOL_VERSION,
        "actual": list(range(16)),
        "prediction": list(range(16)),
    }
    path.write_text(json.dumps(result))
    assert runner._completed_phase2_result(path, candidate, fold, 500) == result
    assert (
        runner._completed_phase2_result(
            path, candidate, fold, 500, require_tracking=True
        )
        is None
    )
    result["forecast_tracking_version"] = 1
    path.write_text(json.dumps(result))
    assert runner._completed_phase2_result(
        path, candidate, fold, 500, require_tracking=True
    )
    result["prediction"] = [1]
    path.write_text(json.dumps(result))
    assert runner._completed_phase2_result(path, candidate, fold, 500) is None


def test_restore_phase1_rebuilds_selected_best_config(runner, tmp_path):
    candidate = runner.Candidate(
        "GRU", "scratch_hpo", "GRU", [{"value": 1}, {"value": 2}], None
    )
    pd.DataFrame(
        [
            {"candidate": "GRU", "rung": 1, "config_id": 0, "pooled_rmse": 2},
            {"candidate": "GRU", "rung": 2, "config_id": 1, "pooled_rmse": 1},
        ]
    ).to_csv(tmp_path / "phase1_trials.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": "GRU",
                "selected": True,
                "config": json.dumps({"value": 2}),
                "pooled_rmse": 1,
                "naive_rmse": 3,
            }
        ]
    ).to_csv(tmp_path / "phase1_ranking.csv", index=False)
    phase1, failures, ranking, selected = runner._restore_phase1(tmp_path, [candidate])
    assert phase1 and not failures and len(ranking) == 1
    assert selected == [candidate]
    assert candidate.best_config == {"value": 2}


def test_validation_targets_do_not_enter_training_gradient(runner, tmp_path):
    frame = data()
    train, test = frame.iloc[:96].copy(), frame.iloc[96:].copy()
    policy = {"val_size": 16, "interval": 10, "patience": 5}
    weights = []
    infos = []
    predictions = []
    for i in range(2):
        directory = tmp_path / str(i)
        directory.mkdir()
        altered = test.copy()
        if i:
            altered["y"] += 10000
        prediction, checkpoint = runner._fit_trainable(
            runner.model_module.GRU,
            params(),
            train,
            altered,
            30,
            None,
            directory,
            stopping=policy,
        )
        predictions.append(prediction)
        weights.append(
            torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        )
        infos.append(json.loads((directory / "early_stopping.json").read_text()))
    assert all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])
    assert infos[0]["best_validation_loss"] != infos[1]["best_validation_loss"]
    assert all(len(prediction) == 16 for prediction in predictions)
    assert infos[0]["best_step"] in (10, 20, 30)
    assert infos[0]["actual_steps"] == 30


def test_phase1_continues_optimizer_steps(runner, tmp_path):
    frame = data()
    train, test = frame.iloc[:96], frame.iloc[96:]
    _, first = runner._fit_trainable(
        runner.model_module.GRU,
        params(),
        train,
        test,
        4,
        None,
        tmp_path,
        schedule_steps=8,
    )
    assert torch.load(first, weights_only=False)["global_step"] == 4
    _, second = runner._fit_trainable(
        runner.model_module.GRU,
        params(),
        train,
        test,
        8,
        first,
        tmp_path,
        schedule_steps=8,
    )
    state = torch.load(second, weights_only=False)
    assert state["global_step"] == 8
    assert state["optimizer_states"][0]["state"]


def test_validation_stopper_survives_rung_and_directory_change(runner, tmp_path):
    frame = data()
    policy = {"val_size": 16, "interval": 2, "patience": 50}
    _, first = runner._fit_trainable(
        runner.model_module.GRU,
        params(),
        frame.iloc[:96],
        frame.iloc[96:],
        4,
        None,
        tmp_path / "first",
        stopping=policy,
        schedule_steps=8,
    )
    before = torch.load(first, weights_only=False)
    first_stop = next(
        v["stopper"]
        for v in before["callbacks"].values()
        if isinstance(v, dict) and "stopper" in v
    )
    _, second = runner._fit_trainable(
        runner.model_module.GRU,
        params(),
        frame.iloc[:96],
        frame.iloc[96:],
        8,
        first,
        tmp_path / "second",
        stopping=policy,
        schedule_steps=8,
    )
    after = torch.load(second, weights_only=False)
    last_stop = next(
        v["stopper"]
        for v in after["callbacks"].values()
        if isinstance(v, dict) and "stopper" in v
    )
    assert after["global_step"] == 8
    assert last_stop["best"] <= first_stop["best"]
    assert last_stop["last_step"] == 8
    assert not (tmp_path / "second" / "best-validation.pt").exists()


def test_phase2_trainable_does_not_retain_checkpoints(runner, tmp_path):
    frame = data()
    workdir = tmp_path / "phase2"
    _, checkpoint = runner._fit_trainable(
        runner.model_module.GRU,
        params(),
        frame.iloc[:96],
        frame.iloc[96:],
        4,
        None,
        workdir,
        keep_checkpoint=False,
    )

    assert checkpoint is None
    assert not list(workdir.glob("*.ckpt"))


def test_discard_checkpoint_preserves_result_metadata(runner, tmp_path):
    workdir = tmp_path / "rung"
    workdir.mkdir()
    checkpoint = workdir / "resume.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    (workdir / "best-validation.pt").write_bytes(b"weights")
    result = workdir / "result-125.json"
    result.write_text("{}")

    runner._discard_checkpoint(checkpoint)

    assert not checkpoint.exists()
    assert not (workdir / "best-validation.pt").exists()
    assert result.is_file()


def test_naive_uses_only_last_train_observation(runner):
    from neuralforecast.benchmark import Fold

    frame = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 6.0]})
    score = runner._naive_score(frame, [Fold(0, 2, 4), Fold(1, 3, 5)])
    assert score == pytest.approx(np.sqrt((1 + 4 + 1 + 9) / 4))


@pytest.mark.parametrize(
    "n_obs,expected_cutoff,expected_first,expected_count,expected_reps",
    [
        (600, 420, 324, 165, [0, 244, 488]),
        (683, 479, 383, 189, [0, 285, 571]),
    ],
)
def test_phase2_starts_at_70_percent_without_changing_phase1_folds(
    runner, n_obs, expected_cutoff, expected_first, expected_count, expected_reps
):
    all_folds = runner.expanding_folds(n_obs, h=16, min_train=96, step_size=1)
    phase1 = runner.representative_folds(all_folds)
    phase2, cutoff = runner._phase2_window(all_folds, n_obs, 0.7)

    assert cutoff == expected_cutoff
    assert phase2[0].index == expected_first
    assert len(phase2) == expected_count
    assert [fold.index for fold in phase1] == expected_reps


def test_phase2_start_ratio_must_leave_a_complete_fold(runner):
    folds = runner.expanding_folds(100, h=16, min_train=32, step_size=1)
    with pytest.raises(ValueError, match="no complete forecast fold"):
        runner._phase2_window(folds, 100, 0.99)


@pytest.mark.parametrize(
    "scores,expected",
    [
        ([1.0, 2.0, 3.0, float("inf")], 1),
        ([2.0, 3.0], 0),
        ([0.5] * 12, 10),
    ],
)
def test_naive_gate_strictly_filters_and_caps(runner, scores, expected):
    candidates = [
        runner.Candidate(
            str(i),
            "scratch_hpo",
            "GRU",
            [{}],
            None,
            best_score=s,
            best_config={},
            best_fold_scores=(s, s, s),
        )
        for i, s in enumerate(scores)
    ]
    selected, table = runner._select_candidates(candidates, 2.0)
    assert len(selected) == expected
    assert table.selected.sum() == expected
    assert all(c.best_score < 2.0 for c in selected)


@pytest.mark.parametrize("stopped_folds,expected_jobs", [({0, 1, 2}, 30), ({0}, 42)])
def test_stopped_folds_are_reused_without_submitting_more_work(
    runner, tmp_path, monkeypatch, stopped_folds, expected_jobs
):
    from neuralforecast.benchmark import Fold, SHPlan

    calls = []
    results = {}

    class Job:
        def remote(self, data, candidate, cid, config, fold, budget, checkpoint, root):
            token = len(calls)
            calls.append((cid, fold.index, budget))
            results[token] = dict(
                ok=True,
                rmse=cid + 1.0,
                actual=[0.0],
                prediction=[cid + 1.0],
                forecast_origin=0.0,
                stop_reason=(
                    "early_stopping" if fold.index in stopped_folds else "max_steps"
                ),
            )
            return token

    monkeypatch.setattr(runner, "_evaluate_job", Job())
    monkeypatch.setattr(runner.ray, "wait", lambda refs, **kw: ([refs[0]], refs[1:]))
    monkeypatch.setattr(runner.ray, "get", lambda ref: results[ref])
    candidate = runner.Candidate("GRU", "scratch_hpo", "GRU", [{}] * 10, SHPlan())
    trials, failures = runner._run_phase1(
        "unused", [candidate], [Fold(i, 10 + i, 11 + i) for i in range(3)], tmp_path
    )
    assert len(calls) == expected_jobs
    assert all(budget == 100 for _, fold, budget in calls if fold in stopped_folds)
    assert len(trials) == 16
    assert not failures
    assert candidate.best_score == 1.0


def test_no_naive_winner_skips_phase2_and_preserves_results(
    runner, tmp_path, monkeypatch
):
    frame = data()
    source = tmp_path / "input.csv"
    frame.to_csv(source, index=False)
    output = tmp_path / "output"
    candidate = runner.Candidate(
        "GRU",
        "scratch_hpo",
        "GRU",
        [{}],
        None,
        best_score=100.0,
        best_config={},
        best_fold_scores=(100.0,) * 3,
    )
    monkeypatch.setattr(runner, "_minimum_train", lambda *a: 32)
    monkeypatch.setattr(runner, "_build_candidates", lambda *a: ([candidate], []))
    monkeypatch.setattr(runner, "_run_phase1", lambda *a: ([], []))
    monkeypatch.setattr(
        runner, "_phase2", lambda *a: pytest.fail("Phase 2 must be skipped")
    )
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.ray, "init", lambda **kw: None)
    monkeypatch.setattr(runner.ray, "shutdown", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--data",
            str(source),
            "--target",
            "y",
            "--output",
            str(output),
            "--scheduler",
            "ray",
        ],
    )
    runner.main()
    config = json.loads((output / "run_config.json").read_text())
    assert config["status"] == "no_models_above_naive"
    assert config["selected"] == []
    assert config["sh_budgets"] == [100, 250, 500]
    assert config["sampling_policy"] == runner.SAMPLING_POLICY
    assert pd.read_csv(output / "phase2_predictions.csv").empty
    assert pd.read_csv(output / "leaderboard.csv").candidate.tolist() == ["Naive"]
    assert pd.read_csv(output / "phase1_leaderboard.csv").candidate.tolist() == [
        "Naive",
        "GRU",
    ]
    definitions = json.loads((output / "metric_definitions.json").read_text())
    assert definitions["phase2"]["points_per_model"] > 0
    before = (output / "run_config.json").read_bytes()
    with pytest.raises(ValueError, match="fresh output"):
        runner.main()
    assert before == (output / "run_config.json").read_bytes()
    monkeypatch.setattr(sys, "argv", sys.argv + ["--resume"])
    with pytest.raises(ValueError, match="already complete"):
        runner.main()
    legacy = dict(config)
    legacy.pop("sampling_policy")
    (output / "run_config.json").write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="sampling_policy"):
        runner.main()
    report = tmp_path / "smoke.json"
    report.write_text(json.dumps({"fingerprint": config["fingerprint"], "passed": ["GRU"]}))
    monkeypatch.setattr(sys, "argv", sys.argv + ["--validated-candidates", str(report)])
    monkeypatch.setattr(runner, "SAMPLING_POLICY", {**runner.SAMPLING_POLICY, "version": 2})
    with pytest.raises(ValueError, match="Smoke report does not match"):
        runner.main()


def test_leaderboard_uses_identical_points_and_includes_naive(runner):
    reference = pd.DataFrame(
        {
            "fold": [0, 0, 1, 1],
            "horizon": [1, 2, 1, 2],
            "actual": [1.0, 2.0, 2.0, 4.0],
            "forecast_origin": [1.0, 1.0, 2.0, 2.0],
        }
    )
    predictions = reference.assign(
        candidate="Perfect", protocol="scratch_hpo", prediction=reference.actual
    ).to_dict("records")
    incomplete = [dict(row, candidate="Incomplete") for row in predictions[:3]]
    board = runner._leaderboard(predictions + incomplete, reference)
    assert [row["candidate"] for row in board] == ["Perfect", "Naive"]
    assert board[0]["rmse"] == 0.0
    assert board[0]["da_pct"] == 100.0
    assert board[1]["rmse"] == pytest.approx(np.sqrt(5 / 4))
    assert board[1]["da_pct"] == 50.0
    assert board[0]["beats_naive"] and not board[1]["beats_naive"]
    with pytest.raises(ValueError, match="Duplicate"):
        runner._leaderboard(predictions + predictions[:1], reference)
    with pytest.raises(ValueError, match="targets differ"):
        runner._leaderboard([dict(row, actual=99.0) for row in predictions], reference)


def test_direction_reference_baselines_are_reported(runner):
    reference = pd.DataFrame(
        {"actual": [2.0, 0.0, 1.0], "forecast_origin": [1.0, 1.0, 1.0]}
    )
    definitions = runner._metric_definitions(reference)
    for direction in ("up", "down", "flat"):
        assert definitions[f"always_{direction}_da_pct"] == pytest.approx(100 / 3)


def test_difference_boundaries_exogenous_and_restoration(runner):
    frame = data().iloc[:8].copy()
    frame["x"] = [1, 3, 2, 7, 5, 4, 9, 11]
    train, valid = frame.iloc[:5], frame.iloc[5:]
    td, vd = runner._difference_fold(train, valid)
    np.testing.assert_allclose(td.y, np.diff(train.y))
    np.testing.assert_allclose(vd.y, np.diff(frame.y.iloc[4:]))
    np.testing.assert_allclose(vd.x, [-1, 5, 2])
    assert td.ds.tolist() == train.ds.iloc[1:].tolist()
    assert vd.ds.tolist() == valid.ds.tolist()
    np.testing.assert_allclose(
        runner._restore_prediction(vd.y, train.y.iloc[-1], "first_difference"), valid.y
    )
    np.testing.assert_allclose(
        runner._restore_prediction(np.zeros(3), train.y.iloc[-1], "first_difference"),
        np.repeat(train.y.iloc[-1], 3),
    )
    changed = valid.copy()
    changed.y += 100
    other_train, _ = runner._difference_fold(train, changed)
    pd.testing.assert_frame_equal(td, other_train)
    np.testing.assert_allclose(
        runner._restore_prediction([1, 2], 9, "identity"), [1, 2]
    )
    assert (
        runner._transform_metadata("uni-gasoline-diff")["training_scale"]
        == "difference"
    )
    assert runner._transform_metadata("uni-gasoline")["training_scale"] == "level"


@pytest.mark.parametrize("protocol", ["scratch_hpo", "lora", "zeroshot"])
def test_diff_job_scores_restored_prices(runner, monkeypatch, tmp_path, protocol):
    from types import SimpleNamespace

    frame = data()
    frame.attrs["transform_metadata"] = runner._transform_metadata("uni-gasoline-diff")
    path = tmp_path / "weekly.pkl"
    frame.to_pickle(path)
    fold = SimpleNamespace(
        index=0, train_slice=slice(0, 96), valid_slice=slice(96, 112)
    )

    def predict(model, config, train, valid, *args, **kwargs):
        np.testing.assert_allclose(train.y, np.diff(frame.y.iloc[:96]))
        np.testing.assert_allclose(valid.y, np.diff(frame.y.iloc[95:]))
        return np.zeros(16), None

    for name in ["_fit_trainable", "_fit_lora", "_fit_inference"]:
        monkeypatch.setattr(runner, name, predict)
    result = runner._evaluate_job._function(
        str(path),
        {
            "name": "GRU",
            "model_name": "GRU",
            "protocol": protocol,
            "protocol_version": "worker-protocol-v2",
        },
        0,
        {},
        fold,
        2,
        None,
        tmp_path / "checkpoints_phase1",
    )
    assert result["ok"], result
    np.testing.assert_allclose(result["actual"], frame.y.iloc[96:])
    np.testing.assert_allclose(result["prediction"], np.repeat(frame.y.iloc[95], 16))
    assert result["rmse"] == pytest.approx(runner._naive_score(frame, [fold]))
    assert result["evaluation_scale"] == "level"
    assert result["protocol_version"] == "worker-protocol-v2"
