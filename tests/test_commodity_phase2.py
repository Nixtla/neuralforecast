"""Actual small-model regression checks, with a CPU Trainer for CI."""

import importlib.util
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import torch
import pytest


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


def test_test_targets_cannot_change_selected_checkpoint(runner, tmp_path):
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
        weights.append(torch.load(checkpoint, map_location="cpu", weights_only=True))
        infos.append(json.loads((directory / "early_stopping.json").read_text()))
    assert infos[0] == infos[1]
    assert all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])
    np.testing.assert_allclose(predictions[0], predictions[1])
    assert infos[0]["best_step"] in (10, 20, 30)
    assert infos[0]["actual_steps"] == 30


def test_phase1_continues_optimizer_steps(runner, tmp_path):
    frame = data()
    train, test = frame.iloc[:96], frame.iloc[96:]
    _, first = runner._fit_trainable(
        runner.model_module.GRU, params(), train, test, 4, None, tmp_path
    )
    assert torch.load(first, weights_only=False)["global_step"] == 4
    _, second = runner._fit_trainable(
        runner.model_module.GRU, params(), train, test, 8, first, tmp_path
    )
    state = torch.load(second, weights_only=False)
    assert state["global_step"] == 8
    assert state["optimizer_states"][0]["state"]
