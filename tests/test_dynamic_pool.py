"""Admission, lifecycle and scientific-equivalence gates for pooled workers."""

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture
def pool(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "experiments/commodity_sota"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "pool_test", directory / "dynamic_pool.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reservations_subtract_only_unallocated_memory(pool, monkeypatch):
    monkeypatch.setattr(pool, "tree_rss", lambda pid: 12)
    jobs = [
        dict(
            device=0,
            process=types.SimpleNamespace(pid=123),
            gpu_reserve=40,
            ram_reserve=20,
        )
    ]
    assert pool.remaining_reservations(jobs, 0, {123: 30}) == (10, 8)
    assert pool.remaining_reservations(jobs, 1, {123: 30}) == (0, 8)
    assert pool.remaining_reservations(jobs, 0, None) == (40, 8)
    assert pool.remaining_reservations(jobs, 0, {123: 50}) == (0, 8)


def configure_queue(pool, monkeypatch, tmp_path):
    queue = pool.PoolQueue(tmp_path, reuse_models={"B"})
    monkeypatch.setattr(
        pool,
        "gpu_snapshot",
        lambda: {
            0: {"used": 10, "total": 100, "utilization": 0},
            1: {"used": 0, "total": 100, "utilization": 0},
        },
    )
    monkeypatch.setattr(pool, "process_gpu_memory", lambda: {1: 10})
    monkeypatch.setattr(pool, "tree_rss", lambda pid: 10)
    monkeypatch.setattr(
        pool.psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(total=1000, available=900),
    )
    monkeypatch.setattr(queue, "collect", lambda stats: None)
    monkeypatch.setattr(queue, "write_state", lambda stats: None)
    queue.running["old"] = dict(
        device=0,
        process=types.SimpleNamespace(pid=1),
        gpu_reserve=85,
        ram_reserve=20,
        unknown=True,
        exclusive=False,
        args=[None, {"name": "A"}],
        attempt=0,
    )
    queue.pending = [
        dict(
            enqueued=pool.time.monotonic(),
            sequence=1,
            key="B",
            length=1,
            budget=100,
            exclusive=False,
            args=[None, {"name": "B"}],
        )
    ]
    monkeypatch.setattr(queue, "estimate", lambda job: dict(gpu=30, ram=20))
    return queue


def test_unknown_on_one_gpu_does_not_block_other_gpu(pool, monkeypatch, tmp_path):
    queue = configure_queue(pool, monkeypatch, tmp_path)
    launched = []
    monkeypatch.setattr(
        queue, "launch", lambda job, device, *args: launched.append(device)
    )
    queue.tick()
    assert launched == [1]


def test_ram_shortage_prevents_admission(pool, monkeypatch, tmp_path):
    queue = configure_queue(pool, monkeypatch, tmp_path)
    monkeypatch.setattr(
        pool.psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(total=1000, available=165),
    )
    monkeypatch.setattr(
        queue, "launch", lambda *args: pytest.fail("RAM must be reserved")
    )
    queue.tick()
    assert (
        queue.waiting_reason == "gpu_profiling"
        or queue.waiting_reason == "memory_headroom"
    )


def test_oldest_unknown_drains_only_one_gpu(pool, monkeypatch, tmp_path):
    queue = configure_queue(pool, monkeypatch, tmp_path)
    queue.running.clear()
    first = queue.pending[0]
    first["enqueued"] = 0
    second = dict(first, key="C", sequence=2)
    queue.pending.append(second)
    # First cannot fit, but a younger job may use the other GPU.
    monkeypatch.setattr(
        queue, "estimate", lambda j: dict(gpu=30, ram=9999 if j is first else 20)
    )
    launched = []
    monkeypatch.setattr(
        queue,
        "launch",
        lambda job, device, *args: launched.append((job["key"], device)),
    )
    queue.tick()
    assert launched == [("C", 1)]


def test_stale_validation_report_never_enables_reuse(pool, monkeypatch, tmp_path):
    path = tmp_path / "validation.json"
    path.write_text(
        json.dumps(dict(execution_fingerprint="old", approved_models=["GRU"]))
    )
    monkeypatch.setattr(pool, "execution_fingerprint", lambda: "new")
    assert not pool.validated_models(path)
    path.write_text(
        json.dumps(dict(execution_fingerprint="new", approved_models=["GRU"]))
    )
    assert pool.validated_models(path) == {"GRU"}


def test_oom_replaces_worker_and_preserves_checkpoint(pool, monkeypatch, tmp_path):
    queue = pool.PoolQueue(tmp_path, reuse_models={"GRU"})
    monkeypatch.setattr(pool, "process_gpu_memory", lambda: {1: 10})
    monkeypatch.setattr(pool, "tree_rss", lambda pid: 12)
    retired = []

    def retire(worker):
        retired.append(worker["id"])
        queue.workers.pop(worker["id"])

    monkeypatch.setattr(queue, "retire", retire)
    path = tmp_path / "output.json"
    path.write_text(json.dumps(dict(ok=False, kind="OOM")))
    args = [
        None,
        {"name": "GRU"},
        0,
        {},
        types.SimpleNamespace(index=0),
        250,
        "original.ckpt",
        tmp_path / "phase1",
    ]
    job = dict(
        args=args,
        key="GRU",
        device=0,
        gpu_peak=0,
        ram_peak=0,
        process=types.SimpleNamespace(pid=1, poll=lambda: None),
        result_path=path,
        attempt=0,
        pooled=True,
        worker_id="w",
        started=pool.time.monotonic(),
        enqueued=0,
    )
    for attempt in range(3):
        queue.workers["w"] = dict(id="w", token="t", process=job["process"])
        queue.running["t"] = job
        queue.pending.clear()
        queue.collect({0: {"used": 10}})
        if attempt < 2:
            assert queue.pending == [job]
            assert job["args"][6] == "original.ckpt"
            assert job["exclusive"]
        else:
            assert queue.finished["t"]["kind"] == "OOM"
    assert queue.retries == 2
    assert retired == ["w", "w", "w"]


def test_dead_worker_publishes_failure_once(pool, monkeypatch, tmp_path):
    queue = pool.PoolQueue(tmp_path)
    monkeypatch.setattr(pool, "process_gpu_memory", lambda: {})
    monkeypatch.setattr(pool, "tree_rss", lambda pid: 0)
    process = types.SimpleNamespace(pid=1, returncode=9, poll=lambda: 9)
    queue.workers["w"] = dict(id="w", token="t", process=process)
    monkeypatch.setattr(queue, "retire", lambda w: queue.workers.pop(w["id"]))
    queue.running["t"] = dict(
        args=[
            None,
            {"name": "GRU"},
            0,
            {},
            types.SimpleNamespace(index=0),
            100,
            None,
            tmp_path / "phase1",
        ],
        key="GRU",
        device=0,
        gpu_peak=0,
        ram_peak=0,
        process=process,
        result_path=tmp_path / "absent.json",
        attempt=0,
        pooled=True,
        worker_id="w",
        started=pool.time.monotonic(),
        enqueued=0,
    )
    queue.collect({0: {"used": 0}})
    assert queue.get("t")["kind"] == "WORKER_FAILURE"
    assert not queue.running and not queue.finished and not queue.workers


def test_real_cpu_worker_reuses_imports_but_not_model_state(tmp_path):
    import os
    import pickle
    import shutil
    import subprocess
    import time

    import numpy as np
    import pandas as pd
    import torch
    from neuralforecast.benchmark import Fold

    root = Path(__file__).resolve().parents[1]
    shutil.copy2(
        root / "experiments/commodity_sota/pool_worker.py", tmp_path / "pool_worker.py"
    )
    wrapper = """import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('cpu_runner', ROOT / 'experiments/commodity_sota/run.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
original = module.pl.Trainer
def trainer(**kwargs):
    kwargs.update(accelerator='cpu', devices=1, enable_model_summary=False)
    return original(**kwargs)
module.pl.Trainer = trainer
pl = module.pl
_evaluate_job = module._evaluate_job
"""
    (tmp_path / "run.py").write_text(wrapper.replace("ROOT", repr(root)))
    # repr(Path) requires its concrete class; use a plain Path expression instead.
    wrapper_path = tmp_path / "run.py"
    wrapper_path.write_text(
        wrapper_path.read_text().replace(repr(root), f"Path({str(root)!r})")
    )
    frame = pd.DataFrame(
        dict(
            unique_id="test",
            ds=pd.date_range("2020-01-05", periods=112, freq="W-SUN"),
            y=3 + np.sin(np.arange(112) / 7),
        )
    )
    frame.attrs["transform_metadata"] = {
        "target_transform": "first_difference",
        "training_scale": "difference",
        "evaluation_scale": "level",
    }
    data = tmp_path / "weekly.pkl"
    frame.to_pickle(data)
    fold = Fold(index=0, train_end=96, valid_end=112)
    config = dict(
        h=16,
        input_size=16,
        encoder_hidden_size=8,
        decoder_hidden_size=8,
        encoder_n_layers=1,
        windows_batch_size=16,
        batch_size=1,
    )
    workers = []

    def spawn(name):
        log = (tmp_path / f"{name}.log").open("w")
        proc = subprocess.Popen(
            [
                sys.executable,
                str(tmp_path / "pool_worker.py"),
                "--ready",
                str(tmp_path / f"{name}.json"),
            ],
            stdin=subprocess.PIPE,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="",
                PYTHONPATH=str(root),
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
            ),
        )
        workers.append((proc, log))
        return proc

    def job(proc, name, budget, checkpoint=None, model="GRU"):
        folder = tmp_path / name
        folder.mkdir()
        parameters = (
            config
            if model == "GRU"
            else dict(
                h=16, input_size=16, num_layers=1, hidden_size=8, windows_batch_size=16
            )
        )
        candidate = dict(
            name=model,
            model_name=model,
            protocol="scratch_hpo",
            phase2_policy=dict(max_steps=4, interval=1, patience=5, val_size=16),
            tracking=dict(
                entity="test",
                project="pool-test",
                group=name,
                directory=str(folder / "wandb"),
                mode="offline",
            ),
        )
        args = [
            str(data),
            candidate,
            0,
            parameters,
            fold,
            budget,
            checkpoint,
            folder / "checkpoints_phase1",
        ]
        path = folder / "input.pkl"
        path.write_bytes(pickle.dumps(args))
        proc.stdin.write(json.dumps({"input": str(path)}) + "\n")
        proc.stdin.flush()
        result = folder / "output.json"
        deadline = time.monotonic() + 45
        while (
            not result.exists() and time.monotonic() < deadline and proc.poll() is None
        ):
            time.sleep(0.05)
        assert result.exists(), (
            (folder / "worker.log").read_text()
            if (folder / "worker.log").exists()
            else "worker did not start"
        )
        value = json.loads(result.read_text())
        assert value["ok"], value
        assert value["worker_reusable"], value
        return value

    try:
        reused = spawn("reused")
        first = job(reused, "first", 2)
        job(reused, "between", 2, model="MLP")
        fresh = job(reused, "fresh", 2)
        np.testing.assert_allclose(
            first["prediction"], fresh["prediction"], rtol=1e-5, atol=1e-6
        )
        resumed = job(reused, "resumed", 4, first["checkpoint"])
        isolated = spawn("isolated")
        reference = job(isolated, "reference", 4, first["checkpoint"])
        np.testing.assert_allclose(
            resumed["prediction"], reference["prediction"], rtol=1e-5, atol=1e-6
        )
        for key in ("actual_steps", "best_step", "stop_reason", "best_validation_loss"):
            assert resumed[key] == reference[key]
        assert first["worker_pid"] == resumed["worker_pid"] != reference["worker_pid"]
        state_a = torch.load(
            resumed["checkpoint"], map_location="cpu", weights_only=False
        )
        state_b = torch.load(
            reference["checkpoint"], map_location="cpu", weights_only=False
        )
        assert all(
            torch.equal(state_a["state_dict"][k], state_b["state_dict"][k])
            for k in state_a["state_dict"]
        )
    finally:
        for proc, log in workers:
            if proc.poll() is None:
                proc.stdin.close()
                proc.wait(timeout=15)
            log.close()


def test_validation_gate_rejects_prediction_or_stopping_changes(pool, monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "experiments/commodity_sota"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "pool_validation_test", directory / "validate_dynamic_pool.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = dict(
        ok=True,
        actual=[2.0, 3.0],
        prediction=[1.9, 2.9],
        forecast_origin=1.0,
        rmse=0.1,
        actual_steps=100,
        best_step=80,
        stop_reason="max_steps",
    )
    assert module.comparable(reference, dict(reference))
    assert not module.comparable(reference, dict(reference, prediction=[2.0, 3.0]))
    assert not module.comparable(reference, dict(reference, best_step=90))
    assert not module.comparable(reference, dict(reference, actual_steps=90))
    assert not module.comparable(reference, dict(reference, ok=False))
    assert module.gate([10, 10, 10], [8, 8, 8], {"GRU"}, 0, 0)["default_eligible"]
    assert not module.gate([10, 10, 10], [9, 9, 9], {"GRU"}, 0, 0)["default_eligible"]
    assert not module.gate([10, 10, 10], [8, 8, 8], {"GRU"}, 0, 1)["default_eligible"]
    assert not module.gate([10, 10, 10], [8, 8, 8], set(), 0, 0)["default_eligible"]


def test_external_gpu_memory_is_not_credited(pool, monkeypatch, tmp_path):
    queue = configure_queue(pool, monkeypatch, tmp_path)
    monkeypatch.setattr(
        pool,
        "gpu_snapshot",
        lambda: {
            0: {"used": 10, "total": 100, "utilization": 0},
            1: {"used": 70, "total": 100, "utilization": 0},
        },
    )
    monkeypatch.setattr(
        queue, "launch", lambda *args: pytest.fail("external memory is occupied")
    )
    queue.tick()


def test_per_gpu_worker_limit(pool, monkeypatch, tmp_path):
    queue = configure_queue(pool, monkeypatch, tmp_path)
    for name in ("first", "second"):
        queue.running[name] = dict(queue.running["old"], device=1, unknown=False)
    monkeypatch.setattr(
        queue, "launch", lambda *args: pytest.fail("both slots are occupied")
    )
    queue.tick()
