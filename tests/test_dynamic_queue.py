import importlib.util
from pathlib import Path
import types

spec = importlib.util.spec_from_file_location(
    "dynamic_queue",
    Path(__file__).resolve().parents[1] / "experiments/commodity_sota/dynamic_queue.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_reservations_cannot_be_double_spent():
    gpu = {"used": 10, "total": 100}
    running = [{"device": 0, "gpu_reserve": 40, "ram_reserve": 20}]
    assert not module.reservation_fits(gpu, 100, running, 0, 40, 10)
    assert module.reservation_fits(gpu, 100, running, 0, 20, 10)
    assert not module.reservation_fits(gpu, 25, running, 0, 20, 10)


def test_estimates_require_at_least_same_history_and_budget(tmp_path):
    queue = module.DynamicQueue(tmp_path)
    queue.estimates["model"] = [{"length": 100, "budget": 125, "gpu": 20, "ram": 10}]
    assert queue.estimate({"key": "model", "length": 90, "budget": 100}) == {
        "gpu": 25,
        "ram": 12.5,
    }
    assert queue.estimate({"key": "model", "length": 101, "budget": 100}) is None
    assert queue.estimate({"key": "model", "length": 90, "budget": 250}) is None


def test_finished_results_consumed_only_once(tmp_path):
    queue = module.DynamicQueue(tmp_path)
    queue.finished["one"] = {"ok": True}
    assert queue.get("one")["ok"]
    assert not queue.finished


def test_unknown_task_waits_until_existing_tasks_exit(tmp_path, monkeypatch):
    queue = module.DynamicQueue(tmp_path)
    monkeypatch.setattr(
        module, "gpu_snapshot", lambda: {0: {"used": 0, "total": 100, "utilization": 0}}
    )
    monkeypatch.setattr(
        module.psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(total=100, available=100),
    )
    monkeypatch.setattr(queue, "collect", lambda stats: None)
    queue.running["old"] = {
        "unknown": True,
        "args": [None, {"name": "A"}],
        "device": 0,
        "gpu_reserve": 85,
        "ram_reserve": 75,
        "attempt": 0,
    }
    queue.pending = [
        {"enqueued": 0, "sequence": 1, "key": "B", "length": 1, "budget": 1}
    ]
    monkeypatch.setattr(
        queue,
        "launch",
        lambda *args: (_ for _ in ()).throw(AssertionError("must wait")),
    )
    queue.tick()


def test_oom_retries_are_bounded_and_final_result_written(tmp_path, monkeypatch):
    import io
    import json

    queue = module.DynamicQueue(tmp_path)
    monkeypatch.setattr(module, "kill_group", lambda proc: None)
    monkeypatch.setattr(module, "tree_rss", lambda pid: 1)
    path = tmp_path / "output.json"
    path.write_text(json.dumps({"ok": False, "kind": "OOM"}))
    process = types.SimpleNamespace(pid=123, poll=lambda: 0, returncode=0)
    args = (
        None,
        {"name": "GRU"},
        0,
        {},
        types.SimpleNamespace(index=0),
        125,
        None,
        tmp_path / "phase1",
    )
    job = {
        "args": args,
        "key": "GRU",
        "device": 0,
        "gpu_peak": 0,
        "ram_peak": 0,
        "baseline_gpu": 0,
        "process": process,
        "result_path": path,
        "attempt": 0,
        "log": io.StringIO(),
    }
    for attempt in range(3):
        queue.running["one"] = job
        queue.pending.clear()
        queue.collect({0: {"used": 10}})
        if attempt < 2:
            assert len(queue.pending) == 1
            assert not queue.finished
            assert job["attempt"] == attempt + 1
        else:
            assert not queue.pending
            assert queue.finished["one"]["kind"] == "OOM"
    assert queue.retries == 2
    assert (tmp_path / "phase1/GRU/0/0/result-125.json").exists()
