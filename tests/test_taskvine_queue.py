"""Resource admission and durable result boundaries for the TaskVine adapter."""

import importlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace
import sys
import types

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "experiments/commodity_sota")
    )
    return (
        importlib.import_module("taskvine_resources"),
        importlib.import_module("taskvine_queue"),
    )


def args(tmp_path):
    return [
        str(tmp_path / "data.pkl"),
        {"name": "GRU", "protocol": "scratch_hpo"},
        0,
        dict(batch_size=32, input_size=96, hidden_size=64, learning_rate=0.01),
        SimpleNamespace(index=0, train_end=96, valid_end=112),
        100,
        None,
        tmp_path / "checkpoints_phase1",
    ]


def snapshot(resources, used=0, processes=None):
    return dict(
        timestamp=time.time(),
        total=24 * resources.GIB,
        used=used * resources.GIB,
        processes=processes,
    )


def test_category_reuses_rungs_but_separates_memory_shapes(modules, tmp_path):
    r, _ = modules
    a = args(tmp_path)
    original = r.category_for(a)
    a[3]["learning_rate"] = 0.1
    a[5] = 500
    a[4].index = 10
    assert r.category_for(a) == original
    a[3]["hidden_size"] = 128
    assert r.category_for(a) != original
    a[3]["hidden_size"] = 64
    a[4] = SimpleNamespace(index=10, train_end=130, valid_end=146)
    assert r.category_for(a) != original


def test_unknown_gpu_is_exclusive_without_blocking_another_gpu(modules):
    r, _ = modules
    peers = [dict(exclusive=True, reservation=24 * r.GIB)]
    assert r.gpu_admission(snapshot(r), peers, None) is None
    assert r.gpu_admission(snapshot(r), [], None) == 24 * r.GIB
    assert r.gpu_admission(snapshot(r, used=1), [], None) is None


def test_shared_gpu_reserves_only_unallocated_usage(modules):
    r, _ = modules
    peers = [dict(exclusive=False, reservation=10 * r.GIB, pid=123)]
    s = snapshot(r, used=8, processes={"123": 8 * r.GIB})
    assert r.gpu_admission(s, peers, 8 * r.GIB) == 10 * r.GIB
    # An unrelated process consumes the remaining headroom.
    s["used"] = 12 * r.GIB
    assert r.gpu_admission(s, peers, 8 * r.GIB) is None


def test_missing_pid_accounting_and_stale_telemetry_fail_closed(modules):
    r, _ = modules
    peers = [dict(exclusive=False, reservation=10 * r.GIB, pid=123)]
    assert r.gpu_admission(snapshot(r, used=8), peers, 8 * r.GIB) is None
    s = snapshot(r)
    s["timestamp"] -= 20
    assert r.gpu_admission(s, [], None) is None
    assert r.gpu_admission(snapshot(r), peers * 2, r.GIB) is None


def test_submit_stages_portable_input_and_get_publishes_checkpoint(modules, tmp_path):
    r, qmod = modules
    q = qmod.TaskVineQueue.__new__(qmod.TaskVineQueue)
    q.session = tmp_path / "session"
    q.jobs = {}
    a = args(tmp_path)
    original_checkpoint = tmp_path / "old.ckpt"
    original_checkpoint.write_bytes(b"old optimizer and RNG")
    a[6] = str(original_checkpoint)
    token = q.submit(*a)
    folder = q.session / "jobs" / token
    import pickle

    payload = pickle.loads((folder / "input.pkl").read_bytes())
    assert payload[0] == "data.pkl"
    assert payload[6] == "prior"
    request = json.loads((folder / "request.json").read_text())
    assert request["checkpoint"] == str(original_checkpoint)
    checkpoint = folder / "attempt-0/products/resume.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"next optimizer and RNG")
    r.atomic_json(
        folder / "done.json", dict(ok=True, checkpoint="attempt-0/products/resume.ckpt")
    )
    result = q.get(token)
    assert result["checkpoint"] == str(checkpoint)
    saved = json.loads((a[7] / "GRU/0/0/result-100.json").read_text())
    assert saved == result
    assert original_checkpoint.read_bytes() == b"old optimizer and RNG"


@pytest.mark.parametrize("path", ["../escape.ckpt", "attempt-0/products/missing.ckpt"])
def test_missing_or_escaping_checkpoint_is_never_published(modules, tmp_path, path):
    r, qmod = modules
    q = qmod.TaskVineQueue.__new__(qmod.TaskVineQueue)
    q.session = tmp_path / "session"
    q.jobs = {}
    a = args(tmp_path)
    token = q.submit(*a)
    r.atomic_json(
        q.session / "jobs" / token / "done.json", dict(ok=True, checkpoint=path)
    )
    with pytest.raises(RuntimeError, match="checkpoint"):
        q.get(token)
    assert not (a[7] / "GRU/0/0/result-100.json").exists()


def test_remote_eligibility_checks_nested_paths(modules):
    r, _ = modules
    assert r.portable(dict(hidden_size=32, layers=[1, 2]))
    assert not r.portable(dict(weights={"path": Path("/private/model")}))


def test_colab_recreation_resets_only_owned_ephemeral_host_key(modules, tmp_path):
    colab = importlib.import_module("taskvine_colab")
    session = colab.Session.__new__(colab.Session)
    session.directory = tmp_path
    session.name = "owned-session"
    key = tmp_path / "known_hosts"
    key.write_text("old runtime key")
    unrelated = tmp_path / "other_known_hosts"
    unrelated.write_text("unrelated host")
    session.cli = lambda *a, **kw: None
    session.ssh = lambda: []

    def stop_before_connect(*a, **kw):
        raise RuntimeError("test stops before connecting")

    session.run = stop_before_connect
    with pytest.raises(RuntimeError, match="test stops"):
        session.prepare()
    assert not key.exists()
    assert unrelated.read_text() == "unrelated host"


def test_received_result_survives_parent_crash_before_get(modules, tmp_path):
    r, qmod = modules
    q = qmod.TaskVineQueue.__new__(qmod.TaskVineQueue)
    q.root = tmp_path / "taskvine"
    q.session = q.root / "session-interrupted"
    q.jobs = {}
    a = args(tmp_path)
    token = q.submit(*a)
    folder = q.session / "jobs" / token
    r.atomic_json(
        folder / "done.json",
        dict(
            ok=True, checkpoint=None, candidate="GRU", config_id=0, fold=0, budget=100
        ),
    )
    q.recover_results()
    target = a[7] / "GRU/0/0/result-100.json"
    assert json.loads(target.read_text())["ok"]
    # Never overwrite an already committed result on another restart.
    expected = dict(ok=True, checkpoint=None, marker="already committed")
    r.atomic_json(target, expected)
    q.recover_results()
    assert json.loads(target.read_text()) == expected


def test_worker_isolates_parent_wandb_service_and_uses_absolute_directory(
    modules, tmp_path, monkeypatch
):
    import pickle

    worker = importlib.import_module("taskvine_worker")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WANDB_SERVICE", "parent-socket")
    monkeypatch.setenv("_WANDB_SERVICE", "parent-socket")
    payload = args(tmp_path)
    payload[1]["tracking"] = dict(directory="wandb")
    (tmp_path / "input.pkl").write_bytes(pickle.dumps(payload))

    def evaluate(*actual):
        assert "WANDB_SERVICE" not in worker.os.environ
        assert "_WANDB_SERVICE" not in worker.os.environ
        assert Path(actual[1]["tracking"]["directory"]).is_absolute()
        return dict(ok=True, checkpoint=None)

    monkeypatch.setitem(
        sys.modules,
        "run",
        SimpleNamespace(_evaluate_job=SimpleNamespace(_function=evaluate)),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)),
    )
    worker.evaluate()
    assert json.loads((tmp_path / "result.json").read_text())["ok"]


@pytest.fixture
def manager_module(modules, monkeypatch):
    package = types.ModuleType("ndcctools")
    package.taskvine = types.ModuleType("ndcctools.taskvine")
    monkeypatch.setitem(sys.modules, "ndcctools", package)
    monkeypatch.setitem(sys.modules, "ndcctools.taskvine", package.taskvine)
    spec = importlib.util.spec_from_file_location(
        "test_vine_manager",
        Path(__file__).parents[1] / "experiments/commodity_sota/taskvine_manager.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_oom_retries_once_exclusively_and_invalidates_profile(
    manager_module, tmp_path
):
    m = manager_module.Manager.__new__(manager_module.Manager)
    folder = tmp_path / "attempt-0"
    folder.mkdir()
    (folder / "result.json").write_text(json.dumps(dict(ok=False, kind="OOM")))
    job = dict(
        token="t",
        folder=tmp_path,
        attempt_folder=folder,
        task=1,
        profile_key="shape",
        attempt=0,
        device="gpu0",
        category="GRU",
        peak=0,
        exclusive=False,
        reservation=100,
        started=time.time(),
        submitted=time.time(),
    )
    m.jobs = {"t": job}
    m.tasks = {1: ("job", "t")}
    m.profiles = {"shape": 100}
    m.profiles_path = tmp_path / "profiles.json"
    m.retries = 0
    m.category_completions = {}
    m.devices = {"gpu0": {"snapshot": {}}}
    task = SimpleNamespace(
        id=1,
        result="success",
        exit_code=0,
        output="",
        resources_measured=SimpleNamespace(memory=100),
        resources_allocated=SimpleNamespace(memory=4096),
    )
    m.complete(task)
    assert job["force_exclusive"] and job["attempt"] == 1
    assert not m.profiles
    assert not (tmp_path / "done.json").exists()
    m.tasks[1] = ("job", "t")
    m.complete(task)
    assert json.loads((tmp_path / "done.json").read_text())["kind"] == "OOM"
    assert m.retries == 1


def test_confirmed_colab_loss_requeues_without_changing_checkpoint(
    manager_module, tmp_path
):
    m = manager_module.Manager.__new__(manager_module.Manager)
    m.session = tmp_path
    (tmp_path / "devices").mkdir()
    (tmp_path / "devices/colab0.lost").write_text(
        json.dumps(dict(timestamp=time.time()))
    )
    cancelled = []
    m.m = SimpleNamespace(cancel_by_task_id=cancelled.append)
    job = dict(
        task=17,
        device="colab0",
        started=0,
        attempt=0,
        checkpoint="/durable/prior.ckpt",
        folder=tmp_path,
    )
    m.jobs = {"t": job}
    m.tasks = {17: ("job", "t")}
    m.devices = {"colab0": dict(last_probe=0, snapshot={})}
    m.retries = 0
    m.poll_devices()
    assert cancelled == [17]
    assert job["task"] is None and job["attempt"] == 1
    assert job["checkpoint"] == "/durable/prior.ckpt"
    m.poll_devices()
    assert cancelled == [17]


def test_frozen_migration_changes_only_scheduler_interface():
    path = Path(__file__).parents[1] / "scripts/migrate_commodity_taskvine.py"
    spec = importlib.util.spec_from_file_location("migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # A minimal legacy interface is sufficient; source content surrounding the
    # scheduler must remain byte-for-byte unchanged.
    source = """MODEL_POLICY = "preserved"
choices=["dynamic", "dynamic-pool", "ray"]
if args.scheduler in {"dynamic", "dynamic-pool"}:
        if args.scheduler == "dynamic-pool":
            pass
if (prior_config.get("scheduler") == "dynamic"
            and args.scheduler == "dynamic-pool"
        ):
    pass
"""
    result = module.patch_runner(source)
    assert result.startswith('MODEL_POLICY = "preserved"\n')
    assert "TaskVineQueue(output, SUMMARY, resume=args.resume)" in result
    with pytest.raises(ValueError):
        module.patch_runner(result)
