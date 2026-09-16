"""Drain the reserved exogenous experiment, then switch only its scheduler.

The legacy manager has no drain command. Pause that process alone, let its
children finish, publish their durable output, and only then stop the service.
Run with the training Python environment, from the repository root.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/commodity_sota"))
from taskvine_resources import atomic_json


def patch_runner(text):
    replacements = [
        (
            'choices=["dynamic", "dynamic-pool", "ray"]',
            'choices=["dynamic", "dynamic-pool", "taskvine", "ray"]',
        ),
        (
            'if args.scheduler in {"dynamic", "dynamic-pool"}:',
            'if args.scheduler in {"dynamic", "dynamic-pool", "taskvine"}:',
        ),
        (
            '        if args.scheduler == "dynamic-pool":\n',
            '        if args.scheduler == "taskvine":\n'
            "            from taskvine_queue import TaskVineQueue\n\n"
            "            queue = TaskVineQueue(output, SUMMARY, resume=args.resume)\n"
            '        elif args.scheduler == "dynamic-pool":\n',
        ),
        (
            '            and args.scheduler == "dynamic-pool"\n        ):\n',
            '            and args.scheduler == "dynamic-pool"\n'
            "        ) and not (\n"
            '            "taskvine" in {prior_config.get("scheduler"), args.scheduler}\n'
            '            and {prior_config.get("scheduler"), args.scheduler}\n'
            '            <= {"dynamic", "dynamic-pool", "taskvine"}\n        ):\n',
        ),
    ]
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError(
                f"Frozen runner differs from expected scheduler interface: {old}"
            )
        text = text.replace(old, new)
    compile(text, "frozen-run.py", "exec")
    return text


def canonical_output(experiment, input_path, result):
    args = pickle.loads(input_path.read_bytes())
    expected = dict(
        candidate=args[1]["name"], config_id=args[2], fold=args[4].index, budget=args[5]
    )
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Worker result identity mismatch: {input_path}")
    checkpoint = result.get("checkpoint")
    if checkpoint and not Path(checkpoint).exists():
        raise ValueError(f"Worker checkpoint missing: {checkpoint}")
    phase = Path(args[7]).name
    if phase not in {"checkpoints_phase1", "checkpoints_phase2"}:
        raise ValueError(f"Unexpected phase: {phase}")
    return (
        experiment
        / phase
        / args[1]["name"]
        / str(args[2])
        / str(args[4].index)
        / f"result-{args[5]}.json"
    )


def active_inputs(experiment, workers, state):
    if not state["running"]:
        return []
    oldest = min(p.create_time() for p in workers)
    expected = Counter(j["candidate"] for j in state["jobs"])
    matches = []
    for path in (experiment / "scheduler").glob("*/*/input.pkl"):
        if path.stat().st_mtime < oldest:
            continue
        args = pickle.loads(path.read_bytes())
        if args[1]["name"] not in expected:
            continue
        target = (
            experiment
            / Path(args[7]).name
            / args[1]["name"]
            / str(args[2])
            / str(args[4].index)
            / f"result-{args[5]}.json"
        )
        # Previous completed attempts may still live in a reused worker's tree.
        if target.exists():
            try:
                if json.loads(target.read_text()).get("ok"):
                    continue
            except ValueError:
                pass
        matches.append(path)
    if Counter(pickle.loads(p.read_bytes())[1]["name"] for p in matches) != expected:
        raise RuntimeError(
            "Cannot uniquely identify legacy running jobs; manager will resume"
        )
    return matches


def migrate(service, interrupt=False):
    queue = ROOT / "results/exogenous-queue"
    experiment = ROOT / "results/wti-exog"
    source = queue / "exog-source"
    runner = source / "experiments/commodity_sota/run.py"
    replacement = patch_runner(runner.read_text())
    reservation = json.loads((queue / "reservation.json").read_text())
    for name, digest in reservation["hashes"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Reserved source/data changed before migration: {name}")
    launch_path = queue / "start.py"
    launcher = launch_path.read_text()
    old = '        "dynamic-pool",\n'
    if launcher.count(old) != 1:
        raise RuntimeError("Unexpected launcher scheduler declaration")
    new_launcher = launcher.replace(
        old, '        "taskvine" if source == EXOG_SOURCE else "dynamic-pool",\n'
    )
    main_pid = int(
        subprocess.check_output(
            ["systemctl", "--user", "show", service, "--property=MainPID", "--value"],
            text=True,
        )
    )
    parent = psutil.Process(main_pid)
    managers = [
        p for p in parent.children(recursive=True) if str(runner) in p.cmdline()
    ]
    if len(managers) != 1:
        raise RuntimeError("Expected exactly one active frozen experiment manager")
    manager = managers[0]
    backup = queue / ("taskvine-migration-" + datetime.now().strftime("%Y%m%dT%H%M%S"))
    backup.mkdir()
    for path in (
        runner,
        launch_path,
        queue / "reservation.json",
        experiment / "run_config.json",
    ):
        shutil.copy2(path, backup / path.name)
    state_path = backup / "migration.json"
    record = dict(
        status="draining",
        started=datetime.now(timezone.utc).isoformat(),
        manager_pid=manager.pid,
        manager_created=manager.create_time(),
        backup=str(backup),
        interrupted=interrupt,
    )
    atomic_json(state_path, record)
    print(f"MIGRATION {state_path}", flush=True)
    paused = False
    stopped = False
    try:
        manager.send_signal(signal.SIGSTOP)
        paused = True
        # Wait for the OS to confirm the manager cannot submit another job.
        while manager.status() != psutil.STATUS_STOPPED:
            time.sleep(0.05)
        workers = [
            p
            for p in manager.children(recursive=True)
            if any(
                Path(s).name in {"pool_worker.py", "dynamic_queue.py"}
                for s in p.cmdline()
            )
        ]
        state = json.loads((experiment / "scheduler/state.json").read_text())
        inputs = active_inputs(experiment, workers, state)
        record["inputs"] = [str(p) for p in inputs]
        atomic_json(state_path, record)
        while not interrupt and any(
            not (p.parent / "output.json").exists() for p in inputs
        ):
            if any(not p.is_running() for p in workers):
                # An exited worker has no in-flight computation to preserve;
                # resume will submit that incomplete job again.
                break
            time.sleep(5)
        published = []
        for path in inputs:
            output = path.parent / "output.json"
            if not output.exists():
                continue
            result = json.loads(output.read_text())
            if not result.get("ok"):
                continue
            target = canonical_output(experiment, path, result)
            atomic_json(target, result)
            published.append(str(target))
        record.update(status="switching", published=published)
        atomic_json(state_path, record)
        # Queue SIGTERM while paused; on SIGCONT its handler runs before the
        # manager can dispatch again. systemd stop prevents automatic restart.
        stop = subprocess.Popen(["systemctl", "--user", "stop", service])
        manager.send_signal(signal.SIGTERM)
        manager.send_signal(signal.SIGCONT)
        paused = False
        stop.wait(timeout=90)
        if stop.returncode:
            raise RuntimeError("Could not stop legacy experiment service")
        stopped = True
        runner.write_text(replacement)
        changed = [runner]
        for path in (ROOT / "experiments/commodity_sota").glob("taskvine_*.py"):
            target = runner.parent / path.name
            shutil.copy2(path, target)
            changed.append(target)
        launch_path.write_text(new_launcher)
        for path in changed:
            reservation["hashes"][str(path.relative_to(ROOT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        atomic_json(queue / "reservation.json", reservation)
        record.update(status="starting", changed=[str(p) for p in changed])
        atomic_json(state_path, record)
        subprocess.run(["systemctl", "--user", "start", service], check=True)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            health = experiment / "taskvine/state.json"
            if health.exists():
                health_state = json.loads(health.read_text())
                if (
                    health_state.get("timestamp", 0) > time.time() - 15
                    and health_state.get("workers", 0) >= 2
                ):
                    break
            time.sleep(2)
        else:
            raise RuntimeError(
                "TaskVine local workers did not become healthy; restoring legacy backend"
            )
        record.update(status="started", finished=datetime.now(timezone.utc).isoformat())
        atomic_json(state_path, record)
        print("TASKVINE STARTED", flush=True)
    except BaseException as exc:
        record.update(status="failed", error=str(exc))
        atomic_json(state_path, record)
        if paused and manager.is_running():
            manager.send_signal(signal.SIGCONT)
        if stopped:
            subprocess.run(["systemctl", "--user", "stop", service], check=False)
            shutil.copy2(backup / "run.py", runner)
            shutil.copy2(backup / "start.py", launch_path)
            shutil.copy2(backup / "reservation.json", queue / "reservation.json")
            subprocess.run(["systemctl", "--user", "start", service], check=False)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", default="neuralforecast-exogenous-queue.service")
    parser.add_argument("--interrupt-running", action="store_true")
    options = parser.parse_args()
    migrate(options.service, options.interrupt_running)
