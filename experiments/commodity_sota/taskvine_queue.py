"""Benchmark adapter to an isolated TaskVine manager via durable file messages."""

import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import uuid

from dynamic_queue import kill_group
from taskvine_resources import atomic_json, category_for, portable


class TaskVineQueue:
    """Preserve the benchmark's submit/wait/get/shutdown interface."""

    def __init__(self, directory, summary=None, resume=False):
        self.root = Path(directory).resolve() / "taskvine"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / "owner.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError(
                "This experiment already has a TaskVine manager"
            ) from None
        self.summary = summary
        self.closed = False
        self.jobs = {}
        self.last_log = 0
        if resume:
            self.recover_results()
        self.session = self.root / ("session-" + uuid.uuid4().hex)
        self.session.mkdir()
        self.environment = Path(
            os.environ.get(
                "NF_TASKVINE_ENV", str(Path.home() / ".local/share/taskvine/env")
            )
        )
        python = self.environment / "bin/python"
        if not python.is_file():
            raise RuntimeError("TaskVine missing: run scripts/setup_taskvine.sh")
        source = Path(__file__).resolve().parents[2]
        digest = hashlib.sha256()
        for p in sorted((source / "neuralforecast").rglob("*.py")):
            digest.update(p.read_bytes())
        digest.update(Path(__file__).with_name("run.py").read_bytes())
        versions = {
            p: importlib.metadata.version(p)
            for p in ("torch", "pytorch-lightning", "numpy", "wandb")
        }
        digest.update(json.dumps(versions, sort_keys=True).encode())
        settings = dict(
            source=str(source),
            training_python=sys.executable,
            environment=str(self.environment),
            identity=digest.hexdigest(),
            package_metadata=str(importlib.metadata.metadata("neuralforecast")),
            directory=str(self.root),
            colab_sessions=int(os.environ.get("NF_COLAB_SESSIONS", "2")),
        )
        if settings["colab_sessions"] not in (0, 1, 2):
            self.lock.close()
            raise ValueError("NF_COLAB_SESSIONS must be 0, 1, or 2")
        atomic_json(self.session / "settings.json", settings)
        self.log = (self.session / "manager.log").open("w")
        self.process = subprocess.Popen(
            [
                str(python),
                str(Path(__file__).with_name("taskvine_manager.py")),
                str(self.session),
            ],
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 60
        try:
            while not (self.session / "ready.json").exists():
                self.check_manager()
                if time.monotonic() > deadline:
                    raise TimeoutError(f"TaskVine startup timed out: {self.log.name}")
                time.sleep(0.2)
        except BaseException:
            self.shutdown()
            raise
        self.colab = None
        if settings["colab_sessions"]:
            self.colab_log = (self.session / "colab.log").open("w")
            self.colab = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("taskvine_colab.py")),
                    str(self.session),
                ],
                stdout=self.colab_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    def check_manager(self):
        if self.process.poll() is not None:
            raise RuntimeError(f"TaskVine manager exited; see {self.log.name}")

    def recover_results(self):
        """Publish results received before an interrupted parent's get call."""
        for path in self.root.glob("session-*/jobs/*/done.json"):
            result = json.loads(path.read_text())
            if not result.get("ok"):
                continue
            args = pickle.loads((path.parent / "input.pkl").read_bytes())
            checkpoint = result.get("checkpoint")
            if checkpoint:
                checkpoint = (path.parent / checkpoint).resolve()
                if (
                    not checkpoint.is_relative_to(path.parent.resolve())
                    or not checkpoint.exists()
                ):
                    continue
                result["checkpoint"] = str(checkpoint)
            target = (
                self.root.parent
                / Path(args[7]).name
                / args[1]["name"]
                / str(args[2])
                / str(args[4].index)
                / f"result-{args[5]}.json"
            )
            try:
                existing = json.loads(target.read_text())
            except (OSError, ValueError):
                existing = {}
            if not existing.get("ok"):
                atomic_json(target, result)

    def submit(self, *args):
        token = uuid.uuid4().hex
        folder = self.session / "jobs" / token
        folder.mkdir(parents=True)
        original = list(args)
        payload = list(args)
        payload[0] = "data.pkl"
        payload[6] = "prior" if args[6] else None
        payload[7] = Path("products") / Path(args[7]).name
        payload[1] = dict(args[1])
        if payload[1].get("tracking"):
            payload[1]["tracking"] = dict(payload[1]["tracking"], directory="wandb")
        (folder / "input.pkl").write_bytes(pickle.dumps(payload))
        request = dict(
            token=token,
            category=category_for(args),
            data=str(Path(args[0]).resolve()),
            checkpoint=str(Path(args[6]).resolve()) if args[6] else None,
            local_only=not portable(args[3])
            or args[1].get("protocol") not in {"scratch_hpo", "lora"},
            submitted=time.time(),
        )
        self.jobs[token] = original
        atomic_json(folder / "request.json", request)
        return token

    def wait(self, tokens, num_returns=1):
        while True:
            self.check_manager()
            ready = [
                t for t in tokens if (self.session / "jobs" / t / "done.json").is_file()
            ][:num_returns]
            if ready:
                return ready, [t for t in tokens if t not in ready]
            if self.summary and time.monotonic() - self.last_log >= 10:
                path = self.root / "state.json"
                if path.exists():
                    state = json.loads(path.read_text())
                    self.summary.log(
                        {
                            f"scheduler/{k}": state[k]
                            for k in ("queued", "running", "workers", "retries")
                        }
                    )
                self.last_log = time.monotonic()
            time.sleep(0.2)

    def get(self, token):
        args = self.jobs.pop(token)
        folder = self.session / "jobs" / token
        result = json.loads((folder / "done.json").read_text())
        for key, value in dict(
            candidate=args[1]["name"],
            config_id=args[2],
            fold=args[4].index,
            budget=args[5],
        ).items():
            result.setdefault(key, value)
        checkpoint = result.get("checkpoint")
        if checkpoint:
            path = (folder / checkpoint).resolve()
            if not path.is_relative_to(folder.resolve()) or not path.exists():
                raise RuntimeError("TaskVine returned a missing/unsafe checkpoint")
            result["checkpoint"] = str(path)
        target = Path(args[7]) / args[1]["name"] / str(args[2]) / str(args[4].index)
        atomic_json(target / f"result-{args[5]}.json", result)
        if not checkpoint:
            shutil.rmtree(folder / "products", ignore_errors=True)
        return result

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        # The supervisor closes only its owned Colab sessions.
        if getattr(self, "colab", None):
            self.colab.terminate()
            try:
                self.colab.wait(timeout=90)
            except subprocess.TimeoutExpired:
                kill_group(self.colab)
            self.colab_log.close()
        if getattr(self, "process", None):
            kill_group(self.process)
        if getattr(self, "log", None):
            self.log.close()
        if getattr(self, "lock", None):
            self.lock.close()
