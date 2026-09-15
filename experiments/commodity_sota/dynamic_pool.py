"""Reusable workers with per-GPU admission and measured reservation headroom."""

import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import uuid

import psutil

from dynamic_queue import DynamicQueue, gpu_snapshot, kill_group, tree_rss

EXECUTION_VERSION = 1


def execution_fingerprint():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    paths = sorted((root / "neuralforecast").rglob("*.py"))
    paths += [
        Path(__file__).with_name(name)
        for name in (
            "run.py",
            "dynamic_queue.py",
            "dynamic_pool.py",
            "pool_worker.py",
        )
    ]
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    digest.update(sys.version.encode())
    for package in ("torch", "pytorch-lightning", "numpy", "wandb"):
        digest.update(importlib.metadata.version(package).encode())
    try:
        hardware = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        hardware = "gpu-accounting-unavailable"
    digest.update(hardware.encode())
    return digest.hexdigest()


def validated_report(path):
    """Unverified, incomplete or stale execution reports cannot change defaults."""
    if not path:
        return {}
    try:
        report = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    if (
        not isinstance(report, dict)
        or report.get("execution_fingerprint") != execution_fingerprint()
    ):
        return {}
    return report


def validated_models(path):
    return set(validated_report(path).get("approved_models", []))


def process_gpu_memory():
    """Return PID GPU usage; None selects conservative reservation accounting."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        values = {}
        for row in csv.reader(out.splitlines()):
            pid, mib = int(row[0]), float(row[1])
            values[pid] = values.get(pid, 0) + mib * 2**20
        return values
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def remaining_reservations(running, device, gpu_usage):
    gpu, ram = 0.0, 0.0
    for job in running:
        # GPU usage can be missing on systems without per-process accounting.
        used = gpu_usage.get(job["process"].pid, 0) if gpu_usage is not None else 0
        if job["device"] == device:
            gpu += max(0, job["gpu_reserve"] - used)
        ram += max(0, job["ram_reserve"] - tree_rss(job["process"].pid))
    return gpu, ram


class PoolQueue(DynamicQueue):
    """Same job interface as DynamicQueue; only validated models reuse workers."""

    def __init__(self, directory, summary=None, reuse_models=(), resume=False):
        super().__init__(directory, summary, resume=resume)
        self.reuse_models = set(reuse_models)
        self.workers = {}
        self.restarts = 0
        self.waiting_reason = None
        self.worker_limit = 2

    def submit(self, *args):
        token = super().submit(*args)
        self.pending[-1]["queued_at"] = time.monotonic()
        return token

    def retire(self, worker):
        kill_group(worker["process"])
        if worker["process"].stdin:
            worker["process"].stdin.close()
        worker["log"].close()
        self.workers.pop(worker["id"], None)

    def new_worker(self, device):
        wid = uuid.uuid4().hex
        folder = self.root / "workers" / wid
        folder.mkdir(parents=True)
        log = (folder / "bootstrap.log").open("w")
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("pool_worker.py")),
                "--ready",
                str(folder / "ready.json"),
            ],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(device),
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                TOKENIZERS_PARALLELISM="false",
            ),
        )
        worker = dict(id=wid, device=device, process=process, log=log, token=None)
        self.workers[wid] = worker
        return worker

    def idle_worker(self, device):
        return next(
            (
                w
                for w in self.workers.values()
                if w["device"] == device and w["token"] is None
            ),
            None,
        )

    def launch(self, job, device, stats, estimate):
        if job["args"][1]["name"] not in self.reuse_models:
            for worker in list(self.workers.values()):
                if worker["device"] == device and worker["token"] is None:
                    self.retire(worker)
            super().launch(job, device, stats, estimate)
            job["pooled"] = False
            return
        worker = self.idle_worker(device) or self.new_worker(device)
        job.pop("shared", None)
        job.update(
            device=device,
            unknown=estimate is None,
            gpu_reserve=estimate["gpu"] if estimate else 0.85 * stats["total"],
            ram_reserve=(
                estimate["ram"]
                if estimate
                else max(2**30, psutil.virtual_memory().available * 0.75)
            ),
            baseline_gpu=stats["used"],
            gpu_peak=0,
            ram_peak=0,
            started=time.monotonic(),
            dispatched_at=time.time(),
            process=worker["process"],
            worker_id=worker["id"],
            pooled=True,
        )
        folder = self.root / job["token"] / str(job["attempt"])
        folder.mkdir(parents=True, exist_ok=False)
        job["folder"] = folder
        args = list(job["args"])
        args[1] = dict(args[1], attempt=job["attempt"])
        args[7] = folder / Path(args[7]).name
        path = folder / "input.pkl"
        path.write_bytes(pickle.dumps(args))
        job["result_path"] = folder / "output.json"
        worker["token"] = job["token"]
        self.pending.remove(job)
        self.running[job["token"]] = job
        self.last_launch = time.monotonic()
        try:
            worker["process"].stdin.write(json.dumps({"input": str(path)}) + "\n")
            worker["process"].stdin.flush()
        except (BrokenPipeError, OSError):
            # collect() publishes a normal worker failure once the process exits.
            kill_group(worker["process"])

    def collect(self, stats):
        usage = process_gpu_memory()
        for token, job in list(self.running.items()):
            job["ram_peak"] = max(job["ram_peak"], tree_rss(job["process"].pid))
            if usage is not None:
                job["gpu_peak"] = max(job["gpu_peak"], usage.get(job["process"].pid, 0))
            elif sum(j["device"] == job["device"] for j in self.running.values()) == 1:
                job["gpu_peak"] = max(job["gpu_peak"], stats[job["device"]]["used"])
            else:
                job["shared"] = True
            done = (
                job["result_path"].exists()
                if job.get("pooled")
                else job["process"].poll() is not None
            )
            if not done and job["process"].poll() is None:
                continue
            if job["result_path"].exists():
                result = json.loads(job["result_path"].read_text())
            else:
                result = dict(
                    ok=False,
                    kind="WORKER_FAILURE",
                    candidate=job["args"][1]["name"],
                    fold=job["args"][4].index,
                    error=f"Worker exited {job['process'].returncode}",
                )
            self.running.pop(token)
            if job.get("pooled"):
                worker = self.workers[job["worker_id"]]
                worker["token"] = None
                if (
                    not result.get("worker_reusable")
                    or job["process"].poll() is not None
                ):
                    self.retire(worker)
                    self.restarts += 1
            else:
                kill_group(job["process"])
                job["log"].close()
            if result.get("kind") == "OOM" and job["attempt"] < 2:
                job["attempt"] += 1
                self.retries += 1
                job["exclusive"] = True
                self.estimates.pop(job["key"], None)
                job["queued_at"] = time.monotonic()
                job["enqueued"] = job["queued_at"] - 60
                self.pending.insert(0, job)
                continue
            job["gpu_peak"] = max(
                job["gpu_peak"], result.get("allocator_peak_bytes", 0)
            )
            if result.get("ok") and not job.get("shared") and job["gpu_peak"] > 0:
                self.estimates.setdefault(job["key"], []).append(
                    dict(
                        length=job["length"],
                        budget=job["budget"],
                        gpu=job["gpu_peak"],
                        ram=max(job["ram_peak"], 256 * 2**20),
                    )
                )
            result.update(
                attempt=job["attempt"],
                resource_peak=dict(
                    gpu_bytes=job["gpu_peak"], ram_bytes=job["ram_peak"]
                ),
                execution_version=EXECUTION_VERSION,
                worker_mode=(
                    "reused_process" if job.get("pooled") else "isolated_process"
                ),
                queue_seconds=job["started"] - job.get("queued_at", job["enqueued"]),
                wall_seconds=time.monotonic() - job["started"],
            )
            if "worker_started_at" in result:
                result["preparation_seconds"] = (
                    result["worker_started_at"] - job["dispatched_at"]
                )
            self.finished[token] = result
            args = job["args"]
            target = Path(args[7]) / args[1]["name"] / str(args[2]) / str(args[4].index)
            target.mkdir(parents=True, exist_ok=True)
            (target / f"result-{args[5]}.json").write_text(json.dumps(result))
            if result.get("ok") and not result.get("checkpoint") and job.get("folder"):
                self.cleanup_paths[token] = job["folder"].parent
            with (self.root / "timings.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            k: result[k]
                            for k in (
                                "candidate",
                                "fold",
                                "budget",
                                "worker_mode",
                                "queue_seconds",
                                "wall_seconds",
                                "preparation_seconds",
                                "evaluation_seconds",
                                "cleanup_seconds",
                                "worker_pid",
                                "worker_reusable",
                                "ok",
                            )
                            if k in result
                        }
                    )
                    + "\n"
                )
        for worker in list(self.workers.values()):
            if worker["token"] is None and worker["process"].poll() is not None:
                self.retire(worker)
                self.restarts += 1

    def tick(self):
        if getattr(self, "colab", None):
            self.colab.tick()
        stats = gpu_snapshot()
        self.collect(stats)
        usage = process_gpu_memory()
        ram = psutil.virtual_memory()
        available = max(0, ram.available - 0.15 * ram.total)
        waiting = sorted(self.pending, key=lambda j: (j["enqueued"], j["sequence"]))
        drain = None
        if waiting and time.monotonic() - waiting[0]["enqueued"] >= 60:
            # Drain the least busy GPU, never the whole machine.
            drain = min(
                stats,
                key=lambda d: sum(j["device"] == d for j in self.running.values()),
            )
        self.waiting_reason = None
        launched = False
        for job in waiting:
            if len(self.running) >= self.cpu_limit:
                self.waiting_reason = "cpu_limit"
                break
            estimate = self.estimate(job)
            exclusive = estimate is None or job["exclusive"]
            for device in sorted(
                stats,
                key=lambda d: (
                    sum(j["device"] == d for j in self.running.values()),
                    stats[d]["used"],
                ),
            ):
                if drain == device and job is not waiting[0]:
                    continue
                peers = [j for j in self.running.values() if j["device"] == device]
                if len(peers) >= self.worker_limit or (exclusive and peers):
                    self.waiting_reason = "gpu_slots_or_exclusive"
                    continue
                if any(j["exclusive"] or j["unknown"] for j in peers):
                    self.waiting_reason = "gpu_profiling"
                    continue
                gpu_need = (
                    estimate["gpu"] if estimate else 0.85 * stats[device]["total"]
                )
                ram_need = (
                    estimate["ram"] if estimate else max(2**30, ram.available * 0.75)
                )
                idle = (
                    self.idle_worker(device)
                    if job["args"][1]["name"] in self.reuse_models
                    else None
                )
                gpu_headroom, ram_headroom = remaining_reservations(
                    self.running.values(), device, usage
                )
                # An idle worker's resident memory is already counted by the OS.
                idle_gpu = (
                    usage.get(idle["process"].pid, 0)
                    if idle and usage is not None
                    else 0
                )
                idle_ram = tree_rss(idle["process"].pid) if idle else 0
                fits = (
                    stats[device]["used"] + gpu_headroom + max(0, gpu_need - idle_gpu)
                    <= 0.85 * stats[device]["total"]
                    and ram_headroom + max(0, ram_need - idle_ram) <= available
                )
                if not fits:
                    self.waiting_reason = "memory_headroom"
                    # Release idle contexts that prevent a larger task from fitting.
                    idle_workers = [
                        w for w in self.workers.values() if w["token"] is None
                    ]
                    if idle_workers:
                        for worker in idle_workers:
                            self.retire(worker)
                        self.write_state(stats)
                        return
                    continue
                # Refresh actual memory before the next admission; include the new
                # job's reservation even before its allocation becomes visible.
                self.launch(job, device, stats[device], estimate)
                launched = True
                break
            if launched:
                break
        self.write_state(stats)

    def write_state(self, stats):
        state = dict(
            queued=len(self.pending),
            running=len(self.running),
            oom_retries=self.retries,
            worker_restarts=self.restarts,
            idle_workers=sum(w["token"] is None for w in self.workers.values()),
            execution_version=EXECUTION_VERSION,
            completed_buffered=len(self.finished),
            observations=[
                dict(candidate=key[0], config_hash=key[1], phase2=key[2], samples=rows)
                for key, rows in self.estimates.items()
            ],
            gpus=stats,
            waiting_reason=self.waiting_reason,
            jobs=[
                dict(
                    candidate=j["args"][1]["name"],
                    gpu=j["device"],
                    gpu_reserved=j["gpu_reserve"],
                    ram_reserved=j["ram_reserve"],
                    attempt=j["attempt"],
                    unknown=j["unknown"],
                )
                for j in self.running.values()
            ],
        )
        temporary = self.root / "state.tmp"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(self.root / "state.json")
        with (self.root / "telemetry.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(timestamp=time.time(), **state)) + "\n")
        if self.summary and time.monotonic() - self.last_log >= 10:
            self.summary.log(
                {
                    **{
                        f"scheduler/{k}": state[k]
                        for k in (
                            "queued",
                            "running",
                            "worker_restarts",
                            "oom_retries",
                            "idle_workers",
                            "waiting_reason",
                        )
                    },
                    **{
                        f"scheduler/gpu_{d}_{k}": v
                        for d, g in stats.items()
                        for k, v in g.items()
                    },
                }
            )
            self.last_log = time.monotonic()

    def wait(self, tokens, num_returns=1):
        while True:
            self.tick()
            ready = [t for t in tokens if t in self.finished][:num_returns]
            if ready:
                return ready, [t for t in tokens if t not in ready]
            time.sleep(0.2)

    def shutdown(self):
        if self.closed:
            return
        for job in list(self.running.values()):
            if not job.get("pooled"):
                kill_group(job["process"])
                job["log"].close()
        self.running.clear()
        for worker in list(self.workers.values()):
            self.retire(worker)
        if getattr(self, "colab", None):
            self.colab.close()
        self.closed = True
