"""Process-isolated resource-aware queue for the commodity benchmark."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import sys
import time
import uuid

import psutil


def gpu_snapshot():
    executable = shutil.which("nvidia-smi") or str(
        Path.home() / ".local/bin/nvidia-smi"
    )
    out = subprocess.check_output(
        [
            executable,
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    return {
        int(r[0]): {
            "used": float(r[1]) * 2**20,
            "total": float(r[2]) * 2**20,
            "utilization": float(r[3]),
        }
        for r in csv.reader(out.splitlines())
    }


def kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
        os.killpg(process.pid, signal.SIGKILL)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def tree_rss(pid):
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def reservation_fits(gpu, ram_available, running, device, gpu_need, ram_need):
    # Subtract entire reservations from currently free resources: conservative
    # when process-level NVML accounting is unavailable; never double-spend slack.
    reserved_gpu = sum(j["gpu_reserve"] for j in running if j["device"] == device)
    reserved_ram = sum(j["ram_reserve"] for j in running)
    return (
        gpu["used"] + reserved_gpu + gpu_need <= 0.85 * gpu["total"]
        and reserved_ram + ram_need <= ram_available
    )


class DynamicQueue:
    """Queue jobs and admit them using observed peaks and explicit reservations."""

    def __init__(self, directory, summary=None):
        self.root = Path(directory).resolve() / "scheduler"
        self.root.mkdir(parents=True, exist_ok=True)
        self.summary = summary
        self.pending, self.running, self.finished = [], {}, {}
        self.estimates = {}
        self.cpu_limit = max(1, len(os.sched_getaffinity(0)) // 2)
        self.sequence = 0
        self.last_launch = 0
        self.last_log = 0
        self.closed = False
        self.retries = 0

    def submit(self, *args):
        self.sequence += 1
        name = args[1]["name"]
        phase = Path(args[7]).name
        # Pickled config preserves loss types and every effective parameter.
        config_hash = hashlib.sha256(pickle.dumps(args[3])).hexdigest()
        key = (name, config_hash, "phase2" in phase)
        token = uuid.uuid4().hex
        self.pending.append(
            {
                "token": token,
                "args": args,
                "key": key,
                "length": args[4].train_end,
                "budget": args[5],
                "enqueued": time.monotonic(),
                "attempt": 0,
                "exclusive": False,
                "sequence": self.sequence,
            }
        )
        return token

    def estimate(self, job):
        options = [
            row
            for row in self.estimates.get(job["key"], [])
            if row["length"] >= job["length"] and row["budget"] >= job["budget"]
        ]
        if not options:
            return None
        return {
            "gpu": max(r["gpu"] for r in options) * 1.25,
            "ram": max(r["ram"] for r in options) * 1.25,
        }

    def launch(self, job, device, stats, estimate):
        job.pop("shared", None)
        job["device"] = device
        job["unknown"] = estimate is None
        job["gpu_reserve"] = estimate["gpu"] if estimate else 0.85 * stats["total"]
        job["ram_reserve"] = (
            estimate["ram"]
            if estimate
            else max(2**30, psutil.virtual_memory().available * 0.75)
        )
        job["baseline_gpu"] = stats["used"]
        job["gpu_peak"], job["ram_peak"] = 0, 0
        job["started"] = time.monotonic()
        folder = self.root / job["token"] / str(job["attempt"])
        folder.mkdir(parents=True, exist_ok=False)
        args = list(job["args"])
        args[1] = dict(args[1], attempt=job["attempt"])
        # Isolate attempts and preserve checkpoints from successful prior rungs.
        args[7] = folder / Path(args[7]).name
        path = folder / "input.pkl"
        path.write_bytes(pickle.dumps(args))
        job["result_path"] = folder / "output.json"
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(device),
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            TOKENIZERS_PARALLELISM="false",
        )
        job["log"] = (folder / "worker.log").open("w")
        job["process"] = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(path)],
            env=env,
            stdout=job["log"],
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.pending.remove(job)
        self.running[job["token"]] = job
        self.last_launch = time.monotonic()

    def collect(self, stats):
        for token, job in list(self.running.items()):
            gpu = stats[job["device"]]
            job["ram_peak"] = max(job["ram_peak"], tree_rss(job["process"].pid))
            peers = [j for j in self.running.values() if j["device"] == job["device"]]
            # Only an exclusive observation can be attributed without NVML PID data.
            if len(peers) == 1:
                job["gpu_peak"] = max(
                    job["gpu_peak"], gpu["used"] - job["baseline_gpu"]
                )
            else:
                job["shared"] = True
            if job["process"].poll() is None:
                continue
            kill_group(job["process"])
            job["log"].close()
            self.running.pop(token)
            if job["result_path"].exists():
                result = json.loads(job["result_path"].read_text())
            else:
                result = {
                    "ok": False,
                    "kind": "WORKER_FAILURE",
                    "error": f"Worker exited {job['process'].returncode}",
                    "candidate": job["args"][1]["name"],
                    "fold": job["args"][4].index,
                }
            if result.get("kind") == "OOM" and job["attempt"] < 2:
                job["attempt"] += 1
                self.retries += 1
                job["exclusive"] = True
                self.estimates.pop(job["key"], None)
                job["enqueued"] = time.monotonic() - 60
                self.pending.insert(0, job)
                continue
            if not job.get("shared"):
                job["gpu_peak"] = max(
                    job["gpu_peak"], result.get("allocator_peak_bytes", 0)
                )
            if result.get("ok") and not job.get("shared") and job["gpu_peak"] > 0:
                self.estimates.setdefault(job["key"], []).append(
                    {
                        "length": job["length"],
                        "budget": job["budget"],
                        "gpu": job["gpu_peak"],
                        "ram": max(job["ram_peak"], 256 * 2**20),
                    }
                )
            result["attempt"] = job["attempt"]
            result["resource_peak"] = {
                "gpu_bytes": job["gpu_peak"],
                "ram_bytes": job["ram_peak"],
            }
            self.finished[token] = result
            # Keep driver-compatible final result layout; only terminal attempt counts.
            args = job["args"]
            target = Path(args[7]) / args[1]["name"] / str(args[2]) / str(args[4].index)
            target.mkdir(parents=True, exist_ok=True)
            (target / f"result-{args[5]}.json").write_text(json.dumps(result))

    def tick(self):
        stats = gpu_snapshot()
        self.collect(stats)
        ram = psutil.virtual_memory()
        available = max(0, ram.available - 0.15 * ram.total)
        waiting = sorted(self.pending, key=lambda j: (j["enqueued"], j["sequence"]))
        drain = (
            min(stats)
            if waiting and time.monotonic() - waiting[0]["enqueued"] >= 60
            else None
        )
        drain_all = bool(
            drain is not None
            and waiting
            and (self.estimate(waiting[0]) is None or waiting[0]["exclusive"])
        )
        for job in waiting:
            if drain_all and job is not waiting[0]:
                continue
            if (
                len(self.running) >= self.cpu_limit
                or time.monotonic() - self.last_launch < 1
            ):
                break
            estimate = self.estimate(job)
            if any(j["unknown"] for j in self.running.values()):
                break
            for device in sorted(stats, key=lambda i: stats[i]["used"]):
                if device == drain and job is not waiting[0]:
                    continue
                peers = [j for j in self.running.values() if j["device"] == device]
                exclusive = estimate is None or job["exclusive"]
                if exclusive and peers:
                    continue
                if any(j["exclusive"] or j["unknown"] for j in peers):
                    continue
                if exclusive:
                    # Unknown job is real work, measured conservatively in isolation.
                    if (
                        self.running
                        or stats[device]["used"] > 0.15 * stats[device]["total"]
                        or available < 0.5 * ram.total
                    ):
                        continue
                elif not reservation_fits(
                    stats[device],
                    available,
                    list(self.running.values()),
                    device,
                    estimate["gpu"],
                    estimate["ram"],
                ):
                    continue
                self.launch(job, device, stats[device], estimate)
                break
        state = {
            "queued": len(self.pending),
            "oom_retries": self.retries,
            "observations": [
                dict(candidate=key[0], config_hash=key[1], phase2=key[2], samples=rows)
                for key, rows in self.estimates.items()
            ],
            "running": len(self.running),
            "completed_buffered": len(self.finished),
            "gpus": stats,
            "jobs": [
                {
                    "candidate": j["args"][1]["name"],
                    "gpu": j["device"],
                    "gpu_reserved": j["gpu_reserve"],
                    "ram_reserved": j["ram_reserve"],
                    "attempt": j["attempt"],
                    "unknown": j["unknown"],
                }
                for j in self.running.values()
            ],
            "waiting_reason": "resource reservation or unknown-setting isolation"
            if self.pending
            else None,
        }
        temp = self.root / "state.tmp"
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(self.root / "state.json")
        if self.summary and time.monotonic() - self.last_log >= 10:
            self.summary.log(
                {
                    "scheduler/queued": len(self.pending),
                    "scheduler/running": len(self.running),
                    "scheduler/oom_retries": self.retries,
                    "scheduler/ram_available_bytes": ram.available,
                    **{
                        f"scheduler/gpu_{device}_{name}": value
                        for device, gpu in stats.items()
                        for name, value in gpu.items()
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
            time.sleep(1)

    def get(self, token):
        return self.finished.pop(token)

    def shutdown(self):
        if self.closed:
            return
        for job in self.running.values():
            kill_group(job["process"])
            job["log"].close()
        self.running.clear()
        self.closed = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    args = parser.parse_args()
    path = Path(args.worker)
    import run as runner

    result = runner._evaluate_job._function(*pickle.loads(path.read_bytes()))
    import torch

    if torch.cuda.is_initialized():
        # Include context/driver overhead beyond the Torch allocator.
        result["allocator_peak_bytes"] = torch.cuda.max_memory_reserved() + 512 * 2**20
    temporary = path.parent / "output.tmp"
    temporary.write_text(json.dumps(result, default=str))
    temporary.replace(path.parent / "output.json")


if __name__ == "__main__":
    main()
