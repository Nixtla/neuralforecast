"""TaskVine manager sidecar; never imports the training dependency environment."""

import csv
import io
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time

import ndcctools.taskvine as vine

from taskvine_resources import atomic_json, gpu_admission


def memory_mb():
    values = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    return int(values["MemAvailable"].split()[0]) // 1024


class Manager:
    def __init__(self, session):
        self.session = Path(session).resolve()
        self.settings = json.loads((self.session / "settings.json").read_text())
        self.root = Path(self.settings["directory"])
        self.m = vine.Manager(0, run_info_path=str(self.session / "vine"))
        self.m.disable_peer_transfers()
        self.m.enable_monitoring(watchdog=True)
        self.m.tune("hungry-minimum", 1)
        # Default warm-up is 25 completions, but an SH category can have fewer
        # folds/rungs. Recompute after every measurement instead of reverting
        # to whole-worker allocations while waiting for 25 samples.
        self.m.tune("category-steady-n-tasks", 1)
        # RAM must not expand merely because a task requests two CPU cores.
        self.m.tune("proportional-resources", 0)
        self.password = self.session / "password"
        self.password.write_text(secrets.token_hex(32))
        self.password.chmod(0o600)
        self.m.set_password_file(str(self.password))
        self.jobs, self.tasks, self.devices, self.local = {}, {}, {}, []
        self.configured = set()
        self.category_completions = {}
        self.retries = 0
        self.last_telemetry = 0
        self.closed = False
        self.profiles_path = self.root / "profiles.json"
        self.profiles = (
            json.loads(self.profiles_path.read_text())
            if self.profiles_path.exists()
            else {}
        )
        bundle = self.session / "source.tar.gz"
        source = Path(self.settings["source"])
        with tarfile.open(bundle, "w:gz") as archive:
            for directory in ("neuralforecast", "experiments/commodity_sota"):
                for path in sorted((source / directory).rglob("*.py")):
                    archive.add(path, arcname=str(path.relative_to(source)))
            # Carry the installed distribution metadata with the frozen code.
            # No pip dependency resolution/upgrades on the training source.
            metadata = self.settings["package_metadata"].encode()
            info = tarfile.TarInfo("neuralforecast.dist-info/METADATA")
            info.size = len(metadata)
            archive.addfile(info, io.BytesIO(metadata))
        self.source = self.m.declare_untar(
            self.m.declare_file(str(bundle), cache="workflow")
        )
        self.probe_file = self.m.declare_file(
            str(Path(__file__).with_name("taskvine_worker.py")), cache="workflow"
        )
        self.credential = None
        if os.environ.get("WANDB_API_KEY"):
            path = self.session / "credential.json"
            path.write_text(json.dumps({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]}))
            path.chmod(0o600)
            self.credential = self.m.declare_file(
                str(path), cache=False, peer_transfer=False
            )
        self.start_local()
        atomic_json(
            self.session / "ready.json", dict(port=self.m.port, pid=os.getpid())
        )

    def start_local(self):
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
        )
        rows = list(csv.reader(output.splitlines()))
        selected = os.environ.get("NF_TASKVINE_LOCAL_GPUS")
        if selected is not None:
            rows = [r for r in rows if r[0].strip() in selected.split(",")]
        if not rows:
            return
        cores = max(2, len(os.sched_getaffinity(0)) // len(rows))
        ram = max(512, int(memory_mb() * 0.85) // len(rows))
        disk = max(
            1024, int(shutil.disk_usage(self.root).free / 2**20 * 0.8) // len(rows)
        )
        for row in rows:
            index, uuid = [s.strip() for s in row]
            name = "local-" + index
            spec = dict(
                name=name,
                device=uuid,
                local=True,
                memory=ram,
                cores=cores,
                disk=disk,
                python=self.settings["training_python"],
            )
            atomic_json(self.session / "devices" / f"{name}.json", spec)
            log = (self.session / f"{name}.log").open("w")
            command = [
                str(Path(self.settings["environment"]) / "bin/vine_worker"),
                "--parent-death",
                "--gpus=0",
                f"--cores={cores}",
                f"--memory={ram}",
                f"--disk={disk}",
                "--feature",
                name,
                "--password",
                str(self.password),
                "--idle-timeout=86400",
                "--workspace",
                str(self.session / (name + "-workspace")),
                "localhost",
                str(self.m.port),
            ]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            self.local.append(
                dict(process=process, log=log, command=command, name=name)
            )

    def task(self, command, device):
        task = vine.Task(command)
        task.add_feature(device["name"])
        task.set_gpus(0)  # Explicit VRAM admission owns shared GPU placement.
        task.set_env_var("CUDA_VISIBLE_DEVICES", device["device"])
        task.set_env_var(
            "LD_LIBRARY_PATH",
            device.get("ld_library_path", os.environ.get("LD_LIBRARY_PATH", "")),
        )
        task.set_env_var("OMP_NUM_THREADS", "2")
        task.set_env_var("MKL_NUM_THREADS", "2")
        task.set_env_var(
            "NF_VINE_REGISTRY", f"/tmp/nf-vine-{self.session.name}/{device['name']}"
        )
        return task

    def poll_devices(self):
        for path in (self.session / "devices").glob("*.lost"):
            lost_at = json.loads(path.read_text())["timestamp"]
            for job in self.jobs.values():
                if (
                    job.get("task")
                    and job["device"] == path.stem
                    and job["started"] < lost_at
                ):
                    self.m.cancel_by_task_id(job["task"])
                    self.tasks.pop(job["task"], None)
                    job.update(task=None, attempt=job["attempt"] + 1)
                    self.retries += 1
                    if job["attempt"] > 3:
                        atomic_json(
                            job["folder"] / "done.json",
                            dict(
                                ok=False,
                                kind="WORKER_FAILURE",
                                error="Colab recovery attempts exhausted",
                            ),
                        )
                        job["done"] = True
            device = self.devices.get(path.stem)
            if device and device["last_probe"] < lost_at:
                device["snapshot"] = None
        for path in (self.session / "devices").glob("*.json"):
            spec = json.loads(path.read_text())
            device = self.devices.setdefault(
                spec["name"], dict(spec, last_probe=0, snapshot=None, probe=None)
            )
            device.update(spec)
            if not device["probe"] and time.time() - device["last_probe"] >= 2:
                task = self.task(
                    f"python3 probe.py --probe {shlex.quote(spec['device'])}", device
                )
                task.add_input(self.probe_file, "probe.py")
                task.set_cores(0)
                task.set_memory(64)
                task.set_disk(1)
                task.set_time_max(15)
                task.set_retries(0)
                task.set_priority(100)
                task_id = self.m.submit(task)
                self.tasks[task_id] = ("probe", device["name"])
                device["probe"] = task_id
                device["last_probe"] = time.time()

    def profile_key(self, job, device):
        return "/".join(
            (self.settings["identity"], device["snapshot"]["hardware"], job["category"])
        )

    def submit_job(self, job, device, reservation, key):
        category = job["category"] + "-" + device["name"]
        if category not in self.configured:
            self.m.set_category_mode(category, "max throughput")
            self.m.set_category_resources_max(
                category,
                # Leave room for telemetry and the worker's input cache even
                # on a maximum-resource retry; otherwise it cannot be placed.
                dict(
                    cores=2,
                    memory=device["memory"] - 256,
                    disk=device["disk"] - 1024,
                    gpus=0,
                ),
            )
            self.m.set_category_resources_min(
                category, dict(cores=2, memory=2048, disk=1, gpus=0)
            )
            self.m.set_category_first_allocation_guess(
                category,
                dict(cores=2, memory=min(4096, device["memory"]), disk=1024, gpus=0),
            )
            self.configured.add(category)
        attempt = job["folder"] / f"attempt-{job['attempt']}"
        attempt.mkdir(exist_ok=True)
        command = f"{shlex.quote(device['python'])} source/experiments/commodity_sota/taskvine_worker.py"
        task = self.task(command, device)
        task.set_category(category)
        task.set_retries(3)
        task.set_env_var("NF_VINE_ATTEMPT", str(job["attempt"]))
        task.set_env_var("NF_VINE_DEVICE", device["name"])
        task.set_env_var("NF_VINE_TOKEN", job["token"])
        task.set_env_var("PYTHONPATH", "source:source/experiments/commodity_sota")
        task.add_input(self.source, "source")
        task.add_input(
            self.m.declare_file(str(job["folder"] / "input.pkl"), cache=False),
            "input.pkl",
        )
        task.add_input(self.m.declare_file(job["data"], cache="workflow"), "data.pkl")
        if job["checkpoint"]:
            task.add_input(
                self.m.declare_file(job["checkpoint"], cache="workflow"), "prior"
            )
        if self.credential:
            task.add_input(self.credential, "credential.json")
        task.add_output(
            self.m.declare_file(str(attempt / "result.json"), cache=False),
            "result.json",
        )
        task.add_output(
            self.m.declare_file(str(attempt / "products"), cache=False), "products"
        )
        tid = self.m.submit(task)
        self.tasks[tid] = ("job", job["token"])
        job.update(
            task=tid,
            device=device["name"],
            reservation=reservation,
            exclusive=job.get("force_exclusive", False) or key not in self.profiles,
            baseline=device["snapshot"]["used"],
            peak=0,
            profile_key=key,
            started=time.time(),
            attempt_folder=attempt,
            vine_task=task,
        )

    def complete(self, task):
        entry = self.tasks.pop(task.id, None)
        if entry is None:
            return
        kind, name = entry
        if kind == "probe":
            device = self.devices[name]
            device["probe"] = None
            if task.exit_code == 0 and task.result == "success":
                try:
                    device["snapshot"] = json.loads(
                        task.output.strip().splitlines()[-1]
                    )
                except (ValueError, TypeError, IndexError):
                    device["snapshot"] = None
            else:
                device["snapshot"] = None
            return
        job = self.jobs[name]
        folder = job["attempt_folder"]
        (folder / "worker.log").write_text(task.output or "")
        result_path = folder / "result.json"
        if task.result == "success" and task.exit_code == 0 and result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            result = dict(
                ok=False,
                kind="WORKER_FAILURE",
                error=f"TaskVine {task.result}; exit={task.exit_code}",
            )
        job["task"] = None
        if result.get("kind") == "OOM" and not job.get("force_exclusive"):
            self.profiles.pop(job["profile_key"], None)
            atomic_json(self.profiles_path, self.profiles)
            job.update(force_exclusive=True, attempt=job["attempt"] + 1)
            self.retries += 1
            return
        if result.get("ok") and job["exclusive"]:
            peak = max(job["peak"], result.get("allocator_peak_bytes", 0))
            if peak:
                self.profiles[job["profile_key"]] = max(
                    self.profiles.get(job["profile_key"], 0), peak
                )
                atomic_json(self.profiles_path, self.profiles)
        if result.get("checkpoint"):
            result["checkpoint"] = str(
                folder.relative_to(job["folder"]) / result["checkpoint"]
            )
        measured = task.resources_measured
        category = job["category"] + "-" + job["device"]
        count = self.category_completions.get(category, 0)
        if result.get("ok"):
            self.category_completions[category] = count + 1
            # 7.17.1 updates steady_state before incrementing the first sample
            # count. Keep the bootstrap guess through that one-sample gap;
            # subsequent allocations are learned entirely by TaskVine.
            if count == 0 and 0 < getattr(measured, "memory", 0) <= 4096:
                device = self.devices[job["device"]]
                self.m.set_category_first_allocation_guess(
                    category,
                    dict(
                        cores=2, memory=min(4096, device["memory"]), disk=1024, gpus=0
                    ),
                )
        result.update(
            worker_mode="taskvine",
            execution_host=job["device"],
            attempt=job["attempt"],
            queue_seconds=job["started"] - job["submitted"],
            wall_seconds=time.time() - job["started"],
            resource_peak=dict(
                gpu_bytes=max(job["peak"], result.get("allocator_peak_bytes", 0)),
                ram_bytes=max(0, getattr(measured, "memory", 0)) * 2**20,
            ),
        )
        allocated = task.resources_allocated
        result["resource_reserved"] = dict(
            ram_bytes=max(0, getattr(allocated, "memory", 0)) * 2**20,
            gpu_bytes=job["reservation"],
        )
        result["execution_interval"] = dict(
            start=getattr(measured, "start", 0), end=getattr(measured, "end", 0)
        )
        atomic_json(job["folder"] / "done.json", result)
        job["done"] = True
        # Avoid using a pre-completion GPU snapshot for the next admission.
        self.devices[job["device"]]["snapshot"] = None

    def tick(self):
        for worker in self.local:
            if worker["process"].poll() is not None:
                worker["process"] = subprocess.Popen(
                    worker["command"], stdout=worker["log"], stderr=subprocess.STDOUT
                )
        for path in (self.session / "jobs").glob("*/request.json"):
            token = path.parent.name
            if token not in self.jobs:
                request = json.loads(path.read_text())
                self.jobs[token] = dict(
                    request, folder=path.parent, task=None, done=False, attempt=0
                )
        self.poll_devices()
        task = self.m.wait(1)
        if task:
            self.complete(task)
        running = [j for j in self.jobs.values() if j["task"]]
        for job in running:
            snapshot = self.devices[job["device"]]["snapshot"]
            if snapshot:
                job["pid"] = snapshot.get("pids", {}).get(job["token"])
                if job["exclusive"]:
                    job["peak"] = max(job["peak"], snapshot["used"] - job["baseline"])
                elif snapshot.get("processes"):
                    job["peak"] = max(
                        job["peak"], snapshot["processes"].get(str(job["pid"]), 0)
                    )
        pending = sorted(
            [j for j in self.jobs.values() if not j["task"] and not j["done"]],
            key=lambda j: j["submitted"],
        )
        drain = None
        if pending and time.time() - pending[0]["submitted"] > 60 and running:
            # Reserve a busy GPU to eventually profile the oldest job. Never
            # block a free GPU merely because an older job cannot fit elsewhere.
            drain = min(
                {j["device"] for j in running},
                key=lambda d: sum(j["device"] == d for j in running),
            )
        for job in pending:
            for device in sorted(
                self.devices.values(),
                key=lambda d: sum(j["device"] == d["name"] for j in running),
            ):
                if not device["snapshot"] or (
                    job["local_only"] and not device["local"]
                ):
                    continue
                if drain == device["name"] and job is not pending[0]:
                    continue
                key = self.profile_key(job, device)
                peers = [j for j in running if j["device"] == device["name"]]
                reservation = gpu_admission(
                    device["snapshot"],
                    peers,
                    self.profiles.get(key),
                    job.get("force_exclusive", False),
                )
                if reservation is not None:
                    self.submit_job(job, device, reservation, key)
                    running.append(job)
                    break
        state = dict(
            queued=sum(not j["task"] and not j["done"] for j in self.jobs.values()),
            running=len(running),
            workers=self.m.stats.workers_connected,
            retries=self.retries,
            timestamp=time.time(),
            devices={k: v["snapshot"] for k, v in self.devices.items()},
            jobs=[
                dict(
                    {
                        k: j.get(k)
                        for k in (
                            "token",
                            "category",
                            "device",
                            "reservation",
                            "exclusive",
                            "started",
                        )
                    },
                    ram_reserved=max(
                        0, getattr(j["vine_task"].resources_allocated, "memory", 0)
                    )
                    * 2**20,
                )
                for j in running
            ],
        )
        atomic_json(self.root / "state.json", state)
        if time.time() - self.last_telemetry >= 10:
            with (self.root / "telemetry.jsonl").open("a") as stream:
                stream.write(json.dumps(state) + "\n")
            self.last_telemetry = time.time()

    def close(self):
        self.m.cancel_all()
        for worker in self.local:
            process, log = worker["process"], worker["log"]
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            log.close()
        for path in (self.password, self.session / "credential.json"):
            path.unlink(missing_ok=True)


def main():
    manager = Manager(sys.argv[1])

    def stop(signum, frame):
        manager.closed = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not manager.closed:
            manager.tick()
            time.sleep(0.1)
    finally:
        manager.close()


if __name__ == "__main__":
    main()
