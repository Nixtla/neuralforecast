"""Task entrypoint and inexpensive GPU telemetry (also runs on Colab)."""

import argparse
import csv
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time


def probe(device):
    fields = "uuid,name,driver_version,memory.total,memory.used,utilization.gpu"
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(device),
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    row = next(csv.reader(output.splitlines()))
    processes = None
    try:
        values = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(device),
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        processes = {
            str(int(r[0])): int(r[1]) * 2**20 for r in csv.reader(values.splitlines())
        }
    except (ValueError, subprocess.SubprocessError):
        pass
    pids = {}
    for path in Path(os.environ.get("NF_VINE_REGISTRY", "/nonexistent")).glob("*.json"):
        try:
            pid = json.loads(path.read_text())["pid"]
            if Path(f"/proc/{pid}").exists():
                pids[path.stem] = pid
        except (OSError, ValueError, KeyError):
            continue
    return dict(
        uuid=row[0].strip(),
        hardware="/".join(x.strip() for x in row[1:3]),
        total=int(row[3]) * 2**20,
        used=int(row[4]) * 2**20,
        utilization=int(row[5]),
        processes=processes,
        pids=pids,
        timestamp=time.time(),
    )


def evaluate():
    # A local worker inherits the summary run's service socket, whereas a Colab
    # worker cannot access it. Each sandbox must own its W&B service and paths.
    for key in ("WANDB_SERVICE", "_WANDB_SERVICE", "WANDB_RUN_ID"):
        os.environ.pop(key, None)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import run as runner

    args = pickle.loads(Path("input.pkl").read_bytes())
    if args[1].get("tracking"):
        args[1]["tracking"] = dict(
            args[1]["tracking"], directory=str(Path("wandb").resolve())
        )
    args[1] = dict(
        args[1],
        attempt=int(os.environ.get("NF_VINE_ATTEMPT", "0")),
        execution_host=os.environ.get("NF_VINE_DEVICE", "taskvine"),
    )
    Path("products").mkdir(exist_ok=True)
    credential = Path("credential.json")
    if credential.exists():
        os.environ.update(json.loads(credential.read_text()))
        # TaskVine input is a cached hardlink: never rewrite its content.
        credential.unlink()
    result = runner._evaluate_job._function(*args)
    import torch

    if torch.cuda.is_initialized():
        result["allocator_peak_bytes"] = torch.cuda.max_memory_reserved() + 512 * 2**20
    if result.get("checkpoint"):
        checkpoint = Path(result["checkpoint"]).resolve()
        result["checkpoint"] = str(checkpoint.relative_to(Path.cwd().resolve()))
    result["worker_pid"] = os.getpid()
    Path("result.json").write_text(json.dumps(result, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe")
    options = parser.parse_args()
    if options.probe is not None:
        print(json.dumps(probe(options.probe)))
    else:
        registry = Path(os.environ["NF_VINE_REGISTRY"])
        registry.mkdir(parents=True, exist_ok=True)
        path = registry / (os.environ["NF_VINE_TOKEN"] + ".json")
        path.write_text(json.dumps(dict(pid=os.getpid())))
        try:
            evaluate()
        finally:
            path.unlink(missing_ok=True)
