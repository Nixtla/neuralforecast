"""Sequential, GPU-pinned benchmark worker; no model objects survive a job."""

import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import pickle
import sys
import time
import warnings

import psutil


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, default=str))
    temporary.replace(path)


@contextlib.contextmanager
def job_log(path):
    """Capture Python and native output, including backend subprocess output."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    with Path(path).open("a") as stream:
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for target, source in zip((1, 2), saved):
                os.dup2(source, target)
                os.close(source)


def _wandb_service(process):
    try:
        command = " ".join(process.cmdline()).lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    return "wandb-core" in command or "wandb-service" in command


def stop_children():
    # W&B owns one process-global service which must survive between logical
    # runs. Killing it leaves the SDK singleton connected to a closed mailbox.
    children = [
        child
        for child in psutil.Process().children(recursive=True)
        if not _wandb_service(child)
    ]
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(children, timeout=2)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=2)
    return not alive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready", required=True)
    args = parser.parse_args()
    import run as runner
    import numpy as np
    import torch

    baseline_env = dict(os.environ)
    baseline_warnings = list(warnings.filters)
    baseline_print = np.get_printoptions()
    baseline_numpy_errors = np.geterr()
    baseline_dtype = torch.get_default_dtype()
    baseline_device = torch.get_default_device()
    baseline_grad = torch.is_grad_enabled()
    baseline_tf32 = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    baseline_precision = torch.get_float32_matmul_precision()
    baseline_cudnn = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    baseline_deterministic = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    atomic_json(args.ready, {"pid": os.getpid(), "ready_at": time.time()})
    for line in sys.stdin:
        path = Path(json.loads(line)["input"])
        task = pickle.loads(path.read_bytes())
        started = time.time()
        result = None
        healthy = True
        with job_log(path.parent / "worker.log"):
            try:
                if torch.cuda.is_initialized():
                    torch.cuda.reset_peak_memory_stats()
                # Every backend receives its existing seed=42; rung resume restores
                # the checkpoint RNG inside fit, after this fresh-task reset.
                runner.pl.seed_everything(42, workers=True, verbose=False)
                result = runner._evaluate_job._function(*task)
            except Exception as exc:
                result = {
                    "ok": False,
                    "kind": "WORKER_FAILURE",
                    "candidate": task[1]["name"],
                    "fold": task[4].index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            evaluated = time.time()
            try:
                # W&B must finish before terminating backend helper processes.
                import wandb

                if wandb.run is not None:
                    wandb.finish(exit_code=0 if result["ok"] else 1)
                # `teardown` permanently joins W&B's process-global async manager.
                # A reused worker must be able to initialize tracking for its next
                # logical job; finishing the active run is sufficient isolation.
                healthy = stop_children()
                task = None
                gc.collect()
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
                    result["allocator_peak_bytes"] = (
                        torch.cuda.max_memory_reserved() + 512 * 2**20
                    )
                    torch.cuda.empty_cache()
                    # Live tensor allocations mean task state escaped cleanup.
                    healthy = healthy and torch.cuda.memory_allocated() == 0
                os.environ.clear()
                os.environ.update(baseline_env)
                warnings.filters[:] = baseline_warnings
                np.set_printoptions(**baseline_print)
                np.seterr(**baseline_numpy_errors)
                torch.set_default_dtype(baseline_dtype)
                if torch.get_default_device() != baseline_device:
                    torch.set_default_device(baseline_device)
                torch.set_grad_enabled(baseline_grad)
                torch.set_float32_matmul_precision(baseline_precision)
                (
                    torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32,
                ) = baseline_tf32
                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = (
                    baseline_cudnn
                )
                torch.use_deterministic_algorithms(
                    baseline_deterministic[0], warn_only=baseline_deterministic[1]
                )
            except Exception as exc:
                healthy = False
                result["cleanup_error"] = type(exc).__name__
            result.update(
                evaluation_seconds=evaluated - started,
                cleanup_seconds=time.time() - evaluated,
                worker_started_at=started,
                worker_pid=os.getpid(),
                worker_reusable=healthy
                and result["ok"]
                and not result.get("tracking_error"),
            )
        atomic_json(path.parent / "output.json", result)
        if not result["worker_reusable"]:
            return


if __name__ == "__main__":
    main()
