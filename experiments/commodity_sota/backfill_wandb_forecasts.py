"""Backfill fold-level forecast arrays from local results into existing W&B runs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
import time

from neuralforecast.benchmark_tracking import run_id


def collect_forecasts(output, phases=("phase1", "phase2")):
    """Return the highest-budget successful result for each W&B fold run."""
    selected = {}
    for phase in phases:
        for path in (Path(output) / f"checkpoints_{phase}").rglob("result-*.json"):
            try:
                result = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not result.get("ok"):
                continue
            if not result.get("actual") or len(result["actual"]) != len(
                result.get("prediction", [])
            ):
                continue
            key = (
                phase,
                result["candidate"],
                int(result["config_id"]),
                int(result["fold"]),
            )
            prior = selected.get(key)
            if prior is None or result.get("budget", 0) > prior.get("budget", 0):
                selected[key] = result
    return {
        key: result
        for key, result in selected.items()
        if result.get("forecast_tracking_version") != 1
    }


def _publish(api, path, result):
    run = api.run(path)
    actual = [float(value) for value in result["actual"]]
    prediction = [float(value) for value in result["prediction"]]
    run.summary.update(
        {
            "forecast/actual": actual,
            "forecast/prediction": prediction,
            "forecast/horizon": list(range(1, len(actual) + 1)),
            "forecast/origin": float(result["forecast_origin"]),
            "forecast/error": [
                predicted - observed for observed, predicted in zip(actual, prediction)
            ],
            "forecast/tracking_version": 1,
            "forecast/backfilled": True,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", action="append", choices=("phase1", "phase2"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    metadata = json.loads((args.output / "wandb_run.json").read_text())
    forecasts = collect_forecasts(args.output, args.phase or ("phase1", "phase2"))
    state_path = args.output / "wandb_forecast_backfill.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    completed = set(state.get("completed", []))
    jobs = []
    for key, result in sorted(forecasts.items()):
        identity = run_id(metadata["group"], *key)
        if identity not in completed:
            jobs.append((identity, result))
    if args.limit is not None:
        jobs = jobs[: args.limit]

    import wandb

    api = wandb.Api(timeout=60)
    lock = threading.Lock()
    failures = {}

    def publish(job):
        identity, result = job
        path = f"{metadata['entity']}/{metadata['project']}/{identity}"
        for attempt in range(4):
            try:
                _publish(api, path, result)
                return identity, None
            except Exception as exc:
                if attempt == 3:
                    return identity, f"{type(exc).__name__}: {exc}"
                time.sleep(2**attempt)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(publish, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            identity, error = future.result()
            with lock:
                if error:
                    failures[identity] = error
                else:
                    completed.add(identity)
                if index % 25 == 0 or index == len(futures):
                    temporary = state_path.with_suffix(".tmp")
                    temporary.write_text(
                        json.dumps(
                            {
                                "completed": sorted(completed),
                                "failures": failures,
                                "remaining": len(futures) - index,
                            },
                            indent=2,
                        )
                    )
                    temporary.replace(state_path)
                    print(
                        f"processed={index}/{len(futures)} "
                        f"completed={len(completed)} failures={len(failures)}",
                        flush=True,
                    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
