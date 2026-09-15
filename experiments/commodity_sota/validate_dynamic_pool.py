"""A/B validation gate; never starts a production experiment."""

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import pandas as pd

import dynamic_queue
import dynamic_pool
import run as runner


def comparable(reference, pooled):
    if not reference.get("ok") or not pooled.get("ok"):
        return False
    for key in ("actual_steps", "best_step", "stop_reason"):
        if reference.get(key) != pooled.get(key):
            return False
    for key in (
        "prediction",
        "actual",
        "best_validation_loss",
        "mae",
        "mape_pct",
        "mse",
        "rmse",
        "da_pct",
    ):
        a, b = reference.get(key), pooled.get(key)
        if a is None or b is None:
            if a != b:
                return False
        elif not np.allclose(a, b, rtol=1e-5, atol=1e-6):
            return False
    # Constant-price Naive must produce the same admission decision.
    for values in (reference, pooled):
        actual = np.asarray(values["actual"])
        naive = float(np.sqrt(np.mean((actual - values["forecast_origin"]) ** 2)))
        if (reference["rmse"] < naive) != (pooled["rmse"] < naive):
            return False
    return True


def pooled_admission(rows, candidate):
    latest = {}
    for key, row in rows.items():
        name, fold, budget = key.split("/")
        if name != candidate or not row.get("ok"):
            continue
        if fold not in latest or int(budget) > latest[fold][0]:
            latest[fold] = int(budget), row
    if not latest:
        return None
    actual = np.concatenate([r["actual"] for _, r in latest.values()])
    prediction = np.concatenate([r["prediction"] for _, r in latest.values()])
    origins = np.concatenate(
        [np.full(len(r["actual"]), r["forecast_origin"]) for _, r in latest.values()]
    )
    return bool(np.mean((actual - prediction) ** 2) < np.mean((actual - origins) ** 2))


def gate(reference_times, pooled_times, approved, failures_reference, failures_pool):
    speedup = 1 - statistics.median(pooled_times) / statistics.median(reference_times)
    return dict(
        speedup_fraction=speedup,
        default_eligible=bool(approved)
        and speedup >= 0.15
        and failures_pool <= failures_reference,
        reference_seconds=reference_times,
        pooled_seconds=pooled_times,
    )


def run_workload(mode, output, data_path, candidates, folds):
    started = time.monotonic()
    queue = (
        dynamic_pool.PoolQueue(output, reuse_models={c.name for c in candidates})
        if mode == "dynamic-pool"
        else dynamic_queue.DynamicQueue(output)
    )
    rows = {}

    def payload(candidate):
        value = runner._payload(candidate)
        value["tracking"] = dict(
            entity="Beat-Sun",
            project="uni-gasoline-diff",
            group=f"pool-validation-{output.name}",
            directory=str(output / "wandb"),
            mode="offline",
        )
        return value

    try:
        pending = {}
        for candidate in candidates:
            for fold in folds:
                budget = 100 if candidate.plan else 0
                token = queue.submit(
                    str(data_path),
                    payload(candidate),
                    0,
                    candidate.configs[0],
                    fold,
                    budget,
                    None,
                    output / "checkpoints_phase1",
                )
                pending[token] = (candidate, fold, budget)
        while pending:
            ready, _ = queue.wait(list(pending), num_returns=1)
            for token in ready:
                candidate, fold, budget = pending.pop(token)
                result = queue.get(token)
                key = f"{candidate.name}/{fold.index}/{budget}"
                rows[key] = result
                if len(rows) % 10 == 0:
                    print(
                        f"{mode}: {len(rows)} jobs completed, {len(pending)} pending",
                        flush=True,
                    )
                # Exercise the actual 100 -> 250 -> 500 checkpoint transitions,
                # including the production terminal-result reuse rule.
                if (
                    candidate.plan
                    and result.get("ok")
                    and result.get("stop_reason") != "early_stopping"
                    and budget < 500
                ):
                    next_budget = 250 if budget == 100 else 500
                    token = queue.submit(
                        str(data_path),
                        payload(candidate),
                        0,
                        candidate.configs[0],
                        fold,
                        next_budget,
                        result["checkpoint"],
                        output / "checkpoints_phase1",
                    )
                    pending[token] = candidate, fold, next_budget
                (output / "results.json").write_text(json.dumps(rows, indent=2))
    finally:
        queue.shutdown()
    return rows, time.monotonic() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-config", default="model_config.json", type=Path)
    parser.add_argument(
        "--models", nargs="+", help="Default: every smoke-validated candidate"
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh validation output directory")
    current = json.loads((args.experiment / "run_config.json").read_text())
    if current["status"] not in {"completed", "no_models_above_naive", "failed"}:
        raise ValueError(
            "Production experiment is still running; GPU validation must wait"
        )
    # One physical GPU for both methods makes the equivalence comparison explicit.
    # Multi-GPU placement itself is covered by scheduler regression tests.
    snapshot = dynamic_queue.gpu_snapshot
    available = snapshot()
    if args.gpu not in available or available[args.gpu]["used"] > 256 * 2**20:
        raise ValueError("Validation GPU is unavailable or busy")

    def selected_gpu():
        return {args.gpu: snapshot()[args.gpu]}

    dynamic_queue.gpu_snapshot = selected_gpu
    dynamic_pool.gpu_snapshot = selected_gpu
    args.output.mkdir(parents=True)
    fingerprint = dynamic_pool.execution_fingerprint()
    frame = pd.read_pickle(args.experiment / "weekly.pkl")
    config = json.loads(args.model_config.read_text())
    runner.POLICY = current["phase2_policy"]
    h = current["horizon"]
    minimum = runner._minimum_train(h, config)
    candidates, _ = runner._build_candidates(h, config, minimum)
    allowed = set(current.get("candidates", []))
    if not allowed:
        eligibility = pd.read_csv(args.experiment / "eligibility.csv")
        for row in eligibility.itertuples():
            if row.status == "READY":
                allowed.add(
                    row.model
                    if row.protocol == "scratch_hpo"
                    else f"{row.model}-{'LoRA' if row.protocol == 'lora' else 'ZeroShot'}"
                )
    if args.models:
        allowed &= set(args.models)
    candidates = [c for c in candidates if c.name in allowed]
    if not candidates:
        raise ValueError("No candidates to validate")
    folds = runner.expanding_folds(len(frame), h=h, min_train=minimum + h, step_size=1)
    folds = [folds[0], folds[-1]]
    data_path = args.experiment.resolve() / "weekly.pkl"
    reference_times, pooled_times = [], []
    approved = {c.name for c in candidates}
    failures = {"dynamic": 0, "dynamic-pool": 0}
    comparisons = []
    for repeat in range(3):
        results = {}
        # Alternate order to limit filesystem/model-file cache bias.
        order = (
            ["dynamic", "dynamic-pool"]
            if repeat % 2 == 0
            else ["dynamic-pool", "dynamic"]
        )
        for mode in order:
            print(f"Validation repeat={repeat + 1}/3 mode={mode}", flush=True)
            directory = args.output / f"{repeat}-{mode}"
            results[mode], duration = run_workload(
                mode, directory, data_path, candidates, folds
            )
            (reference_times if mode == "dynamic" else pooled_times).append(duration)
            failures[mode] += sum(not r.get("ok") for r in results[mode].values())
        for key in sorted(set(results["dynamic"]) | set(results["dynamic-pool"])):
            reference, pooled = results["dynamic"].get(key, {}), results[
                "dynamic-pool"
            ].get(key, {})
            ok = comparable(reference, pooled)
            candidate = key.split("/")[0]
            if not ok:
                approved.discard(candidate)
            comparisons.append(dict(repeat=repeat, job=key, equivalent=ok))
        for candidate in candidates:
            if pooled_admission(results["dynamic"], candidate.name) != pooled_admission(
                results["dynamic-pool"], candidate.name
            ):
                approved.discard(candidate.name)
        (args.output / "comparisons.json").write_text(json.dumps(comparisons, indent=2))
    report = dict(
        execution_fingerprint=fingerprint,
        approved_models=sorted(approved),
        isolated_models=sorted({c.name for c in candidates} - approved),
        repetitions=3,
        gpu=args.gpu,
        timing_scope="single GPU, W&B offline lifecycle enabled",
        **gate(
            reference_times,
            pooled_times,
            approved,
            failures["dynamic"],
            failures["dynamic-pool"],
        ),
        failures=failures,
    )
    # Any divergent model changes the workload for fallback; do not extrapolate
    # measured speed to that unmeasured hybrid execution mode.
    if len(approved) != len(candidates):
        report["default_eligible"] = False
    if dynamic_pool.execution_fingerprint() != fingerprint:
        report.update(
            default_eligible=False,
            approved_models=[],
            error="Source changed during validation",
        )
    temporary = args.output / "validation.tmp"
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(args.output / "validation.json")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
