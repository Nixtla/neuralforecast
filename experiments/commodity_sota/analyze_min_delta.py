"""Replay relative min_delta policies on a stratified W&B validation sample.

Reads full, unsampled history through the public API. Does not train models or
write to W&B. Results describe only the already observed portions of curves.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd


def replay(curve, relative_delta, patience=5):
    """Keep true best loss; reset patience only for cumulative material progress."""
    reference = best = float("inf")
    bad = 0
    for step, loss in curve:
        if not np.isfinite(loss) or loss < 0:
            raise ValueError("Expected finite non-negative validation MSE")
        best = min(best, loss)
        if loss < reference * (1 - relative_delta):
            reference, bad = loss, 0
        else:
            bad += 1
        if bad >= patience:
            break
    return {"stop_step": step, "best_loss": best}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-folds", type=int, default=21)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh analysis output directory")
    import wandb

    meta = json.loads((args.experiment / "wandb_run.json").read_text())
    config = json.loads((args.experiment / "run_config.json").read_text())
    fold_ids = sorted(
        set(
            np.linspace(0, config["phase2_folds"] - 1, args.sample_folds)
            .astype(int)
            .tolist()
            + config["phase1_folds"]
        )
    )
    api = wandb.Api(timeout=60)
    runs = list(
        api.runs(
            f"{meta['entity']}/{meta['project']}",
            filters={
                "group": meta["group"],
                "jobType": "phase2",
                "state": "finished",
                "config.fold": {"$in": fold_ids},
            },
            per_page=100,
        )
    )
    print(f"Reading {len(runs)} runs across {len(fold_ids)} folds", flush=True)

    def read(run):
        history = list(
            run.scan_history(
                keys=["train/global_step", "validation/loss"], page_size=1000
            )
        )
        points = {
            int(r["train/global_step"]): float(r["validation/loss"]) for r in history
        }
        curve = sorted(points.items())
        if not curve:
            return None
        expected_step = int(run.summary["actual_steps"])
        interval = config["phase2_policy"]["interval"]
        if [s for s, _ in curve] != list(range(interval, expected_step + 1, interval)):
            raise ValueError(f"Incomplete validation history: {run.id}")
        baseline = replay(curve, 0.0, config["phase2_policy"]["patience"])
        if baseline["stop_step"] != expected_step or not np.isclose(
            baseline["best_loss"], run.summary["best_validation_loss"]
        ):
            raise ValueError(f"Baseline replay does not match run: {run.id}")
        return {
            "run_id": run.id,
            "candidate": run.summary["candidate"],
            "fold": run.config["fold"],
            "curve": curve,
        }

    curves = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, curve in enumerate(executor.map(read, runs), 1):
            if curve is not None:
                curves.append(curve)
            if i % 25 == 0:
                print(f"Read {i}/{len(runs)}", flush=True)
    if not curves:
        raise RuntimeError("No complete validation curves found")
    records = []
    for item in curves:
        base = replay(item["curve"], 0.0, config["phase2_policy"]["patience"])
        for delta in (0.0, 0.001, 0.0025, 0.005, 0.01):
            result = replay(item["curve"], delta, config["phase2_policy"]["patience"])
            degradation = result["best_loss"] - base["best_loss"]
            records.append(
                {
                    "candidate": item["candidate"],
                    "fold": item["fold"],
                    "relative_delta": delta,
                    "baseline_steps": base["stop_step"],
                    "stop_step": result["stop_step"],
                    "steps_saved": base["stop_step"] - result["stop_step"],
                    "best_loss_increase": degradation,
                    "best_loss_increase_pct": (
                        100 * degradation / base["best_loss"]
                        if base["best_loss"] > 0
                        else 0.0
                    ),
                }
            )
    detail = pd.DataFrame(records)
    summaries = []
    for name, part in [("ALL", detail), *detail.groupby("candidate")]:
        for delta, rows in part.groupby("relative_delta"):
            summaries.append(
                {
                    "candidate": name,
                    "relative_delta": delta,
                    "runs": len(rows),
                    "steps_saved_pct": 100
                    * rows.steps_saved.sum()
                    / rows.baseline_steps.sum(),
                    "loss_increase_pct_mean": rows.best_loss_increase_pct.mean(),
                    "loss_increase_pct_p95": rows.best_loss_increase_pct.quantile(0.95),
                    "loss_increase_pct_max": rows.best_loss_increase_pct.max(),
                    "runs_loss_increase_over_1pct": int(
                        (rows.best_loss_increase_pct > 1).sum()
                    ),
                }
            )
    summary = pd.DataFrame(summaries)
    args.output.mkdir(parents=True)
    (args.output / "curves.json").write_text(json.dumps(curves))
    (args.output / "method.json").write_text(
        json.dumps(
            {
                "group": meta["group"],
                "folds": fold_ids,
                "relative_deltas": [0.0, 0.001, 0.0025, 0.005, 0.01],
                "patience": config["phase2_policy"]["patience"],
                "semantics": "Reference resets only on material cumulative improvement; true best weights still retained.",
                "limits": "Stratified sample of selected configurations under the OLD protocol; step savings are not wall-time savings. No predictions from alternative checkpoints were evaluated.",
            },
            indent=2,
        )
    )
    detail.to_csv(args.output / "details.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    print(summary[summary.candidate == "ALL"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
