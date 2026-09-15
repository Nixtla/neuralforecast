"""Tighten a completed experiment's Phase 2 window without retraining."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import time

import pandas as pd


PHASE2_NAME = re.compile(r"^phase2/.+/config-\d+/fold-(\d+)$")


def _delete_run(run):
    for attempt in range(5):
        try:
            run.delete(delete_artifacts=True)
            return run.id
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def _inventory(api, config, first_fold):
    tracking = config["wandb"]
    path = f"{tracking['entity']}/{tracking['project']}"
    runs = list(
        api.runs(path, filters={"group": tracking["group"]}, per_page=200)
    )
    early = []
    retained = []
    summaries = []
    phase1 = []
    unexpected = []
    for run in runs:
        match = PHASE2_NAME.match(run.name)
        if match:
            (early if int(match.group(1)) < first_fold else retained).append(run)
        elif run.name.startswith("phase1/"):
            phase1.append(run)
        elif run.name.startswith("summary/"):
            summaries.append(run)
        else:
            unexpected.append(run.name)
    return path, early, retained, phase1, summaries, unexpected


def _local_inventory(output, config, first_fold, last_fold):
    selected = config["selected"]
    root = output / "checkpoints_phase2"
    retained = []
    early = []
    for candidate in selected:
        parent = root / candidate / "0"
        folders = sorted(
            (path for path in parent.iterdir() if path.is_dir()),
            key=lambda path: int(path.name),
        )
        for folder in folders:
            fold = int(folder.name)
            (early if fold < first_fold else retained).append(folder)
    expected = len(selected) * (last_fold - first_fold + 1)
    retained_results = [
        folder
        for folder in retained
        if any(folder.glob("result-*.json"))
        and first_fold <= int(folder.name) <= last_fold
    ]
    if len(retained_results) != expected:
        raise ValueError(
            f"Expected {expected} retained local results, found {len(retained_results)}"
        )
    return early, retained_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-fold", type=int, required=True)
    parser.add_argument("--last-fold", type=int, required=True)
    parser.add_argument("--cutoff", type=int, required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    config_path = output / "run_config.json"
    config = json.loads(config_path.read_text())
    if config.get("status") != "completed":
        raise ValueError("Migration requires a completed experiment")
    if config.get("phase2_folds") != args.last_fold + 1:
        raise ValueError("Unexpected existing Phase 2 fold range")

    early_local, retained_local = _local_inventory(
        output, config, args.first_fold, args.last_fold
    )
    import wandb

    api = wandb.Api(timeout=120)
    path, early, retained, phase1, summaries, unexpected = _inventory(
        api, config, args.first_fold
    )
    expected_retained = len(config["selected"]) * (
        args.last_fold - args.first_fold + 1
    )
    if len(retained) != expected_retained:
        raise ValueError(
            f"Expected {expected_retained} retained W&B runs, found {len(retained)}"
        )
    if len(summaries) != 1 or unexpected:
        raise ValueError(
            f"Expected one summary and no unknown runs; got {len(summaries)}, {unexpected}"
        )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "applied": False,
        "project": path,
        "group": config["wandb"]["group"],
        "phase2_start_ratio": 0.7,
        "phase2_start_cutoff": args.cutoff,
        "phase2_first_fold": args.first_fold,
        "phase2_last_fold": args.last_fold,
        "phase1_runs_preserved": len(phase1),
        "phase2_runs_to_delete": len(early),
        "phase2_runs_retained": len(retained),
        "local_folders_to_delete": len(early_local),
        "local_results_retained": len(retained_local),
        "wandb_run_ids_to_delete": sorted(run.id for run in early),
        "local_paths_to_delete": sorted(str(path) for path in early_local),
    }
    manifest_path = output / "phase2_window_migration.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if not isinstance(v, list)}))
    if not args.apply:
        return

    artifacts = list(summaries[0].logged_artifacts())
    for artifact in artifacts:
        artifact.delete(delete_aliases=True)
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(_delete_run, run) for run in early]
        for done, future in enumerate(as_completed(futures), start=1):
            future.result()
            if done % 100 == 0 or done == len(futures):
                print(f"deleted_wandb_runs={done}/{len(futures)}", flush=True)

    for folder in early_local:
        shutil.rmtree(folder)
    predictions = pd.read_csv(output / "phase2_predictions.csv")
    predictions = predictions.loc[predictions["fold"] >= args.first_fold]
    predictions.to_csv(output / "phase2_predictions.csv", index=False)

    smoke = json.loads(args.smoke_report.read_text())
    config.update(
        fingerprint=args.fingerprint,
        status="phase2_migrating",
        candidates=smoke["passed"],
        phase2_folds=args.last_fold - args.first_fold + 1,
        phase2_start_ratio=0.7,
        phase2_start_cutoff=args.cutoff,
        phase2_first_fold=args.first_fold,
        phase2_last_fold=args.last_fold,
    )
    config_path.write_text(json.dumps(config, indent=2))
    smoke["fingerprint"] = args.fingerprint
    args.smoke_report.write_text(json.dumps(smoke, indent=2))
    manifest["applied"] = True
    manifest["applied_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
