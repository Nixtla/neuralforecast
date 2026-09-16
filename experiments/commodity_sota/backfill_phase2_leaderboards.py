"""Publish saved Phase 2 results, optionally watching an active experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from neuralforecast.benchmark import Fold
from neuralforecast.benchmark_leaderboard import (
    Phase2Publisher,
    TERMINAL_STATUSES,
    leaderboard,
    metric_definitions,
    publisher_lock,
    validation_reference,
)
from neuralforecast.benchmark_tracking import safe_config


def evaluation_reference(output, config):
    """Reconstruct the stored evaluation window using original fold identities."""
    frame = pd.read_pickle(Path(output) / "weekly.pkl")
    first, last = config["phase2_first_fold"], config["phase2_last_fold"]
    count = config["phase2_folds"]
    if config.get("step_size", 1) != 1 or count != last - first + 1:
        raise ValueError("Expected contiguous Phase 2 folds with step_size=1")
    horizon = config["horizon"]
    first_train = len(frame) - horizon - count + 1
    if first_train != config["phase2_start_cutoff"]:
        raise ValueError("Saved data does not match the Phase 2 evaluation window")
    folds = [
        Fold(first + i, first_train + i, first_train + i + horizon)
        for i in range(count)
    ]
    return validation_reference(frame, folds)


def selected_models(output, config):
    """Read admission decisions rather than admitting any available checkpoint."""
    ranking = pd.read_csv(Path(output) / "phase1_ranking.csv")
    chosen = ranking.selected.astype(str).str.lower().eq("true")
    selected = ranking.loc[chosen, ["candidate", "protocol"]]
    if selected.candidate.duplicated().any():
        raise ValueError("Duplicate selected candidates")
    models = dict(zip(selected.candidate, selected.protocol))
    if "selected" in config and set(config["selected"]) != set(models):
        raise ValueError("Phase 1 admission disagrees with run configuration")
    return models


def saved_predictions(output, config, models):
    """Collect complete, finite, matching results only from the retained window."""
    predictions = []
    for candidate, protocol in models.items():
        budget = (
            config["phase2_policy"]["max_steps"]
            if protocol in {"scratch_hpo", "lora"}
            else 0
        )
        for fold in range(config["phase2_first_fold"], config["phase2_last_fold"] + 1):
            path = (
                Path(output)
                / "checkpoints_phase2"
                / candidate
                / "0"
                / str(fold)
                / f"result-{budget}.json"
            )
            try:
                result = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue  # The worker may still be writing this result.
            if not result.get("ok"):
                continue
            if (
                result.get("candidate") != candidate
                or result.get("config_id") != 0
                or result.get("fold") != fold
                or result.get("budget") != budget
                or result.get("protocol_version") != config["protocol_version"]
            ):
                raise ValueError(f"Incompatible Phase 2 result: {path}")
            actual, prediction = result.get("actual", []), result.get("prediction", [])
            if (
                len(actual) != config["horizon"]
                or len(prediction) != len(actual)
                or not np.isfinite(actual).all()
                or not np.isfinite(prediction).all()
            ):
                continue
            predictions.extend(
                dict(
                    candidate=candidate,
                    protocol=protocol,
                    fold=fold,
                    horizon=i,
                    actual=y,
                    prediction=p,
                )
                for i, (y, p) in enumerate(zip(actual, prediction), 1)
            )
    return predictions


def collect_snapshot(output):
    """Validate a final CSV or build a complete-model snapshot from saved folds."""
    output = Path(output)
    config = json.loads((output / "run_config.json").read_text())
    if not (output / "phase1_ranking.csv").exists():
        if config.get("status") in TERMINAL_STATUSES:
            raise ValueError("Terminal experiment has no Phase 1 ranking")
        return config, None
    models = selected_models(output, config)
    reference = evaluation_reference(output, config)
    definitions = metric_definitions(reference)
    status = config.get("status", "phase2")
    if status in TERMINAL_STATUSES and (output / "leaderboard.csv").exists():
        predictions = pd.read_csv(output / "phase2_predictions.csv")
        if not set(predictions.candidate).issubset(models):
            raise ValueError("Final predictions contain an unselected candidate")
        board = pd.DataFrame(leaderboard(predictions.to_dict("records"), reference))
        stored = pd.read_csv(output / "leaderboard.csv")
        pd.testing.assert_frame_equal(
            stored[board.columns].reset_index(drop=True),
            board,
            check_dtype=False,
            check_exact=False,
            rtol=1e-10,
            atol=1e-12,
        )
        stored_definitions = json.loads(
            (output / "metric_definitions.json").read_text()
        )
        if stored_definitions["phase2"] != definitions:
            raise ValueError(
                "Saved metric definitions differ from the evaluation window"
            )
    else:
        board = pd.DataFrame(
            leaderboard(saved_predictions(output, config, models), reference)
        )
        if status not in TERMINAL_STATUSES:
            status = "phase2"
    return config, (board, definitions, len(models), status)


def publish_experiment(output, *, watch=False, poll_seconds=60):
    """Publish once or follow local results without touching the training process."""
    output = Path(output).resolve()
    with publisher_lock(output):
        publisher = None
        try:
            while True:
                try:
                    config, snapshot = collect_snapshot(output)
                    if snapshot is not None:
                        if publisher is None:
                            metadata = json.loads(
                                (output / "wandb_run.json").read_text()
                            )
                            for key in ("entity", "project", "group"):
                                if metadata[key] != config["wandb"][key]:
                                    raise ValueError(
                                        f"W&B {key} differs from saved config"
                                    )
                            publisher = Phase2Publisher(output, metadata, config)
                        board, definitions, total, status = snapshot
                        changed = publisher.publish(
                            board,
                            definitions,
                            total_models=total,
                            status=status,
                        )
                        if changed:
                            print(
                                f"{output.name}: {len(board) - 1}/{total} models; "
                                f"{publisher.url}",
                                flush=True,
                            )
                        if status in TERMINAL_STATUSES:
                            return
                    else:
                        print(f"{output.name}: waiting for Phase 1", flush=True)
                    if not watch:
                        return
                except Exception as exc:
                    if not watch:
                        raise
                    print(
                        f"{output.name}: retrying: {safe_config(str(exc))}", flush=True
                    )
                time.sleep(poll_seconds)
        finally:
            if publisher is not None:
                publisher.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    publish_experiment(args.output, watch=args.watch, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
