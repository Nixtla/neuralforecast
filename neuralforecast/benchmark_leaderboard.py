"""Shared Phase 2 ranking and independent W&B leaderboard publication."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from neuralforecast.benchmark import fold_rmse, forecast_metrics, summary_rank_key
from neuralforecast.benchmark_tracking import run_id, safe_config

TABLE_KEY = "phase2/leaderboard_with_naive"
TERMINAL_STATUSES = {"completed", "no_models_above_naive", "failed"}


@contextmanager
def publisher_lock(output):
    """Allow only one leaderboard writer for a local experiment."""
    with (Path(output) / "phase2_leaderboard.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def phase2_publication(output, options, config):
    """Own a publisher unless tracking is disabled or a watcher already owns it."""
    if not options:
        yield None
        return
    lock = publisher_lock(output)
    try:
        lock.__enter__()
    except BlockingIOError:
        # An existing watcher publishes these same saved results.
        yield None
        return
    publisher = Phase2Publisher(output, options, config)
    try:
        yield publisher
    finally:
        try:
            publisher.finish()
        finally:
            lock.__exit__(None, None, None)


class Phase2Publisher:
    """Publish complete ranking snapshots without changing the training run.

    Args:
        output: Experiment output directory; callers must hold publisher_lock.
        options: Existing experiment's W&B entity, project, and group.
        config: Benchmark configuration identifying the evaluation protocol.
    """

    def __init__(self, output, options, config):
        self.output = Path(output)
        self.options = options
        self.config = config
        self.run = None
        self.digest = None
        self.state_path = self.output / "phase2_leaderboard.json"
        self.identity = run_id(
            options["group"], "phase2-leaderboard", "benchmark", 0, -1
        )
        self.url = (
            f"https://wandb.ai/{options['entity']}/{options['project']}"
            f"/runs/{self.identity}"
        )

    def publish(self, frame, definitions, *, total_models, status):
        """Log a complete table only when its content or status changes."""
        import wandb

        frame = frame.copy()
        frame["evaluation_folds"] = self.config["phase2_folds"]
        leading = [
            "rank",
            "candidate",
            "rmse",
            "mae",
            "mape_pct",
            "mse",
            "da_pct",
            "rmse_vs_naive",
            "evaluation_folds",
        ]
        frame = frame[leading + [col for col in frame if col not in leading]]
        # W&B JSON requires null for undefined metrics, e.g. MAPE on all-zero targets.
        records = json.loads(frame.to_json(orient="records", double_precision=15))
        summary = {
            "phase2/completed_models": int(frame.candidate.ne("Naive").sum()),
            "phase2/total_models": total_models,
            "phase2/evaluation_folds": self.config["phase2_folds"],
            "phase2/leaderboard_with_naive_definitions": definitions,
            "phase2/leaderboard_with_naive_rows": len(frame),
            "status": status,
        }
        payload = json.dumps([records, summary], sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        if digest == self.digest:
            return False
        for attempt in range(4):
            try:
                if self.run is None:
                    directory = self.output / "wandb"
                    directory.mkdir(exist_ok=True)
                    self.run = wandb.init(
                        entity=self.options["entity"],
                        project=self.options["project"],
                        group=self.options["group"],
                        id=self.identity,
                        name="Phase 2 Leaderboard",
                        job_type="phase2-leaderboard",
                        resume="allow",
                        reinit="create_new",
                        mode=self.options.get("mode", "online"),
                        dir=str(directory),
                        config=safe_config(
                            {
                                key: self.config[key]
                                for key in (
                                    "target",
                                    "fingerprint",
                                    "protocol_version",
                                    "evaluation_split",
                                    "evaluation_scale",
                                    "phase2_first_fold",
                                    "phase2_last_fold",
                                    "phase2_folds",
                                )
                                if key in self.config
                            }
                        ),
                        save_code=False,
                        settings=wandb.Settings(init_timeout=120),
                    )
                if self.run.summary.get("phase2/leaderboard_digest") != digest:
                    self.run.log(
                        {
                            TABLE_KEY: wandb.Table(
                                columns=list(frame.columns),
                                data=[[row[col] for col in frame] for row in records],
                                log_mode="MUTABLE",
                            )
                        }
                    )
                    self.run.summary.update(
                        {
                            **summary,
                            "phase2/leaderboard_digest": digest,
                            "phase2/updated_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        self.digest = digest
        state = {
            "url": self.url,
            "run_id": self.identity,
            "digest": digest,
            **summary,
        }
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False))
        temporary.replace(self.state_path)
        return True

    def finish(self):
        """Flush only the leaderboard run, preserving the training run."""
        if self.run is not None:
            self.run.finish()
            self.run = None


def leaderboard(predictions, reference):
    """Compare complete model predictions and naive on exactly the same points."""
    naive = reference.assign(
        candidate="Naive", protocol="naive", prediction=reference.forecast_origin
    )
    frame = pd.concat([pd.DataFrame(predictions), naive], ignore_index=True)
    naive_rmse = fold_rmse(reference.actual, reference.forecast_origin)
    rows = []
    for (candidate, protocol), values in frame.groupby(["candidate", "protocol"]):
        if values.duplicated(["fold", "horizon"]).any():
            raise ValueError(f"Duplicate fold/horizon predictions for {candidate}")
        joined = values.drop(columns=["forecast_origin"], errors="ignore").merge(
            reference,
            on=["fold", "horizon"],
            suffixes=("", "_reference"),
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        if not joined["_merge"].eq("both").all():
            continue
        if not np.allclose(joined.actual, joined.actual_reference, rtol=0, atol=1e-10):
            raise ValueError(f"Actual targets differ from reference for {candidate}")
        metrics = forecast_metrics(
            joined.actual_reference, joined.prediction, joined.forecast_origin
        )
        fold_scores = [
            fold_rmse(fold["actual"], fold["prediction"])
            for _, fold in values.groupby("fold")
        ]
        score = fold_rmse(values["actual"], values["prediction"])
        rows.append(
            {
                "candidate": candidate,
                "protocol": protocol,
                "pooled_rmse": score,
                "fold_rmse_std": float(np.std(fold_scores)),
                "worst_fold_rmse": float(np.max(fold_scores)),
                **metrics,
                "naive_rmse": naive_rmse,
                "rmse_vs_naive": metrics["rmse"] / naive_rmse if naive_rmse else None,
                "beats_naive": metrics["rmse"] < naive_rmse,
                "evaluation_split": "validation",
            }
        )
    rows.sort(
        key=lambda row: summary_rank_key(
            row["pooled_rmse"], row["fold_rmse_std"], row["worst_fold_rmse"]
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def metric_definitions(reference):
    difference = reference.actual - reference.forecast_origin
    return {
        "evaluation_split": "validation",
        "evaluation_scale": "level",
        "aggregation": "Equal weight per fold/horizon point; RMSE ascending",
        "naive": "Last training observation repeated for every validation horizon",
        "mape": "Percent; zero actuals excluded, mape_n reports denominator",
        "da": "Percent matching rise/fall/flat signs relative to last training observation; origin fixed within each fold",
        "points_per_model": len(reference),
        "always_up_da_pct": float(100 * (difference > 0).mean()),
        "always_down_da_pct": float(100 * (difference < 0).mean()),
        "always_flat_da_pct": float(100 * (difference == 0).mean()),
    }


def validation_reference(frame, folds):
    rows = []
    for fold in folds:
        train = frame.iloc[fold.train_slice].copy()
        train["y"] = train.y.ffill()
        valid = frame.iloc[fold.valid_slice].copy()
        valid["y"] = valid.y.ffill()
        if train.y.isna().any() or valid.y.isna().any():
            raise ValueError("target begins with missing observations")
        rows.extend(
            {
                "fold": fold.index,
                "horizon": horizon,
                "actual": float(actual),
                "forecast_origin": float(train.y.iloc[-1]),
            }
            for horizon, actual in enumerate(valid.y, 1)
        )
    return pd.DataFrame(rows)
