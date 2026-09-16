"""Opt-in tracking for the commodity benchmark; imports never start a run."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re


_SECRET = re.compile(r"api.?key|password|secret|token", re.I)


def ensure_open_project(entity, project):
    """Set the benchmark project's W&B visibility to Open before launching runs."""
    import netrc

    import requests
    from urllib.parse import urlparse

    base_url = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai").rstrip("/")
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        credentials = netrc.netrc().authenticators(urlparse(base_url).hostname)
        key = credentials[2] if credentials else None
    if not key:
        raise ValueError("W&B credentials are required to set project visibility")
    response = requests.post(
        f"{base_url}/graphql",
        auth=("api", key),
        timeout=30,
        json={
            "query": """
                mutation OpenProject($entity: String!, $project: String!) {
                    upsertModel(input: {
                        entityName: $entity, name: $project, access: "USER_WRITE"
                    }) { project { name access } }
                }
            """,
            "variables": {"entity": entity, "project": project},
        },
    )
    response.raise_for_status()
    result = response.json()
    if result.get("errors"):
        raise RuntimeError(f"Could not set W&B project {entity}/{project} to Open")
    if result["data"]["upsertModel"]["project"]["access"] != "USER_WRITE":
        raise RuntimeError(f"W&B project {entity}/{project} is not Open")


def safe_config(value):
    """Serialize configuration without credentials or executable objects."""
    if isinstance(value, dict):
        return {
            str(k): safe_config(v)
            for k, v in value.items()
            if not _SECRET.search(str(k))
        }
    if isinstance(value, (list, tuple)):
        return [safe_config(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, str):
            key = os.environ.get("WANDB_API_KEY")
            if key:
                value = value.replace(key, "[REDACTED]")
        return value
    return repr(value)


def run_id(group, phase, candidate, config_id, fold):
    """Stable identity across SH rungs, distinct across phases and folds."""
    identity = json.dumps([group, phase, candidate, config_id, fold])
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


class Tracking:
    """One explicitly owned W&B run. No logging when options are absent."""

    def __init__(
        self,
        options=None,
        *,
        phase="summary",
        candidate="benchmark",
        config_id=0,
        fold=-1,
        config=None,
    ):
        self.run = None
        if not options:
            return
        import wandb

        directory = Path(options["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(
            entity=options["entity"],
            project=options["project"],
            group=options["group"],
            job_type=phase,
            id=run_id(options["group"], phase, candidate, config_id, fold),
            name=f"{phase}/{candidate}/config-{config_id}/fold-{fold}",
            resume="allow",
            mode=options.get("mode", "online"),
            dir=str(directory),
            config=safe_config(config or {}),
            save_code=False,
            settings=wandb.Settings(init_timeout=120),
        )
        self.run.define_metric("train/global_step")
        self.run.define_metric("train/*", step_metric="train/global_step")
        self.run.define_metric("validation/*", step_metric="train/global_step")

    def log(self, metrics):
        if self.run:
            self.run.log(safe_config(metrics))

    def summary(self, metrics):
        if self.run:
            self.run.summary.update(safe_config(metrics))

    def table(self, name, frame):
        if self.run and len(frame):
            import wandb

            # Stringify object/config cells; no source frames or model weights.
            frame = frame.copy()
            for col in frame:
                frame[col] = frame[col].map(safe_config)
            self.run.log({name: wandb.Table(dataframe=frame)})

    def forecast(self, actual, prediction, forecast_origin):
        """Publish one fold's point forecast as arrays and an ordered W&B table."""
        if not self.run:
            return
        import pandas as pd

        actual = [float(value) for value in actual]
        prediction = [float(value) for value in prediction]
        if len(actual) != len(prediction) or not actual:
            raise ValueError("Forecast tracking requires aligned nonempty arrays")
        origin = float(forecast_origin)
        self.summary(
            {
                "forecast/actual": actual,
                "forecast/prediction": prediction,
                "forecast/horizon": list(range(1, len(actual) + 1)),
                "forecast/origin": origin,
            }
        )
        self.table(
            "forecast/series",
            pd.DataFrame(
                {
                    "horizon": range(1, len(actual) + 1),
                    "actual": actual,
                    "prediction": prediction,
                    "error": [
                        predicted - observed
                        for observed, predicted in zip(actual, prediction)
                    ],
                    "forecast_origin": origin,
                }
            ),
        )

    def artifact(self, root):
        if self.run:
            import wandb

            artifact = wandb.Artifact(
                f"results-{self.run.group}", type="benchmark-results"
            )
            for name in (
                "eligibility.csv",
                "phase1_trials.csv",
                "phase1_ranking.csv",
                "phase1_leaderboard.csv",
                "phase2_predictions.csv",
                "leaderboard.csv",
                "metric_definitions.json",
                "failures.csv",
                "run_config.json",
                "data_manifest.json",
                "preparation.json",
            ):
                path = Path(root) / name
                if path.is_file():
                    artifact.add_file(str(path), name=name)
            self.run.log_artifact(artifact)

    def evaluation_table(self, phase, frame, definitions):
        """Publish a baseline-inclusive table and individually comparable metrics."""
        if not self.run:
            return
        self.table(f"{phase}/leaderboard_with_naive", frame)
        summary = {
            f"{phase}/leaderboard_with_naive_definitions": definitions,
            f"{phase}/leaderboard_with_naive_rows": len(frame),
        }
        for row in frame.to_dict(orient="records"):
            for metric in ("mae", "mape_pct", "mse", "rmse", "da_pct", "mape_n"):
                value = row.get(metric)
                if isinstance(value, float) and not math.isfinite(value):
                    value = None
                summary[f"{phase}_with_naive/{row['candidate']}/{metric}"] = value
        self.summary(summary)

    def finish(self, failed=False):
        if self.run:
            self.run.finish(exit_code=1 if failed else 0)
            self.run = None


def training_callback(tracker):
    """Attach explicitly to this runner's Lightning trainers only."""
    from pytorch_lightning.callbacks import Callback

    class Metrics(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            step = int(trainer.global_step)
            if step != 1 and step % 10:
                return
            metrics = {"train/global_step": step}
            for name, value in trainer.callback_metrics.items():
                if hasattr(value, "numel") and value.numel() == 1:
                    metrics[f"train/{name}"] = float(value.detach().cpu())
            if trainer.optimizers:
                metrics["train/learning_rate"] = trainer.optimizers[0].param_groups[0][
                    "lr"
                ]
            tracker.log(metrics)

    return Metrics()
