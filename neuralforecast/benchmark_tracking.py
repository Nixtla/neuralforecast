"""Opt-in tracking for the commodity benchmark; imports never start a run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re


_SECRET = re.compile(r"api.?key|password|secret|token", re.I)


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
            mode="online",
            dir=str(directory),
            config=safe_config(config or {}),
            save_code=False,
            settings=wandb.Settings(init_timeout=120),
        )
        self.run.define_metric("train/global_step")
        self.run.define_metric("train/*", step_metric="train/global_step")

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
                "phase2_predictions.csv",
                "leaderboard.csv",
                "failures.csv",
                "run_config.json",
                "data_manifest.json",
                "preparation.json",
            ):
                path = Path(root) / name
                if path.is_file():
                    artifact.add_file(str(path), name=name)
            self.run.log_artifact(artifact)

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
