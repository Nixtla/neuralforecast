"""Two-phase weekly commodity benchmark with pooled-RMSE model selection."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import hashlib
import shutil
import time
import uuid
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import ray
import torch

import neuralforecast.auto as auto_module
import neuralforecast.models as model_module
from neuralforecast.benchmark import (
    SAMPLING_POLICY,
    SHPlan,
    benchmark_search_space,
    expanding_folds,
    fold_rmse,
    forecast_metrics,
    minimum_history,
    pooled_rmse,
    rank_key,
    representative_folds,
    restrict_input_size,
    sample_ray_configs,
)
from neuralforecast.inference_tuning import (
    INFERENCE_TUNING_MODELS,
    get_inference_tuning_config,
)
from neuralforecast.benchmark_tracking import Tracking, training_callback, safe_config
from neuralforecast.benchmark_leaderboard import (
    leaderboard as _leaderboard,
    metric_definitions as _metric_definitions,
    phase2_publication,
    validation_reference as _validation_reference,
)
from neuralforecast.benchmark_numerics import (
    NUMERICS_VERSION,
    FiniteTraining,
    guard_config,
    numerical_search_space,
)
from neuralforecast.tsdataset import TimeSeriesDataModule, TimeSeriesDataset


LLM_MODELS = {"Aurora", "ChatTime", "GPT4MTS", "LangTime", "UniTime"}
# Temporarily disabled after benchmark review; remove an entry to re-enable it.
# Keep protocol-specific exclusions separate so TimesFM-LoRA remains available.
DISABLED_CANDIDATES = {
    "Moirai-ZeroShot",
    "Moirai2-ZeroShot",
    "MoiraiMoE-ZeroShot",
    "Chronos2-ZeroShot",
    "TimesFM-ZeroShot",
    "TimesFM3-ZeroShot",
    "Toto-ZeroShot",
    "NLinear",
    "Chronos2-LoRA",
}
SKIP_MODELS = {
    "HINT",
    "SearchCast",
    "VoT",
    "SpecTF",
    "TGForecaster",
    "APT",
    "GLAFF",
    "ChronosX",
    "BaguanTS",
    "RAG4CTS",
    "TabPFNTS",
}
SKIP_REASONS = {
    "HINT": "requires a hierarchical forecast reconciliation setup",
    "SearchCast": "internal HPO has no fixed-configuration Phase 2 path",
    **{
        name: "requires exogenous or text inputs; experiment is target-only"
        for name in SKIP_MODELS - {"HINT", "SearchCast"}
    },
}
TRACKING = None
SUMMARY = None
POLICY = {"max_steps": 500, "interval": 10, "patience": 5, "val_size": 16}
PROTOCOL_VERSION = "validation-only-sh3-naive-v1"
EXOG_PROTOCOL_VERSION = "validation-only-sh3-naive-hist-exog-v2"


@dataclass
class Candidate:
    name: str
    protocol: str
    model_name: str
    configs: list[dict]
    plan: SHPlan | None
    alive: list[int] = field(default_factory=list)
    rung: int = 0
    checkpoints: dict[tuple[int, int], str] = field(default_factory=dict)
    rung_results: dict[tuple[int, int], dict] = field(default_factory=dict)
    best_config: dict | None = None
    best_score: float = float("inf")
    best_fold_scores: tuple[float, ...] = ()
    best_metrics: dict = field(default_factory=dict)


def _lora_api():
    try:
        from neuralforecast.foundation_lora import (
            FOUNDATION_LORA_MODELS,
            fit_foundation_lora,
            get_foundation_lora_config,
        )
    except ImportError:
        return (), None, None
    return FOUNDATION_LORA_MODELS, fit_foundation_lora, get_foundation_lora_config


def _weekly_frame(
    path,
    date_col,
    target_col,
    start_date=None,
    end_date=None,
    include_exogenous=False,
):
    frame = pd.read_csv(path)
    if date_col not in frame or target_col not in frame:
        raise ValueError(f"CSV must contain {date_col!r} and {target_col!r}.")
    value_columns = [target_col]
    if include_exogenous:
        value_columns.extend(
            column for column in frame if column not in {date_col, target_col}
        )
    frame = frame[[date_col, *value_columns]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise", utc=True)
    frame[value_columns] = frame[value_columns].apply(pd.to_numeric, errors="coerce")
    if start_date:
        frame = frame[frame[date_col] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        frame = frame[frame[date_col] <= pd.Timestamp(end_date, tz="UTC")]
    frame = frame.set_index(date_col)[value_columns].resample("W-SUN").mean()
    frame = frame.rename(columns={target_col: "y"}).reset_index().rename(
        columns={date_col: "ds"}
    )
    valid = frame["y"].notna()
    if not valid.any():
        raise ValueError("target has no weekly observations.")
    first, last = np.flatnonzero(valid)[[0, -1]]
    frame = frame.iloc[first : last + 1].reset_index(drop=True)
    frame["unique_id"] = target_col
    return frame[["unique_id", "ds", "y", *value_columns[1:]]]


def _select_historical_exogenous(
    frame, training_rows, max_abs_correlation=0.8, max_missing_ratio=0.05
):
    """Select one fixed, leakage-safe historical exogenous feature set."""
    if not 0 < max_abs_correlation <= 1:
        raise ValueError("max_abs_correlation must be in (0, 1]")
    if not 0 < max_missing_ratio <= 1:
        raise ValueError("max_missing_ratio must be in (0, 1]")
    if not 2 <= training_rows <= len(frame):
        raise ValueError("training_rows must identify at least two rows in the frame")

    report = []
    selected = []
    candidates = [c for c in frame if c not in {"unique_id", "ds", "y"}]
    for column in candidates:
        missing_ratio = float(frame[column].isna().mean())
        pair = frame.iloc[:training_rows][["y", column]].dropna()
        correlation = (
            float(pair[column].corr(pair["y"]))
            if len(pair) >= 2
            and pair[column].nunique() > 1
            and pair["y"].nunique() > 1
            else float("nan")
        )
        reason = "selected"
        keep = True
        if missing_ratio >= max_missing_ratio:
            keep, reason = False, "missing_ratio_at_or_above_threshold"
        elif not np.isfinite(correlation):
            keep, reason = False, "correlation_unavailable"
        elif pd.isna(frame[column].iloc[0]):
            keep, reason = False, "missing_at_period_start"
        elif abs(correlation) >= max_abs_correlation:
            keep, reason = False, "absolute_correlation_at_or_above_threshold"
        if keep:
            selected.append(column)
        report.append(
            {
                "feature": column,
                "missing_ratio": missing_ratio,
                "correlation": correlation,
                "absolute_correlation": abs(correlation),
                "selection_rows": training_rows,
                "selected": keep,
                "reason": reason,
            }
        )
    if not selected:
        raise ValueError("No historical exogenous feature passed the selection policy")
    output = frame[["unique_id", "ds", "y", *selected]].copy()
    output[selected] = output[selected].ffill()
    if output[selected].isna().any().any():
        raise ValueError("Selected historical exogenous features remain missing")
    return output, pd.DataFrame(report), selected


def _fill(values):
    """Forward-fill observations using only values available in the same slice."""
    values = values.copy()
    values["y"] = values["y"].ffill()
    if values["y"].isna().any():
        raise ValueError(
            "target begins with missing observations; cannot forward-fill."
        )
    return values


def _transform_metadata(project):
    return {
        "target_transform": (
            "first_difference" if project.endswith("-diff") else "identity"
        ),
        "training_scale": "difference" if project.endswith("-diff") else "level",
        "evaluation_scale": "level",
        "transform_version": 1,
    }


def _difference_fold(train, valid):
    """Difference selected series after filling; keep the forecast origin fixed."""
    columns = [column for column in train if column not in {"ds", "unique_id"}]
    train_diff, valid_diff = train.copy(), valid.copy()
    train_diff[columns] = train[columns].diff()
    train_diff = train_diff.iloc[1:].reset_index(drop=True)
    boundary = pd.concat([train.iloc[-1:][columns], valid[columns]], ignore_index=True)
    valid_diff[columns] = boundary.diff().iloc[1:].to_numpy()
    if (
        train_diff.empty
        or not np.isfinite(
            pd.concat([train_diff[columns], valid_diff[columns]]).to_numpy(dtype=float)
        ).all()
    ):
        raise ValueError(
            "Differencing requires finite observations and two training rows"
        )
    return train_diff, valid_diff


def _restore_prediction(prediction, origin, transform):
    prediction = np.asarray(prediction, dtype=float)
    if transform == "first_difference":
        return float(origin) + np.cumsum(prediction)
    return prediction


def _dataset(frame):
    dataset, *_ = TimeSeriesDataset.from_df(
        frame,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
    )
    return dataset


def _point_prediction(model, dataset, h):
    model.set_test_size(h)
    values = model.predict(dataset, test_size=h, step_size=1)
    output_size = len(model.loss.output_names)
    values = np.asarray(values).reshape(-1, output_size)
    if len(values) != h:
        raise ValueError(f"expected {h} predictions, received {len(values)}.")
    return values[:, 0]


def _fit_trainable(
    model_cls,
    config,
    train,
    valid,
    budget,
    checkpoint,
    workdir,
    tracker=None,
    stopping=None,
    schedule_steps=None,
    keep_checkpoint=True,
):
    config = dict(config)
    numerics = config.pop("_benchmark_numerics", None)
    if numerics and numerics["version"] != NUMERICS_VERSION:
        raise ValueError("Unsupported benchmark numerical policy version")
    if model_cls.__name__ == "xLSTM":
        # Already queued configurations retain their original backend behavior.
        config.setdefault("numerical_stability", False)
    if numerics:
        config.setdefault("gradient_clip_val", numerics["gradient_clip_val"])
    config["max_steps"] = schedule_steps or budget
    if stopping:
        from neuralforecast.losses.pytorch import MSE

        config["valid_loss"] = MSE()
        config["val_check_steps"] = stopping["interval"]
    config["random_seed"] = 42
    config["early_stop_patience_steps"] = -1
    config["enable_checkpointing"] = False
    config["logger"] = False
    config["accelerator"] = "gpu"
    config["devices"] = 1
    config["enable_progress_bar"] = False
    model = model_cls(**config)
    train_dataset = _dataset(
        pd.concat([train, valid], ignore_index=True) if stopping else train
    )
    model._check_exog(train_dataset)
    model._restart_seed(42)
    model.val_size = stopping["val_size"] if stopping else 0
    model.test_size = 0
    datamodule = TimeSeriesDataModule(
        dataset=train_dataset,
        batch_size=model.batch_size,
        valid_batch_size=model.valid_batch_size,
        drop_last=model.drop_last_loader,
        shuffle_train=True,
        **(model.dataloader_kwargs or {}),
    )
    trainer_kwargs = dict(model.trainer_kwargs)
    callbacks = list(trainer_kwargs.get("callbacks") or [])
    callbacks.append(FiniteTraining(check_forward=model_cls.__name__ == "FEDformer"))
    stopper = None
    if stopping:
        from neuralforecast.benchmark_stopping import (
            ValidationStopper,
            rng_state,
            restore_rng,
        )

        stopper = ValidationStopper(
            workdir,
            interval=stopping["interval"],
            patience=stopping["patience"],
            metrics_callback=tracker.log if tracker and tracker.run else None,
        )

        class StopCallback(pl.Callback):
            saved_rng = None

            def state_dict(self):
                return {"stopper": stopper.state_dict(), "rng": rng_state()}

            def load_state_dict(self, state):
                stopper.load_state_dict(state["stopper"])
                self.saved_rng = state["rng"]

            def on_train_start(self, trainer, pl_module):
                if self.saved_rng is not None:
                    restore_rng(self.saved_rng)

            def on_validation_end(self, trainer, pl_module):
                if trainer.sanity_checking:
                    return
                loss = float(trainer.callback_metrics["ptl/val_loss"])
                if stopper.update(pl_module, trainer.global_step, loss):
                    trainer.should_stop = True

        callbacks.append(StopCallback())
        trainer_kwargs.update(
            val_check_interval=stopping["interval"],
            check_val_every_n_epoch=None,
            num_sanity_val_steps=0,
        )
    if tracker and tracker.run:
        callbacks.append(training_callback(tracker))
    trainer_kwargs["callbacks"] = callbacks
    trainer_kwargs["max_steps"] = budget
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint)
    next_checkpoint = None
    if keep_checkpoint:
        next_checkpoint = str(Path(workdir) / "resume.ckpt")
        trainer.save_checkpoint(next_checkpoint)
    if stopper:
        stopper.restore(model)
        stopper.path.unlink(missing_ok=True)
        (Path(workdir) / "early_stopping.json").write_text(
            json.dumps(stopper.summary())
        )
    model.metrics = trainer.callback_metrics
    model.val_size = 0
    model.__dict__.pop("_trainer", None)
    prediction = _point_prediction(
        model, _dataset(pd.concat([train, valid], ignore_index=True)), len(valid)
    )
    return prediction, next_checkpoint


def _fit_inference(model_cls, config, train, valid):
    model = model_cls(**config)
    dataset = _dataset(pd.concat([train, valid], ignore_index=True))
    model.fit(dataset, val_size=0, test_size=len(valid), random_seed=42)
    return _point_prediction(model, dataset, len(valid)), None


def _clear_training_artifacts(directory):
    """Remove model artifacts while leaving the job directory reusable."""
    directory = Path(directory)
    if not directory.is_dir():
        return
    for path in directory.iterdir():
        shutil.rmtree(path) if path.is_dir() else path.unlink()


def _discard_checkpoint(checkpoint):
    """Discard an obsolete rung checkpoint and its sibling training artifacts."""
    if not checkpoint:
        return
    path = Path(checkpoint)
    workdir = path.parent
    if not workdir.is_dir():
        return
    for child in workdir.iterdir():
        if child.suffix == ".json" or child.suffix == ".log":
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()


def _fit_lora(
    model_cls,
    config,
    train,
    valid,
    budget,
    workdir,
    tracker=None,
    stopping=None,
    checkpoint=None,
    schedule_steps=None,
    keep_checkpoint=True,
):
    _, fit_foundation_lora, _ = _lora_api()
    if fit_foundation_lora is None:
        raise RuntimeError("foundation LoRA module is unavailable.")
    options = {}
    if stopping:
        options = {
            "validation": valid.copy(),
            "stopping": {
                "interval": stopping["interval"],
                "patience": stopping["patience"],
            },
        }
    model = fit_foundation_lora(
        model_cls,
        config,
        train,
        h=len(valid),
        steps=budget,
        output_dir=workdir,
        metrics_callback=tracker.log if tracker and tracker.run else None,
        checkpoint=checkpoint,
        schedule_steps=schedule_steps,
        **options,
    )
    if stopping:
        (Path(workdir) / "early_stopping.json").write_text(
            json.dumps(model.early_stopping_info)
        )
    model.val_size = 0
    model.test_size = len(valid)
    dataset = _dataset(pd.concat([train, valid], ignore_index=True))
    prediction = _point_prediction(model, dataset, len(valid))
    next_checkpoint = model.resume_checkpoint if keep_checkpoint else None
    if not keep_checkpoint:
        for path in Path(workdir).iterdir():
            if path.name == "early_stopping.json":
                continue
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    return prediction, next_checkpoint


@ray.remote(num_gpus=1, max_calls=1)
def _evaluate_job(
    data_path, candidate, config_id, config, fold, budget, checkpoint, root
):
    started = time.monotonic()
    tracker = None
    result = None
    protocol_version = candidate.get("protocol_version", PROTOCOL_VERSION)
    try:
        # Backend auto-integrations must not create additional runs.
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["WANDB_DISABLE_CODE"] = "true"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_num_threads(2)
        frame = pd.read_pickle(data_path)
        phase = Path(root).name.removeprefix("checkpoints_")
        stopping = candidate.get("phase2_policy", POLICY)
        if phase == "smoke":
            stopping = dict(stopping, interval=1)
        history = frame.iloc[fold.train_slice]
        train = _fill(history)
        valid = _fill(frame.iloc[fold.valid_slice])
        actual = valid.y.to_numpy(dtype=float)
        forecast_origin = float(train.y.iloc[-1])
        transform_metadata = frame.attrs.get(
            "transform_metadata", _transform_metadata("")
        )
        transform = transform_metadata["target_transform"]
        if transform == "first_difference":
            train, valid = _difference_fold(train, valid)
        workdir = Path(root) / candidate["name"] / str(config_id) / str(fold.index)
        workdir.mkdir(parents=True, exist_ok=True)
        tracker = Tracking(
            candidate.get("tracking"),
            phase=phase,
            candidate=candidate["name"],
            config_id=config_id,
            fold=fold.index,
            config={
                **transform_metadata,
                "model": candidate["model_name"],
                "protocol": candidate["protocol"],
                "parameters": config,
                "fold": fold.index,
                "train_end": str(train.ds.iloc[-1]),
                "validation_start": str(valid.ds.iloc[0]),
                "validation_end": str(valid.ds.iloc[-1]),
                "protocol_version": protocol_version,
                "evaluation_split": "validation",
                "phase2_policy": stopping,
                "attempt": candidate.get("attempt", 0),
                "valid_start": str(valid.ds.iloc[0]),
                "valid_end": str(valid.ds.iloc[-1]),
            },
        )
        model_cls = getattr(model_module, candidate["model_name"])
        if candidate["protocol"] == "scratch_hpo":
            prediction, next_checkpoint = _fit_trainable(
                model_cls,
                config,
                train,
                valid,
                budget,
                checkpoint,
                workdir,
                tracker,
                stopping,
                schedule_steps=(
                    SHPlan().budgets[-1] if phase == "phase1" else stopping["max_steps"]
                ),
                keep_checkpoint=phase == "phase1",
            )
        elif candidate["protocol"] == "lora":
            prediction, next_checkpoint = _fit_lora(
                model_cls,
                config,
                train,
                valid,
                budget,
                workdir,
                tracker,
                stopping,
                checkpoint=checkpoint,
                schedule_steps=(
                    SHPlan().budgets[-1] if phase == "phase1" else stopping["max_steps"]
                ),
                keep_checkpoint=phase == "phase1",
            )
        else:
            prediction, next_checkpoint = _fit_inference(
                model_cls, config, train, valid
            )
        prediction = _restore_prediction(prediction, forecast_origin, transform)
        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite predictions")
        result = {
            "ok": True,
            "candidate": candidate["name"],
            "config_id": config_id,
            "fold": fold.index,
            "actual": actual.tolist(),
            "prediction": np.asarray(prediction, dtype=float).tolist(),
            "rmse": fold_rmse(actual, prediction),
            "checkpoint": next_checkpoint,
            "budget": budget,
            "evaluation_split": "validation",
            "protocol_version": protocol_version,
            "forecast_origin": forecast_origin,
            **transform_metadata,
            **forecast_metrics(
                actual, prediction, np.full(len(valid), forecast_origin)
            ),
        }
        early_path = workdir / "early_stopping.json"
        if early_path.exists():
            result.update(json.loads(early_path.read_text()))
        result["attempt"] = candidate.get("attempt", 0)
    except Exception as exc:
        kind = (
            "OOM"
            if isinstance(exc, torch.cuda.OutOfMemoryError)
            or "out of memory" in str(exc).lower()
            else "FAILURE"
        )
        result = {
            "ok": False,
            "candidate": candidate["name"],
            "config_id": config_id,
            "fold": fold.index,
            "budget": budget,
            "kind": kind,
            "error": safe_config(f"{type(exc).__name__}: {exc}"),
        }
    finally:
        if result is not None:
            result["seconds"] = time.monotonic() - started
        if tracker:
            try:
                if result.get("ok"):
                    tracker.forecast(
                        result["actual"],
                        result["prediction"],
                        result["forecast_origin"],
                    )
                    result["forecast_tracking_version"] = 1
                tracker.log(
                    {
                        k: v
                        for k, v in result.items()
                        if k not in {"actual", "prediction", "checkpoint"}
                    }
                )
                tracker.summary(
                    {
                        "status": "success" if result["ok"] else "failed",
                        "budget": budget,
                    }
                )
                tracker.finish(failed=not result["ok"])
            except Exception as exc:
                result["tracking_error"] = type(exc).__name__
        if (
            "workdir" in locals()
            and result is not None
            and (phase != "phase1" or not result.get("ok"))
        ):
            _clear_training_artifacts(workdir)
        (
            (workdir / f"result-{budget}.json").write_text(
                json.dumps(result, default=str)
            )
            if "workdir" in locals()
            else None
        )
    return result


def _fixed_kwargs(config, name, protocol=None):
    fixed = dict(config.get(name, config.get(f"Auto{name}", {})))
    if protocol:
        fixed.update(config.get(f"{name}-{protocol}", {}))
    return fixed


def _skip_reason(name, hist_exog_list, protocol=None):
    candidate = f"{name}-{protocol}" if protocol else name
    if candidate in DISABLED_CANDIDATES:
        return "temporarily disabled after benchmark review"
    if name in LLM_MODELS:
        return "LLM protocol excluded by runner"
    if name in SKIP_MODELS and not (
        hist_exog_list and name in {"KITE", "ChronosX"}
    ):
        return SKIP_REASONS.get(name, "excluded by runner")
    return None


def _historical_exog_reason(model_cls, hist_exog_list):
    if hist_exog_list and not getattr(model_cls, "EXOGENOUS_HIST", False):
        return "does not support historical exogenous inputs"
    return None


def _with_historical_exog(configs, hist_exog_list):
    if not hist_exog_list:
        return configs
    for config in configs:
        config["hist_exog_list"] = list(hist_exog_list)
        config["futr_exog_list"] = None
    return configs


def _build_candidates(h, model_config, first_train, hist_exog_list=None):
    candidates, eligibility = [], []
    for auto_name in getattr(auto_module, "__all__", []):
        if not auto_name.startswith("Auto"):
            continue
        name = auto_name[4:]
        reason = _skip_reason(name, hist_exog_list)
        if reason:
            eligibility.append(
                (name, "scratch_hpo", "SKIP", reason)
            )
            continue
        auto_cls = getattr(auto_module, auto_name, None)
        if auto_cls is None:
            continue
        kwargs = _fixed_kwargs(model_config, name)
        if "n_series" in inspect.signature(auto_cls.__init__).parameters:
            kwargs.setdefault("n_series", 1)
        try:
            auto = auto_cls(h=h, backend="ray", num_samples=10, **kwargs)
            reason = _historical_exog_reason(auto.cls_model, hist_exog_list)
            if reason:
                eligibility.append((name, "scratch_hpo", "SKIP", reason))
                continue
            if getattr(auto.cls_model, "MULTIVARIATE", False):
                eligibility.append(
                    (
                        name,
                        "scratch_hpo",
                        "SKIP",
                        "multivariate without exogenous inputs",
                    )
                )
                continue
            space = restrict_input_size(
                numerical_search_space(name, benchmark_search_space(auto.config)),
                first_train - h,
            )
            configs = [
                guard_config(name, c) for c in sample_ray_configs(space, n=10, seed=42)
            ]
            configs = _with_historical_exog(configs, hist_exog_list)
        except Exception as exc:
            eligibility.append(
                (name, "scratch_hpo", "SKIP", f"{type(exc).__name__}: {exc}")
            )
            continue
        candidates.append(
            Candidate(name, "scratch_hpo", auto.cls_model.__name__, configs, SHPlan())
        )
        eligibility.append((name, "scratch_hpo", "READY", ""))

    for name in INFERENCE_TUNING_MODELS:
        reason = _skip_reason(name, hist_exog_list, "ZeroShot")
        if reason:
            eligibility.append(
                (name, "zero_shot", "SKIP", reason)
            )
            continue
        try:
            model_cls = getattr(model_module, name)
            reason = _historical_exog_reason(model_cls, hist_exog_list)
            if reason:
                eligibility.append((name, "zero_shot", "SKIP", reason))
                continue
            space = get_inference_tuning_config(
                name,
                h=h,
                fixed=_fixed_kwargs(model_config, name),
                backend="ray",
            )
            space = restrict_input_size(space, first_train)
            configs = sample_ray_configs(space, n=10, seed=42)
            for config in configs:
                config["h"] = h
            configs = _with_historical_exog(configs, hist_exog_list)
        except Exception as exc:
            eligibility.append(
                (name, "zero_shot", "SKIP", f"{type(exc).__name__}: {exc}")
            )
            continue
        candidates.append(
            Candidate(f"{name}-ZeroShot", "zero_shot", name, configs, None)
        )
        eligibility.append((name, "zero_shot", "READY", ""))

    lora_models, _, get_lora_config = _lora_api()
    for name in lora_models:
        if f"{name}-LoRA" in DISABLED_CANDIDATES:
            eligibility.append(
                (name, "lora", "SKIP", "temporarily disabled after benchmark review")
            )
            continue
        try:
            model_cls = getattr(model_module, name)
            reason = _historical_exog_reason(model_cls, hist_exog_list)
            if reason:
                eligibility.append((name, "lora", "SKIP", reason))
                continue
            space = get_lora_config(
                name,
                h=h,
                fixed=_fixed_kwargs(model_config, name, "LoRA"),
                backend="ray",
            )
            space = restrict_input_size(space, first_train - h)
            configs = sample_ray_configs(space, n=10, seed=42)
            for config in configs:
                config["h"] = h
            configs = _with_historical_exog(configs, hist_exog_list)
        except Exception as exc:
            eligibility.append((name, "lora", "SKIP", f"{type(exc).__name__}: {exc}"))
            continue
        candidates.append(Candidate(f"{name}-LoRA", "lora", name, configs, SHPlan()))
        eligibility.append((name, "lora", "READY", "three validation-stopped rungs"))
    return candidates, eligibility


def _minimum_train(h, model_config):
    minima = [h]
    for auto_name in getattr(auto_module, "__all__", []):
        if not auto_name.startswith("Auto"):
            continue
        name = auto_name[4:]
        if (
            name in LLM_MODELS
            or name in SKIP_MODELS
            or name in DISABLED_CANDIDATES
        ):
            continue
        auto_cls = getattr(auto_module, auto_name, None)
        if auto_cls is None:
            continue
        kwargs = _fixed_kwargs(model_config, name)
        if "n_series" in inspect.signature(auto_cls.__init__).parameters:
            kwargs.setdefault("n_series", 1)
        try:
            auto = auto_cls(h=h, backend="ray", num_samples=10, **kwargs)
            if not getattr(auto.cls_model, "MULTIVARIATE", False):
                minima.append(
                    minimum_history(benchmark_search_space(auto.config), h) + h
                )
        except Exception:
            pass
    for name in INFERENCE_TUNING_MODELS:
        if (
            name in LLM_MODELS
            or name in SKIP_MODELS
            or f"{name}-ZeroShot" in DISABLED_CANDIDATES
        ):
            continue
        try:
            space = get_inference_tuning_config(
                name,
                h=h,
                fixed=_fixed_kwargs(model_config, name),
                backend="ray",
            )
            minima.append(minimum_history(space, h))
        except Exception:
            pass
    return max(minima)


def _payload(candidate):
    return {
        "name": candidate.name,
        "protocol": candidate.protocol,
        "model_name": candidate.model_name,
        "protocol_version": PROTOCOL_VERSION,
        "tracking": TRACKING,
        "phase2_policy": POLICY,
    }


def _compound_score(candidate, folds):
    for config_id in candidate.alive:
        rows = [candidate.rung_results[(config_id, fold.index)] for fold in folds]
        if not all(row["ok"] for row in rows):
            yield config_id, float("inf"), (float("inf"),) * len(folds)
            continue
        actual = [row["actual"] for row in rows]
        prediction = [row["prediction"] for row in rows]
        fold_scores = tuple(row["rmse"] for row in rows)
        yield config_id, pooled_rmse(actual, prediction), fold_scores


def _trial_metrics(rows):
    if not all(row["ok"] for row in rows):
        return {}
    return forecast_metrics(
        np.concatenate([row["actual"] for row in rows]),
        np.concatenate([row["prediction"] for row in rows]),
        np.concatenate(
            [np.full(len(row["actual"]), row["forecast_origin"]) for row in rows]
        ),
    )


def _run_phase1(data_path, candidates, folds, checkpoint_root, resume=False):
    pending, trials, failures = {}, [], []
    terminal = {}
    restored = 0

    def submit(candidate, config_id, fold, budget):
        nonlocal restored
        cached = terminal.get((candidate.name, config_id, fold.index))
        if cached is not None:
            candidate.rung_results[(config_id, fold.index)] = cached
            return
        if resume:
            path = (
                Path(checkpoint_root) / candidate.name / str(config_id)
                / str(fold.index) / f"result-{budget}.json"
            )
            cached = _completed_fold_result(
                path, candidate, fold, budget,
                require_tracking=TRACKING is not None, config_id=config_id,
            )
            if cached is not None:
                candidate.rung_results[(config_id, fold.index)] = cached
                restored += 1
                if cached.get("stop_reason") == "early_stopping":
                    terminal[(candidate.name, config_id, fold.index)] = cached
                checkpoint = cached.get("checkpoint")
                if checkpoint and Path(checkpoint).is_file():
                    candidate.checkpoints[(config_id, fold.index)] = checkpoint
                return
        checkpoint = candidate.checkpoints.get((config_id, fold.index))
        ref = _evaluate_job.remote(
            data_path,
            _payload(candidate),
            config_id,
            candidate.configs[config_id],
            fold,
            budget,
            checkpoint,
            checkpoint_root,
        )
        pending[ref] = (candidate, config_id, fold, budget, checkpoint)

    def advance(candidate):
        while len(candidate.rung_results) == len(candidate.alive) * len(folds):
            budget = candidate.plan.budgets[candidate.rung] if candidate.plan else 0
            scored = list(_compound_score(candidate, folds))
            for cid, score, fold_scores in scored:
                trials.append(
                    {
                        "candidate": candidate.name,
                        "protocol": candidate.protocol,
                        "rung": candidate.rung,
                        "budget": budget,
                        "config_id": cid,
                        "pooled_rmse": score,
                        "fold_rmse_std": float(np.std(fold_scores)),
                        "worst_fold_rmse": float(np.max(fold_scores)),
                        **_trial_metrics(
                            [candidate.rung_results[(cid, f.index)] for f in folds]
                        ),
                    }
                )
            pd.DataFrame(trials).to_csv(
                Path(checkpoint_root).parent / "phase1_trials.csv", index=False
            )
            pd.DataFrame(failures).to_csv(
                Path(checkpoint_root).parent / "phase1_failures.csv", index=False
            )
            scored.sort(key=lambda row: rank_key(row[1], row[2]))
            if (
                candidate.plan is None
                or candidate.rung == len(candidate.plan.budgets) - 1
            ):
                best = scored[0]
                candidate.best_config = candidate.configs[best[0]]
                candidate.best_score = best[1]
                candidate.best_fold_scores = best[2]
                candidate.best_metrics = _trial_metrics(
                    [candidate.rung_results[(best[0], f.index)] for f in folds]
                )
                return
            keep = candidate.plan.survivors[candidate.rung + 1]
            candidate.alive = [row[0] for row in scored[:keep]]
            candidate.rung += 1
            candidate.rung_results = {}
            for cid in candidate.alive:
                for fold in folds:
                    submit(candidate, cid, fold, candidate.plan.budgets[candidate.rung])

    for candidate in candidates:
        candidate.alive = list(range(len(candidate.configs)))
        if candidate.plan:
            budget = candidate.plan.budgets[0]
        else:
            budget = 0
        for config_id in candidate.alive:
            for fold in sorted(folds, key=lambda f: f.train_end, reverse=True):
                submit(candidate, config_id, fold, budget)
        advance(candidate)

    if resume:
        print(f"phase1 resume restored={restored} pending={len(pending)}", flush=True)
        if SUMMARY:
            SUMMARY.summary({"phase1/restored_jobs": restored, "status": "phase1_resuming"})

    while pending:
        done, _ = ray.wait(list(pending), num_returns=1)
        ref = done[0]
        candidate, config_id, fold, budget, prior_checkpoint = pending.pop(ref)
        try:
            result = ray.get(ref)
        except Exception as exc:
            result = {
                "ok": False,
                "candidate": candidate.name,
                "fold": fold.index,
                "kind": "WORKER_FAILURE",
                "error": safe_config(str(exc)),
            }
        print(
            f"completed {candidate.name} fold={fold.index} ok={result['ok']} "
            f"rmse={result.get('rmse')} error={result.get('error', '')}",
            flush=True,
        )
        if SUMMARY:
            SUMMARY.log(
                {
                    "progress/completed_job": 1,
                    "progress/last_model": candidate.name,
                    "progress/last_ok": result["ok"],
                    "progress/last_rmse": result.get("rmse"),
                }
            )
        candidate.rung_results[(config_id, fold.index)] = result
        if not result["ok"] or result.get("stop_reason") == "early_stopping":
            terminal[(candidate.name, config_id, fold.index)] = result
        if result["ok"] and result.get("checkpoint"):
            candidate.checkpoints[(config_id, fold.index)] = result["checkpoint"]
            if prior_checkpoint != result["checkpoint"]:
                _discard_checkpoint(prior_checkpoint)
        elif not result["ok"]:
            failures.append(result)

        advance(candidate)
    for candidate in candidates:
        for checkpoint in candidate.checkpoints.values():
            _discard_checkpoint(checkpoint)
        candidate.checkpoints.clear()
    return trials, failures


def _naive_score(frame, folds):
    actual, predictions = [], []
    for fold in folds:
        train = _fill(frame.iloc[fold.train_slice])
        valid = _fill(frame.iloc[fold.valid_slice])
        actual.append(valid.y.to_numpy(dtype=float))
        predictions.append(np.full(len(valid), float(train.y.iloc[-1])))
    return pooled_rmse(actual, predictions)


def _select_candidates(ranking, naive_rmse):
    selected = [
        c for c in ranking if np.isfinite(c.best_score) and c.best_score < naive_rmse
    ][:10]
    names = {c.name for c in selected}
    rows = []
    for c in ranking:
        passed = bool(np.isfinite(c.best_score) and c.best_score < naive_rmse)
        rows.append(
            {
                "candidate": c.name,
                "protocol": c.protocol,
                "pooled_rmse": c.best_score,
                "fold_rmse_std": float(np.std(c.best_fold_scores)),
                "worst_fold_rmse": float(np.max(c.best_fold_scores)),
                "config": json.dumps(safe_config(c.best_config), sort_keys=True),
                "naive_rmse": naive_rmse,
                "beats_naive": passed,
                "selected": c.name in names,
                "exclusion_reason": (
                    ""
                    if c.name in names
                    else (
                        "failed"
                        if not np.isfinite(c.best_score)
                        else "not_better_than_naive" if not passed else "outside_top_10"
                    )
                ),
                "evaluation_split": "validation",
                **c.best_metrics,
            }
        )
    return selected, pd.DataFrame(rows)


def _completed_fold_result(
    path, candidate, fold, budget, require_tracking=False, config_id=0,
):
    """Load a complete compatible fold result, or return None for recomputation."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text())
        valid = (
            result.get("ok") is True
            and result.get("candidate") == candidate.name
            and int(result.get("config_id")) == config_id
            and int(result.get("fold")) == fold.index
            and int(result.get("budget")) == budget
            and result.get("protocol_version") == PROTOCOL_VERSION
            and (not require_tracking or result.get("forecast_tracking_version") == 1)
            and len(result.get("actual", [])) == fold.valid_end - fold.train_end
            and len(result.get("prediction", [])) == len(result["actual"])
            and np.isfinite(result["actual"]).all()
            and np.isfinite(result["prediction"]).all()
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return None
    return result if valid else None


def _completed_phase2_result(path, candidate, fold, budget, require_tracking=False):
    """Load a complete Phase 2 result for the selected configuration."""
    return _completed_fold_result(
        path, candidate, fold, budget, require_tracking=require_tracking,
    )


def _phase2(
    data_path,
    selected,
    folds,
    checkpoint_root,
    resume=False,
    require_tracking=True,
    on_model_complete=None,
):
    pending, predictions, failures = {}, [], []
    completed_folds = {candidate.name: set() for candidate in selected}

    def record(candidate, fold, result, notify=True):
        for horizon, (actual, prediction) in enumerate(
            zip(result["actual"], result["prediction"]), start=1
        ):
            predictions.append(
                {
                    "candidate": candidate.name,
                    "protocol": candidate.protocol,
                    "fold": fold.index,
                    "horizon": horizon,
                    "actual": actual,
                    "prediction": prediction,
                }
            )

        completed_folds[candidate.name].add(fold.index)
        if (
            notify
            and on_model_complete
            and len(completed_folds[candidate.name]) == len(folds)
        ):
            on_model_complete(predictions)

    for candidate in selected:
        budget = (
            POLICY["max_steps"] if candidate.protocol in {"scratch_hpo", "lora"} else 0
        )
        for fold in sorted(folds, key=lambda f: f.train_end, reverse=True):
            result_path = (
                Path(checkpoint_root)
                / candidate.name
                / "0"
                / str(fold.index)
                / f"result-{budget}.json"
            )
            completed = (
                _completed_phase2_result(
                    result_path,
                    candidate,
                    fold,
                    budget,
                    require_tracking=TRACKING is not None and require_tracking,
                )
                if resume
                else None
            )
            if completed is not None:
                record(candidate, fold, completed, notify=False)
                continue
            ref = _evaluate_job.remote(
                data_path,
                _payload(candidate),
                0,
                candidate.best_config,
                fold,
                budget,
                None,
                checkpoint_root,
            )
            pending[ref] = (candidate, fold)
    if resume and on_model_complete:
        on_model_complete(predictions)
    while pending:
        done, _ = ray.wait(list(pending), num_returns=1)
        ref = done[0]
        candidate, fold = pending.pop(ref)
        try:
            result = ray.get(ref)
        except Exception as exc:
            result = {
                "ok": False,
                "candidate": candidate.name,
                "fold": fold.index,
                "kind": "WORKER_FAILURE",
                "error": safe_config(str(exc)),
            }
        print(
            f"completed {candidate.name} fold={fold.index} ok={result['ok']} "
            f"rmse={result.get('rmse')} error={result.get('error', '')}",
            flush=True,
        )
        if SUMMARY:
            SUMMARY.log(
                {
                    "progress/completed_job": 1,
                    "progress/last_model": candidate.name,
                    "progress/last_ok": result["ok"],
                    "progress/last_rmse": result.get("rmse"),
                }
            )
        if not result["ok"]:
            failures.append(result)
            continue
        record(candidate, fold, result)
        pd.DataFrame(predictions).to_csv(
            Path(checkpoint_root).parent / "phase2_predictions.csv", index=False
        )
        pd.DataFrame(failures).to_csv(
            Path(checkpoint_root).parent / "phase2_failures.csv", index=False
        )
    return predictions, failures


def _restore_phase1(output, candidates):
    """Restore selected candidates from a completed compatible Phase 1."""
    output = Path(output)
    trials = pd.read_csv(output / "phase1_trials.csv")
    ranking = pd.read_csv(output / "phase1_ranking.csv")
    failures_path = output / "phase1_failures.csv"
    failures = (
        pd.read_csv(failures_path).to_dict(orient="records")
        if failures_path.exists() and failures_path.stat().st_size > 1
        else []
    )
    by_name = {candidate.name: candidate for candidate in candidates}
    selected = []
    chosen = ranking["selected"].astype(str).str.lower().eq("true")
    for row in ranking.loc[chosen].to_dict(orient="records"):
        candidate = by_name.get(row["candidate"])
        if candidate is None:
            raise ValueError(f"Resume candidate is unavailable: {row['candidate']}")
        rows = trials.loc[trials["candidate"] == candidate.name]
        rows = rows.loc[rows["rung"] == rows["rung"].max()].sort_values("pooled_rmse")
        if rows.empty:
            raise ValueError(f"Resume Phase 1 result is missing: {candidate.name}")
        config_id = int(rows.iloc[0]["config_id"])
        candidate.best_config = candidate.configs[config_id]
        expected = json.loads(row["config"])
        if safe_config(candidate.best_config) != expected:
            raise ValueError(f"Resume config differs for {candidate.name}")
        candidate.best_score = float(row["pooled_rmse"])
        selected.append(candidate)
    return trials.to_dict(orient="records"), failures, ranking, selected


def _phase2_window(folds, n_obs, start_ratio):
    """Keep full-series fold identities at or after the requested cutoff."""
    cutoff = max(folds[0].train_end, int(np.ceil(n_obs * start_ratio)))
    selected = [fold for fold in folds if fold.train_end >= cutoff]
    if not selected:
        raise ValueError("Phase 2 start ratio leaves no complete forecast fold")
    return selected, cutoff



def main():
    global TRACKING, SUMMARY, POLICY, PROTOCOL_VERSION, ray, _evaluate_job
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--date-col", default="ds")
    parser.add_argument("--target", required=True)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--auto-hist-exog", action="store_true")
    parser.add_argument("--exog-max-abs-corr", type=float, default=0.8)
    parser.add_argument("--exog-max-missing-ratio", type=float, default=0.05)
    parser.add_argument(
        "--model-config", help="JSON mapping model names to fixed kwargs"
    )
    parser.add_argument("--output", default="results/commodity_sota")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument(
        "--scheduler", choices=["dynamic", "dynamic-pool", "taskvine", "ray"], default="dynamic"
    )
    parser.add_argument("--phase2-max-steps", type=int, default=500)
    parser.add_argument("--phase2-val-check-steps", type=int, default=10)
    parser.add_argument("--phase2-patience", type=int, default=5)
    parser.add_argument("--phase2-val-size", type=int, default=16)
    parser.add_argument("--phase2-start-ratio", type=float, default=0.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="uni-gasoline")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--validated-candidates", help="JSON report from --smoke-test")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--phase2-window-migration", action="store_true")
    args = parser.parse_args()
    if args.auto_hist_exog:
        PROTOCOL_VERSION = EXOG_PROTOCOL_VERSION
    POLICY = {
        "max_steps": args.phase2_max_steps,
        "interval": args.phase2_val_check_steps,
        "patience": args.phase2_patience,
        "val_size": args.phase2_val_size,
    }
    if (
        any(v < 1 for v in POLICY.values())
        or not 0 <= args.phase2_start_ratio < 1
        or POLICY["val_size"] != args.horizon
        or POLICY["max_steps"] % POLICY["interval"]
        or any(b % POLICY["interval"] for b in SHPlan().budgets)
    ):
        parser.error(
            "Need positive settings, val_size=horizon, and both phase budgets divisible by validation interval"
        )
    if not 0 < args.exog_max_abs_corr <= 1:
        parser.error("--exog-max-abs-corr must be in (0, 1]")
    if not 0 < args.exog_max_missing_ratio <= 1:
        parser.error("--exog-max-missing-ratio must be in (0, 1]")
    if args.end_date and args.start_date and args.end_date < args.start_date:
        parser.error("--end-date must not precede --start-date")
    if args.wandb and (args.preflight or args.smoke_test):
        parser.error("W&B is enabled only for the main experiment")
    if args.resume and (args.preflight or args.smoke_test):
        parser.error("--resume is available only for the main experiment")
    if args.phase2_window_migration and not args.resume:
        parser.error("--phase2-window-migration requires --resume")

    output = Path(args.output).resolve()
    run_config_path = output / "run_config.json"
    if run_config_path.exists() and not args.resume:
        raise ValueError(
            "Use a fresh output directory; existing experiment results are preserved."
        )
    if args.resume and not run_config_path.exists():
        raise ValueError("--resume requires an existing run_config.json")
    output.mkdir(parents=True, exist_ok=True)
    model_config = (
        json.loads(Path(args.model_config).read_text()) if args.model_config else {}
    )
    frame = _weekly_frame(
        args.data,
        args.date_col,
        args.target,
        args.start_date,
        args.end_date,
        include_exogenous=args.auto_hist_exog,
    )
    transform_metadata = _transform_metadata(args.wandb_project)
    data_path = str(output / "weekly.pkl")

    min_train = _minimum_train(args.horizon, model_config)
    all_folds = expanding_folds(
        len(frame),
        h=args.horizon,
        min_train=min_train + POLICY["val_size"],
        step_size=1,
    )
    if not all_folds:
        raise ValueError(
            "dataset is too short for the common feasible cutoff and horizon."
        )
    exog_report = pd.DataFrame()
    hist_exog_list = []
    if args.auto_hist_exog:
        frame, exog_report, hist_exog_list = _select_historical_exogenous(
            frame,
            all_folds[0].train_end,
            max_abs_correlation=args.exog_max_abs_corr,
            max_missing_ratio=args.exog_max_missing_ratio,
        )
        exog_report.to_csv(output / "exogenous_selection.csv", index=False)
    exog_metadata = {
        "auto_hist_exog": args.auto_hist_exog,
        "hist_exog_list": hist_exog_list,
        "exog_selection_rows": all_folds[0].train_end if args.auto_hist_exog else None,
        "exog_max_abs_correlation": (
            args.exog_max_abs_corr if args.auto_hist_exog else None
        ),
        "exog_max_missing_ratio": (
            args.exog_max_missing_ratio if args.auto_hist_exog else None
        ),
        "exog_imputation": "forward_fill" if args.auto_hist_exog else None,
    }
    frame.attrs["transform_metadata"] = transform_metadata
    frame.attrs["exog_metadata"] = exog_metadata
    reps = representative_folds(all_folds)
    folds, phase2_start_cutoff = _phase2_window(
        all_folds, len(frame), args.phase2_start_ratio
    )
    candidates, eligibility = _build_candidates(
        args.horizon, model_config, min_train, hist_exog_list
    )
    if not candidates:
        raise ValueError("no eligible candidate model was found.")

    pd.DataFrame(eligibility, columns=["model", "protocol", "status", "reason"]).to_csv(
        output / "eligibility.csv", index=False
    )
    fingerprint_payload = (
        Path(args.data).read_bytes()
        + json.dumps(model_config, sort_keys=True).encode()
        + str(args.horizon).encode()
        + str(args.start_date).encode()
        + json.dumps(POLICY, sort_keys=True).encode()
        + str(args.phase2_start_ratio).encode()
        + str(NUMERICS_VERSION).encode()
        + json.dumps(SAMPLING_POLICY, sort_keys=True).encode()
        + json.dumps(transform_metadata, sort_keys=True).encode()
        + PROTOCOL_VERSION.encode()
        + json.dumps(SHPlan().__dict__, sort_keys=True).encode()
    )
    if args.end_date or args.auto_hist_exog:
        fingerprint_payload += str(args.end_date).encode()
    if args.auto_hist_exog:
        fingerprint_payload += json.dumps(exog_metadata, sort_keys=True).encode()
    fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()
    print(
        f"rows={len(frame)} min_train={min_train} folds={len(folds)} "
        f"phase2_cutoff={phase2_start_cutoff} candidates={len(candidates)} "
        f"hist_exog={len(hist_exog_list)}",
        flush=True,
    )
    if args.preflight:
        (output / "preflight.json").write_text(
            json.dumps(
                {
                    "rows": len(frame),
                    "folds": len(folds),
                    "min_train": min_train,
                    "phase1_folds": [fold.index for fold in reps],
                    "phase2_start_ratio": args.phase2_start_ratio,
                    "phase2_start_cutoff": phase2_start_cutoff,
                    "phase2_first_fold": folds[0].index,
                    "phase2_last_fold": folds[-1].index,
                    "candidates": [c.name for c in candidates],
                    "fingerprint": fingerprint,
                    "protocol_version": PROTOCOL_VERSION,
                    **transform_metadata,
                    **exog_metadata,
                    "phase2_policy": POLICY,
                    "scheduler": args.scheduler,
                },
                indent=2,
            )
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires at least one available GPU")
    if args.validated_candidates:
        report = json.loads(Path(args.validated_candidates).read_text())
        if report["fingerprint"] != fingerprint:
            raise ValueError(
                "Smoke report does not match data/config/horizon/date-range/exogenous-policy"
            )
        allowed = set(report["passed"])
        candidates = [c for c in candidates if c.name in allowed]
        eligibility = [
            (
                name,
                protocol,
                (
                    status
                    if (
                        name
                        if protocol == "scratch_hpo"
                        else f"{name}-{'LoRA' if protocol == 'lora' else 'ZeroShot'}"
                    )
                    in allowed
                    or status == "SKIP"
                    else "SKIP"
                ),
                (
                    reason
                    if status == "SKIP"
                    or (
                        name
                        if protocol == "scratch_hpo"
                        else f"{name}-{'LoRA' if protocol == 'lora' else 'ZeroShot'}"
                    )
                    in allowed
                    else "failed smoke test; see preparation.json"
                ),
            )
            for name, protocol, status, reason in eligibility
        ]
        pd.DataFrame(
            eligibility, columns=["model", "protocol", "status", "reason"]
        ).to_csv(output / "eligibility.csv", index=False)
        if not candidates:
            raise ValueError("No smoke-validated candidates")
    prior_config = None
    if args.resume:
        prior_config = json.loads(run_config_path.read_text())
        expected = {
            "sampling_policy": SAMPLING_POLICY,
            "fingerprint": fingerprint,
            "horizon": args.horizon,
            "phase2_policy": POLICY,
            "protocol_version": PROTOCOL_VERSION,
            "candidates": [candidate.name for candidate in candidates],
        }
        for key, value in expected.items():
            prior_value = prior_config.get(key)
            differs = (
                set(prior_value or []) != set(value)
                if key == "candidates"
                else prior_value != value
            )
            if differs:
                raise ValueError(f"Resume {key} differs from the existing experiment")
        if prior_config.get("status") in {"completed", "no_models_above_naive"}:
            raise ValueError("The existing experiment is already complete")
        if prior_config.get("scheduler") != args.scheduler and not (
            prior_config.get("scheduler") == "dynamic"
            and args.scheduler == "dynamic-pool"
        ) and not (
            "taskvine" in {prior_config.get("scheduler"), args.scheduler}
            and {prior_config.get("scheduler"), args.scheduler}
            <= {"dynamic", "dynamic-pool", "taskvine"}
        ):
            raise ValueError("Resume scheduler differs from the existing experiment")
        if bool(prior_config.get("wandb")) != args.wandb:
            raise ValueError("Resume must preserve the existing W&B setting")
    frame.to_pickle(data_path)
    if args.wandb:
        if not args.wandb_entity or not os.environ.get("WANDB_API_KEY"):
            raise ValueError("W&B entity and WANDB_API_KEY are required")
        from neuralforecast.benchmark_tracking import ensure_open_project

        TRACKING = (
            prior_config["wandb"]
            if args.resume
            else {
                "entity": args.wandb_entity,
                "project": args.wandb_project,
                "group": f"{args.wandb_project}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}",
                "directory": str(output / "wandb"),
            }
        )
        if (
            TRACKING["entity"] != args.wandb_entity
            or TRACKING["project"] != args.wandb_project
        ):
            raise ValueError("Resume W&B destination differs from the existing run")
        ensure_open_project(args.wandb_entity, args.wandb_project)
        SUMMARY = Tracking(
            TRACKING,
            config={
                "target": args.target,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "horizon": args.horizon,
                "seed": 42,
                "rows": len(frame),
                "folds": len(folds),
                "phase2_start_ratio": args.phase2_start_ratio,
                "phase2_start_cutoff": phase2_start_cutoff,
                "phase2_first_fold": folds[0].index,
                "phase2_last_fold": folds[-1].index,
                "fingerprint": fingerprint,
                **transform_metadata,
                **exog_metadata,
                "phase2_policy": POLICY,
                "scheduler": args.scheduler,
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
            },
        )
        (output / "wandb_run.json").write_text(
            json.dumps({**TRACKING, "url": SUMMARY.run.url}, indent=2)
        )
        SUMMARY.table(
            "eligibility",
            pd.DataFrame(
                eligibility, columns=["model", "protocol", "status", "reason"]
            ),
        )
    if args.scheduler in {"dynamic", "dynamic-pool", "taskvine"}:
        import atexit
        import signal
        from types import SimpleNamespace
        from dynamic_queue import DynamicQueue

        if args.scheduler == "taskvine":
            from taskvine_queue import TaskVineQueue

            queue = TaskVineQueue(output, SUMMARY, resume=args.resume)
        elif args.scheduler == "dynamic-pool":
            from dynamic_pool import PoolQueue

            queue = PoolQueue(
                output,
                SUMMARY,
                reuse_models={candidate.name for candidate in candidates},
                resume=args.resume,
            )
        else:
            queue = DynamicQueue(output, SUMMARY, resume=args.resume)
        atexit.register(queue.shutdown)

        def shutdown_signal(signum, frame):
            queue.shutdown()
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, shutdown_signal)
        signal.signal(signal.SIGINT, shutdown_signal)
        _evaluate_job = SimpleNamespace(remote=queue.submit)
        ray = SimpleNamespace(wait=queue.wait, get=queue.get, shutdown=queue.shutdown)
    else:
        ray.init(ignore_reinit_error=True, num_cpus=8, include_dashboard=False)
    if args.smoke_test:
        pending = {}
        results = []
        for candidate in candidates:
            budget = 2 if candidate.protocol in {"scratch_hpo", "lora"} else 0
            ref = _evaluate_job.remote(
                data_path,
                _payload(candidate),
                0,
                candidate.configs[0],
                folds[-1],
                budget,
                None,
                output / "checkpoints_smoke",
            )
            pending[ref] = candidate
        while pending:
            done, _ = ray.wait(list(pending), num_returns=1)
            ref = done[0]
            candidate = pending.pop(ref)
            try:
                result = ray.get(ref)
            except Exception as exc:
                result = {
                    "ok": False,
                    "candidate": candidate.name,
                    "error": safe_config(str(exc)),
                }
            results.append(result)
            print(json.dumps(result, default=str), flush=True)
            (output / "smoke_results.json").write_text(
                json.dumps(
                    {
                        "fingerprint": fingerprint,
                        "protocol_version": PROTOCOL_VERSION,
                        **transform_metadata,
                        **exog_metadata,
                        "phase2_policy": POLICY,
                        "scheduler": args.scheduler,
                        "passed": [r["candidate"] for r in results if r["ok"]],
                        "results": results,
                    },
                    indent=2,
                )
            )
        ray.shutdown()
        return
    phase1_reference = _validation_reference(frame, reps)
    resume_phase1 = (
        args.resume
        and prior_config.get("status") in {"phase1", "phase1_resuming"}
        and not (output / "phase1_ranking.csv").is_file()
    )
    if not args.resume:
        (output / "run_config.json").write_text(
            json.dumps(
                {
                    "target": args.target,
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                    "horizon": args.horizon,
                    "step_size": 1,
                    "seed": 42,
                    "phase1_folds": [fold.index for fold in reps],
                    "phase2_folds": len(folds),
                    "phase2_start_ratio": args.phase2_start_ratio,
                    "phase2_start_cutoff": phase2_start_cutoff,
                    "phase2_first_fold": folds[0].index,
                    "phase2_last_fold": folds[-1].index,
                    "sh_budgets": list(SHPlan().budgets),
                    "sh_rung_counts": list(SHPlan().survivors),
                    "sampling_policy": SAMPLING_POLICY,
                    "protocol_version": PROTOCOL_VERSION,
                    "evaluation_split": "validation",
                    "phase2_top_k": 10,
                    "fingerprint": fingerprint,
                    **transform_metadata,
                    **exog_metadata,
                    "phase2_policy": POLICY,
                    "scheduler": args.scheduler,
                    "wandb": TRACKING,
                    "candidates": [c.name for c in candidates],
                    "status": "phase1",
                },
                indent=2,
            )
        )
    if resume_phase1:
        prior_config.update(status="phase1_resuming", scheduler=args.scheduler)
        run_config_path.write_text(json.dumps(prior_config, indent=2))
    if not args.resume or resume_phase1:
        phase1, failures1 = _run_phase1(
            data_path, candidates, reps, output / "checkpoints_phase1",
            **({"resume": True} if resume_phase1 else {}),
        )
        ranking = sorted(
            candidates,
            key=lambda item: rank_key(item.best_score, item.best_fold_scores),
        )
        naive_rmse = _naive_score(frame, reps)
        selected, ranking_frame = _select_candidates(ranking, naive_rmse)
        ranking_frame.to_csv(output / "phase1_ranking.csv", index=False)
    else:
        phase1, failures1, ranking_frame, selected = _restore_phase1(output, candidates)
        naive_rmse = float(ranking_frame["naive_rmse"].dropna().iloc[0])
        restored = sum(
            _completed_phase2_result(
                output
                / "checkpoints_phase2"
                / candidate.name
                / "0"
                / str(fold.index)
                / f"result-{POLICY['max_steps'] if candidate.protocol in {'scratch_hpo', 'lora'} else 0}.json",
                candidate,
                fold,
                (
                    POLICY["max_steps"]
                    if candidate.protocol in {"scratch_hpo", "lora"}
                    else 0
                ),
                require_tracking=(
                    TRACKING is not None and not args.phase2_window_migration
                ),
            )
            is not None
            for candidate in selected
            for fold in folds
        )
        prior_config.update(
            status="phase2_resuming",
            scheduler=args.scheduler,
            restored_phase2_folds=restored,
            remaining_phase2_folds=len(selected) * len(folds) - restored,
        )
        run_config_path.write_text(json.dumps(prior_config, indent=2))
        print(
            f"resume restored={restored} remaining={len(selected) * len(folds) - restored}",
            flush=True,
        )
    baseline = _leaderboard([], phase1_reference)[0]
    baseline.update(selected=False, exclusion_reason="baseline")
    phase1_report = pd.concat(
        [ranking_frame, pd.DataFrame([baseline])], ignore_index=True
    )
    phase1_report = phase1_report.sort_values("pooled_rmse").reset_index(drop=True)
    phase1_report["rank"] = np.arange(1, len(phase1_report) + 1)
    phase1_report.to_csv(output / "phase1_leaderboard.csv", index=False)
    if SUMMARY:
        SUMMARY.table("phase1/ranking", pd.read_csv(output / "phase1_ranking.csv"))
        SUMMARY.evaluation_table(
            "phase1", phase1_report, _metric_definitions(phase1_reference)
        )
    reference = _validation_reference(frame, folds)
    publication_config = json.loads(run_config_path.read_text())
    with phase2_publication(output, TRACKING, publication_config) as publisher:
        def publish_completed(predictions):
            publisher.publish(
                pd.DataFrame(_leaderboard(predictions, reference)),
                _metric_definitions(reference),
                total_models=len(selected), status="phase2",
            )

        if publisher and not args.resume:
            publish_completed([])
        predictions, failures2 = (
            _phase2(
                data_path,
                selected,
                folds,
                output / "checkpoints_phase2",
                resume=args.resume,
                require_tracking=not args.phase2_window_migration,
                on_model_complete=publish_completed if publisher else None,
            )
            if selected
            else ([], [])
        )
        leaderboard = _leaderboard(predictions, reference)
        completed_models = sum(row["protocol"] != "naive" for row in leaderboard)
        definitions = {
            "phase1": _metric_definitions(phase1_reference),
            "phase2": _metric_definitions(reference),
        }
        (output / "metric_definitions.json").write_text(json.dumps(definitions, indent=2))

        pd.DataFrame(eligibility, columns=["model", "protocol", "status", "reason"]).to_csv(
            output / "eligibility.csv", index=False
        )
        if not args.resume:
            pd.DataFrame(phase1).to_csv(output / "phase1_trials.csv", index=False)
        ranking_frame.to_csv(output / "phase1_ranking.csv", index=False)
        pd.DataFrame(
            predictions,
            columns=["candidate", "protocol", "fold", "horizon", "actual", "prediction"],
        ).to_csv(output / "phase2_predictions.csv", index=False)
        pd.DataFrame(leaderboard).to_csv(output / "leaderboard.csv", index=False)
        status = (
            "no_models_above_naive"
            if not selected
            else "completed" if completed_models else "failed"
        )
        pd.DataFrame(failures1 + failures2).to_csv(output / "failures.csv", index=False)
        (output / "run_config.json").write_text(
            json.dumps(
                {
                    "target": args.target,
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                    "horizon": args.horizon,
                    "fingerprint": fingerprint,
                    **transform_metadata,
                    **exog_metadata,
                    "phase2_policy": POLICY,
                    "scheduler": args.scheduler,
                    "wandb": TRACKING,
                    "step_size": 1,
                    "seed": 42,
                    "phase1_folds": [fold.index for fold in reps],
                    "phase2_folds": len(folds),
                    "phase2_start_ratio": args.phase2_start_ratio,
                    "phase2_start_cutoff": phase2_start_cutoff,
                    "phase2_first_fold": folds[0].index,
                    "phase2_last_fold": folds[-1].index,
                    "sh_budgets": list(SHPlan().budgets),
                    "sh_rung_counts": list(SHPlan().survivors),
                    "sampling_policy": SAMPLING_POLICY,
                    "protocol_version": PROTOCOL_VERSION,
                    "evaluation_split": "validation",
                    "naive_rmse": naive_rmse,
                    "status": status,
                    "phase2_top_k": 10,
                    "candidates": [candidate.name for candidate in candidates],
                    "selected": [item.name for item in selected],
                },
                indent=2,
            )
        )

        if SUMMARY:
            SUMMARY.table("leaderboard", pd.DataFrame(leaderboard))
            SUMMARY.evaluation_table(
                "phase2", pd.DataFrame(leaderboard), definitions["phase2"]
            )
            SUMMARY.summary(
                {
                    "phase2/completed_models": completed_models,
                    "status": status,
                    "phase1/naive_rmse": naive_rmse,
                }
            )
            SUMMARY.artifact(output)
            SUMMARY.finish(failed=bool(selected) and not completed_models)
        if publisher:
            publisher.publish(
                pd.DataFrame(leaderboard), definitions["phase2"],
                total_models=len(selected), status=status,
            )
    ray.shutdown()
    if selected and not completed_models:
        raise RuntimeError("No candidate completed all Phase 2 folds")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if SUMMARY:
            SUMMARY.summary({"status": "failed"})
            SUMMARY.finish(failed=True)
        raise
