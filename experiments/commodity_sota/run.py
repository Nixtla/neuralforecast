"""Two-phase weekly commodity benchmark with pooled-RMSE model selection."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import hashlib
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
from pytorch_lightning.callbacks import ModelCheckpoint

import neuralforecast.auto as auto_module
import neuralforecast.models as model_module
from neuralforecast.benchmark import (
    SHPlan,
    benchmark_search_space,
    expanding_folds,
    fold_rmse,
    minimum_history,
    pooled_rmse,
    rank_key,
    representative_folds,
    restrict_input_size,
    sample_ray_configs,
    summary_rank_key,
)
from neuralforecast.inference_tuning import (
    INFERENCE_TUNING_MODELS,
    get_inference_tuning_config,
)
from neuralforecast.benchmark_tracking import Tracking, training_callback, safe_config
from neuralforecast.tsdataset import TimeSeriesDataModule, TimeSeriesDataset


LLM_MODELS = {"Aurora", "ChatTime", "GPT4MTS", "LangTime", "UniTime"}
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


def _weekly_frame(path, date_col, target_col, start_date=None):
    frame = pd.read_csv(path)
    if date_col not in frame or target_col not in frame:
        raise ValueError(f"CSV must contain {date_col!r} and {target_col!r}.")
    frame = frame[[date_col, target_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise", utc=True)
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    if start_date:
        frame = frame[frame[date_col] >= pd.Timestamp(start_date, tz="UTC")]
    frame = (
        frame.set_index(date_col)[target_col]
        .resample("W-SUN")
        .mean()
        .rename("y")
        .reset_index()
        .rename(columns={date_col: "ds"})
    )
    valid = frame["y"].notna()
    if not valid.any():
        raise ValueError("target has no weekly observations.")
    first, last = np.flatnonzero(valid)[[0, -1]]
    frame = frame.iloc[first : last + 1].reset_index(drop=True)
    frame["unique_id"] = target_col
    return frame[["unique_id", "ds", "y"]]


def _fill(values):
    """Forward-fill observations using only values available in the same slice."""
    values = values.copy()
    values["y"] = values["y"].ffill()
    if values["y"].isna().any():
        raise ValueError("target begins with missing observations; cannot forward-fill.")
    return values


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
):
    config = dict(config)
    config["max_steps"] = budget if stopping else 1000
    if stopping:
        from neuralforecast.losses.pytorch import MSE

        config["valid_loss"] = MSE()
        config["val_check_steps"] = stopping["interval"]
    config["random_seed"] = 42
    config["early_stop_patience_steps"] = -1
    config["enable_checkpointing"] = True
    config["logger"] = False
    config["accelerator"] = "gpu"
    config["devices"] = 1
    config["enable_progress_bar"] = False
    model = model_cls(**config)
    train_dataset = _dataset(train)
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
    checkpoint_cb = ModelCheckpoint(dirpath=workdir, save_last=True, save_top_k=0)
    trainer_kwargs = dict(model.trainer_kwargs)
    callbacks = list(trainer_kwargs.get("callbacks") or [])
    callbacks.append(checkpoint_cb)
    stopper = None
    if stopping:
        from neuralforecast.benchmark_stopping import ValidationStopper

        stopper = ValidationStopper(
            workdir,
            interval=stopping["interval"],
            patience=stopping["patience"],
            metrics_callback=tracker.log if tracker and tracker.run else None,
        )

        class StopCallback(pl.Callback):
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
    next_checkpoint = str(Path(workdir) / "resume.ckpt")
    trainer.save_checkpoint(next_checkpoint)
    if stopper:
        stopper.restore(model)
        next_checkpoint = str(stopper.path)
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


def _fit_lora(
    model_cls, config, train, valid, budget, workdir, tracker=None, stopping=None
):
    _, fit_foundation_lora, _ = _lora_api()
    if fit_foundation_lora is None:
        raise RuntimeError("foundation LoRA module is unavailable.")
    options = {}
    if stopping:
        options = {
            "validation": train.iloc[-stopping["val_size"] :].copy(),
            "stopping": {
                "interval": stopping["interval"],
                "patience": stopping["patience"],
            },
        }
    model = fit_foundation_lora(
        model_cls,
        config,
        train.iloc[: -stopping["val_size"]].copy() if stopping else train,
        h=len(valid),
        steps=budget,
        output_dir=workdir,
        metrics_callback=tracker.log if tracker and tracker.run else None,
        **options,
    )
    if stopping:
        (Path(workdir) / "early_stopping.json").write_text(
            json.dumps(model.early_stopping_info)
        )
    model.val_size = 0
    model.test_size = len(valid)
    dataset = _dataset(pd.concat([train, valid], ignore_index=True))
    return _point_prediction(model, dataset, len(valid)), None


@ray.remote(num_gpus=1, max_calls=1)
def _evaluate_job(
    data_path, candidate, config_id, config, fold, budget, checkpoint, root
):
    started = time.monotonic()
    tracker = None
    result = None
    try:
        # Backend auto-integrations must not create additional runs.
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["WANDB_DISABLE_CODE"] = "true"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_num_threads(2)
        frame = pd.read_pickle(data_path)
        phase = Path(root).name.removeprefix("checkpoints_")
        stopping = candidate.get("phase2_policy", POLICY) if phase == "phase2" else None
        history = frame.iloc[fold.train_slice]
        if stopping and candidate["protocol"] != "zero_shot":
            n = stopping["val_size"]
            train = pd.concat([_fill(history.iloc[:-n]), _fill(history.iloc[-n:])])
        else:
            train = _fill(history)
        valid = _fill(frame.iloc[fold.valid_slice])
        workdir = Path(root) / candidate["name"] / str(config_id) / str(fold.index)
        workdir.mkdir(parents=True, exist_ok=True)
        tracker = Tracking(
            candidate.get("tracking"),
            phase=phase,
            candidate=candidate["name"],
            config_id=config_id,
            fold=fold.index,
            config={
                "model": candidate["model_name"],
                "protocol": candidate["protocol"],
                "parameters": config,
                "fold": fold.index,
                "train_end": str(train.ds.iloc[-stopping["val_size"] - 1])
                if stopping and candidate["protocol"] != "zero_shot"
                else str(train.ds.iloc[-1]),
                "validation_start": str(train.ds.iloc[-stopping["val_size"]])
                if stopping and candidate["protocol"] != "zero_shot"
                else None,
                "validation_end": str(train.ds.iloc[-1])
                if stopping and candidate["protocol"] != "zero_shot"
                else None,
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
            )
        elif candidate["protocol"] == "lora":
            prediction, next_checkpoint = _fit_lora(
                model_cls, config, train, valid, budget, workdir, tracker, stopping
            )
        else:
            prediction, next_checkpoint = _fit_inference(
                model_cls, config, train, valid
            )
        actual = valid["y"].to_numpy(dtype=float)
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
        (workdir / f"result-{budget}.json").write_text(
            json.dumps(result, default=str)
        ) if "workdir" in locals() else None
    return result


def _fixed_kwargs(config, name, protocol=None):
    fixed = dict(config.get(name, config.get(f"Auto{name}", {})))
    if protocol:
        fixed.update(config.get(f"{name}-{protocol}", {}))
    return fixed


def _build_candidates(h, model_config, first_train):
    candidates, eligibility = [], []
    for auto_name in getattr(auto_module, "__all__", []):
        if not auto_name.startswith("Auto"):
            continue
        name = auto_name[4:]
        if name in LLM_MODELS or name in SKIP_MODELS:
            eligibility.append(
                (
                    name,
                    "scratch_hpo",
                    "SKIP",
                    SKIP_REASONS.get(name, "LLM protocol excluded by runner"),
                )
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
                benchmark_search_space(auto.config), first_train - h
            )
            configs = sample_ray_configs(space, n=10, seed=42)
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
        if name in LLM_MODELS or name in SKIP_MODELS:
            eligibility.append(
                (
                    name,
                    "zero_shot",
                    "SKIP",
                    SKIP_REASONS.get(name, "LLM protocol excluded by runner"),
                )
            )
            continue
        try:
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
        try:
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
        except Exception as exc:
            eligibility.append((name, "lora", "SKIP", f"{type(exc).__name__}: {exc}"))
            continue
        candidates.append(Candidate(f"{name}-LoRA", "lora", name, configs, None))
        eligibility.append((name, "lora", "READY", "one 1000-step Phase 1 rung"))
    return candidates, eligibility


def _minimum_train(h, model_config):
    minima = [h]
    for auto_name in getattr(auto_module, "__all__", []):
        if not auto_name.startswith("Auto"):
            continue
        name = auto_name[4:]
        if name in LLM_MODELS or name in SKIP_MODELS:
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
        if name in LLM_MODELS or name in SKIP_MODELS:
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


def _run_phase1(data_path, candidates, folds, checkpoint_root):
    pending, trials, failures = {}, [], []

    def submit(candidate, config_id, fold, budget):
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
        pending[ref] = (candidate, config_id, fold, budget)

    for candidate in candidates:
        candidate.alive = list(range(len(candidate.configs)))
        if candidate.plan:
            budget = candidate.plan.budgets[0]
        else:
            budget = 1000 if candidate.protocol == "lora" else 0
        for config_id in candidate.alive:
            for fold in sorted(folds, key=lambda f: f.train_end, reverse=True):
                submit(candidate, config_id, fold, budget)

    while pending:
        done, _ = ray.wait(list(pending), num_returns=1)
        ref = done[0]
        candidate, config_id, fold, budget = pending.pop(ref)
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
        if result["ok"] and result.get("checkpoint"):
            candidate.checkpoints[(config_id, fold.index)] = result["checkpoint"]
        elif not result["ok"]:
            failures.append(result)

        required = len(candidate.alive) * len(folds)
        if len(candidate.rung_results) != required:
            continue
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
                }
            )
        pd.DataFrame(trials).to_csv(
            Path(checkpoint_root).parent / "phase1_trials.csv", index=False
        )
        pd.DataFrame(failures).to_csv(
            Path(checkpoint_root).parent / "phase1_failures.csv", index=False
        )
        scored.sort(key=lambda row: rank_key(row[1], row[2]))
        final_rung = (
            candidate.plan is None or candidate.rung == len(candidate.plan.budgets) - 1
        )
        if final_rung:
            best = scored[0]
            candidate.best_config = candidate.configs[best[0]]
            candidate.best_score = best[1]
            candidate.best_fold_scores = best[2]
            continue
        keep = candidate.plan.survivors[candidate.rung + 1]
        candidate.alive = [row[0] for row in scored[:keep]]
        candidate.rung += 1
        candidate.rung_results = {}
        next_budget = candidate.plan.budgets[candidate.rung]
        for cid in candidate.alive:
            for rep_fold in folds:
                submit(candidate, cid, rep_fold, next_budget)
    return trials, failures


def _phase2(data_path, selected, folds, checkpoint_root):
    pending, predictions, failures = {}, [], []
    for candidate in selected:
        budget = (
            POLICY["max_steps"] if candidate.protocol in {"scratch_hpo", "lora"} else 0
        )
        for fold in sorted(folds, key=lambda f: f.train_end, reverse=True):
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
        pd.DataFrame(predictions).to_csv(
            Path(checkpoint_root).parent / "phase2_predictions.csv", index=False
        )
        pd.DataFrame(failures).to_csv(
            Path(checkpoint_root).parent / "phase2_failures.csv", index=False
        )
    return predictions, failures


def _leaderboard(predictions, expected_folds):
    frame = pd.DataFrame(predictions)
    if frame.empty:
        return []
    rows = []
    for (candidate, protocol), values in frame.groupby(["candidate", "protocol"]):
        if values["fold"].nunique() != expected_folds:
            continue
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


def main():
    global TRACKING, SUMMARY, POLICY, ray, _evaluate_job
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--date-col", default="ds")
    parser.add_argument("--target", required=True)
    parser.add_argument("--start-date")
    parser.add_argument(
        "--model-config", help="JSON mapping model names to fixed kwargs"
    )
    parser.add_argument("--output", default="results/commodity_sota")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--scheduler", choices=["dynamic", "ray"], default="dynamic")
    parser.add_argument("--phase2-max-steps", type=int, default=500)
    parser.add_argument("--phase2-val-check-steps", type=int, default=10)
    parser.add_argument("--phase2-patience", type=int, default=5)
    parser.add_argument("--phase2-val-size", type=int, default=16)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="uni-gasoline")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--validated-candidates", help="JSON report from --smoke-test")
    args = parser.parse_args()
    POLICY = {
        "max_steps": args.phase2_max_steps,
        "interval": args.phase2_val_check_steps,
        "patience": args.phase2_patience,
        "val_size": args.phase2_val_size,
    }
    if (
        any(v < 1 for v in POLICY.values())
        or POLICY["val_size"] != args.horizon
        or POLICY["max_steps"] % POLICY["interval"]
    ):
        parser.error(
            "Phase 2 needs positive settings, val_size=horizon and max_steps divisible by validation interval"
        )
    if args.wandb and (args.preflight or args.smoke_test):
        parser.error("W&B is enabled only for the main experiment")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_config = (
        json.loads(Path(args.model_config).read_text()) if args.model_config else {}
    )
    frame = _weekly_frame(args.data, args.date_col, args.target, args.start_date)
    data_path = str(output / "weekly.pkl")
    frame.to_pickle(data_path)

    min_train = _minimum_train(args.horizon, model_config)
    folds = expanding_folds(
        len(frame),
        h=args.horizon,
        min_train=min_train + POLICY["val_size"],
        step_size=1,
    )
    if not folds:
        raise ValueError(
            "dataset is too short for the common feasible cutoff and horizon."
        )
    reps = representative_folds(folds)
    candidates, eligibility = _build_candidates(args.horizon, model_config, min_train)
    if not candidates:
        raise ValueError("no eligible candidate model was found.")

    pd.DataFrame(eligibility, columns=["model", "protocol", "status", "reason"]).to_csv(
        output / "eligibility.csv", index=False
    )
    fingerprint = hashlib.sha256(
        Path(args.data).read_bytes()
        + json.dumps(model_config, sort_keys=True).encode()
        + str(args.horizon).encode()
        + str(args.start_date).encode()
        + json.dumps(POLICY, sort_keys=True).encode()
    ).hexdigest()
    print(
        f"rows={len(frame)} min_train={min_train} folds={len(folds)} candidates={len(candidates)}",
        flush=True,
    )
    if args.preflight:
        (output / "preflight.json").write_text(
            json.dumps(
                {
                    "rows": len(frame),
                    "folds": len(folds),
                    "min_train": min_train,
                    "candidates": [c.name for c in candidates],
                    "fingerprint": fingerprint,
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
                "Smoke report does not match data/config/horizon/start-date"
            )
        allowed = set(report["passed"])
        candidates = [c for c in candidates if c.name in allowed]
        eligibility = [
            (
                name,
                protocol,
                status
                if (
                    name
                    if protocol == "scratch_hpo"
                    else f"{name}-{'LoRA' if protocol == 'lora' else 'ZeroShot'}"
                )
                in allowed
                or status == "SKIP"
                else "SKIP",
                reason
                if status == "SKIP"
                or (
                    name
                    if protocol == "scratch_hpo"
                    else f"{name}-{'LoRA' if protocol == 'lora' else 'ZeroShot'}"
                )
                in allowed
                else "failed smoke test; see preparation.json",
            )
            for name, protocol, status, reason in eligibility
        ]
        pd.DataFrame(
            eligibility, columns=["model", "protocol", "status", "reason"]
        ).to_csv(output / "eligibility.csv", index=False)
        if not candidates:
            raise ValueError("No smoke-validated candidates")
    if args.wandb:
        if not args.wandb_entity or not os.environ.get("WANDB_API_KEY"):
            raise ValueError("W&B entity and WANDB_API_KEY are required")
        TRACKING = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "group": f"gasoline-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}",
            "directory": str(output / "wandb"),
        }
        SUMMARY = Tracking(
            TRACKING,
            config={
                "target": args.target,
                "start_date": args.start_date,
                "horizon": args.horizon,
                "seed": 42,
                "rows": len(frame),
                "folds": len(folds),
                "fingerprint": fingerprint,
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
    if args.scheduler == "dynamic":
        import atexit
        import signal
        from types import SimpleNamespace
        from dynamic_queue import DynamicQueue

        queue = DynamicQueue(output, SUMMARY)
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
    (output / "run_config.json").write_text(
        json.dumps(
            {
                "horizon": args.horizon,
                "step_size": 1,
                "seed": 42,
                "phase1_folds": [fold.index for fold in reps],
                "phase2_folds": len(folds),
                "sh_budgets": [125, 250, 500, 1000],
                "sh_rung_counts": [10, 5, 2, 1],
                "phase2_top_k": 10,
                "fingerprint": fingerprint,
                "phase2_policy": POLICY,
                "scheduler": args.scheduler,
                "wandb": TRACKING,
                "candidates": [c.name for c in candidates],
                "status": "phase1",
            },
            indent=2,
        )
    )
    phase1, failures1 = _run_phase1(
        data_path, candidates, reps, output / "checkpoints_phase1"
    )
    ranking = sorted(
        candidates,
        key=lambda item: rank_key(item.best_score, item.best_fold_scores),
    )
    selected = [item for item in ranking if np.isfinite(item.best_score)][:10]
    if not selected:
        raise RuntimeError("every Phase 1 candidate failed.")
    pd.DataFrame(
        [
            {
                "candidate": c.name,
                "protocol": c.protocol,
                "pooled_rmse": c.best_score,
                "config": json.dumps(safe_config(c.best_config), sort_keys=True),
            }
            for c in ranking
        ]
    ).to_csv(output / "phase1_ranking.csv", index=False)
    if SUMMARY:
        SUMMARY.table("phase1/ranking", pd.read_csv(output / "phase1_ranking.csv"))
    predictions, failures2 = _phase2(
        data_path, selected, folds, output / "checkpoints_phase2"
    )
    leaderboard = _leaderboard(predictions, len(folds))

    pd.DataFrame(eligibility, columns=["model", "protocol", "status", "reason"]).to_csv(
        output / "eligibility.csv", index=False
    )
    pd.DataFrame(phase1).to_csv(output / "phase1_trials.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": item.name,
                "protocol": item.protocol,
                "pooled_rmse": item.best_score,
                "fold_rmse_std": float(np.std(item.best_fold_scores)),
                "worst_fold_rmse": float(np.max(item.best_fold_scores)),
                "config": json.dumps(item.best_config, default=str, sort_keys=True),
            }
            for item in ranking
        ]
    ).to_csv(output / "phase1_ranking.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "phase2_predictions.csv", index=False)
    pd.DataFrame(leaderboard).to_csv(output / "leaderboard.csv", index=False)
    pd.DataFrame(failures1 + failures2).to_csv(output / "failures.csv", index=False)
    (output / "run_config.json").write_text(
        json.dumps(
            {
                "horizon": args.horizon,
                "fingerprint": fingerprint,
                "phase2_policy": POLICY,
                "scheduler": args.scheduler,
                "wandb": TRACKING,
                "step_size": 1,
                "seed": 42,
                "phase1_folds": [fold.index for fold in reps],
                "phase2_folds": len(folds),
                "sh_budgets": [125, 250, 500, 1000],
                "sh_rung_counts": [10, 5, 2, 1],
                "phase2_top_k": 10,
                "selected": [item.name for item in selected],
            },
            indent=2,
        )
    )

    if SUMMARY:
        SUMMARY.table("leaderboard", pd.DataFrame(leaderboard))
        SUMMARY.summary(
            {
                "phase2/completed_models": len(leaderboard),
                "status": "completed" if leaderboard else "failed",
            }
        )
        SUMMARY.artifact(output)
        SUMMARY.finish(failed=not leaderboard)
    ray.shutdown()
    if not leaderboard:
        raise RuntimeError("No candidate completed all Phase 2 folds")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if SUMMARY:
            SUMMARY.summary({"status": "failed"})
            SUMMARY.finish(failed=True)
        raise
