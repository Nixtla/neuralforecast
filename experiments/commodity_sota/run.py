"""Two-phase weekly commodity benchmark with pooled-RMSE model selection."""

from __future__ import annotations

import argparse
import inspect
import json
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
    Fold,
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
)
from neuralforecast.inference_tuning import (
    INFERENCE_TUNING_MODELS,
    get_inference_tuning_config,
)
from neuralforecast.tsdataset import TimeSeriesDataModule, TimeSeriesDataset


LLM_MODELS = {"Aurora", "ChatTime", "GPT4MTS", "LangTime", "UniTime"}
SKIP_MODELS = {"HINT", "SearchCast"}


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
    values = values.copy()
    values["y"] = values["y"].interpolate().ffill().bfill()
    if values["y"].isna().any():
        raise ValueError("target interpolation left missing values.")
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



def _fit_trainable(model_cls, config, train, valid, budget, checkpoint, workdir):
    config = dict(config)
    config["max_steps"] = budget
    config["random_seed"] = 42
    config["early_stop_patience_steps"] = -1
    config["enable_checkpointing"] = True
    model = model_cls(**config)
    train_dataset = _dataset(train)
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
    trainer_kwargs["callbacks"] = callbacks
    trainer_kwargs["max_steps"] = budget
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint)
    next_checkpoint = str(Path(workdir) / "resume.ckpt")
    trainer.save_checkpoint(next_checkpoint)
    model.metrics = trainer.callback_metrics
    model.__dict__.pop("_trainer", None)
    prediction = _point_prediction(model, _dataset(pd.concat([train, valid])), len(valid))
    return prediction, next_checkpoint



def _fit_inference(model_cls, config, train, valid):
    model = model_cls(**config)
    dataset = _dataset(pd.concat([train, valid]))
    model.fit(dataset, val_size=0, test_size=len(valid), random_seed=42)
    return _point_prediction(model, dataset, len(valid)), None


@ray.remote(num_gpus=1)
def _evaluate_job(data_path, candidate, config_id, config, fold, budget, checkpoint, root):
    try:
        frame = pd.read_pickle(data_path)
        train = _fill(frame.iloc[fold.train_slice])
        valid = _fill(frame.iloc[fold.valid_slice])
        workdir = Path(root) / candidate["name"] / str(config_id) / str(fold.index)
        workdir.mkdir(parents=True, exist_ok=True)
        model_cls = getattr(model_module, candidate["model_name"])
        if candidate["protocol"] == "scratch_hpo":
            prediction, next_checkpoint = _fit_trainable(
                model_cls, config, train, valid, budget, checkpoint, workdir
            )
        else:
            prediction, next_checkpoint = _fit_inference(model_cls, config, train, valid)
        actual = valid["y"].to_numpy(dtype=float)
        return {
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
    except Exception as exc:
        kind = "OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "FAILURE"
        return {
            "ok": False,
            "candidate": candidate["name"],
            "config_id": config_id,
            "fold": fold.index,
            "budget": budget,
            "kind": kind,
            "error": f"{type(exc).__name__}: {exc}",
        }



def _fixed_kwargs(config, name):
    return dict(config.get(name, config.get(f"Auto{name}", {})))



def _build_candidates(h, model_config, first_train):
    candidates, eligibility = [], []
    for auto_name in getattr(auto_module, "__all__", []):
        if not auto_name.startswith("Auto"):
            continue
        name = auto_name[4:]
        if name in LLM_MODELS or name in SKIP_MODELS:
            eligibility.append((name, "scratch_hpo", "SKIP", "excluded protocol"))
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
                eligibility.append((name, "scratch_hpo", "SKIP", "multivariate without exogenous inputs"))
                continue
            space = benchmark_search_space(auto.config)
            space = restrict_input_size(space, first_train)
            configs = sample_ray_configs(space, n=10, seed=42)
        except Exception as exc:
            eligibility.append((name, "scratch_hpo", "SKIP", f"{type(exc).__name__}: {exc}"))
            continue
        candidates.append(Candidate(name, "scratch_hpo", auto.cls_model.__name__, configs, SHPlan()))
        eligibility.append((name, "scratch_hpo", "READY", ""))

    for name in INFERENCE_TUNING_MODELS:
        if name in LLM_MODELS or name in SKIP_MODELS:
            eligibility.append((name, "zero_shot", "SKIP", "excluded protocol"))
            continue
        try:
            space = get_inference_tuning_config(
                name, h=h, fixed=_fixed_kwargs(model_config, name), backend="ray"
            )
            space = restrict_input_size(space, first_train)
            configs = sample_ray_configs(space, n=10, seed=42)
        except Exception as exc:
            eligibility.append((name, "zero_shot", "SKIP", f"{type(exc).__name__}: {exc}"))
            continue
        candidates.append(Candidate(f"{name}-ZeroShot", "zero_shot", name, configs, None))
        eligibility.append((name, "zero_shot", "READY", ""))
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
                minima.append(minimum_history(benchmark_search_space(auto.config), h))
        except Exception:
            pass
    for name in INFERENCE_TUNING_MODELS:
        if name in LLM_MODELS or name in SKIP_MODELS:
            continue
        try:
            space = get_inference_tuning_config(
                name, h=h, fixed=_fixed_kwargs(model_config, name), backend="ray"
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
    }



def _compound_score(candidate, folds):
    actual, prediction, fold_scores = [], [], []
    for config_id in candidate.alive:
        rows = [candidate.rung_results[(config_id, fold.index)] for fold in folds]
        if not all(row["ok"] for row in rows):
            yield config_id, float("inf"), (float("inf"),) * len(folds)
            continue
        actual_i = [row["actual"] for row in rows]
        pred_i = [row["prediction"] for row in rows]
        scores_i = tuple(row["rmse"] for row in rows)
        yield config_id, pooled_rmse(actual_i, pred_i), scores_i



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
        budget = candidate.plan.budgets[0] if candidate.plan else 0
        for config_id in candidate.alive:
            for fold in folds:
                submit(candidate, config_id, fold, budget)

    while pending:
        done, _ = ray.wait(list(pending), num_returns=1)
        ref = done[0]
        candidate, config_id, fold, budget = pending.pop(ref)
        result = ray.get(ref)
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
        scored.sort(key=lambda row: rank_key(row[1], row[2]))
        keep = 1 if candidate.plan is None else candidate.plan.survivors[candidate.rung]
        candidate.alive = [row[0] for row in scored[:keep]]
        if candidate.plan is None or candidate.rung == len(candidate.plan.budgets) - 1:
            best = scored[0]
            candidate.best_config = candidate.configs[best[0]]
            candidate.best_score = best[1]
            candidate.best_fold_scores = best[2]
            continue
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
        budget = 1000 if candidate.protocol == "scratch_hpo" else 0
        for fold in folds:
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
        result = ray.get(ref)
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
    return predictions, failures



def _leaderboard(predictions):
    frame = pd.DataFrame(predictions)
    rows = []
    for (candidate, protocol), values in frame.groupby(["candidate", "protocol"]):
        fold_scores = []
        for _, fold in values.groupby("fold"):
            fold_scores.append(fold_rmse(fold["actual"], fold["prediction"]))
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
    rows.sort(key=lambda row: rank_key(row["pooled_rmse"], [row["fold_rmse_std"], row["worst_fold_rmse"]]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--date-col", default="ds")
    parser.add_argument("--target", required=True)
    parser.add_argument("--start-date")
    parser.add_argument("--model-config", help="JSON mapping model names to fixed kwargs")
    parser.add_argument("--output", default="results/commodity_sota")
    parser.add_argument("--horizon", type=int, default=16)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model_config = json.loads(Path(args.model_config).read_text()) if args.model_config else {}
    frame = _weekly_frame(args.data, args.date_col, args.target, args.start_date)
    data_path = str(output / "weekly.pkl")
    frame.to_pickle(data_path)

    min_train = _minimum_train(args.horizon, model_config)
    folds = expanding_folds(len(frame), h=args.horizon, min_train=min_train, step_size=1)
    if not folds:
        raise ValueError("dataset is too short for the common feasible cutoff and horizon.")
    reps = representative_folds(folds)
    candidates, eligibility = _build_candidates(args.horizon, model_config, min_train)
    if not candidates:
        raise ValueError("no eligible candidate model was found.")

    ray.init(ignore_reinit_error=True)
    phase1, failures1 = _run_phase1(data_path, candidates, reps, output / "checkpoints_phase1")
    ranking = sorted(
        candidates,
        key=lambda item: rank_key(item.best_score, item.best_fold_scores),
    )
    selected = ranking[: min(10, len(ranking))]
    predictions, failures2 = _phase2(
        data_path, selected, folds, output / "checkpoints_phase2"
    )
    leaderboard = _leaderboard(predictions)

    pd.DataFrame(eligibility, columns=["model", "protocol", "status", "reason"]).to_csv(output / "eligibility.csv", index=False)
    pd.DataFrame(phase1).to_csv(output / "phase1_trials.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": item.name,
                "protocol": item.protocol,
                "pooled_rmse": item.best_score,
                "fold_rmse_std": float(np.std(item.best_fold_scores)),
                "worst_fold_rmse": float(np.max(item.best_fold_scores)),
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
                "step_size": 1,
                "seed": 42,
                "phase1_folds": [fold.index for fold in reps],
                "phase2_folds": len(folds),
                "sh_budgets": [125, 250, 500, 1000],
                "sh_survivors": [10, 5, 2, 1],
                "phase2_top_k": 10,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
