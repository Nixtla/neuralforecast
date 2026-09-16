"""Utilities for chronological two-phase forecasting benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from math import sqrt
from typing import Iterable, Sequence

import numpy as np
from ray import tune
from ray.tune.search.sample import (
    Categorical,
    Domain,
    Float,
    Integer,
    LogUniform,
    Uniform,
)
from scipy.stats import qmc

SAMPLING_POLICY = {
    "version": 1,
    "method": "sobol",
    "seed": 42,
    "scramble": True,
    "n": 10,
}


@dataclass(frozen=True)
class Fold:
    """One expanding-window fold with an evaluation-only validation block."""

    index: int
    train_end: int
    valid_end: int

    @property
    def train_slice(self) -> slice:
        return slice(0, self.train_end)

    @property
    def valid_slice(self) -> slice:
        return slice(self.train_end, self.valid_end)


@dataclass(frozen=True)
class SHPlan:
    """Successive-halving budgets and survivor counts."""

    budgets: tuple[int, ...] = (100, 250, 500)
    survivors: tuple[int, ...] = (10, 5, 1)

    def __post_init__(self):
        if len(self.budgets) != len(self.survivors):
            raise ValueError("budgets and survivors must have the same length.")
        if not self.budgets or any(b <= 0 for b in self.budgets):
            raise ValueError("budgets must be positive.")
        if any(a >= b for a, b in zip(self.budgets, self.budgets[1:])):
            raise ValueError("budgets must be strictly increasing.")
        if any(a <= b for a, b in zip(self.survivors, self.survivors[1:])):
            raise ValueError("survivors must be strictly decreasing.")
        if self.survivors[-1] != 1:
            raise ValueError("the final rung must keep one configuration.")


def expanding_folds(
    n_obs: int,
    h: int = 16,
    min_train: int = 1,
    step_size: int = 1,
) -> list[Fold]:
    """Create expanding folds ending at the last complete horizon."""
    if min(n_obs, h, min_train, step_size) < 1:
        raise ValueError("n_obs, h, min_train and step_size must be positive.")
    if min_train + h > n_obs:
        return []
    folds: list[Fold] = []
    for train_end in range(min_train, n_obs - h + 1, step_size):
        folds.append(Fold(len(folds), train_end, train_end + h))
    return folds


def representative_folds(folds: Sequence[Fold]) -> tuple[Fold, Fold, Fold]:
    """Return the first, middle and last fold."""
    if not folds:
        raise ValueError("at least one fold is required.")
    middle = (len(folds) - 1) // 2
    return folds[0], folds[middle], folds[-1]


def pooled_rmse(
    actual: Iterable[Sequence[float]], prediction: Iterable[Sequence[float]]
) -> float:
    """RMSE over every forecast point across folds."""
    y = np.concatenate([np.asarray(values, dtype=float) for values in actual])
    y_hat = np.concatenate([np.asarray(values, dtype=float) for values in prediction])
    if y.shape != y_hat.shape or not y.size:
        raise ValueError("actual and prediction must have the same non-empty shape.")
    if not np.isfinite(y).all() or not np.isfinite(y_hat).all():
        return float("inf")
    return sqrt(float(np.mean(np.square(y - y_hat))))


def fold_rmse(actual: Sequence[float], prediction: Sequence[float]) -> float:
    """RMSE for one fold."""
    return pooled_rmse([actual], [prediction])


def forecast_metrics(actual, prediction, forecast_origin):
    """Compute pooled point errors and direction accuracy against forecast origins.

    Args:
        actual: One-dimensional actual targets.
        prediction: Corresponding point forecasts.
        forecast_origin: Last observation before each forecast, repeated per horizon.

    Returns:
        MAE, MAPE percent, MSE, RMSE and DA percent. MAPE excludes zero actuals
        and is None if all actuals are zero; mape_n records its denominator.
        DA compares exact rise/fall/flat signs relative to the fixed origin.
    """
    y, pred, origin = [
        np.asarray(v, dtype=float) for v in (actual, prediction, forecast_origin)
    ]
    if y.ndim != 1 or not y.size or y.shape != pred.shape or y.shape != origin.shape:
        raise ValueError("Metrics require aligned nonempty one-dimensional arrays")
    if not all(np.isfinite(v).all() for v in (y, pred, origin)):
        raise ValueError("Metrics require finite actuals, predictions and origins")
    error = pred - y
    nonzero = y != 0
    mse = float(np.mean(error**2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "mape_pct": (
            float(100 * np.mean(np.abs(error[nonzero]) / np.abs(y[nonzero])))
            if nonzero.any()
            else None
        ),
        "mape_n": int(nonzero.sum()),
        "mse": mse,
        "rmse": sqrt(mse),
        "da_pct": float(100 * np.mean(np.sign(pred - origin) == np.sign(y - origin))),
    }


def rank_key(score: float, fold_scores: Sequence[float]) -> tuple[float, float, float]:
    """Order by pooled RMSE, fold RMSE std and worst fold RMSE."""
    values = np.asarray(fold_scores, dtype=float)
    if not np.isfinite(score) or not len(values) or not np.isfinite(values).all():
        return float("inf"), float("inf"), float("inf")
    return score, float(values.std()), float(values.max())


def summary_rank_key(
    score: float, fold_std: float, worst_fold: float
) -> tuple[float, float, float]:
    """Order already-aggregated leaderboard metrics."""
    values = np.asarray([score, fold_std, worst_fold], dtype=float)
    if not np.isfinite(values).all():
        return float("inf"), float("inf"), float("inf")
    return score, fold_std, worst_fold


def sample_ray_configs(space: dict, n: int = 10, seed: int = 42) -> list[dict]:
    """Resolve Ray domains with seeded scrambled Sobol points.

    Args:
        space: Nested search space with choice, uniform, loguniform or randint.
        n: Positive number of configurations. Discrete duplicates are retained.
        seed: Seed for Sobol scrambling, independent of training RNG state.

    Returns:
        Independent configurations in Sobol sequence order. Non-power-of-two
        prefixes do not retain the full sequence's balance guarantee.

    Raises:
        ValueError: If n is invalid or a search domain is unsupported.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be positive.")
    domains: list[tuple[tuple, Domain]] = []

    def collect(value, path=()):
        if isinstance(value, Domain):
            sampler = value.get_sampler()
            supported = (
                isinstance(value, Categorical)
                and isinstance(sampler, Uniform)
                or isinstance(value, Float)
                and isinstance(sampler, (Uniform, LogUniform))
                or isinstance(value, Integer)
                and isinstance(sampler, Uniform)
            )
            if not supported:
                raise ValueError(f"Unsupported Sobol domain at {path}: {value}")
            if isinstance(value, Categorical):
                if not value.categories:
                    raise ValueError(f"Empty Sobol choice at {path}")
                # Conditional domains within choices cannot use fixed dimensions.
                for category in value.categories:
                    before = len(domains)
                    collect(category, path)
                    if len(domains) != before:
                        raise ValueError(f"Conditional Sobol choice at {path}")
            elif (
                not np.isfinite([value.lower, value.upper]).all()
                or value.lower >= value.upper
                or isinstance(sampler, LogUniform)
                and value.lower <= 0
            ):
                raise ValueError(f"Invalid Sobol bounds at {path}")
            domains.append((path, value))
        elif isinstance(value, dict):
            if "grid_search" in value:
                raise ValueError(f"Unsupported Sobol grid_search at {path}")
            for key in sorted(value):
                collect(value[key], (*path, key))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                collect(item, (*path, index))

    collect(space)
    if not domains:
        return [deepcopy(space) for _ in range(n)]
    points = qmc.Sobol(d=len(domains), scramble=True, seed=seed).random_base2(
        (int(n) - 1).bit_length()
    )[:n]

    def resolve(value, replacements, path=()):
        if isinstance(value, Domain):
            return deepcopy(replacements[path])
        if isinstance(value, dict):
            return {k: resolve(v, replacements, (*path, k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(
                resolve(v, replacements, (*path, i)) for i, v in enumerate(value)
            )
        return deepcopy(value)

    configs = []
    for point in points:
        replacements = {}
        for (path, domain), u in zip(domains, point):
            if isinstance(domain, Categorical):
                value = domain.categories[int(u * len(domain.categories))]
            elif isinstance(domain, Integer):
                value = min(
                    domain.upper - 1,
                    int(domain.lower + np.floor(u * (domain.upper - domain.lower))),
                )
            elif isinstance(domain.get_sampler(), LogUniform):
                value = float(
                    np.exp(
                        np.log(domain.lower)
                        + u * (np.log(domain.upper) - np.log(domain.lower))
                    )
                )
            else:
                value = float(domain.lower + u * (domain.upper - domain.lower))
            replacements[path] = value
        configs.append(resolve(space, replacements))
    return configs


def minimum_history(space: dict, default: int = 1) -> int:
    """Smallest positive input size represented by a Ray search space."""
    value = space.get("input_size")
    categories = getattr(value, "categories", None)
    if categories is None:
        return value if isinstance(value, int) and value > 0 else default
    positive = [item for item in categories if isinstance(item, int) and item > 0]
    return min(positive) if positive else default


def restrict_input_size(space: dict, max_history: int) -> dict:
    """Keep only input-size choices executable on the first representative fold."""
    if max_history < 1:
        raise ValueError("max_history must be positive.")
    space = dict(space)
    value = space.get("input_size")
    categories = getattr(value, "categories", None)
    if categories is None:
        if isinstance(value, int) and value > max_history:
            raise ValueError("fixed input_size exceeds available history.")
        return space
    allowed = [
        item
        for item in categories
        if not isinstance(item, int) or item <= 0 or item <= max_history
    ]
    if not allowed:
        raise ValueError("no feasible input_size remains.")
    space["input_size"] = tune.choice(allowed)
    return space


def benchmark_search_space(space: dict, seed: int = 42) -> dict:
    """Apply the benchmark training protocol to a trainable Auto search space."""
    space = dict(space)
    space.pop("max_steps", None)
    space["random_seed"] = seed
    if "early_stop_patience_steps" in space:
        space["early_stop_patience_steps"] = -1
    return space
