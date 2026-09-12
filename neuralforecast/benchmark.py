"""Utilities for chronological two-phase forecasting benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence

import numpy as np
from ray import tune
from ray.tune.search.variant_generator import generate_variants


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

    budgets: tuple[int, ...] = (125, 250, 500, 1000)
    survivors: tuple[int, ...] = (10, 5, 2, 1)

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
    y_hat = np.concatenate(
        [np.asarray(values, dtype=float) for values in prediction]
    )
    if y.shape != y_hat.shape or not y.size:
        raise ValueError("actual and prediction must have the same non-empty shape.")
    if not np.isfinite(y).all() or not np.isfinite(y_hat).all():
        return float("inf")
    return sqrt(float(np.mean(np.square(y - y_hat))))


def fold_rmse(actual: Sequence[float], prediction: Sequence[float]) -> float:
    """RMSE for one fold."""
    return pooled_rmse([actual], [prediction])


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
    """Resolve ``ray.tune`` domains without launching training actors."""
    if n < 1:
        raise ValueError("n must be positive.")
    random_state = np.random.RandomState(seed)
    configs = []
    for _ in range(n):
        _, config = next(generate_variants(space, random_state=random_state))
        configs.append(config)
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