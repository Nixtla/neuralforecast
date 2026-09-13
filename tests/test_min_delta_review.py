"""Offline replay is diagnostic and does not change training policy."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "min_delta_review",
    Path(__file__).parents[1] / "experiments/commodity_sota/analyze_min_delta.py",
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_replay_zero_delta_matches_strict_patience():
    assert review.replay(
        [(10, 2.0), (20, 1.0), (30, 1.0), (40, 1.0), (50, 0.1)], 0.0, 2
    ) == {
        "stop_step": 40,
        "best_loss": 1.0,
    }


def test_replay_keeps_true_best_even_when_progress_is_not_material():
    assert review.replay(
        [(10, 1.0), (20, 0.9999), (30, 0.9998), (40, 0.9)], 0.001, 2
    ) == {
        "stop_step": 30,
        "best_loss": 0.9998,
    }


def test_replay_accumulates_small_improvements_against_reference():
    curve = [(10, 1.0), (20, 0.9994), (30, 0.9988), (40, 0.9982), (50, 0.9976)]
    assert review.replay(curve, 0.001, 2)["stop_step"] == 50
    with pytest.raises(ValueError):
        review.replay([(10, float("nan"))], 0.001)
