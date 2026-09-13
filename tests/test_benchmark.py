import numpy as np
import pytest
from ray import tune

from neuralforecast.benchmark import (
    SHPlan,
    benchmark_search_space,
    expanding_folds,
    forecast_metrics,
    pooled_rmse,
    rank_key,
    representative_folds,
    restrict_input_size,
    sample_ray_configs,
)


def test_expanding_fold_boundaries_and_representatives():
    folds = expanding_folds(40, h=4, min_train=20, step_size=1)
    assert len(folds) == 17
    assert folds[0].train_slice == slice(0, 20)
    assert folds[-1].valid_slice == slice(36, 40)
    first, middle, last = representative_folds(folds)
    assert (first.index, middle.index, last.index) == (0, 8, 16)


def test_pooled_rmse_uses_all_forecast_points():
    actual = [[0.0, 0.0], [0.0, 0.0]]
    prediction = [[0.0, 0.0], [2.0, 2.0]]
    assert pooled_rmse(actual, prediction) == pytest.approx(np.sqrt(2.0))


def test_benchmark_space_fixes_training_protocol():
    space = benchmark_search_space(
        {
            "input_size": tune.choice([16, 64, 256]),
            "max_steps": tune.choice([500, 1000]),
            "random_seed": tune.randint(1, 20),
            "early_stop_patience_steps": 5,
        }
    )
    space = restrict_input_size(space, 64)
    configs = sample_ray_configs(space, n=10, seed=42)
    assert {config["random_seed"] for config in configs} == {42}
    assert {config["early_stop_patience_steps"] for config in configs} == {-1}
    assert "max_steps" not in configs[0]
    assert {config["input_size"] for config in configs} <= {16, 64}


def test_sh_plan_matches_benchmark_contract():
    plan = SHPlan()
    assert plan.budgets == (100, 250, 500)
    assert plan.survivors == (10, 5, 1)
    assert rank_key(1.0, [1.0, 1.1, 0.9]) < rank_key(2.0, [2.0, 2.0, 2.0])


def test_five_metrics_use_fixed_origins_and_report_zero_mape_denominator():
    result = forecast_metrics(
        [2.0, 0.0, -2.0, 3.0], [3.0, 1.0, -1.0, 3.0], [1.0, 1.0, -1.0, 3.0]
    )
    assert result == pytest.approx(
        {
            "mae": 0.75,
            "mse": 0.75,
            "rmse": np.sqrt(0.75),
            "mape_pct": 100 / 3,
            "mape_n": 3,
            "da_pct": 50.0,
        }
    )
    assert forecast_metrics([0.0], [1.0], [0.0])["mape_pct"] is None


@pytest.mark.parametrize(
    "actual,prediction,origin",
    [
        ([], [], []),
        ([1.0], [1.0, 2.0], [1.0]),
        ([1.0], [float("nan")], [1.0]),
    ],
)
def test_metrics_reject_misaligned_or_nonfinite_values(actual, prediction, origin):
    with pytest.raises(ValueError):
        forecast_metrics(actual, prediction, origin)
