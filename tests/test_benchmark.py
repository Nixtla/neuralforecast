import numpy as np
import pytest
from ray import tune
from scipy.stats import qmc

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


def test_sobol_mapping_and_reproducibility():
    space = {
        "a": tune.choice([None, [1, 2], "robust"]),
        "b": tune.uniform(-2, 3),
        "c": tune.loguniform(1e-4, 1e-1),
        "d": tune.randint(-3, 5),
    }
    configs = sample_ray_configs(space)
    points = qmc.Sobol(d=4, scramble=True, seed=42).random_base2(4)[:10]
    assert len(configs) == 10
    for config, (a, b, c, d) in zip(configs, points):
        assert config["a"] == [None, [1, 2], "robust"][int(a * 3)]
        assert config["b"] == pytest.approx(-2 + 5 * b)
        assert config["c"] == pytest.approx(10 ** (-4 + 3 * c))
        assert config["d"] == -3 + int(d * 8)
        assert isinstance(config["d"], int)
    assert configs == sample_ray_configs(dict(reversed(list(space.items()))))
    assert configs == sample_ray_configs(space)
    assert configs != sample_ray_configs(space, seed=43)


def test_sobol_nested_values_and_independent_copies():
    space = {
        "nested": [{"x": tune.randint(2, 3)}, (tune.choice([[4]]),)],
        "fixed": [1, 2],
    }
    configs = sample_ray_configs(space, n=3)
    assert configs == [{"nested": [{"x": 2}, ([4],)], "fixed": [1, 2]}] * 3
    configs[0]["nested"][1][0].append(5)
    configs[0]["fixed"].append(3)
    assert configs[1]["nested"][1][0] == [4]
    assert space["fixed"] == configs[1]["fixed"] == [1, 2]
    fixed = sample_ray_configs({"x": [1]}, n=2)
    fixed[0]["x"].append(2)
    assert fixed[1] == {"x": [1]}


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_sobol_rejects_invalid_count(n):
    with pytest.raises(ValueError, match="n must be positive"):
        sample_ray_configs({}, n=n)


@pytest.mark.parametrize(
    "domain",
    [
        tune.sample_from(lambda spec: 1),
        tune.randn(),
        tune.quniform(0, 1, 0.1),
        tune.grid_search([1, 2]),
        tune.choice([{"x": tune.uniform(0, 1)}]),
    ],
)
def test_sobol_rejects_unsupported_domains_with_path(domain):
    with pytest.raises(ValueError, match="nested"):
        sample_ray_configs({"nested": domain})


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
