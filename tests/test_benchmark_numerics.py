import types

import pytest
import torch
from ray import tune

from neuralforecast.benchmark import sample_ray_configs
from neuralforecast.benchmark_numerics import (
    FiniteTraining,
    guard_config,
    numerical_search_space,
)


@pytest.mark.parametrize("name", ["NBEATS", "NBEATSx", "Autoformer", "xLSTM"])
def test_sampled_rates_cannot_reenter_failed_high_rate_region(name):
    space = numerical_search_space(name, {"learning_rate": tune.loguniform(1e-4, 1e-1)})
    configs = sample_ray_configs(space, n=100, seed=42)
    assert all(1e-4 <= c["learning_rate"] <= 1e-3 for c in configs)


def test_explicit_rates_and_unaffected_models_are_preserved():
    assert (
        numerical_search_space("NBEATS", {"learning_rate": 0.002})["learning_rate"]
        == 0.002
    )
    rate = tune.loguniform(1e-4, 1e-1)
    assert (
        numerical_search_space("DeepAR", {"learning_rate": rate})["learning_rate"]
        is rate
    )
    assert guard_config("DeepAR", {}) == {}
    assert guard_config("xLSTM", {})["numerical_stability"] is True
    assert (
        guard_config("FEDformer", {})["_benchmark_numerics"]["gradient_clip_val"] == 1
    )


def test_nonfinite_gradients_rejected_before_optimizer_mutation():
    model = torch.nn.Linear(1, 1)
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    before = model.weight.detach().clone()
    with pytest.raises(FloatingPointError, match="weight.*step 12"):
        FiniteTraining().on_before_optimizer_step(
            types.SimpleNamespace(global_step=12), model, None
        )
    torch.testing.assert_close(model.weight, before)


def test_forward_diagnostic_names_module_and_removes_hooks():
    model = torch.nn.Sequential(torch.nn.Linear(1, 1))
    callback = FiniteTraining(check_forward=True)
    trainer = types.SimpleNamespace(global_step=7)
    callback.on_fit_start(trainer, model)
    with pytest.raises(FloatingPointError, match="output at 0, optimizer step 7"):
        model(torch.tensor([[float("nan")]]))
    callback.teardown(trainer, model, "fit")
    assert not callback.handles
    assert not model[0]._forward_hooks
