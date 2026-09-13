"""Opt-in safeguards for new commodity benchmark configurations."""

import torch
from pytorch_lightning import Callback

NUMERICS_VERSION = 1
RATE_LIMITED_MODELS = frozenset({"NBEATS", "NBEATSx", "Autoformer", "xLSTM"})
GUARDED_MODELS = RATE_LIMITED_MODELS | {"FEDformer"}


def numerical_search_space(name, space):
    """Restrict unstable sampled learning rates; preserve explicit fixed rates.

    Args:
        name: Forecasting model name.
        space: Existing Ray search space.

    Returns:
        A copy with sampled learning rates capped at 0.001 for affected models.
    """
    from ray import tune
    from ray.tune.search.sample import Float

    space = dict(space)
    rate = space.get("learning_rate")
    if name in RATE_LIMITED_MODELS and isinstance(rate, Float):
        lower, upper = rate.lower, min(rate.upper, 1e-3)
        if lower > upper:
            raise ValueError(f"{name}: learning-rate range exceeds stability cap")
        space["learning_rate"] = (
            lower if lower == upper else tune.loguniform(lower, upper)
        )
    return space


def guard_config(name, config):
    """Mark a newly sampled configuration with versioned numerical safeguards."""
    config = dict(config)
    if name in GUARDED_MODELS:
        config["_benchmark_numerics"] = {
            "version": NUMERICS_VERSION,
            "gradient_clip_val": 1.0,
        }
    if name == "xLSTM":
        config["numerical_stability"] = True
    return config


def _finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, (tuple, list)):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return True


class FiniteTraining(Callback):
    """Stop before an optimizer update when forward values or gradients diverge.

    Args:
        check_forward: Also inspect module outputs, useful for FEDformer diagnosis.
    """

    def __init__(self, check_forward=False):
        self.check_forward = check_forward
        self.handles = []

    def on_fit_start(self, trainer, pl_module):
        if self.check_forward:
            for name, module in pl_module.named_modules():

                def check(module, inputs, output, name=name):
                    if not _finite(output):
                        raise FloatingPointError(
                            f"Non-finite forward output at {name or 'model'}, "
                            f"optimizer step {trainer.global_step}"
                        )

                self.handles.append(module.register_forward_hook(check))

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        for name, parameter in pl_module.named_parameters():
            if parameter.grad is not None and not _finite(parameter.grad):
                raise FloatingPointError(
                    f"Non-finite gradient at {name}, optimizer step {trainer.global_step}"
                )

    def teardown(self, trainer, pl_module, stage):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
