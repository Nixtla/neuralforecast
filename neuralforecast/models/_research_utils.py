"""Shared validation helpers for official-source trainable model adapters."""

import torch


def _positive(**values):
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")


def _full_windows(kwargs):
    options = dict(kwargs)
    threshold = options.get("training_data_availability_threshold", 1.0)
    if threshold != 1.0 and threshold != [1.0, 1.0]:
        raise ValueError(
            "Official-source training requires fully observed history and forecast windows."
        )
    options["training_data_availability_threshold"] = 1.0
    if options.get("start_padding_enabled", False):
        raise ValueError("Official-source training does not support start padding.")
    return options


def _no_sample_weights(batch):
    if "sample_weight" in batch["temporal_cols"]:
        raise ValueError(
            "This model's auxiliary/flow objective does not support sample_weight."
        )


def _finite_loss(loss):
    if (
        not isinstance(loss, torch.Tensor)
        or loss.ndim != 0
        or not torch.isfinite(loss)
    ):
        raise ValueError("Official training objective must be a finite scalar tensor.")
    return loss
