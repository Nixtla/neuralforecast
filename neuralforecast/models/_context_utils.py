"""Shared validation helpers for context-conditioned model adapters."""

from pathlib import Path

import torch


def _positive(**values):
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")


def _options(kwargs):
    if kwargs.get("scaler_type", "identity") != "identity":
        raise ValueError(
            "Use identity scaling; embeddings/context IDs must not be rescaled."
        )
    if kwargs.get("start_padding_enabled", False):
        raise ValueError("These integrations require complete history.")
    kwargs.setdefault("training_data_availability_threshold", [1.0, 1.0])
    return kwargs


def _local_path(path, directory=True):
    if not isinstance(path, str) or not path:
        raise ValueError(
            "An explicit local checkpoint path is required; no implicit downloads."
        )
    resolved = Path(path).expanduser().resolve()
    if not (resolved.is_dir() if directory else resolved.is_file()):
        raise FileNotFoundError(f"Missing local checkpoint: {resolved}")
    if directory and not any(resolved.glob("*.safetensors")):
        raise ValueError("The checkpoint directory must contain safetensors weights.")
    return str(resolved)


def _contexts(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("contexts must be a nonempty list of external descriptions.")
    if any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError("Every context must be a nonempty string.")
    return list(values)


def _context_indices(model, batch, y):
    value = batch.get("stat_exog")
    if value is None or value.shape != (len(y), 1):
        raise ValueError("stat_exog must contain one context_id per series/window.")
    ids = value[:, 0]
    if not torch.isfinite(ids).all() or (ids != ids.round()).any():
        raise ValueError("context_id must be a finite integer.")
    if (ids < 0).any() or (ids >= len(model.contexts)).any():
        raise ValueError("context_id must index the contexts list.")
    return ids.long()


def _context_schema(model):
    if model.stat_exog_size != 1:
        raise ValueError(
            "Provide exactly one stat_exog_list column containing context_id."
        )
