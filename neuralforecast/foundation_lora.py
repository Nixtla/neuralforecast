"""LoRA fine-tuning protocol for foundation adapters with verified native support."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from ray import tune

from .inference_tuning import get_inference_tuning_config

__all__ = [
    "FOUNDATION_LORA_MODELS",
    "fit_foundation_lora",
    "get_foundation_lora_config",
]

FOUNDATION_LORA_MODELS = ("Chronos2",)

_CHRONOS2_TARGET_MODULES = [
    "self_attention.q",
    "self_attention.v",
    "self_attention.k",
    "self_attention.o",
    "output_patch_embedding.output_layer",
]


def get_foundation_lora_config(model, h, fixed=None, backend="ray"):
    """Return the benchmark LoRA search space for a supported foundation model."""
    name = model if isinstance(model, str) else model.__name__
    if name not in FOUNDATION_LORA_MODELS:
        raise ValueError(
            f"LoRA is unavailable for {name!r}; supported models: "
            f"{FOUNDATION_LORA_MODELS}."
        )
    if backend != "ray":
        raise ValueError("foundation LoRA benchmark tuning currently uses backend='ray'.")
    space = get_inference_tuning_config(name, h=h, fixed=fixed, backend="ray")
    space.pop("max_steps", None)
    space.update(
        {
            "learning_rate": tune.loguniform(1e-5, 3e-4),
            "lora_r": tune.choice([4, 8, 16]),
            "lora_alpha": tune.choice([8, 16, 32]),
            "lora_dropout": tune.choice([0.0, 0.05, 0.1]),
            "lora_batch_size": tune.choice([8, 16, 32]),
        }
    )
    return space


def _require_peft():
    if importlib.util.find_spec("peft") is None:
        raise ImportError(
            "Chronos2 LoRA requires the optional 'peft' package. "
            "Install peft in the benchmark environment."
        )


def fit_foundation_lora(model_cls, config, train, h, steps, output_dir):
    """Fine-tune one Chronos2 adapter with the official pipeline LoRA path."""
    if model_cls.__name__ not in FOUNDATION_LORA_MODELS:
        raise ValueError(f"LoRA is unavailable for {model_cls.__name__}.")
    if steps < 1:
        raise ValueError("steps must be positive.")
    _require_peft()

    config = dict(config)
    learning_rate = config.pop("learning_rate")
    rank = config.pop("lora_r")
    alpha = config.pop("lora_alpha")
    dropout = config.pop("lora_dropout")
    batch_size = config.pop("lora_batch_size")
    config.pop("max_steps", None)
    adapter = model_cls(**config)
    pipeline = adapter._get_backend()
    if not callable(getattr(pipeline, "fit", None)):
        raise RuntimeError("Chronos2 backend does not expose its official fit API.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(train["y"], dtype=np.float32)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("LoRA training target must be a finite one-dimensional series.")
    adapter.__dict__["_backend"] = pipeline.fit(
        inputs=[values],
        prediction_length=h,
        validation_inputs=None,
        finetune_mode="lora",
        lora_config={
            "r": rank,
            "lora_alpha": alpha,
            "lora_dropout": dropout,
            "target_modules": _CHRONOS2_TARGET_MODULES,
        },
        context_length=adapter.input_size,
        learning_rate=learning_rate,
        num_steps=steps,
        batch_size=batch_size,
        output_dir=output_dir,
        remove_printer_callback=True,
    )
    return adapter
