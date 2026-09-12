"""LoRA fine-tuning protocols for foundation adapters with verified support."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ray import tune

from .inference_tuning import get_inference_tuning_config

__all__ = [
    "FOUNDATION_LORA_MODELS",
    "fit_foundation_lora",
    "get_foundation_lora_config",
    "predict_foundation_lora",
]

FOUNDATION_LORA_MODELS = ("Chronos2", "TimesFM")

_CHRONOS2_TARGET_MODULES = [
    "self_attention.q",
    "self_attention.v",
    "self_attention.k",
    "self_attention.o",
    "output_patch_embedding.output_layer",
]


@dataclass
class _TimesFMLoRA:
    model: object
    input_size: int
    device: str


def get_foundation_lora_config(model, h, fixed=None, backend="ray"):
    """Return a benchmark LoRA search space for a supported foundation model."""
    name = model if isinstance(model, str) else model.__name__
    if name not in FOUNDATION_LORA_MODELS:
        raise ValueError(
            f"LoRA is unavailable for {name!r}; supported models: "
            f"{FOUNDATION_LORA_MODELS}."
        )
    if backend != "ray":
        raise ValueError("foundation LoRA benchmark tuning currently uses backend='ray'.")
    fixed = dict(fixed or {})
    if name == "TimesFM" and "model_id" not in fixed:
        fixed["model_id"] = "google/timesfm-2.5-200m-transformers"
    space = get_inference_tuning_config(name, h=h, fixed=fixed, backend="ray")
    space.pop("max_steps", None)
    if name == "TimesFM":
        space.pop("xreg_ridge", None)
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


def _require_package(name):
    if importlib.util.find_spec(name) is None:
        raise ImportError(
            f"foundation LoRA requires the optional {name!r} package. "
            f"Install {name} in the benchmark environment."
        )


def _lora_params(config):
    config = dict(config)
    params = {
        "learning_rate": config.pop("learning_rate"),
        "rank": config.pop("lora_r"),
        "alpha": config.pop("lora_alpha"),
        "dropout": config.pop("lora_dropout"),
        "batch_size": config.pop("lora_batch_size"),
    }
    config.pop("max_steps", None)
    return config, params


def _fit_chronos2(model_cls, config, params, train, h, steps, output_dir):
    _require_package("peft")
    adapter = model_cls(**config)
    pipeline = adapter._get_backend()
    if not callable(getattr(pipeline, "fit", None)):
        raise RuntimeError("Chronos2 backend does not expose its official fit API.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(train["y"], dtype=np.float32)
    adapter.__dict__["_backend"] = pipeline.fit(
        inputs=[values],
        prediction_length=h,
        validation_inputs=None,
        finetune_mode="lora",
        lora_config={
            "r": params["rank"],
            "lora_alpha": params["alpha"],
            "lora_dropout": params["dropout"],
            "target_modules": _CHRONOS2_TARGET_MODULES,
        },
        context_length=adapter.input_size,
        learning_rate=params["learning_rate"],
        num_steps=steps,
        batch_size=params["batch_size"],
        output_dir=output_dir,
        remove_printer_callback=True,
    )
    return adapter


def _fit_timesfm(config, params, train, h, steps):
    _require_package("peft")
    _require_package("transformers")
    from peft import LoraConfig, get_peft_model
    from transformers import TimesFm2_5ModelForPrediction

    values = np.asarray(train["y"], dtype=np.float32)
    input_size = int(config.pop("input_size"))
    if len(values) < input_size + h:
        raise ValueError("TimesFM LoRA requires input_size + horizon training rows.")
    model_id = config.pop("model_id", "google/timesfm-2.5-200m-transformers")
    device = str(torch.device(config.pop("backend_device", "cuda" if torch.cuda.is_available() else "cpu")))
    config.pop("h", None)
    config.pop("revision", None)
    if config:
        raise ValueError(f"unsupported TimesFM LoRA fixed arguments: {sorted(config)}")

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
    )
    input_size = min(input_size, int(model.config.context_length))
    model = get_peft_model(
        model,
        LoraConfig(
            r=params["rank"],
            lora_alpha=params["alpha"],
            target_modules="all-linear",
            lora_dropout=params["dropout"],
            bias="none",
        ),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    rng = np.random.default_rng(42)
    max_start = len(values) - input_size - h
    model.train()
    for _ in range(steps):
        starts = rng.integers(0, max_start + 1, size=params["batch_size"])
        past = torch.as_tensor(
            np.stack([values[start : start + input_size] for start in starts]),
            device=device,
        )
        future = torch.as_tensor(
            np.stack(
                [values[start + input_size : start + input_size + h] for start in starts]
            ),
            device=device,
        )
        output = model(
            past_values=past,
            future_values=future,
            forecast_context_len=input_size,
        )
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError("TimesFM LoRA returned a non-finite training loss.")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    model.eval()
    return _TimesFMLoRA(model=model, input_size=input_size, device=device)


def fit_foundation_lora(model_cls, config, train, h, steps, output_dir):
    """Fine-tune one supported foundation adapter through its verified LoRA path."""
    name = model_cls.__name__
    if name not in FOUNDATION_LORA_MODELS:
        raise ValueError(f"LoRA is unavailable for {name}.")
    if steps < 1:
        raise ValueError("steps must be positive.")
    values = np.asarray(train["y"], dtype=np.float32)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("LoRA training target must be a finite one-dimensional series.")
    config, params = _lora_params(config)
    if name == "Chronos2":
        return _fit_chronos2(model_cls, config, params, train, h, steps, output_dir)
    return _fit_timesfm(config, params, train, h, steps)


def predict_foundation_lora(fitted, dataset, h):
    """Return point predictions from a fitted LoRA protocol."""
    if isinstance(fitted, _TimesFMLoRA):
        values = dataset.temporal[:, dataset.y_idx].detach().cpu().numpy()
        if len(values) < fitted.input_size + h:
            raise ValueError("TimesFM LoRA prediction history is too short.")
        context = torch.as_tensor(
            values[-h - fitted.input_size : -h],
            dtype=torch.float32,
            device=fitted.device,
        ).unsqueeze(0)
        with torch.no_grad():
            output = fitted.model(past_values=context)
        prediction = output.mean_predictions[0, :h].float().cpu().numpy()
        if len(prediction) != h or not np.isfinite(prediction).all():
            raise RuntimeError("TimesFM LoRA returned invalid predictions.")
        return prediction

    fitted.set_test_size(h)
    values = fitted.predict(dataset, test_size=h, step_size=1)
    output_size = len(fitted.loss.output_names)
    prediction = np.asarray(values).reshape(-1, output_size)[:, 0]
    if len(prediction) != h or not np.isfinite(prediction).all():
        raise RuntimeError("foundation LoRA returned invalid predictions.")
    return prediction
