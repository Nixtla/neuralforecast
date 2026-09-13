"""LoRA fine-tuning protocols for foundation adapters with verified support."""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from ray import tune

from .inference_tuning import get_inference_tuning_config
from .losses.pytorch import MAE
from .benchmark_stopping import rng_state, restore_rng

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

    def __post_init__(self):
        self.loss = MAE()
        self.test_size = 0
        self.val_size = 0

    def set_test_size(self, test_size):
        self.test_size = test_size

    def predict(self, dataset, test_size=None, step_size=1, **kwargs):
        if step_size != 1:
            raise ValueError("TimesFM LoRA benchmark prediction requires step_size=1.")
        h = self.test_size if test_size is None else test_size
        values = dataset.temporal[:, dataset.y_idx].detach().cpu().numpy()
        if h < 1 or len(values) < self.input_size + h:
            raise ValueError("TimesFM LoRA prediction history is too short.")
        context = torch.as_tensor(
            values[-h - self.input_size : -h],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            output = self.model(past_values=context)
        prediction = output.mean_predictions[0, :h].float().cpu().numpy()
        if len(prediction) != h or not np.isfinite(prediction).all():
            raise RuntimeError("TimesFM LoRA returned invalid predictions.")
        return prediction.reshape(-1, 1)


def get_foundation_lora_config(model, h, fixed=None, backend="ray"):
    """Return a benchmark LoRA search space for a supported foundation model."""
    name = model if isinstance(model, str) else model.__name__
    if name not in FOUNDATION_LORA_MODELS:
        raise ValueError(
            f"LoRA is unavailable for {name!r}; supported models: "
            f"{FOUNDATION_LORA_MODELS}."
        )
    if backend != "ray":
        raise ValueError(
            "foundation LoRA benchmark tuning currently uses backend='ray'."
        )
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


def _seed():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@contextmanager
def _preserve_precision():
    """Restore global Torch precision changed by external HF training."""
    objects = [torch.backends, torch.backends.cuda.matmul, torch.backends.cudnn]
    for name in ("conv", "rnn"):
        if hasattr(torch.backends.cudnn, name):
            objects.append(getattr(torch.backends.cudnn, name))
    saved = [
        (obj, obj.fp32_precision) for obj in objects if hasattr(obj, "fp32_precision")
    ]
    try:
        yield
    finally:
        for obj, value in saved:
            obj.fp32_precision = value


def _fit_chronos2(
    model_cls,
    config,
    params,
    train,
    h,
    steps,
    output_dir,
    metrics_callback=None,
    validation=None,
    stopping=None,
    checkpoint=None,
    schedule_steps=None,
):
    _require_package("peft")
    _seed()
    adapter = model_cls(**config)
    pipeline = adapter._get_backend()
    if not callable(getattr(pipeline, "fit", None)):
        raise RuntimeError("Chronos2 backend does not expose its official fit API.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(train["y"], dtype=np.float32)
    callbacks = []
    if metrics_callback is not None:
        from transformers import TrainerCallback

        class MetricsCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                metrics_callback(
                    {
                        "train/global_step": state.global_step,
                        **{f"train/{k}": v for k, v in (logs or {}).items()},
                    }
                )

        callbacks.append(MetricsCallback())
    stopper = None
    if validation is not None:
        from transformers import TrainerCallback
        from chronos.chronos2.pipeline import Chronos2Pipeline
        from .benchmark_stopping import ValidationStopper

        stopper = ValidationStopper(
            output_dir,
            metrics_callback=metrics_callback,
            adapters_only=True,
            **(stopping or {}),
        )
        target = np.asarray(validation["y"], dtype=np.float32)

        class ValidationCallback(TrainerCallback):
            def on_step_end(self, args, state, control, model=None, **kwargs):
                if state.global_step % stopper.interval:
                    return control
                was_training = model.training
                model.eval()
                with torch.no_grad(), torch.random.fork_rng():
                    backend = Chronos2Pipeline(model=model)
                    forecast = backend.predict(
                        [values[-adapter.input_size :]],
                        prediction_length=h,
                        context_length=adapter.input_size,
                        cross_learning=False,
                    )[0]
                    median = int(np.flatnonzero(np.isclose(backend.quantiles, 0.5))[0])
                    prediction = forecast[0, median, :h].float().cpu().numpy()
                model.train(was_training)
                loss = float(np.mean((prediction - target) ** 2))
                stopped = stopper.update(model, state.global_step, loss)
                control.should_training_stop = control.should_training_stop or stopped
                return control

        callbacks.append(ValidationCallback())
    from chronos.chronos2 import trainer as trainer_module

    class ContinuingTrainer(trainer_module.Chronos2Trainer):
        def create_scheduler(self, num_training_steps, optimizer=None):
            return super().create_scheduler(schedule_steps or steps, optimizer)

        def train(self, *args, **kwargs):
            if checkpoint:
                if stopper:
                    stopper.load_state_dict(
                        torch.load(
                            Path(checkpoint) / "benchmark_stopper.pt",
                            map_location="cpu",
                            weights_only=False,
                        )
                    )
                kwargs["resume_from_checkpoint"] = checkpoint
            result = super().train(*args, **kwargs)
            # Save last training state before restoring weights for evaluation.
            self._save_checkpoint(self.model, trial=None)
            path = output_dir / f"checkpoint-{self.state.global_step}"
            adapter.resume_checkpoint = str(path)
            if stopper:
                torch.save(stopper.state_dict(), path / "benchmark_stopper.pt")
                stopper.restore(self.model)
            return result

    with (
        _preserve_precision(),
        patch.object(trainer_module, "Chronos2Trainer", ContinuingTrainer),
    ):
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
            seed=42,
            data_seed=42,
            report_to="none",
            callbacks=callbacks,
            save_only_model=False,
            logging_steps=10,
        )
    if stopper:
        adapter.early_stopping_info = stopper.summary()
    return adapter


def _fit_timesfm(
    config,
    params,
    train,
    h,
    steps,
    metrics_callback=None,
    validation=None,
    stopping=None,
    output_dir=None,
    checkpoint=None,
    schedule_steps=None,
):
    _require_package("peft")
    _require_package("transformers")
    from peft import LoraConfig, get_peft_model
    from transformers import TimesFm2_5ModelForPrediction

    _seed()
    values = np.asarray(train["y"], dtype=np.float32)
    input_size = int(config.pop("input_size"))
    if len(values) < input_size + h:
        raise ValueError("TimesFM LoRA requires input_size + horizon training rows.")
    model_id = config.pop("model_id", "google/timesfm-2.5-200m-transformers")
    revision = config.pop("revision", None)
    device = str(
        torch.device(
            config.pop("backend_device", "cuda" if torch.cuda.is_available() else "cpu")
        )
    )
    config.pop("h", None)
    if config:
        raise ValueError(f"unsupported TimesFM LoRA fixed arguments: {sorted(config)}")

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    hub_kwargs = {"revision": revision} if revision is not None else {}
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        **hub_kwargs,
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=schedule_steps or steps
    )
    rng = np.random.default_rng(42)
    max_start = len(values) - input_size - h
    stopper = None
    if validation is not None:
        from .benchmark_stopping import ValidationStopper

        stopper = ValidationStopper(
            output_dir,
            metrics_callback=metrics_callback,
            adapters_only=True,
            **(stopping or {}),
        )
        target = np.asarray(validation["y"], dtype=np.float32)
    start_step = 0
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["weights"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        rng.bit_generator.state = state["sampling_rng"]
        restore_rng(state["rng"])
        start_step = state["global_step"]
        if stopper:
            stopper.load_state_dict(state["stopper"])
    last_step = start_step
    model.train()
    for step in range(start_step, steps):
        starts = rng.integers(0, max_start + 1, size=params["batch_size"])
        past = torch.as_tensor(
            np.stack([values[start : start + input_size] for start in starts]),
            device=device,
        )
        future = torch.as_tensor(
            np.stack(
                [
                    values[start + input_size : start + input_size + h]
                    for start in starts
                ]
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
        last_step = step + 1
        if metrics_callback is not None and (step == 0 or (step + 1) % 10 == 0):
            metrics_callback(
                {
                    "train/global_step": step + 1,
                    "train/loss": float(output.loss.detach()),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
        if stopper and (step + 1) % stopper.interval == 0:
            model.eval()
            with torch.no_grad(), torch.random.fork_rng():
                context = torch.as_tensor(
                    values[-input_size:], device=device
                ).unsqueeze(0)
                prediction = (
                    model(past_values=context)
                    .mean_predictions[0, :h]
                    .float()
                    .cpu()
                    .numpy()
                )
            model.train()
            if stopper.update(
                model, step + 1, float(np.mean((prediction - target) ** 2))
            ):
                break
    resume_path = Path(output_dir) / "resume.pt"
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    torch.save(
        {
            "weights": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
                if name in names
            },
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": rng_state(),
            "sampling_rng": rng.bit_generator.state,
            "global_step": last_step,
            "stopper": stopper.state_dict() if stopper else None,
        },
        resume_path,
    )
    if stopper:
        stopper.restore(model)
    model.eval()
    adapter = _TimesFMLoRA(model=model, input_size=input_size, device=device)
    adapter.resume_checkpoint = str(resume_path)
    if stopper:
        adapter.early_stopping_info = stopper.summary()
    return adapter


def fit_foundation_lora(
    model_cls,
    config,
    train,
    h,
    steps,
    output_dir,
    metrics_callback=None,
    validation=None,
    stopping=None,
    checkpoint=None,
    schedule_steps=None,
):
    """Fine-tune a supported foundation adapter.

    Args:
        model_cls: Chronos2 or TimesFM adapter class.
        config: Sampled model and LoRA settings.
        train: Training frame, excluding validation and test observations.
        h: Forecast horizon.
        steps: Maximum optimizer steps.
        output_dir: Checkpoint directory.
        metrics_callback: Optional benchmark-only metrics sink.
        validation: Optional held-out frame of exactly h observations.
        stopping: Optional interval and patience for validation-only selection.
        checkpoint: Locally produced full training state from a previous rung.
        schedule_steps: Fixed total scheduler horizon across cumulative rungs.

    Returns:
        Fitted adapter with best validation weights restored when requested.
    """
    name = model_cls.__name__
    if name not in FOUNDATION_LORA_MODELS:
        raise ValueError(f"LoRA is unavailable for {name}.")
    if steps < 1:
        raise ValueError("steps must be positive.")
    if schedule_steps is not None and schedule_steps < steps:
        raise ValueError("schedule_steps must cover the cumulative step budget")
    values = np.asarray(train["y"], dtype=np.float32)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(
            "LoRA training target must be a finite one-dimensional series."
        )
    if validation is not None:
        target = np.asarray(validation["y"], dtype=np.float32)
        if len(target) != h or not np.isfinite(target).all():
            raise ValueError("Validation must contain h finite targets")
        interval = (stopping or {}).get("interval", 10)
        if interval < 1 or steps % interval:
            raise ValueError("steps must be divisible by validation interval")
    config, params = _lora_params(config)
    if name == "Chronos2":
        return _fit_chronos2(
            model_cls,
            config,
            params,
            train,
            h,
            steps,
            output_dir,
            metrics_callback,
            validation,
            stopping,
            checkpoint,
            schedule_steps,
        )
    return _fit_timesfm(
        config,
        params,
        train,
        h,
        steps,
        metrics_callback,
        validation,
        stopping,
        output_dir,
        checkpoint,
        schedule_steps,
    )


def predict_foundation_lora(fitted, dataset, h):
    """Return point predictions from a fitted LoRA protocol."""
    fitted.set_test_size(h)
    values = fitted.predict(dataset, test_size=h, step_size=1)
    output_size = len(fitted.loss.output_names)
    prediction = np.asarray(values).reshape(-1, output_size)[:, 0]
    if len(prediction) != h or not np.isfinite(prediction).all():
        raise RuntimeError("foundation LoRA returned invalid predictions.")
    return prediction
