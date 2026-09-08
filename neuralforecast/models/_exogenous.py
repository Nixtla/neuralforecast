"""Shared NeuralForecast window validation for the exogenous model adapters."""

import math
from typing import Any, Optional

import torch

from ..common._base_model import BaseModel
from ..losses.pytorch import MAE, MSE, quantiles_to_outputs


class _NativeQuantileMAE(MAE):
    """Inference metadata for pretrained backends that natively return quantiles."""

    def __init__(self):
        super().__init__()
        self.quantiles = None

    def update_quantile(self, q=None):
        if q is None:
            self.quantiles = None
            self.outputsize_multiplier = 1
            self.output_names = [""]
            return
        try:
            quantiles = [float(value) for value in q]
        except (TypeError, ValueError) as exc:
            raise ValueError("quantiles must be a nonempty sequence of numbers in (0, 1).") from exc
        if not quantiles or any(not math.isfinite(value) or not 0 < value < 1 for value in quantiles):
            raise ValueError("quantiles must be a nonempty sequence of numbers in (0, 1).")
        quantiles, names = quantiles_to_outputs(quantiles)
        self.quantiles = list(quantiles)
        self.outputsize_multiplier = 1 + len(self.quantiles)
        self.output_names = [""] + names


class ExogenousModel(BaseModel):
    """Point-forecast base; the standard NeuralForecast trainer remains in charge.

    Only numerical historical/future covariates and MAE/MSE are supported.
    Static/categorical inputs and probabilistic loss heads are rejected rather
    than silently ignored. Additional keyword arguments are BaseModel options.
    """

    EXOGENOUS_HIST = True
    EXOGENOUS_FUTR = True
    EXOGENOUS_STAT = False
    EXOGENOUS_CAT = False
    MULTIVARIATE = False
    RECURRENT = False

    def __init__(
        self,
        h: int,
        input_size: int,
        hist_exog_list=None,
        futr_exog_list=None,
        stat_exog_list=None,
        loss=None,
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        **kwargs,
    ):
        if not isinstance(h, int) or isinstance(h, bool) or h < 1:
            raise ValueError("h must be a positive integer.")
        if not isinstance(input_size, int) or isinstance(input_size, bool) or input_size < 2:
            raise ValueError("input_size must be an integer >= 2.")
        hist = list(hist_exog_list or [])
        futr = list(futr_exog_list or [])
        if len(set(hist + futr)) != len(hist + futr):
            raise ValueError("Historical and future covariate names must be distinct.")
        for name in hist + futr:
            if not isinstance(name, str) or not name:
                raise ValueError("Covariate names must be nonempty strings.")
        loss = MAE() if loss is None else loss
        valid_loss = loss if valid_loss is None else valid_loss
        if not isinstance(loss, (MAE, MSE)) or not isinstance(valid_loss, (MAE, MSE)):
            raise ValueError("These adapters currently support MAE() and MSE() only.")
        options: dict[str, Any] = dict(
            val_check_steps=100,
            batch_size=32,
            valid_batch_size=None,
            windows_batch_size=32,
            inference_windows_batch_size=32,
            start_padding_enabled=False,
            num_lr_decays=-1,
            scaler_type="identity",
        )
        options.update(kwargs)
        if options.get("exclude_insample_y", False):
            raise ValueError("These models require the historical target.")
        super().__init__(
            h=h,
            input_size=input_size,
            hist_exog_list=hist,
            futr_exog_list=futr,
            stat_exog_list=stat_exog_list,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            **options,
        )

    def _inputs(self, windows_batch):
        """Return only observed history and declared future covariates, never y_future."""
        y = windows_batch["insample_y"]
        if y.ndim != 3 or y.shape[1:] != (self.input_size, 1):
            raise ValueError(f"insample_y must have shape [batch, {self.input_size}, 1].")
        mask = windows_batch.get("insample_mask")
        mask = torch.ones_like(y, dtype=torch.bool) if mask is None else mask.bool()
        if mask.shape != y.shape or not mask.any(dim=1).all():
            raise ValueError("Each history must contain at least one observed target.")
        if not torch.isfinite(y[mask]).all():
            raise ValueError("Observed target values must be finite.")
        y = torch.where(mask, y, torch.zeros_like(y))
        exog = []
        for key, size, length in (
            ("hist_exog", self.hist_exog_size, self.input_size),
            ("futr_exog", self.futr_exog_size, self.input_size + self.h),
        ):
            value = windows_batch.get(key)
            if not size:
                exog.append(None)
                continue
            if value is None or value.shape != (y.shape[0], length, size):
                raise ValueError(f"{key} must have shape [batch, {length}, {size}].")
            observed = mask
            if key == "futr_exog":
                observed = torch.cat((mask, mask.new_ones(y.shape[0], self.h, 1)), dim=1)
            if not torch.isfinite(value[observed.expand_as(value)]).all():
                raise ValueError(f"Observed {key} values must be finite.")
            exog.append(torch.where(observed, value, torch.zeros_like(value)))
        return y, mask, exog[0], exog[1]

    @staticmethod
    def _complete_history(mask):
        if not mask.all():
            raise ValueError("This adapter requires complete history; padding/missing targets are unsupported.")

    def _point_output(self, output, y):
        output = torch.as_tensor(output, device=y.device, dtype=y.dtype)
        if output.ndim == 2:
            output = output.unsqueeze(-1)
        if output.shape != (y.shape[0], self.h, 1):
            raise ValueError(f"Backend returned unexpected forecast shape {tuple(output.shape)}.")
        if not torch.isfinite(output).all():
            raise ValueError("Backend returned non-finite forecasts.")
        return output


class PretrainedExogenousModel(ExogenousModel):
    """Inference-only adapter. fit validates inputs but does not fine-tune weights.

    Checkpoints save the model ID/revision, NOT the external weights. Keep a
    local Hugging Face cache or pass a local model directory for offline reload.
    Optional packages and weights are loaded on first prediction, not import.
    """

    DEFAULT_MODEL_ID = ""
    NATIVE_QUANTILES = False

    def __init__(
        self,
        h: int,
        input_size: int,
        model_id: Optional[str] = None,
        revision: Optional[str] = None,
        backend_device: str = "cpu",
        num_samples: int = 100,
        max_steps: int = 0,
        **kwargs,
    ):
        if max_steps != 0:
            raise ValueError("This pretrained adapter is inference-only; use max_steps=0.")
        if kwargs.get("early_stop_patience_steps", -1) > 0:
            raise ValueError("Early stopping is unavailable for inference-only adapters.")
        if not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer.")
        if self.NATIVE_QUANTILES and kwargs.get("loss") is None:
            native_loss = _NativeQuantileMAE()
            kwargs["loss"] = native_loss
            if kwargs.get("valid_loss") is None:
                kwargs["valid_loss"] = native_loss
        super().__init__(h=h, input_size=input_size, max_steps=0, **kwargs)
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.revision = revision
        self.backend_device = str(torch.device(backend_device))
        self.num_samples = num_samples
        # External modules must not enter NF's state_dict: reload from ID/revision.
        self.__dict__["_backend"] = None

    def _get_backend(self):
        if self.__dict__["_backend"] is None:
            self.__dict__["_backend"] = self._load_backend()
        return self.__dict__["_backend"]

    def _hub_kwargs(self):
        return {"revision": self.revision} if self.revision is not None else {}

    def _fit(
        self, dataset, batch_size, valid_batch_size=1024, val_size=0,
        test_size=0, random_seed=None, shuffle_train=True, distributed_config=None,
    ):
        if distributed_config is not None:
            raise ValueError("Distributed fitting is unavailable for pretrained adapters.")
        self._check_exog(dataset)
        self._restart_seed(random_seed)
        self.val_size, self.test_size = val_size, test_size
        self.metrics = {}
        return self

    def predict(self, *args, **kwargs):
        if kwargs.get("explainer_config") is not None:
            raise ValueError("Gradient explanations are unavailable for pretrained adapters.")
        if kwargs.get("quantiles") is not None and not isinstance(self.loss, _NativeQuantileMAE):
            raise ValueError("This adapter exposes point forecasts only.")
        return super().predict(*args, **kwargs)
