"""APT trainable official-source forecasting adapter."""

import torch

from ._exogenous import ExogenousModel
from ._research_source import source_module
from ._research_utils import _finite_loss, _full_windows, _no_sample_weights, _positive
from .dlinear import DLinear

__all__ = ["APT"]


class APT(ExogenousModel):
    """Official APT timestamp/prototype affine transform with NF DLinear.

    Args:
        h: Forecast horizon.
        input_size: Complete history length.
        source_dir: Checkout of blisky-li/APT.
        time_of_day_size: Calendar bins per day (24 for hourly data).
        warmup_steps: APT regularizer-only steps with the backbone frozen,
            followed by one forecast-loss step with the backbone frozen, then
            joint training. This translates the source's epoch phases to NF steps.
        station_lambda: Coefficient of official warmup regularizers.
        **kwargs: NF training options. Exactly two future columns, ordered as
            time_of_day and day_of_week, encoded as index/cardinality - 0.5.

    Uses the original dependent/shared-prototype mode without an extra RevIN
    module. APT performs the source's affine transform and its exact inverse;
    near-singular scales raise an error rather than silently changing the model.
    """

    EXOGENOUS_HIST = False

    def __init__(
        self,
        h,
        input_size,
        source_dir=None,
        timestamp_dim=16,
        timestamp_hidden=32,
        num_prototypes=8,
        top_k=3,
        time_of_day_size=24,
        warmup_steps=10,
        station_lambda=0.01,
        moving_avg_window=25,
        **kwargs,
    ):
        _positive(
            timestamp_dim=timestamp_dim,
            timestamp_hidden=timestamp_hidden,
            num_prototypes=num_prototypes,
            top_k=top_k,
            time_of_day_size=time_of_day_size,
            moving_avg_window=moving_avg_window,
        )
        if top_k > num_prototypes or moving_avg_window % 2 == 0:
            raise ValueError(
                "top_k <= num_prototypes and odd moving_avg_window are required."
            )
        if (
            not isinstance(warmup_steps, int)
            or isinstance(warmup_steps, bool)
            or warmup_steps < 0
        ):
            raise ValueError("warmup_steps must be a nonnegative integer.")
        if not 0 < station_lambda < float("inf"):
            raise ValueError("station_lambda must be positive and finite.")
        if kwargs.get("scaler_type", "identity") != "identity":
            raise ValueError(
                "APT's encoded timestamps require scaler_type='identity'."
            )
        super().__init__(h=h, input_size=input_size, **_full_windows(kwargs))
        if self.futr_exog_size != 2:
            raise ValueError(
                "APT requires [time_of_day, day_of_week] in futr_exog_list."
            )
        self.source_dir, self.time_of_day_size = source_dir, time_of_day_size
        self.warmup_steps, self.station_lambda = warmup_steps, station_lambda
        config = dict(
            tan_timestamp=["time_of_day", "day_of_week"],
            timestamp_dim=timestamp_dim,
            timestamp_hidden=timestamp_hidden,
            num_prototypes=num_prototypes,
            top_k=top_k,
            is_xformer=False,
            time_of_day_size=time_of_day_size,
            day_of_week_size=7,
            independent=False,
            enc_in=1,
            model_name="DLinear",
            normalization_name="None",
            datasets_name="NeuralForecast",
            use_tan=True,
        )
        self.affine = source_module(source_dir, "APT").APT(**config)
        self.backbone = DLinear(
            h=h,
            input_size=input_size,
            moving_avg_window=moving_avg_window,
            random_seed=self.random_seed,
        )
        self.__dict__["_orthogonality"] = source_module(
            source_dir,
            "APT_orthogonality",
        ).orthogonality
        self.__dict__["_balance_loss"] = source_module(
            source_dir,
            "APT_balance",
        ).balance_loss
        self.__dict__["_affine_loss"] = source_module(
            source_dir,
            "APT_affine",
        ).l2

    def forward(self, windows_batch):
        y, mask, _, calendar = self._inputs(windows_batch)
        self._complete_history(mask)
        for i, cardinality in enumerate((self.time_of_day_size, 7)):
            indices = (calendar[:, :, i] + 0.5) * cardinality
            rounded = indices.round()
            if (
                ((rounded < 0) | (rounded >= cardinality)).any()
                or not torch.allclose(
                    indices,
                    rounded,
                    atol=1e-4,
                    rtol=0,
                )
            ):
                raise ValueError(
                    "APT timestamps must equal index/cardinality - 0.5 "
                    "within valid calendar bins."
                )
        weight, bias = self.affine(
            calendar[:, : self.input_size],
            calendar[:, self.input_size :],
            self.training,
        )
        if not torch.isfinite(weight).all() or (weight.abs() < 1e-7).any():
            raise ValueError(
                "APT produced a singular affine scale; "
                "change initialization/training settings."
            )
        output = (
            self.backbone({"insample_y": y * weight + bias}) - bias
        ) / weight
        if self.training and self.global_step < self.warmup_steps:
            if len(y) < 2:
                raise ValueError(
                    "APT warmup's unbiased variance requires at least "
                    "two sampled windows."
                )
            self.__dict__["_regularizer"] = self.station_lambda * (
                self._orthogonality(self.affine.get_combined_embeddings())
                + self._balance_loss(self.affine.get_load())
                + self._affine_loss(weight)
                + self._affine_loss(bias)
            )
        return self._point_output(output, y)

    def training_step(self, batch, batch_idx):
        _no_sample_weights(batch)
        self.backbone.requires_grad_(self.global_step > self.warmup_steps)
        self.__dict__.pop("_regularizer", None)
        forecast_loss = super().training_step(batch, batch_idx)
        regularizer = self.__dict__.pop("_regularizer", None)
        objective = _finite_loss(
            regularizer
            if self.global_step < self.warmup_steps
            else forecast_loss
        )
        self.log(
            "train_objective",
            objective.detach(),
            on_step=True,
            on_epoch=False,
        )
        return objective
