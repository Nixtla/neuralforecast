"""Trainable official-source integrations: DAG, KITE, GLAFF and APT.

Install the documented source checkouts and optional dependencies explicitly.
No upstream source is copied here. All models forecast one target per NF series;
GLAFF/APT are official calendar modules composed with NF's DLinear backbone.
"""

from types import SimpleNamespace

import torch
from torch import nn

from ._exogenous import ExogenousModel
from ._research_source import source_module
from .dlinear import DLinear

__all__ = ["DAG", "KITE", "GLAFF", "APT"]


def _positive(**values):
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")


def _full_windows(kwargs):
    options = dict(kwargs)
    threshold = options.get("training_data_availability_threshold", 1.0)
    if threshold != 1.0 and threshold != [1.0, 1.0]:
        raise ValueError("Official-source training requires fully observed history and forecast windows.")
    options["training_data_availability_threshold"] = 1.0
    if options.get("start_padding_enabled", False):
        raise ValueError("Official-source training does not support start padding.")
    return options


def _no_sample_weights(batch):
    if "sample_weight" in batch["temporal_cols"]:
        raise ValueError("This model's auxiliary/flow objective does not support sample_weight.")


def _finite_loss(loss):
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("Official training objective must be a finite scalar tensor.")
    return loss


class DAG(ExogenousModel):
    """Official dual-correlation DAG with its reconstruction auxiliary losses.

    Args:
        h: Forecast horizon.
        input_size: Complete history length.
        source_dir: Checkout of decisionintelligence/DAG.
        futr_exog_list: Nonempty known-future numerical covariates. Supply both
            their historical values and h future values through NF's futr_df.
        hidden_size: Encoder width, divisible by n_heads and four.
        n_heads: Attention heads.
        encoder_layers: Layers per encoder.
        patch_len: Temporal patch width, at most input_size.
        stride: Temporal patch stride.
        alpha: Temporal/channel output mixing coefficient.
        beta: Weight of the official auxiliary losses.
        **kwargs: ExogenousModel/NF training options. Only complete windows.

    The exported name is the forecasting model DAG, not a causal graph estimator.
    This initial integration uses the official known-future path; past-only
    covariates, static/categorical features and sample weights are rejected.
    """

    EXOGENOUS_HIST = False

    def __init__(self, h, input_size, source_dir=None, hidden_size=64, n_heads=4,
                 encoder_layers=2, patch_len=8, stride=4, d_ff=128, dropout=0.1,
                 alpha=0.2, beta=0.1, **kwargs):
        _positive(hidden_size=hidden_size, n_heads=n_heads, encoder_layers=encoder_layers,
                  patch_len=patch_len, stride=stride, d_ff=d_ff)
        if hidden_size % n_heads or hidden_size % 4 or patch_len > input_size:
            raise ValueError("hidden_size must divide by n_heads and four; patch_len <= input_size.")
        if not 0 <= dropout < 1 or not 0 <= alpha <= 1 or not 0 <= beta < float("inf"):
            raise ValueError("Invalid dropout, alpha or beta.")
        super().__init__(h=h, input_size=input_size, **_full_windows(kwargs))
        if not self.futr_exog_size:
            raise ValueError("DAG requires nonempty futr_exog_list.")
        self.source_dir = source_dir
        criterion = nn.MSELoss() if type(self.loss).__name__ == "MSE" else nn.L1Loss()
        config = SimpleNamespace(
            seq_len=input_size, pred_len=h, enc_in=1 + self.futr_exog_size,
            series_dim=1, patch_len=patch_len, stride=stride,
            d_model=hidden_size, d_ff=d_ff, n_heads=n_heads, e_layers=encoder_layers,
            dropout=dropout, factor=1, activation="gelu", criterion=criterion,
            use_c=True, use_t=True, use_c_exog=True, use_t_exog=True,
            infer_use_future=True, alpha=alpha, beta=beta,
        )
        self.network = source_module(source_dir, "DAG").DAGModel(config)

    def forward(self, windows_batch):
        y, mask, _, futr = self._inputs(windows_batch)
        self._complete_history(mask)
        output, auxiliary = self.network(torch.cat((y, futr[:, :self.input_size]), -1),
                                         futr[:, self.input_size:])
        if self.training:
            self.__dict__["_auxiliary"] = auxiliary
        return self._point_output(output, y)

    def training_step(self, batch, batch_idx):
        _no_sample_weights(batch)
        self.__dict__.pop("_auxiliary", None)
        forecast_loss = super().training_step(batch, batch_idx)
        auxiliary = self.__dict__.pop("_auxiliary")
        objective = _finite_loss(forecast_loss + auxiliary)
        self.log("train_objective", objective.detach(), on_step=True, on_epoch=False)
        return objective


class KITE(ExogenousModel):
    """Official KITE flow matching, not MSE training on sampled predictions.

    Args:
        h: Forecast horizon >= 2 (the official source normalizes horizon variance).
        input_size: Complete history length.
        source_dir: Checkout of decisionintelligence/KITE.
        hidden_size: Flow-vector-field width.
        n_heads: Flow attention heads.
        depth: Number of flow blocks.
        num_sampling_steps: Euler integration steps at prediction.
        num_samples: Trajectories averaged to form the point forecast.
        **kwargs: NF options. Set exactly one nonempty hist_exog_list or
            futr_exog_list. valid_loss controls forecast evaluation; the training
            objective is always the official conditional flow-matching loss.

    Historical-only mode disables future conditioning. A shape-only zero array
    avoids an upstream zeros_like(None) bug in classifier-free training; no
    fabricated future covariates are consumed by the disabled conditioning path.
    """

    def __init__(self, h, input_size, source_dir=None, hidden_size=64, n_heads=4,
                 depth=2, num_sampling_steps=20, num_samples=20, omega=1.0,
                 p_uncond=0.1, loss=None, **kwargs):
        _positive(hidden_size=hidden_size, n_heads=n_heads, depth=depth,
                  num_sampling_steps=num_sampling_steps, num_samples=num_samples)
        if h < 2 or hidden_size % n_heads:
            raise ValueError("KITE requires h >= 2 and hidden_size divisible by n_heads.")
        if not 0 <= p_uncond <= 1 or not 0 <= omega < float("inf"):
            raise ValueError("p_uncond must be in [0,1] and omega finite/nonnegative.")
        if loss is not None and (type(loss).__name__ != "MAE" or getattr(loss, "horizon_weight", None) is not None):
            raise ValueError("KITE uses its official flow loss; loss must remain default MAE metadata. Set valid_loss for scoring.")
        super().__init__(h=h, input_size=input_size, loss=loss, **_full_windows(kwargs))
        if bool(self.hist_exog_size) == bool(self.futr_exog_size):
            raise ValueError("Set exactly one nonempty hist_exog_list or futr_exog_list for KITE.")
        self.source_dir, self.num_samples = source_dir, num_samples
        config = SimpleNamespace(
            seq_len=input_size, horizon=h, input_dim=1 + self.hist_exog_size + self.futr_exog_size,
            output_dim=1, flow_dim=hidden_size, flow_head=n_heads, flow_depth=depth,
            num_sampling_steps=num_sampling_steps, omega=omega, noise_dropout=0.0,
            p_uncond=p_uncond, structure_max=0.7, rank=min(8, h), min_sigma=0.1,
            fc_type="Linear", rate=2, use_future_exog=bool(self.futr_exog_size),
            agg_method="mean", aux_loss_weight=0.1, mlp_ratio=4.0, prior_level="sample",
        )
        self.network = source_module(source_dir, "KITE").KITEModel(config)

    def _flow_inputs(self, windows_batch):
        y, mask, hist, futr = self._inputs(windows_batch)
        self._complete_history(mask)
        if futr is not None:
            return y, futr[:, :self.input_size], futr[:, self.input_size:]
        return y, hist, y.new_zeros(len(y), self.h, self.hist_exog_size)

    def forward(self, windows_batch):
        y, past, future = self._flow_inputs(windows_batch)
        output = self.network.inference(y, past, future, num_samples=self.num_samples)
        return self._point_output(output, y)

    def training_step(self, batch, batch_idx):
        _no_sample_weights(batch)
        temporal, static, static_cols, selected, weights, columns = self._create_windows(batch, step="train")
        count = len(selected)
        size = self.windows_batch_size
        if size is None:
            indices = torch.arange(count, device=temporal.device)
        elif not isinstance(size, int) or size < 1:
            raise ValueError("windows_batch_size must be a positive integer or None.")
        elif count < size:
            indices = torch.randint(count, (size,), device=temporal.device)
        else:
            indices = torch.randperm(count, device=temporal.device)[:size]
        windows = self._sample_windows(
            windows_temporal=temporal, static=static, static_cols=static_cols,
            temporal_cols=columns, w_idxs=indices, final_condition=selected, sample_weight=weights,
        )
        windows = self._normalization(windows=windows, y_idx=batch["y_idx"])
        y, mask, target, target_mask, hist, futr, _ = self._parse_windows(batch, windows)
        if not target_mask.bool().all() or not torch.isfinite(target).all():
            raise ValueError("KITE flow training requires fully observed finite targets.")
        y, past, future = self._flow_inputs(dict(insample_y=y, insample_mask=mask,
                                               hist_exog=hist, futr_exog=futr))
        objective = _finite_loss(self.network.train_function(y, past, target, future))
        self.log("train_loss", objective.detach(), batch_size=len(y), prog_bar=True, on_epoch=True)
        self.train_trajectories.append((self.global_step, objective.detach().item()))
        return objective


class GLAFF(ExogenousModel):
    """Official GLAFF global/local fusion plugin on a trainable NF DLinear.

    Args:
        h: Forecast horizon.
        input_size: Complete history length.
        source_dir: Checkout of ForestsKing/GLAFF.
        q: Robust quantile in (0.5,1).
        **kwargs: NF training options. futr_exog_list must contain the six
            numerical timestamp features expected by the original plugin, with
            past and future values. scaler_type must remain identity.

    This is GLAFF+DLinear, not an unrelated new backbone or a pretrained model.
    Historical-only, static and categorical NF inputs are unsupported.
    """

    EXOGENOUS_HIST = False

    def __init__(self, h, input_size, source_dir=None, hidden_size=32, n_heads=4,
                 encoder_layers=1, d_ff=64, dropout=0.1, q=0.75,
                 moving_avg_window=25, **kwargs):
        _positive(hidden_size=hidden_size, n_heads=n_heads, encoder_layers=encoder_layers,
                  d_ff=d_ff, moving_avg_window=moving_avg_window)
        if hidden_size % n_heads or moving_avg_window % 2 == 0 or not 0.5 < q < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid GLAFF dimensions, quantile or dropout.")
        if kwargs.get("scaler_type", "identity") != "identity":
            raise ValueError("Calendar features must not be rescaled; use scaler_type='identity'.")
        super().__init__(h=h, input_size=input_size, **_full_windows(kwargs))
        if self.futr_exog_size != 6:
            raise ValueError("GLAFF requires exactly six known-future timestamp columns.")
        self.source_dir = source_dir
        config = SimpleNamespace(hist_len=input_size, pred_len=h, dim=hidden_size,
                                 head_num=n_heads, layer_num=encoder_layers, dff=d_ff,
                                 dropout=dropout, q=q)
        self.plugin = source_module(source_dir, "GLAFF").Plugin(config, channel=1)
        self.backbone = DLinear(h=h, input_size=input_size, moving_avg_window=moving_avg_window,
                                random_seed=self.random_seed)

    def forward(self, windows_batch):
        y, mask, _, calendar = self._inputs(windows_batch)
        self._complete_history(mask)
        baseline = self.backbone({"insample_y": y})
        output = self.plugin(y, calendar[:, :self.input_size], baseline, calendar[:, self.input_size:])
        return self._point_output(output, y)


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

    def __init__(self, h, input_size, source_dir=None, timestamp_dim=16,
                 timestamp_hidden=32, num_prototypes=8, top_k=3,
                 time_of_day_size=24, warmup_steps=10, station_lambda=0.01,
                 moving_avg_window=25, **kwargs):
        _positive(timestamp_dim=timestamp_dim, timestamp_hidden=timestamp_hidden,
                  num_prototypes=num_prototypes, top_k=top_k, time_of_day_size=time_of_day_size,
                  moving_avg_window=moving_avg_window)
        if top_k > num_prototypes or moving_avg_window % 2 == 0:
            raise ValueError("top_k <= num_prototypes and odd moving_avg_window are required.")
        if not isinstance(warmup_steps, int) or isinstance(warmup_steps, bool) or warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer.")
        if not 0 < station_lambda < float("inf"):
            raise ValueError("station_lambda must be positive and finite.")
        if kwargs.get("scaler_type", "identity") != "identity":
            raise ValueError("APT's encoded timestamps require scaler_type='identity'.")
        super().__init__(h=h, input_size=input_size, **_full_windows(kwargs))
        if self.futr_exog_size != 2:
            raise ValueError("APT requires [time_of_day, day_of_week] in futr_exog_list.")
        self.source_dir, self.time_of_day_size = source_dir, time_of_day_size
        self.warmup_steps, self.station_lambda = warmup_steps, station_lambda
        config = dict(tan_timestamp=["time_of_day", "day_of_week"],
                      timestamp_dim=timestamp_dim, timestamp_hidden=timestamp_hidden,
                      num_prototypes=num_prototypes, top_k=top_k, is_xformer=False,
                      time_of_day_size=time_of_day_size, day_of_week_size=7,
                      independent=False, enc_in=1, model_name="DLinear",
                      normalization_name="None", datasets_name="NeuralForecast", use_tan=True)
        self.affine = source_module(source_dir, "APT").APT(**config)
        self.backbone = DLinear(h=h, input_size=input_size, moving_avg_window=moving_avg_window,
                                random_seed=self.random_seed)
        self.__dict__["_orthogonality"] = source_module(source_dir, "APT_orthogonality").orthogonality
        self.__dict__["_balance_loss"] = source_module(source_dir, "APT_balance").balance_loss
        self.__dict__["_affine_loss"] = source_module(source_dir, "APT_affine").l2

    def forward(self, windows_batch):
        y, mask, _, calendar = self._inputs(windows_batch)
        self._complete_history(mask)
        for i, cardinality in enumerate((self.time_of_day_size, 7)):
            indices = (calendar[:, :, i] + 0.5) * cardinality
            rounded = indices.round()
            if ((rounded < 0) | (rounded >= cardinality)).any() or not torch.allclose(indices, rounded, atol=1e-4, rtol=0):
                raise ValueError("APT timestamps must equal index/cardinality - 0.5 within valid calendar bins.")
        weight, bias = self.affine(calendar[:, :self.input_size], calendar[:, self.input_size:], self.training)
        if not torch.isfinite(weight).all() or (weight.abs() < 1e-7).any():
            raise ValueError("APT produced a singular affine scale; change initialization/training settings.")
        output = (self.backbone({"insample_y": y * weight + bias}) - bias) / weight
        if self.training and self.global_step < self.warmup_steps:
            if len(y) < 2:
                raise ValueError("APT warmup's unbiased variance requires at least two sampled windows.")
            self.__dict__["_regularizer"] = self.station_lambda * (
                self._orthogonality(self.affine.get_combined_embeddings())
                + self._balance_loss(self.affine.get_load())
                + self._affine_loss(weight) + self._affine_loss(bias)
            )
        return self._point_output(output, y)

    def training_step(self, batch, batch_idx):
        _no_sample_weights(batch)
        self.backbone.requires_grad_(self.global_step > self.warmup_steps)
        self.__dict__.pop("_regularizer", None)
        forecast_loss = super().training_step(batch, batch_idx)
        regularizer = self.__dict__.pop("_regularizer", None)
        objective = _finite_loss(regularizer if self.global_step < self.warmup_steps else forecast_loss)
        self.log("train_objective", objective.detach(), on_step=True, on_epoch=False)
        return objective
