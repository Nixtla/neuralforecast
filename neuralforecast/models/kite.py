"""KITE trainable official-source forecasting adapter."""

from types import SimpleNamespace

import torch

from ._exogenous import ExogenousModel
from ._research_source import source_module
from ._research_utils import _finite_loss, _full_windows, _no_sample_weights, _positive

__all__ = ["KITE"]


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

    def __init__(
        self,
        h,
        input_size,
        source_dir=None,
        hidden_size=64,
        n_heads=4,
        depth=2,
        num_sampling_steps=20,
        num_samples=20,
        omega=1.0,
        p_uncond=0.1,
        loss=None,
        **kwargs,
    ):
        _positive(
            hidden_size=hidden_size,
            n_heads=n_heads,
            depth=depth,
            num_sampling_steps=num_sampling_steps,
            num_samples=num_samples,
        )
        if h < 2 or hidden_size % n_heads:
            raise ValueError(
                "KITE requires h >= 2 and hidden_size divisible by n_heads."
            )
        if not 0 <= p_uncond <= 1 or not 0 <= omega < float("inf"):
            raise ValueError(
                "p_uncond must be in [0,1] and omega finite/nonnegative."
            )
        if loss is not None and (
            type(loss).__name__ != "MAE"
            or getattr(loss, "horizon_weight", None) is not None
        ):
            raise ValueError(
                "KITE uses its official flow loss; loss must remain default MAE "
                "metadata. Set valid_loss for scoring."
            )
        super().__init__(
            h=h,
            input_size=input_size,
            loss=loss,
            **_full_windows(kwargs),
        )
        if bool(self.hist_exog_size) == bool(self.futr_exog_size):
            raise ValueError(
                "Set exactly one nonempty hist_exog_list or futr_exog_list "
                "for KITE."
            )
        self.source_dir, self.num_samples = source_dir, num_samples
        config = SimpleNamespace(
            seq_len=input_size,
            horizon=h,
            input_dim=1 + self.hist_exog_size + self.futr_exog_size,
            output_dim=1,
            flow_dim=hidden_size,
            flow_head=n_heads,
            flow_depth=depth,
            num_sampling_steps=num_sampling_steps,
            omega=omega,
            noise_dropout=0.0,
            p_uncond=p_uncond,
            structure_max=0.7,
            rank=min(8, h),
            min_sigma=0.1,
            fc_type="Linear",
            rate=2,
            use_future_exog=bool(self.futr_exog_size),
            agg_method="mean",
            aux_loss_weight=0.1,
            mlp_ratio=4.0,
            prior_level="sample",
        )
        self.network = source_module(source_dir, "KITE").KITEModel(config)

    def _flow_inputs(self, windows_batch):
        y, mask, hist, futr = self._inputs(windows_batch)
        self._complete_history(mask)
        if futr is not None:
            return y, futr[:, : self.input_size], futr[:, self.input_size :]
        return y, hist, y.new_zeros(len(y), self.h, self.hist_exog_size)

    def forward(self, windows_batch):
        y, past, future = self._flow_inputs(windows_batch)
        output = self.network.inference(
            y,
            past,
            future,
            num_samples=self.num_samples,
        )
        return self._point_output(output, y)

    def training_step(self, batch, batch_idx):
        _no_sample_weights(batch)
        temporal, static, static_cols, selected, weights, columns = (
            self._create_windows(batch, step="train")
        )
        count = len(selected)
        size = self.windows_batch_size
        if size is None:
            indices = torch.arange(count, device=temporal.device)
        elif not isinstance(size, int) or size < 1:
            raise ValueError(
                "windows_batch_size must be a positive integer or None."
            )
        elif count < size:
            indices = torch.randint(count, (size,), device=temporal.device)
        else:
            indices = torch.randperm(count, device=temporal.device)[:size]
        windows = self._sample_windows(
            windows_temporal=temporal,
            static=static,
            static_cols=static_cols,
            temporal_cols=columns,
            w_idxs=indices,
            final_condition=selected,
            sample_weight=weights,
        )
        windows = self._normalization(
            windows=windows,
            y_idx=batch["y_idx"],
        )
        y, mask, target, target_mask, hist, futr, _ = self._parse_windows(
            batch,
            windows,
        )
        if not target_mask.bool().all() or not torch.isfinite(target).all():
            raise ValueError(
                "KITE flow training requires fully observed finite targets."
            )
        y, past, future = self._flow_inputs(
            dict(
                insample_y=y,
                insample_mask=mask,
                hist_exog=hist,
                futr_exog=futr,
            )
        )
        objective = _finite_loss(
            self.network.train_function(y, past, target, future)
        )
        self.log(
            "train_loss",
            objective.detach(),
            batch_size=len(y),
            prog_bar=True,
            on_epoch=True,
        )
        self.train_trajectories.append(
            (self.global_step, objective.detach().item())
        )
        return objective
