"""DAG trainable official-source forecasting adapter."""

from types import SimpleNamespace

import torch
from torch import nn

from ._exogenous import ExogenousModel
from ._research_source import source_module
from ._research_utils import _finite_loss, _full_windows, _no_sample_weights, _positive

__all__ = ["DAG"]


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

    def __init__(
        self,
        h,
        input_size,
        source_dir=None,
        hidden_size=64,
        n_heads=4,
        encoder_layers=2,
        patch_len=8,
        stride=4,
        d_ff=128,
        dropout=0.1,
        alpha=0.2,
        beta=0.1,
        **kwargs,
    ):
        _positive(
            hidden_size=hidden_size,
            n_heads=n_heads,
            encoder_layers=encoder_layers,
            patch_len=patch_len,
            stride=stride,
            d_ff=d_ff,
        )
        if hidden_size % n_heads or hidden_size % 4 or patch_len > input_size:
            raise ValueError(
                "hidden_size must divide by n_heads and four; "
                "patch_len <= input_size."
            )
        if (
            not 0 <= dropout < 1
            or not 0 <= alpha <= 1
            or not 0 <= beta < float("inf")
        ):
            raise ValueError("Invalid dropout, alpha or beta.")
        super().__init__(h=h, input_size=input_size, **_full_windows(kwargs))
        if not self.futr_exog_size:
            raise ValueError("DAG requires nonempty futr_exog_list.")
        self.source_dir = source_dir
        criterion = (
            nn.MSELoss() if type(self.loss).__name__ == "MSE" else nn.L1Loss()
        )
        config = SimpleNamespace(
            seq_len=input_size,
            pred_len=h,
            enc_in=1 + self.futr_exog_size,
            series_dim=1,
            patch_len=patch_len,
            stride=stride,
            d_model=hidden_size,
            d_ff=d_ff,
            n_heads=n_heads,
            e_layers=encoder_layers,
            dropout=dropout,
            factor=1,
            activation="gelu",
            criterion=criterion,
            use_c=True,
            use_t=True,
            use_c_exog=True,
            use_t_exog=True,
            infer_use_future=True,
            alpha=alpha,
            beta=beta,
        )
        self.network = source_module(source_dir, "DAG").DAGModel(config)

    def forward(self, windows_batch):
        y, mask, _, futr = self._inputs(windows_batch)
        self._complete_history(mask)
        output, auxiliary = self.network(
            torch.cat((y, futr[:, : self.input_size]), -1),
            futr[:, self.input_size :],
        )
        if self.training:
            self.__dict__["_auxiliary"] = auxiliary
        return self._point_output(output, y)

    def training_step(self, batch, batch_idx):
        _no_sample_weights(batch)
        self.__dict__.pop("_auxiliary", None)
        forecast_loss = super().training_step(batch, batch_idx)
        auxiliary = self.__dict__.pop("_auxiliary")
        objective = _finite_loss(forecast_loss + auxiliary)
        self.log(
            "train_objective",
            objective.detach(),
            on_step=True,
            on_epoch=False,
        )
        return objective
