"""Dualformer trainable official-source forecasting adapter."""

from types import SimpleNamespace

import torch

from ._exogenous import ExogenousModel
from ._forecast_source import forecast_source
from ._research_utils import _full_windows, _positive

__all__ = ["Dualformer"]


class Dualformer(ExogenousModel):
    """Train the official time-frequency Dualformer, not the namesake LLM.

    Historical covariates enter the official multivariate encoder. enc_in and
    c_out are kept equal so the upstream RevIN inverse remains valid, then the
    first (target) channel is returned. Known-future/calendar inputs are not
    synthesized. Requires e_layers >= 2 to avoid an upstream sampler division.
    """

    EXOGENOUS_FUTR = False

    def __init__(
        self,
        h,
        input_size,
        source_dir=None,
        hidden_size=128,
        d_ff=256,
        n_heads=8,
        e_layers=2,
        dropout=0.1,
        **kwargs,
    ):
        _positive(
            h=h,
            input_size=input_size,
            hidden_size=hidden_size,
            d_ff=d_ff,
            n_heads=n_heads,
            e_layers=e_layers,
        )
        if (
            input_size < 8
            or e_layers < 2
            or hidden_size % n_heads
            or hidden_size % 2
        ):
            raise ValueError(
                "Dualformer requires input_size >= 8, e_layers >= 2 "
                "and even hidden_size divisible by n_heads."
            )
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1).")
        super().__init__(h=h, input_size=input_size, **_full_windows(kwargs))
        self.source_dir = source_dir
        channels = 1 + self.hist_exog_size
        config = SimpleNamespace(
            seq_len=input_size,
            pred_len=h,
            enc_in=channels,
            c_out=channels,
            d_model=hidden_size,
            d_ff=d_ff,
            n_heads=n_heads,
            e_layers=e_layers,
            dropout=dropout,
            factor=1,
            embed="timeF",
            freq="h",
            activation="gelu",
        )
        self.network = forecast_source(
            source_dir,
            "Dualformer",
        ).Model(config)

    def forward(self, windows_batch):
        y, mask, hist, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        inputs = y if hist is None else torch.cat((y, hist), dim=-1)
        result = self.network(inputs, None)
        if result.shape != (len(y), self.h, inputs.shape[-1]):
            raise ValueError(
                "Dualformer returned an unexpected multi-channel shape."
            )
        return self._point_output(result[:, :, :1], y)
