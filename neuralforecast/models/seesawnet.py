"""SeesawNet trainable official-source forecasting adapter."""

from types import SimpleNamespace

import torch

from ._exogenous import ExogenousModel
from ._forecast_source import forecast_source
from ._research_utils import _full_windows, _positive

__all__ = ["SeesawNet"]


class SeesawNet(ExogenousModel):
    """Train the official SeesawNet architecture using NF's requested point loss.

    source_dir is a pinned dreamone-Lee/SeesawNet checkout. Historical covariates
    are additional observed channels; only the NF target output is supervised.
    Future/static/categorical covariates are unsupported. This adapter does not
    claim reproduction of the paper's multivariate TFMAE training protocol.
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
        patch_len=8,
        stride=4,
        pd_layers=1,
        cr_layers=2,
        down_sample_rate=1.0,
        dropout=0.1,
        **kwargs,
    ):
        _positive(
            h=h,
            input_size=input_size,
            hidden_size=hidden_size,
            d_ff=d_ff,
            n_heads=n_heads,
            patch_len=patch_len,
            stride=stride,
            pd_layers=pd_layers,
            cr_layers=cr_layers,
        )
        if hidden_size % n_heads or patch_len > input_size:
            raise ValueError(
                "hidden_size must divide by n_heads and patch_len <= input_size."
            )
        patch_num = (input_size - patch_len) // stride + 2
        if not 0 <= dropout < 1 or not 0 <= down_sample_rate <= 1:
            raise ValueError(
                "dropout must be in [0,1), down_sample_rate in [0,1]."
            )
        if down_sample_rate and int(patch_num * down_sample_rate) < 1:
            raise ValueError(
                "down_sample_rate would produce zero aggregation tokens."
            )
        super().__init__(h=h, input_size=input_size, **_full_windows(kwargs))
        self.source_dir = source_dir
        config = SimpleNamespace(
            seq_len=input_size,
            pred_len=h,
            enc_in=1 + self.hist_exog_size,
            d_model=hidden_size,
            d_ff=d_ff,
            n_heads=n_heads,
            patch_len=patch_len,
            stride=stride,
            pd_layers=pd_layers,
            cr_layers=cr_layers,
            down_sample_rate=down_sample_rate,
            dropout=dropout,
            pe="zeros",
            learn_pe=True,
            activation="gelu",
            norm="LayerNorm",
            group=False,
        )
        self.network = forecast_source(
            source_dir,
            "SeesawNet",
        ).Model(config)

    def forward(self, windows_batch):
        y, mask, hist, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        inputs = y if hist is None else torch.cat((y, hist), dim=-1)
        result, _, _ = self.network(inputs)
        if result.shape != (len(y), self.h, inputs.shape[-1]):
            raise ValueError(
                "SeesawNet returned an unexpected multi-channel shape."
            )
        return self._point_output(result[:, :, :1], y)
