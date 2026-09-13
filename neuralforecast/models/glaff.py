"""GLAFF trainable official-source forecasting adapter."""

from types import SimpleNamespace

from ._exogenous import ExogenousModel
from ._research_source import source_module
from ._research_utils import _full_windows, _positive
from .dlinear import DLinear

__all__ = ["GLAFF"]


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

    def __init__(
        self,
        h,
        input_size,
        source_dir=None,
        hidden_size=32,
        n_heads=4,
        encoder_layers=1,
        d_ff=64,
        dropout=0.1,
        q=0.75,
        moving_avg_window=25,
        **kwargs,
    ):
        _positive(
            hidden_size=hidden_size,
            n_heads=n_heads,
            encoder_layers=encoder_layers,
            d_ff=d_ff,
            moving_avg_window=moving_avg_window,
        )
        if (
            hidden_size % n_heads
            or moving_avg_window % 2 == 0
            or not 0.5 < q < 1
            or not 0 <= dropout < 1
        ):
            raise ValueError("Invalid GLAFF dimensions, quantile or dropout.")
        if kwargs.get("scaler_type", "identity") != "identity":
            raise ValueError(
                "Calendar features must not be rescaled; "
                "use scaler_type='identity'."
            )
        super().__init__(h=h, input_size=input_size, **_full_windows(kwargs))
        if self.futr_exog_size != 6:
            raise ValueError(
                "GLAFF requires exactly six known-future timestamp columns."
            )
        self.source_dir = source_dir
        config = SimpleNamespace(
            hist_len=input_size,
            pred_len=h,
            dim=hidden_size,
            head_num=n_heads,
            layer_num=encoder_layers,
            dff=d_ff,
            dropout=dropout,
            q=q,
        )
        self.plugin = source_module(source_dir, "GLAFF").Plugin(
            config,
            channel=1,
        )
        self.backbone = DLinear(
            h=h,
            input_size=input_size,
            moving_avg_window=moving_avg_window,
            random_seed=self.random_seed,
        )

    def forward(self, windows_batch):
        y, mask, _, calendar = self._inputs(windows_batch)
        self._complete_history(mask)
        baseline = self.backbone({"insample_y": y})
        output = self.plugin(
            y,
            calendar[:, : self.input_size],
            baseline,
            calendar[:, self.input_size :],
        )
        return self._point_output(output, y)
