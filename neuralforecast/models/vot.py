"""VoT context-conditioned forecasting adapter."""

from pathlib import Path
from types import SimpleNamespace

import torch

from ._context_source import official_module
from ._context_utils import _options, _positive
from ._exogenous import ExogenousModel

__all__ = ["VoT"]


class VoT(ExogenousModel):
    """Official VoT PatchTST text-fusion forecasting stage, trained with point loss.

    hist_exog_list contains precomputed event/text embedding coordinates.
    Reasoning/text extraction and the original contrastive pretraining stages
    remain external; this runs the official forecast=2 path with both branches.
    """

    EXOGENOUS_FUTR = False

    def __init__(
        self,
        h,
        input_size,
        source_dir,
        d_model=128,
        n_heads=4,
        d_ff=256,
        e_layers=2,
        patch_len=4,
        stride=4,
        dropout=0.1,
        **kwargs,
    ):
        _positive(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            patch_len=patch_len,
            stride=stride,
        )
        if (
            d_model % n_heads
            or not 0 <= dropout < 1
            or patch_len > min(input_size, h)
        ):
            raise ValueError(
                "Invalid attention/dropout/patch configuration."
            )
        super().__init__(h=h, input_size=input_size, **_options(kwargs))
        if self.hist_exog_size < 8 or self.hist_exog_size % 8:
            raise ValueError(
                "VoT text embedding width must be a positive multiple of eight."
            )
        self.source_dir = str(Path(source_dir).expanduser().resolve())
        config = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=input_size,
            pred_len=h,
            enc_in=1,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            dropout=dropout,
            factor=1,
            output_attention=False,
            activation="gelu",
            multimodal=True,
            tr_sea=True,
            clip_t=0.07,
            llm_dim=self.hist_exog_size,
        )
        self.model = official_module(
            self.source_dir,
            "VoT",
        ).Model(config, patch_len, stride)

    def forward(self, windows_batch):
        y, mask, hist, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        # ponytail: upstream text rescaling averages over the batch; independent
        # window calls prevent later windows affecting earlier forecasts.
        output = torch.cat(
            [
                self.model(
                    y[i : i + 1],
                    None,
                    None,
                    None,
                    hist[i : i + 1],
                    forecast=2,
                )
                for i in range(len(y))
            ]
        )
        return self._point_output(output, y)
