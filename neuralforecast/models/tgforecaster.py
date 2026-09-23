"""TGForecaster text-conditioned forecasting adapter."""

from pathlib import Path
from types import SimpleNamespace

import torch

from ._exogenous import ExogenousModel
from ._research_utils import _full_windows, _positive
from ._text_source import text_source

__all__ = ["TGForecaster"]


def _options(kwargs):
    if kwargs.get("scaler_type", "identity") != "identity":
        raise ValueError("Use scaler_type='identity' to preserve text embeddings.")
    return _full_windows(kwargs)


class TGForecaster(ExogenousModel):
    """Official TGTSF_torch using origin-known news and description embeddings.

    Args:
        h: Forecast horizon, divisible by patch_len.
        input_size: Complete history, divisible by patch_len.
        source_dir: Reviewed VEWOXIC/TGTSF checkout.
        text_dim: Width of BOTH the news and description embeddings.
        n_heads: Attention heads; must divide text_dim.
        encoder_layers: Layers in the time-series encoder.
        cross_layers: Layers in the text encoder's cross attention.
        mixer_self_layers: Layers in the fusion self-attention decoder.
        patch_len: Non-overlapping patch length.
        dropout: Upstream dropout probability in [0, 1).
        **kwargs: NF options, including 2*text_dim ordered futr_exog_list columns.

    Column order is [news_0..news_D-1, description_0..description_D-1].
    Each forecast patch uses the mean of its per-step vectors. Future embeddings
    MUST describe information already available at the forecast origin, never
    articles published after it. The NF future-frame API also requires historical
    values of these columns; the official model uses only their horizon segment.
    This integration has one news vector per patch, not variable-length news sets.
    """

    EXOGENOUS_HIST = False

    def __init__(
        self,
        h: int,
        input_size: int,
        source_dir: str,
        text_dim: int = 384,
        n_heads: int = 4,
        encoder_layers: int = 2,
        cross_layers: int = 1,
        mixer_self_layers: int = 1,
        patch_len: int = 8,
        dropout: float = 0.1,
        **kwargs,
    ):
        _positive(
            text_dim=text_dim,
            n_heads=n_heads,
            encoder_layers=encoder_layers,
            cross_layers=cross_layers,
            mixer_self_layers=mixer_self_layers,
            patch_len=patch_len,
        )
        if (
            text_dim < 2
            or text_dim % n_heads
            or not 0 <= dropout < 1
        ):
            raise ValueError(
                "text_dim must be >= 2 and divisible by n_heads; "
                "dropout must be in [0, 1)."
            )
        super().__init__(h=h, input_size=input_size, **_options(kwargs))
        if h % patch_len or input_size % patch_len:
            raise ValueError(
                "h and input_size must be divisible by patch_len."
            )
        if self.futr_exog_size != 2 * text_dim:
            raise ValueError(
                "TGForecaster requires 2*text_dim futr_exog columns: "
                "news then description."
            )
        module = text_source(source_dir, "TGForecaster")
        self.source_dir = str(Path(source_dir).expanduser().resolve())
        self.hparams["source_dir"] = self.source_dir
        self.text_dim, self.patch_len = text_dim, patch_len
        config = SimpleNamespace(
            enc_in=1,
            seq_len=input_size,
            pred_len=h,
            e_layers=encoder_layers,
            n_heads=n_heads,
            d_model=text_dim,
            dropout=dropout,
            fc_dropout=dropout,
            head_dropout=dropout,
            individual=False,
            patch_len=patch_len,
            stride=patch_len,
            padding_patch=None,
            revin=True,
            affine=False,
            subtract_last=False,
            out_attn_weights=False,
            cross_layers=cross_layers,
            self_layers=1,
            text_dim=text_dim,
            mixer_self_layers=mixer_self_layers,
        )
        self.model = module.Model(config)

    def forward(self, windows_batch):
        y, mask, _, futr = self._inputs(windows_batch)
        self._complete_history(mask)
        # ponytail: mean-pool one news/description vector per patch; a dedicated
        # ragged-news dataset adapter is needed to retain individual articles.
        text = futr[:, self.input_size :].reshape(
            y.shape[0],
            self.h // self.patch_len,
            self.patch_len,
            2 * self.text_dim,
        ).mean(dim=2)
        news = text[..., : self.text_dim].unsqueeze(2).contiguous()
        description = text[..., self.text_dim :].unsqueeze(2).contiguous()
        news_mask = torch.zeros(
            news.shape[:-1],
            dtype=torch.bool,
            device=y.device,
        )
        return self._point_output(
            self.model(y, news, description, news_mask),
            y,
        )
