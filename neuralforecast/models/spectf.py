"""SpecTF text-conditioned forecasting adapter."""

from pathlib import Path
from types import SimpleNamespace

from ._exogenous import ExogenousModel
from ._research_utils import _full_windows, _positive
from ._text_source import text_source

__all__ = ["SpecTF"]


def _options(kwargs):
    if kwargs.get("scaler_type", "identity") != "identity":
        raise ValueError("Use scaler_type='identity' to preserve text embeddings.")
    return _full_windows(kwargs)


class SpecTF(ExogenousModel):
    """Official spectral fusion of target history and historical text embeddings.

    Args:
        h: Forecast horizon.
        input_size: Complete target/text history, at most 9998 steps.
        source_dir: Reviewed hiepnh137/SpecTF checkout.
        mm_emb_size: Even spectral width, at least two.
        mm_hidden_size: Width used by the official constructor.
        text_emb: Width of the trained text projection.
        dropout: Upstream attention/embedding dropout in [0, 1).
        text_dropout: Text projection dropout in [0, 1).
        **kwargs: NeuralForecast options, including nonempty hist_exog_list.

    Each historical column is one coordinate of a fixed text encoder's embedding.
    This integration selects the official history-fusion path, with no calendar
    or future-text input. Static/categorical inputs and non-point losses are
    rejected by ExogenousModel. All upstream parameters are trained from scratch;
    this does not load a pretrained language model or reproduce its paper scores.
    """

    EXOGENOUS_FUTR = False

    def __init__(
        self,
        h: int,
        input_size: int,
        source_dir: str,
        mm_emb_size: int = 32,
        mm_hidden_size: int = 64,
        text_emb: int = 6,
        dropout: float = 0.1,
        text_dropout: float = 0.1,
        **kwargs,
    ):
        _positive(
            mm_emb_size=mm_emb_size,
            mm_hidden_size=mm_hidden_size,
            text_emb=text_emb,
        )
        if (
            mm_emb_size % 2
            or not 0 <= dropout < 1
            or not 0 <= text_dropout < 1
        ):
            raise ValueError(
                "mm_emb_size must be even; "
                "dropout probabilities must be in [0, 1)."
            )
        super().__init__(h=h, input_size=input_size, **_options(kwargs))
        if input_size > 9998:
            raise ValueError(
                "SpecTF requires input_size <= 9998 "
                "(upstream positional embedding limit)."
            )
        if not self.hist_exog_size:
            raise ValueError(
                "SpecTF requires hist_exog_list containing text embedding columns."
            )
        module = text_source(source_dir, "SpecTF")
        self.source_dir = str(Path(source_dir).expanduser().resolve())
        self.hparams["source_dir"] = self.source_dir
        config = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=input_size,
            pred_len=h,
            n_ts_features=1,
            enc_in=1,
            mm_emb_size=mm_emb_size,
            mm_hidden_size=mm_hidden_size,
            text_emb=text_emb,
            llm_emb_size=self.hist_exog_size,
            text_dropout=text_dropout,
            dropout=dropout,
            embed="timeF",
            freq="h",
            channel_independence="1",
            proj_per_freq=False,
            freq_cut_off_rate=1.0,
            only_text_input=False,
            fuse_history=True,
            use_product=False,
            sum_fusion=False,
        )
        self.text_encoder = module.TextEncoder(config)
        self.model = module.FreqModelHistPred(config)

    def forward(self, windows_batch):
        y, mask, hist, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        text = self.text_encoder(hist, None)
        return self._point_output(
            self.model(y, None, None, None, text),
            y,
        )
