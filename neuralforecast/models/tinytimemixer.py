"""Trainable adapter to IBM's official TinyTimeMixer implementation."""

import torch

from ._exogenous import ExogenousModel

__all__ = ["TinyTimeMixer"]


class TinyTimeMixer(ExogenousModel):
    """Train the official TTM architecture with NF historical/future covariates.

    This adapter initializes from configuration, not pretrained weights. Both
    backbone/decoder channel mixing and forecast-channel mixing are enabled:
    historical inputs and known future covariates cannot be silently discarded.
    Only target channel zero is forecast. Future targets are never fed back.

    Args:
        h (int): Forecast horizon.
        input_size (int): Historical context length.
        patch_len (int): Nonoverlapping patch length; must divide input_size.
        hidden_size (int): Encoder/decoder width.
        n_layers (int): Encoder/decoder mixer depth.
        dropout (float): Dropout probability.
        **kwargs: NF training options, hist_exog_list and futr_exog_list.

    Requires:
        The optional official granite-tsfm package (import name tsfm_public).

    References:
        https://github.com/ibm-granite/granite-tsfm
    """

    def __init__(
        self, h, input_size, patch_len=8, hidden_size=32,
        n_layers=3, dropout=0.1, **kwargs,
    ):
        if any(not isinstance(v, int) or v < 1 for v in (patch_len, hidden_size, n_layers)):
            raise ValueError("Architecture sizes must be positive integers.")
        if input_size % patch_len or input_size < 2 * patch_len:
            raise ValueError("input_size must be a multiple of patch_len with at least two patches.")
        super().__init__(h=h, input_size=input_size, **kwargs)
        try:
            from tsfm_public.models.tinytimemixer import TinyTimeMixerConfig, TinyTimeMixerForPrediction
        except ImportError as exc:
            raise ImportError("TinyTimeMixer requires the official granite-tsfm package. See docs/exogenous_models.md.") from exc
        n_channels = 1 + self.hist_exog_size + self.futr_exog_size
        known = list(range(1 + self.hist_exog_size, n_channels))
        config = TinyTimeMixerConfig(
            context_length=input_size, prediction_length=h,
            patch_length=patch_len, patch_stride=patch_len,
            num_input_channels=n_channels, prediction_channel_indices=[0],
            exogenous_channel_indices=known or None,
            d_model=hidden_size, num_layers=n_layers, dropout=dropout,
            head_dropout=dropout, mode="mix_channel", scaling="std",
            decoder_d_model=hidden_size, decoder_num_layers=n_layers,
            decoder_mode="mix_channel", adaptive_patching_levels=0,
            decoder_adaptive_patching_levels=0,
            enable_forecast_channel_mixing=True, fcm_use_mixer=True,
            fcm_context_length=1, loss="mse",
        )
        self.model = TinyTimeMixerForPrediction(config)

    def forward(self, windows_batch):
        y, mask, hist, futr = self._inputs(windows_batch)
        parts = [y]
        if hist is not None:
            parts.append(hist)
        if futr is not None:
            parts.append(futr[:, :self.input_size])
        past = torch.cat(parts, dim=-1)
        future = past.new_zeros(past.shape[0], self.h, past.shape[-1])
        if futr is not None:
            future[:, :, -self.futr_exog_size:] = futr[:, self.input_size:]
        result = self.model(
            past_values=past, past_observed_mask=mask.expand_as(past),
            future_values=future, return_loss=False, return_dict=True,
        )
        return self._point_output(result.prediction_outputs, y)
