"""Toto-1.0 pretrained forecasting adapter."""

import torch

from ._exogenous import PretrainedExogenousModel

__all__ = ["Toto"]


class Toto(PretrainedExogenousModel):
    """Toto-1.0 official forecaster with known-future exogenous injection.

    Historical-only covariates are joint auxiliary channels; known-future
    channels are last and their values replace generated values during decoding.
    This uses Toto 1.0, NOT Toto 2.0 (which lacks exogenous support in the reviewed
    release). The adapter returns the mean of num_samples forecast trajectories.
    Timestamp placeholders are not used by the reviewed Toto-1.0 model.

    Args:
        h (int): Forecast horizon.
        input_size (int): Context length.
        **kwargs: PretrainedExogenousModel options and NF covariate lists.

    Requires:
        toto-ts, providing toto.model.toto.Toto (the legacy 1.0 API).

    References:
        https://github.com/DataDog/toto
    """

    DEFAULT_MODEL_ID = "Datadog/Toto-Open-Base-1.0"

    def _load_backend(self):
        try:
            from toto.model.toto import Toto as OfficialToto
            from toto.inference.forecaster import TotoForecaster
            from toto.data.util.dataset import MaskedTimeseries
        except ImportError as exc:
            raise ImportError(
                "Install toto-ts with the Toto-1.0 API. "
                "See docs/exogenous_models.md."
            ) from exc
        model = OfficialToto.from_pretrained(
            self.model_id, **self._hub_kwargs()
        )
        model.to(self.backend_device).eval()
        return TotoForecaster(model.model), MaskedTimeseries

    @torch.no_grad()
    def forward(self, windows_batch):
        y, mask, hist, futr = self._inputs(windows_batch)
        forecaster, input_type = self._get_backend()
        parts = [y]
        if hist is not None:
            parts.append(hist)
        if futr is not None:
            parts.append(futr[:, : self.input_size])
        series = (
            torch.cat(parts, dim=-1)
            .transpose(1, 2)
            .to(self.backend_device)
            .float()
        )
        inputs = input_type(
            series=series,
            padding_mask=mask.transpose(1, 2)
            .expand_as(series)
            .to(self.backend_device),
            id_mask=torch.zeros_like(series, dtype=torch.long),
            timestamp_seconds=torch.zeros_like(series, dtype=torch.long),
            time_interval_seconds=torch.ones(
                series.shape[:2],
                device=series.device,
                dtype=torch.long,
            ),
            num_exogenous_variables=self.futr_exog_size,
        )
        future = (
            None
            if futr is None
            else futr[:, self.input_size :]
            .transpose(1, 2)
            .to(self.backend_device)
            .float()
        )
        result = forecaster.forecast(
            inputs,
            prediction_length=self.h,
            num_samples=self.num_samples,
            samples_per_batch=min(10, self.num_samples),
            future_exogenous_variables=future,
        )
        return self._point_output(result.mean[:, 0, :], y)
