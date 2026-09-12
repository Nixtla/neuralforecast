"""Chronos-2 pretrained forecasting adapter."""

import numpy as np
import torch

from ._exogenous import PretrainedExogenousModel

__all__ = ["Chronos2"]


class Chronos2(PretrainedExogenousModel):
    """Chronos-2 point forecasts using official past/future_covariates dictionaries.

    Args:
        h (int): Forecast horizon.
        input_size (int): Historical context length.
        **kwargs: PretrainedExogenousModel options and NF covariate lists.

    Requires:
        chronos-forecasting with chronos.Chronos2Pipeline.

    References:
        https://github.com/amazon-science/chronos-forecasting
    """

    DEFAULT_MODEL_ID = "amazon/chronos-2"

    def _load_backend(self):
        try:
            from chronos import Chronos2Pipeline
        except ImportError as exc:
            raise ImportError(
                "Install chronos-forecasting with Chronos2Pipeline support."
            ) from exc
        return Chronos2Pipeline.from_pretrained(
            self.model_id, device_map=self.backend_device, **self._hub_kwargs()
        )

    @torch.no_grad()
    def forward(self, windows_batch):
        y, mask, hist, futr = self._inputs(windows_batch)
        backend = self._get_backend()
        inputs = []
        for i in range(len(y)):
            target = (
                y[i, :, 0]
                .masked_fill(~mask[i, :, 0], float("nan"))
                .cpu()
                .float()
                .numpy()
            )
            past, future = {}, {}
            for data, names in (
                (hist, self.hist_exog_list),
                (futr, self.futr_exog_list),
            ):
                if data is None:
                    continue
                for j, name in enumerate(names):
                    value = data[i, : self.input_size, j].masked_fill(
                        ~mask[i, :, 0], float("nan")
                    )
                    past[name] = value.cpu().float().numpy()
            if futr is not None:
                future = {
                    name: futr[i, self.input_size :, j].cpu().float().numpy()
                    for j, name in enumerate(self.futr_exog_list)
                }
            inputs.append(
                {
                    "target": target,
                    "past_covariates": past,
                    "future_covariates": future,
                }
            )
        forecasts = backend.predict(
            inputs,
            prediction_length=self.h,
            context_length=self.input_size,
            cross_learning=False,
        )
        quantiles = np.asarray(backend.quantiles, dtype=float)
        median = np.flatnonzero(np.isclose(quantiles, 0.5))
        if len(median) != 1 or len(forecasts) != len(y):
            raise ValueError(
                "Chronos-2 must return one forecast per input and a median quantile."
            )
        prediction = torch.stack(
            [item[0, int(median[0]), :] for item in forecasts]
        )
        return self._point_output(prediction, y)
