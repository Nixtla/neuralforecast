"""TimesFM-2.5 pretrained forecasting adapter."""

import numpy as np
import torch

from ._exogenous import PretrainedExogenousModel

__all__ = ["TimesFM"]


class TimesFM(PretrainedExogenousModel):
    """TimesFM-2.5 with the official XReg known-future covariate pathway.

    Only futr_exog_list is supported: XReg requires historical AND future values
    of each numerical covariate. Historical-only variables are rejected. The
    covariate regression is fitted independently for each NF window to prevent
    pooling later windows' targets into earlier forecasts.

    Args:
        h (int): Forecast horizon.
        input_size (int): Complete context length.
        **kwargs: PretrainedExogenousModel options and futr_exog_list.

    Requires:
        timesfm[torch], jax and scikit-learn for the official XReg implementation.

    References:
        https://github.com/google-research/timesfm
    """

    DEFAULT_MODEL_ID = "google/timesfm-2.5-200m-pytorch"
    EXOGENOUS_HIST = False

    def _load_backend(self):
        try:
            from timesfm import ForecastConfig, TimesFM_2p5_200M_torch
        except ImportError as exc:
            raise ImportError(
                "Install timesfm[torch] with TimesFM-2.5 support, "
                "jax and scikit-learn."
            ) from exc
        backend = TimesFM_2p5_200M_torch.from_pretrained(
            self.model_id, torch_compile=False, **self._hub_kwargs()
        )
        backend.model.device = torch.device(self.backend_device)
        backend.model.device_count = 1
        backend.model.to(self.backend_device).eval()
        backend.compile(
            ForecastConfig(
                max_context=((self.input_size + 31) // 32) * 32,
                max_horizon=((self.h + 127) // 128) * 128,
                per_core_batch_size=1,
                return_backcast=True,
                use_continuous_quantile_head=False,
            )
        )
        return backend

    @torch.no_grad()
    def forward(self, windows_batch):
        y, mask, _, futr = self._inputs(windows_batch)
        self._complete_history(mask)
        backend = self._get_backend()
        outputs = []
        for i in range(len(y)):
            inputs = [y[i, :, 0].cpu().float().numpy()]
            if futr is None:
                result, _ = backend.forecast(horizon=self.h, inputs=inputs)
            else:
                covariates = {
                    name: [futr[i, :, j].cpu().float().numpy()]
                    for j, name in enumerate(self.futr_exog_list)
                }
                result, _ = backend.forecast_with_covariates(
                    inputs=inputs,
                    dynamic_numerical_covariates=covariates,
                    xreg_mode="xreg + timesfm",
                    ridge=1e-3,
                    force_on_cpu=True,
                )
            outputs.append(np.asarray(result[0])[-self.h :])
        return self._point_output(np.stack(outputs), y)
