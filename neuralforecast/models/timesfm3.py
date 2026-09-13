"""TimesFM-3 pretrained forecasting adapter."""

import warnings

import numpy as np
import torch

from ..losses.pytorch import MQLoss
from ._exogenous import PretrainedExogenousModel
from ._research_utils import _positive

__all__ = ["TimesFM3"]


class TimesFM3(PretrainedExogenousModel):
    """Official TimesFM-3 inference with native numerical covariates and deciles.

    One NF target per series; historical covariates use cross-variate attention.
    This is NOT the TimesFM-2.5/XReg adapter. MAE/MSE return the native median;
    MQLoss accepts a subset of native deciles, without quantile interpolation.
    Default pretrained weights are non-commercial and non-production only.
    Complete history is required; no implicit imputation or covariate dropping.
    """

    DEFAULT_MODEL_ID = "google/timesfm-3.0-pytorch"
    NATIVE_QUANTILES = True

    def __init__(
        self,
        h,
        input_size,
        backend_batch_size=16,
        use_symmetric_averaging=False,
        **kwargs,
    ):
        _positive(backend_batch_size=backend_batch_size)
        if not isinstance(use_symmetric_averaging, bool):
            raise ValueError(
                "use_symmetric_averaging must be boolean."
            )
        super().__init__(h=h, input_size=input_size, **kwargs)
        if input_size > 15360:
            raise ValueError(
                "TimesFM3 input_size must not exceed 15360; "
                "implicit truncation is disabled."
            )
        if 1 + self.hist_exog_size + self.futr_exog_size > 32:
            raise ValueError(
                "TimesFM3 accepts at most 32 target/covariate channels; "
                "none are silently dropped."
            )
        self.backend_batch_size = backend_batch_size
        self.use_symmetric_averaging = use_symmetric_averaging
        if isinstance(self.loss, MQLoss):
            self._decile_indices()

    def _decile_indices(self):
        grid = np.arange(1, 10) / 10
        indices = []
        for q in self.loss.quantiles.detach().cpu().numpy():
            found = np.flatnonzero(
                np.isclose(grid, q, rtol=0, atol=1e-6)
            )
            if len(found) != 1:
                raise ValueError(
                    "TimesFM3 MQLoss supports native quantiles "
                    "0.1, 0.2, ..., 0.9 only."
                )
            indices.append(int(found[0]))
        return indices

    def _load_backend(self):
        try:
            from timesfm3 import ModelConfig, TimesFM3Forecaster
        except ImportError as exc:
            raise ImportError(
                "Install the official timesfm checkout with TimesFM-3 support; "
                "see docs/short_horizon_models.md."
            ) from exc
        if self.model_id == self.DEFAULT_MODEL_ID:
            warnings.warn(
                "TimesFM-3 default weights: non-commercial, non-production "
                "use only. Review timesfm-non-commercial-license-v1.0 before use.",
                UserWarning,
                stacklevel=2,
            )
        return TimesFM3Forecaster(
            ModelConfig(
                checkpoint_path=self.model_id,
                revision=self.revision,
                per_core_batch_size=self.backend_batch_size,
                device=self.backend_device,
            )
        )

    @torch.no_grad()
    def forward(self, windows_batch):
        y, mask, hist, futr = self._inputs(windows_batch)
        self._complete_history(mask)
        to_arrays = lambda t: [
            v.transpose(0, 1).cpu().float().numpy() for v in t
        ]
        quantiles = isinstance(self.loss, MQLoss)
        outputs = list(
            self._get_backend().predict_batch(
                contexts=[
                    v[:, 0].cpu().float().numpy() for v in y
                ],
                horizon=self.h,
                past_only_covariates=(
                    None if hist is None else to_arrays(hist)
                ),
                past_future_covariates=(
                    None if futr is None else to_arrays(futr)
                ),
                return_quantiles=quantiles,
                sort_quantiles=True,
                make_positive=False,
                use_symmetric_averaging=self.use_symmetric_averaging,
            )
        )
        if len(outputs) != len(y):
            raise ValueError(
                "TimesFM3 must return one forecast per NF window."
            )
        if quantiles:
            values = [np.asarray(out.quantiles) for out in outputs]
            if any(v.shape != (self.h, 9) for v in values):
                raise ValueError(
                    "TimesFM3 must return [h, 9] native quantiles per target."
                )
            return self._quantile_output(
                np.stack(values)[:, :, self._decile_indices()],
                y,
            )
        values = [np.asarray(out.forecast) for out in outputs]
        if any(v.shape != (self.h,) for v in values):
            raise ValueError(
                "TimesFM3 must return [h] point forecasts per target."
            )
        return self._point_output(np.stack(values), y)
