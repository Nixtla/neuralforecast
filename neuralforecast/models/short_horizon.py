"""TimesFM-3 and trainable official SeesawNet/Dualformer window adapters.

Chronos2 is already exported from foundation.py and is deliberately not duplicated.
See docs/short_horizon_models.md for pinned dependencies, licenses and limitations.
"""

from types import SimpleNamespace
import warnings

import numpy as np
import torch

from ..losses.pytorch import MQLoss
from ._exogenous import ExogenousModel, PretrainedExogenousModel
from ._forecast_source import forecast_source
from .research import _full_windows, _positive

__all__ = ["TimesFM3", "SeesawNet", "Dualformer"]


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

    def __init__(self, h, input_size, backend_batch_size=16,
                 use_symmetric_averaging=False, **kwargs):
        _positive(backend_batch_size=backend_batch_size)
        if not isinstance(use_symmetric_averaging, bool):
            raise ValueError("use_symmetric_averaging must be boolean.")
        super().__init__(h=h, input_size=input_size, **kwargs)
        if input_size > 15360:
            raise ValueError("TimesFM3 input_size must not exceed 15360; implicit truncation is disabled.")
        if 1 + self.hist_exog_size + self.futr_exog_size > 32:
            raise ValueError("TimesFM3 accepts at most 32 target/covariate channels; none are silently dropped.")
        self.backend_batch_size = backend_batch_size
        self.use_symmetric_averaging = use_symmetric_averaging
        if isinstance(self.loss, MQLoss):
            self._decile_indices()

    def _decile_indices(self):
        grid = np.arange(1, 10) / 10
        indices = []
        for q in self.loss.quantiles.detach().cpu().numpy():
            found = np.flatnonzero(np.isclose(grid, q, rtol=0, atol=1e-6))
            if len(found) != 1:
                raise ValueError("TimesFM3 MQLoss supports native quantiles 0.1, 0.2, ..., 0.9 only.")
            indices.append(int(found[0]))
        return indices

    def _load_backend(self):
        try:
            from timesfm3 import ModelConfig, TimesFM3Forecaster
        except ImportError as exc:
            raise ImportError("Install the official timesfm checkout with TimesFM-3 support; see docs/short_horizon_models.md.") from exc
        if self.model_id == self.DEFAULT_MODEL_ID:
            warnings.warn(
                "TimesFM-3 default weights: non-commercial, non-production use only. "
                "Review timesfm-non-commercial-license-v1.0 before use.",
                UserWarning, stacklevel=2,
            )
        return TimesFM3Forecaster(ModelConfig(
            checkpoint_path=self.model_id, revision=self.revision,
            per_core_batch_size=self.backend_batch_size, device=self.backend_device,
        ))

    @torch.no_grad()
    def forward(self, windows_batch):
        y, mask, hist, futr = self._inputs(windows_batch)
        self._complete_history(mask)
        to_arrays = lambda t: [v.transpose(0, 1).cpu().float().numpy() for v in t]
        quantiles = isinstance(self.loss, MQLoss)
        outputs = list(self._get_backend().predict_batch(
            contexts=[v[:, 0].cpu().float().numpy() for v in y], horizon=self.h,
            past_only_covariates=None if hist is None else to_arrays(hist),
            past_future_covariates=None if futr is None else to_arrays(futr),
            return_quantiles=quantiles, sort_quantiles=True, make_positive=False,
            use_symmetric_averaging=self.use_symmetric_averaging,
        ))
        if len(outputs) != len(y):
            raise ValueError("TimesFM3 must return one forecast per NF window.")
        if quantiles:
            values = [np.asarray(out.quantiles) for out in outputs]
            if any(v.shape != (self.h, 9) for v in values):
                raise ValueError("TimesFM3 must return [h, 9] native quantiles per target.")
            return self._quantile_output(np.stack(values)[:, :, self._decile_indices()], y)
        values = [np.asarray(out.forecast) for out in outputs]
        if any(v.shape != (self.h,) for v in values):
            raise ValueError("TimesFM3 must return [h] point forecasts per target.")
        return self._point_output(np.stack(values), y)


class SeesawNet(ExogenousModel):
    """Train the official SeesawNet architecture using NF's requested point loss.

    source_dir is a pinned dreamone-Lee/SeesawNet checkout. Historical covariates
    are additional observed channels; only the NF target output is supervised.
    Future/static/categorical covariates are unsupported. This adapter does not
    claim reproduction of the paper's multivariate TFMAE training protocol.
    """

    EXOGENOUS_FUTR = False

    def __init__(self, h, input_size, source_dir=None, hidden_size=128, d_ff=256,
                 n_heads=8, patch_len=8, stride=4, pd_layers=1, cr_layers=2,
                 down_sample_rate=1.0, dropout=0.1, **kwargs):
        _positive(h=h, input_size=input_size, hidden_size=hidden_size, d_ff=d_ff, n_heads=n_heads,
                  patch_len=patch_len, stride=stride, pd_layers=pd_layers, cr_layers=cr_layers)
        if hidden_size % n_heads or patch_len > input_size:
            raise ValueError("hidden_size must divide by n_heads and patch_len <= input_size.")
        patch_num = (input_size - patch_len) // stride + 2
        if not 0 <= dropout < 1 or not 0 <= down_sample_rate <= 1:
            raise ValueError("dropout must be in [0,1), down_sample_rate in [0,1].")
        if down_sample_rate and int(patch_num * down_sample_rate) < 1:
            raise ValueError("down_sample_rate would produce zero aggregation tokens.")
        super().__init__(h=h, input_size=input_size, **_full_windows(kwargs))
        self.source_dir = source_dir
        config = SimpleNamespace(
            seq_len=input_size, pred_len=h, enc_in=1 + self.hist_exog_size,
            d_model=hidden_size, d_ff=d_ff, n_heads=n_heads, patch_len=patch_len,
            stride=stride, pd_layers=pd_layers, cr_layers=cr_layers,
            down_sample_rate=down_sample_rate, dropout=dropout,
            pe="zeros", learn_pe=True, activation="gelu", norm="LayerNorm", group=False,
        )
        self.network = forecast_source(source_dir, "SeesawNet").Model(config)

    def forward(self, windows_batch):
        y, mask, hist, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        inputs = y if hist is None else torch.cat((y, hist), dim=-1)
        result, _, _ = self.network(inputs)
        if result.shape != (len(y), self.h, inputs.shape[-1]):
            raise ValueError("SeesawNet returned an unexpected multi-channel shape.")
        return self._point_output(result[:, :, :1], y)


class Dualformer(ExogenousModel):
    """Train the official time-frequency Dualformer, not the namesake LLM.

    Historical covariates enter the official multivariate encoder. enc_in and
    c_out are kept equal so the upstream RevIN inverse remains valid, then the
    first (target) channel is returned. Known-future/calendar inputs are not
    synthesized. Requires e_layers >= 2 to avoid an upstream sampler division.
    """

    EXOGENOUS_FUTR = False

    def __init__(self, h, input_size, source_dir=None, hidden_size=128, d_ff=256,
                 n_heads=8, e_layers=2, dropout=0.1, **kwargs):
        _positive(h=h, input_size=input_size, hidden_size=hidden_size, d_ff=d_ff, n_heads=n_heads, e_layers=e_layers)
        if input_size < 8 or e_layers < 2 or hidden_size % n_heads or hidden_size % 2:
            raise ValueError("Dualformer requires input_size >= 8, e_layers >= 2 and even hidden_size divisible by n_heads.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1).")
        super().__init__(h=h, input_size=input_size, **_full_windows(kwargs))
        self.source_dir = source_dir
        channels = 1 + self.hist_exog_size
        config = SimpleNamespace(
            seq_len=input_size, pred_len=h, enc_in=channels, c_out=channels,
            d_model=hidden_size, d_ff=d_ff, n_heads=n_heads, e_layers=e_layers,
            dropout=dropout, factor=1, embed="timeF", freq="h", activation="gelu",
        )
        self.network = forecast_source(source_dir, "Dualformer").Model(config)

    def forward(self, windows_batch):
        y, mask, hist, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        inputs = y if hist is None else torch.cat((y, hist), dim=-1)
        result = self.network(inputs, None)
        if result.shape != (len(y), self.h, inputs.shape[-1]):
            raise ValueError("Dualformer returned an unexpected multi-channel shape.")
        return self._point_output(result[:, :, :1], y)
