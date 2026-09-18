"""Official pretrained forecast backends with explicit numerical covariate routing.

No model weights are bundled. Dependencies are optional; imports are lazy.
Moirai uses an isolated interpreter because uni2ts requires torch<2.5 whereas
this NeuralForecast revision requires torch>=2.9.1.
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from ._exogenous import PretrainedExogenousModel

__all__ = ["Chronos2", "Moirai", "MoiraiMoE", "TimesFM", "Toto"]


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
            raise ImportError("Install chronos-forecasting with Chronos2Pipeline support.") from exc
        return Chronos2Pipeline.from_pretrained(
            self.model_id, device_map=self.backend_device, **self._hub_kwargs()
        )

    @torch.no_grad()
    def forward(self, windows_batch):
        y, mask, hist, futr = self._inputs(windows_batch)
        backend = self._get_backend()
        inputs = []
        for i in range(len(y)):
            target = y[i, :, 0].masked_fill(~mask[i, :, 0], float("nan")).cpu().float().numpy()
            past, future = {}, {}
            for data, names in ((hist, self.hist_exog_list), (futr, self.futr_exog_list)):
                if data is None:
                    continue
                for j, name in enumerate(names):
                    value = data[i, :self.input_size, j].masked_fill(~mask[i, :, 0], float("nan"))
                    past[name] = value.cpu().float().numpy()
            if futr is not None:
                future = {
                    name: futr[i, self.input_size:, j].cpu().float().numpy()
                    for j, name in enumerate(self.futr_exog_list)
                }
            inputs.append({"target": target, "past_covariates": past, "future_covariates": future})
        forecasts = backend.predict(
            inputs, prediction_length=self.h, context_length=self.input_size,
            cross_learning=False,
        )
        quantiles = np.asarray(backend.quantiles, dtype=float)
        median = np.flatnonzero(np.isclose(quantiles, 0.5))
        if len(median) != 1 or len(forecasts) != len(y):
            raise ValueError("Chronos-2 must return one forecast per input and a median quantile.")
        prediction = torch.stack([item[0, int(median[0]), :] for item in forecasts])
        return self._point_output(prediction, y)


class Moirai(PretrainedExogenousModel):
    """Official Moirai-1.1 via a separate uni2ts Python environment.

    Args:
        h (int): Forecast horizon.
        input_size (int): Historical context length.
        backend_python (str): Interpreter path in an environment containing
            official uni2ts (torch<2.5). Required at prediction time.
        patch_size (int): Supported explicit patch width; auto is not supported.
        backend_timeout (int): Maximum seconds for one backend batch.
        **kwargs: PretrainedExogenousModel options and NF covariate lists.

    NF checkpoints retain the interpreter/model reference, not uni2ts weights.
    Both historical and known-future numerical covariates are passed unchanged
    on observed timestamps. See docs/exogenous_models.md for environment setup.

    References:
        https://github.com/SalesforceAIResearch/uni2ts
    """

    DEFAULT_MODEL_ID = "Salesforce/moirai-1.1-R-small"
    BACKEND_KIND = "moirai"

    def __init__(
        self, h, input_size, backend_python=None, patch_size=16,
        backend_timeout=600, **kwargs,
    ):
        if not isinstance(patch_size, int) or patch_size < 1:
            raise ValueError("patch_size must be an explicit positive integer.")
        if not isinstance(backend_timeout, int) or backend_timeout < 1:
            raise ValueError("backend_timeout must be a positive integer.")
        super().__init__(h=h, input_size=input_size, **kwargs)
        self.backend_python = backend_python
        self.patch_size = patch_size
        self.backend_timeout = backend_timeout

    @torch.no_grad()
    def forward(self, windows_batch):
        y, mask, hist, futr = self._inputs(windows_batch)
        if self.backend_python is None:
            raise ValueError("Set backend_python to the separate uni2ts environment's Python. See docs/exogenous_models.md.")
        executable = Path(self.backend_python).expanduser().absolute()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(f"backend_python is not an executable file: {executable}")
        arrays = {
            "past_target": y.cpu().float().numpy(),
            "past_observed_target": mask.cpu().numpy(),
            "past_is_pad": (~mask[:, :, 0].cumsum(dim=1).bool()).cpu().numpy(),
        }
        if hist is not None:
            arrays["past_feat_dynamic_real"] = hist.cpu().float().numpy()
            arrays["past_observed_feat_dynamic_real"] = mask.expand_as(hist).cpu().numpy()
        if futr is not None:
            arrays["feat_dynamic_real"] = futr.cpu().float().numpy()
            known_mask = torch.cat((mask, mask.new_ones(len(y), self.h, 1)), dim=1)
            arrays["observed_feat_dynamic_real"] = known_mask.expand_as(futr).cpu().numpy()
        config = dict(
            kind=self.BACKEND_KIND, model_id=self.model_id, revision=self.revision,
            h=self.h, input_size=self.input_size, patch_size=self.patch_size,
            hist_size=self.hist_exog_size, futr_size=self.futr_exog_size,
            num_samples=self.num_samples, device=self.backend_device,
            random_seed=self.random_seed,
        )
        # ponytail: one process/weight load per batch; persistent workers are the
        # upgrade path for high-throughput rolling-window forecasting.
        with tempfile.TemporaryDirectory(prefix="nf-uni2ts-") as directory:
            directory = Path(directory)
            np.savez(directory / "inputs.npz", **arrays)
            (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
            worker = Path(__file__).with_name("_uni2ts_worker.py")
            try:
                subprocess.run(
                    [str(executable), str(worker), str(directory)],
                    check=True, capture_output=True, text=True,
                    timeout=self.backend_timeout,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"uni2ts backend failed:\n{exc.stderr[-3000:]}") from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("uni2ts backend timed out; increase backend_timeout or reduce the batch size.") from exc
            with np.load(directory / "outputs.npz", allow_pickle=False) as output:
                return self._point_output(output["prediction"].copy(), y)


class MoiraiMoE(Moirai):
    """Moirai-MoE-1.0 using the same isolated covariate interface as Moirai.

    This loads MoiraiMoEModule/MoiraiMoEForecast, not dense Moirai weights.
    """

    DEFAULT_MODEL_ID = "Salesforce/moirai-moe-1.0-R-small"
    BACKEND_KIND = "moirai_moe"


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
            raise ImportError("Install timesfm[torch] with TimesFM-2.5 support, jax and scikit-learn.") from exc
        backend = TimesFM_2p5_200M_torch.from_pretrained(
            self.model_id, torch_compile=False, **self._hub_kwargs()
        )
        backend.model.device = torch.device(self.backend_device)
        backend.model.device_count = 1
        backend.model.to(self.backend_device).eval()
        backend.compile(ForecastConfig(
            max_context=((self.input_size + 31) // 32) * 32,
            max_horizon=((self.h + 127) // 128) * 128,
            per_core_batch_size=1, return_backcast=True,
            use_continuous_quantile_head=False,
        ))
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
                    inputs=inputs, dynamic_numerical_covariates=covariates,
                    xreg_mode="xreg + timesfm", ridge=1e-3, force_on_cpu=True,
                )
            outputs.append(np.asarray(result[0])[-self.h:])
        return self._point_output(np.stack(outputs), y)


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
            raise ImportError("Install toto-ts with the Toto-1.0 API. See docs/exogenous_models.md.") from exc
        model = OfficialToto.from_pretrained(self.model_id, **self._hub_kwargs())
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
            parts.append(futr[:, :self.input_size])
        series = torch.cat(parts, dim=-1).transpose(1, 2).to(self.backend_device).float()
        inputs = input_type(
            series=series,
            padding_mask=mask.transpose(1, 2).expand_as(series).to(self.backend_device),
            id_mask=torch.zeros_like(series, dtype=torch.long),
            timestamp_seconds=torch.zeros_like(series, dtype=torch.long),
            time_interval_seconds=torch.ones(series.shape[:2], device=series.device, dtype=torch.long),
            num_exogenous_variables=self.futr_exog_size,
        )
        future = None if futr is None else futr[:, self.input_size:].transpose(1, 2).to(self.backend_device).float()
        result = forecaster.forecast(
            inputs, prediction_length=self.h, num_samples=self.num_samples,
            samples_per_batch=min(10, self.num_samples),
            future_exogenous_variables=future,
        )
        return self._point_output(result.mean[:, 0, :], y)
