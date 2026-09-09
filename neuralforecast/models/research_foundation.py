"""Official inference integrations added in the second exogenous model batch.

Moirai2, ChronosX and BaguanTS use isolated backend interpreters. RAG4CTS uses
its actual retriever/input-construction pipeline with an official Chronos-2
backend. No external checkpoint weights are included in NF checkpoints.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ._exogenous import PretrainedExogenousModel
from ._research_source import run_worker, source_module
from .foundation import Moirai

__all__ = ["Moirai2", "ChronosX", "BaguanTS", "RAG4CTS"]


class Moirai2(Moirai):
    """Official Moirai 2.0 quantile model, exposing its median (not quantile mean).

    Uses the same isolated uni2ts environment and numerical past/future fields
    as Moirai. Its official architecture fixes patch size and quantile levels;
    inherited patch_size=16/num_samples=100 are compatibility metadata only.
    Nondefault values are rejected instead of pretending to alter this backend.
    """

    DEFAULT_MODEL_ID = "Salesforce/moirai-2.0-R-small"
    BACKEND_KIND = "moirai2"

    def __init__(self, h, input_size, **kwargs):
        if kwargs.get("patch_size", 16) != 16 or kwargs.get("num_samples", 100) != 100:
            raise ValueError("Moirai2 uses the official fixed patches/quantiles; do not change patch_size or num_samples.")
        super().__init__(h=h, input_size=input_size, **kwargs)


class _IsolatedResearchForecast(PretrainedExogenousModel):
    """Shared NPZ transport for the two incompatible legacy research backends."""

    def __init__(self, h, input_size, backend_python=None, backend_timeout=600, **kwargs):
        if not isinstance(backend_timeout, int) or isinstance(backend_timeout, bool) or backend_timeout < 1:
            raise ValueError("backend_timeout must be a positive integer.")
        super().__init__(h=h, input_size=input_size, **kwargs)
        self.backend_python, self.backend_timeout = backend_python, backend_timeout
        if not self.model_id:
            raise ValueError("An explicit locally trained official checkpoint is required as model_id.")
        if self.revision is not None:
            raise ValueError("This backend takes a local checkpoint, not a Hub revision.")
        self.model_id = str(self.model_id)

    def _configuration(self):
        return dict(kind=self.BACKEND_KIND, model_id=str(Path(self.model_id).expanduser().absolute()),
                    h=self.h, input_size=self.input_size, num_samples=self.num_samples,
                    device=self.backend_device, random_seed=self.random_seed)

    @torch.no_grad()
    def forward(self, windows_batch):
        y, mask, hist, futr = self._inputs(windows_batch)
        self._complete_history(mask)
        arrays = {"y": y.cpu().float().numpy()}
        if hist is not None:
            arrays["hist"] = hist.cpu().float().numpy()
        if futr is not None:
            arrays["futr"] = futr.cpu().float().numpy()
        output = run_worker(self.backend_python, self._configuration(), arrays, self.backend_timeout)
        return self._point_output(output, y)


class ChronosX(_IsolatedResearchForecast):
    """Official ChronosX IIB+OIB inference from a *fine-tuned* local checkpoint.

    Args:
        h: Forecast horizon.
        input_size: Complete context, within the checkpoint's context limit.
        model_id: Local ChronosX checkpoint directory containing safetensors.
            Base amazon/chronos-t5 weights are not a trained covariate model.
        hidden_dim: Injection-block width used during original fine-tuning.
        num_layers: Injection-block depth used during original fine-tuning.
        **kwargs: PretrainedExogenousModel options and a separate backend_python.

    Channels follow the official HF data loader: values then missing indicators.
    Historical-only features have -1 and indicator=1 in the future. Known-future
    values are passed unchanged, with indicator=0. Feature order is hist then
    futr and must match the order used to train the checkpoint. NF fit does not
    fine-tune the model; use the original ChronosX training pipeline for that.
    """

    BACKEND_KIND = "chronosx"

    def __init__(self, h, input_size, hidden_dim=256, num_layers=1, **kwargs):
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (hidden_dim, num_layers)):
            raise ValueError("hidden_dim and num_layers must be positive integers.")
        super().__init__(h=h, input_size=input_size, **kwargs)
        if not self.hist_exog_size + self.futr_exog_size:
            raise ValueError("ChronosX requires at least one numerical exogenous feature.")
        self.hidden_dim, self.num_layers = hidden_dim, num_layers

    def _configuration(self):
        return dict(super()._configuration(), hidden_dim=self.hidden_dim, num_layers=self.num_layers)


class BaguanTS(_IsolatedResearchForecast):
    """Official Baguan-TS TS-tabular pipeline with window-local retrieval.

    Args:
        h: Forecast horizon, strictly smaller than input_size.
        input_size: Historical context length.
        source_dir: Checkout of jxgogo/Baguan-TS, in the backend environment.
        config_path: Trusted official model YAML matching the checkpoint.
        model_id: Explicit local Baguan checkpoint (tensor-only state_dict).
        context_size: Retrieved sequence width, h < context_size <= input_size.
        neighbors: Maximum number of retrieved historical contexts.
        num_samples: Official mF/n_repeat count, not forecast quantile count.
        **kwargs: NF options, nonempty futr_exog_list and separate backend_python.

    The worker calls the unchanged official predict method. Its small safe
    initializer avoids the source's unused undefined StandardScaler reference
    and loads checkpoint tensors with weights_only=True, never unsafe fallback.
    No trained Baguan checkpoint is bundled or implicitly substituted.
    """

    BACKEND_KIND = "baguants"
    EXOGENOUS_HIST = False

    def __init__(self, h, input_size, source_dir=None, config_path=None,
                 context_size=None, neighbors=5, num_samples=1, **kwargs):
        context_size = max(h + 1, input_size // 2) if context_size is None else context_size
        if not isinstance(context_size, int) or isinstance(context_size, bool) or not h < context_size <= input_size:
            raise ValueError("BaguanTS requires h < context_size <= input_size.")
        if not isinstance(neighbors, int) or isinstance(neighbors, bool) or neighbors < 1:
            raise ValueError("neighbors must be a positive integer.")
        if source_dir is None or config_path is None:
            raise ValueError("BaguanTS requires source_dir and a trusted official config_path.")
        super().__init__(h=h, input_size=input_size, num_samples=num_samples, **kwargs)
        if not self.futr_exog_size:
            raise ValueError("BaguanTS requires nonempty futr_exog_list.")
        self.source_dir, self.config_path = str(source_dir), str(config_path)
        self.context_size, self.neighbors = context_size, neighbors

    def _configuration(self):
        return dict(super()._configuration(),
                    source_dir=str(Path(self.source_dir).expanduser().absolute()),
                    config_path=str(Path(self.config_path).expanduser().absolute()),
                    context_size=self.context_size, neighbors=self.neighbors)


class RAG4CTS(PretrainedExogenousModel):
    """Official RAG4CTS fixed-k forecasting using only each NF window's past bank.

    Args:
        h: Forecast horizon.
        input_size: History, large enough for a reference and query context.
        source_dir: Checkout of RAG4CTS-Project/RAG4CTS.
        query_size: Final history segment to forecast from. Earlier complete
            query_size+h windows become the retrieval bank; it never crosses
            into the query context or uses target labels from the forecast.
        neighbors: Requested maximum k, capped by the available reference bank.
        retrieval_stride: Stride between candidate historical reference windows.
        **kwargs: PretrainedExogenousModel options and nonempty futr_exog_list.

    Uses the official coarse_wcos_fine_weuc retrieval and predict_with_fixed_k
    path. Target hints, future target observations, value alignment and adaptive
    k-search are deliberately disabled. This is the fixed-k forecast variant,
    not the source's right-context reconstruction/benchmark configuration.
    """

    DEFAULT_MODEL_ID = "amazon/chronos-2"
    EXOGENOUS_HIST = False

    def __init__(self, h, input_size, source_dir=None, query_size=None,
                 neighbors=3, retrieval_stride=1, **kwargs):
        query_size = min(32, (input_size - h) // 2) if query_size is None else query_size
        for name, value in (("query_size", query_size), ("neighbors", neighbors), ("retrieval_stride", retrieval_stride)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if input_size < 2 * query_size + h:
            raise ValueError("RAG4CTS needs input_size >= 2*query_size+h for a strictly historical bank.")
        if source_dir is None:
            raise ValueError("RAG4CTS requires source_dir.")
        super().__init__(h=h, input_size=input_size, **kwargs)
        if not self.futr_exog_size:
            raise ValueError("RAG4CTS requires nonempty futr_exog_list.")
        if set(self.futr_exog_list) & {"y", "id", "time"}:
            raise ValueError("RAG4CTS reserves the column names y, id and time.")
        self.source_dir, self.query_size = str(source_dir), query_size
        self.neighbors, self.retrieval_stride = neighbors, retrieval_stride

    def _load_backend(self):
        try:
            from chronos import Chronos2Pipeline
        except ImportError as exc:
            raise ImportError("RAG4CTS requires chronos-forecasting with Chronos2Pipeline.") from exc
        official = source_module(self.source_dir, "RAG4CTS")
        backend = Chronos2Pipeline.from_pretrained(self.model_id, device_map=self.backend_device, **self._hub_kwargs())
        config = dict(feature_mapping={"target": "y", "covariates": self.futr_exog_list},
                      model_window={"left_padding": self.query_size, "core_window": self.h,
                                    "right_padding": 0, "total_size": self.query_size + self.h})
        return official.RAGPipeline(config, backend, self.futr_exog_list, use_hint=False)

    def _retrieval_inputs(self, target, covariates):
        """Build one query and its bank; zero unknown y has zero retrieval weight."""
        q, length = self.query_size, self.query_size + self.h
        history = pd.DataFrame(covariates[:self.input_size], columns=self.futr_exog_list)
        history["y"] = target
        # Bank end <= input_size-query_size: no overlap with the query's history.
        last_start = self.input_size - q - length
        bank = [{"df": history.iloc[start:start + length].copy()}
                for start in range(0, last_start + 1, self.retrieval_stride)]
        query = pd.DataFrame(covariates[self.input_size - q:], columns=self.futr_exog_list)
        query["y"] = np.concatenate((target[-q:], np.zeros(self.h, dtype=np.float32)))
        return query, bank

    @torch.no_grad()
    def forward(self, windows_batch):
        y, mask, _, futr = self._inputs(windows_batch)
        self._complete_history(mask)
        pipeline = self._get_backend()
        forecasts = []
        for target, covariates in zip(y[:, :, 0].cpu().float().numpy(), futr.cpu().float().numpy()):
            query, bank = self._retrieval_inputs(target, covariates)
            result = pipeline.predict_with_fixed_k(query, bank, k=min(self.neighbors, len(bank)),
                                                  strategy="coarse_wcos_fine_weuc", do_alignment=False)
            if result.get("pred") is None:
                raise RuntimeError("Official RAG4CTS did not return a forecast; check the backend/input schema.")
            forecasts.append(np.asarray(result["pred"]).reshape(self.h))
        return self._point_output(np.stack(forecasts), y)
