"""RAG4CTS pretrained research forecasting adapter."""

import numpy as np
import pandas as pd
import torch

from ._exogenous import PretrainedExogenousModel
from ._research_source import source_module

__all__ = ["RAG4CTS"]


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

    def __init__(
        self,
        h,
        input_size,
        source_dir=None,
        query_size=None,
        neighbors=3,
        retrieval_stride=1,
        **kwargs,
    ):
        query_size = (
            min(32, (input_size - h) // 2)
            if query_size is None
            else query_size
        )
        for name, value in (
            ("query_size", query_size),
            ("neighbors", neighbors),
            ("retrieval_stride", retrieval_stride),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if input_size < 2 * query_size + h:
            raise ValueError(
                "RAG4CTS needs input_size >= 2*query_size+h "
                "for a strictly historical bank."
            )
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
            raise ImportError(
                "RAG4CTS requires chronos-forecasting with Chronos2Pipeline."
            ) from exc
        official = source_module(self.source_dir, "RAG4CTS")
        backend = Chronos2Pipeline.from_pretrained(
            self.model_id,
            device_map=self.backend_device,
            **self._hub_kwargs(),
        )
        config = dict(
            feature_mapping={
                "target": "y",
                "covariates": self.futr_exog_list,
            },
            model_window={
                "left_padding": self.query_size,
                "core_window": self.h,
                "right_padding": 0,
                "total_size": self.query_size + self.h,
            },
        )
        return official.RAGPipeline(
            config,
            backend,
            self.futr_exog_list,
            use_hint=False,
        )

    def _retrieval_inputs(self, target, covariates):
        """Build one query and its bank; zero unknown y has zero retrieval weight."""
        q, length = self.query_size, self.query_size + self.h
        history = pd.DataFrame(
            covariates[: self.input_size],
            columns=self.futr_exog_list,
        )
        history["y"] = target
        # Bank end <= input_size-query_size: no overlap with the query's history.
        last_start = self.input_size - q - length
        bank = [
            {"df": history.iloc[start : start + length].copy()}
            for start in range(
                0,
                last_start + 1,
                self.retrieval_stride,
            )
        ]
        query = pd.DataFrame(
            covariates[self.input_size - q :],
            columns=self.futr_exog_list,
        )
        query["y"] = np.concatenate(
            (target[-q:], np.zeros(self.h, dtype=np.float32))
        )
        return query, bank

    @torch.no_grad()
    def forward(self, windows_batch):
        y, mask, _, futr = self._inputs(windows_batch)
        self._complete_history(mask)
        pipeline = self._get_backend()
        forecasts = []
        for target, covariates in zip(
            y[:, :, 0].cpu().float().numpy(),
            futr.cpu().float().numpy(),
        ):
            query, bank = self._retrieval_inputs(target, covariates)
            result = pipeline.predict_with_fixed_k(
                query,
                bank,
                k=min(self.neighbors, len(bank)),
                strategy="coarse_wcos_fine_weuc",
                do_alignment=False,
            )
            if result.get("pred") is None:
                raise RuntimeError(
                    "Official RAG4CTS did not return a forecast; "
                    "check the backend/input schema."
                )
            forecasts.append(np.asarray(result["pred"]).reshape(self.h))
        return self._point_output(np.stack(forecasts), y)
