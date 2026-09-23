"""ChatTime context-conditioned pretrained forecasting adapter."""

from pathlib import Path

import torch

from ._context_source import official_module
from ._context_utils import (
    _context_indices,
    _context_schema,
    _contexts,
    _local_path,
    _options,
)
from ._exogenous import PretrainedExogenousModel

__all__ = ["ChatTime"]


class ChatTime(PretrainedExogenousModel):
    """Official ChatTime local Llama forecasting API, conditioned on descriptions.

    stat_exog_list contains one context_id. Only a compatible trained ChatTime
    checkpoint is meaningful; a generic Llama checkpoint is not a forecaster.
    Invalid/unparseable numerical generations raise instead of filling forecasts.
    """

    EXOGENOUS_HIST = False
    EXOGENOUS_FUTR = False
    EXOGENOUS_STAT = True

    def __init__(
        self,
        h,
        input_size,
        source_dir,
        model_id,
        contexts,
        stat_exog_list=None,
        **kwargs,
    ):
        self.contexts = _contexts(contexts)
        self.source_dir = str(Path(source_dir).expanduser().resolve())
        super().__init__(
            h=h,
            input_size=input_size,
            model_id=_local_path(model_id),
            stat_exog_list=stat_exog_list,
            **_options(kwargs),
        )
        _context_schema(self)

    def _load_backend(self):
        backend = official_module(
            self.source_dir,
            "ChatTime",
        ).ChatTime(
            self.model_id,
            hist_len=self.input_size,
            pred_len=self.h,
            num_samples=self.num_samples,
        )
        backend.model.to(self.backend_device).eval()
        return backend

    def forward(self, windows_batch):
        import numpy as np

        y, mask, _, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        ids = _context_indices(self, windows_batch, y)
        backend = self._get_backend()
        with torch.inference_mode():
            output = [
                backend.predict(
                    y[i, :, 0].detach().cpu().numpy().copy(),
                    context=self.contexts[idx],
                )
                for i, idx in enumerate(ids.tolist())
            ]
        return self._point_output(np.stack(output), y)
