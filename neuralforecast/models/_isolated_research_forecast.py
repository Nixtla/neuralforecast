"""Shared transport for isolated legacy research forecasting backends."""

from pathlib import Path

import torch

from ._exogenous import PretrainedExogenousModel
from ._research_source import run_worker


class _IsolatedResearchForecast(PretrainedExogenousModel):
    """Shared NPZ transport for incompatible legacy research backends."""

    def __init__(
        self, h, input_size, backend_python=None, backend_timeout=600, **kwargs
    ):
        if (
            not isinstance(backend_timeout, int)
            or isinstance(backend_timeout, bool)
            or backend_timeout < 1
        ):
            raise ValueError("backend_timeout must be a positive integer.")
        super().__init__(h=h, input_size=input_size, **kwargs)
        self.backend_python, self.backend_timeout = backend_python, backend_timeout
        if not self.model_id:
            raise ValueError(
                "An explicit locally trained official checkpoint is required as model_id."
            )
        if self.revision is not None:
            raise ValueError(
                "This backend takes a local checkpoint, not a Hub revision."
            )
        self.model_id = str(self.model_id)

    def _configuration(self):
        return dict(
            kind=self.BACKEND_KIND,
            model_id=str(Path(self.model_id).expanduser().absolute()),
            h=self.h,
            input_size=self.input_size,
            num_samples=self.num_samples,
            device=self.backend_device,
            random_seed=self.random_seed,
        )

    @torch.no_grad()
    def forward(self, windows_batch):
        y, mask, hist, futr = self._inputs(windows_batch)
        self._complete_history(mask)
        arrays = {"y": y.cpu().float().numpy()}
        if hist is not None:
            arrays["hist"] = hist.cpu().float().numpy()
        if futr is not None:
            arrays["futr"] = futr.cpu().float().numpy()
        output = run_worker(
            self.backend_python, self._configuration(), arrays, self.backend_timeout
        )
        return self._point_output(output, y)
