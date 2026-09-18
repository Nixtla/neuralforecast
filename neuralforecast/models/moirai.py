"""Moirai pretrained forecasting adapter."""

import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from ._exogenous import PretrainedExogenousModel

__all__ = ["Moirai"]


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
        self,
        h,
        input_size,
        backend_python=None,
        patch_size=16,
        backend_timeout=600,
        **kwargs,
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
            raise ValueError(
                "Set backend_python to the separate uni2ts environment's Python. "
                "See docs/exogenous_models.md."
            )
        executable = Path(self.backend_python).expanduser().absolute()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(
                f"backend_python is not an executable file: {executable}"
            )
        arrays = {
            "past_target": y.cpu().float().numpy(),
            "past_observed_target": mask.cpu().numpy(),
            "past_is_pad": (~mask[:, :, 0].cumsum(dim=1).bool()).cpu().numpy(),
        }
        if hist is not None:
            arrays["past_feat_dynamic_real"] = hist.cpu().float().numpy()
            arrays["past_observed_feat_dynamic_real"] = (
                mask.expand_as(hist).cpu().numpy()
            )
        if futr is not None:
            arrays["feat_dynamic_real"] = futr.cpu().float().numpy()
            known_mask = torch.cat(
                (mask, mask.new_ones(len(y), self.h, 1)), dim=1
            )
            arrays["observed_feat_dynamic_real"] = (
                known_mask.expand_as(futr).cpu().numpy()
            )
        config = dict(
            kind=self.BACKEND_KIND,
            model_id=self.model_id,
            revision=self.revision,
            h=self.h,
            input_size=self.input_size,
            patch_size=self.patch_size,
            hist_size=self.hist_exog_size,
            futr_size=self.futr_exog_size,
            num_samples=self.num_samples,
            device=self.backend_device,
            random_seed=self.random_seed,
        )
        # ponytail: one process/weight load per batch; persistent workers are the
        # upgrade path for high-throughput rolling-window forecasting.
        with tempfile.TemporaryDirectory(prefix="nf-uni2ts-") as directory:
            directory = Path(directory)
            np.savez(directory / "inputs.npz", **arrays)
            (directory / "config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            worker = Path(__file__).with_name("_uni2ts_worker.py")
            try:
                subprocess.run(
                    [str(executable), str(worker), str(directory)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=self.backend_timeout,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"uni2ts backend failed:\n{exc.stderr[-3000:]}"
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "uni2ts backend timed out; increase backend_timeout "
                    "or reduce the batch size."
                ) from exc
            with np.load(directory / "outputs.npz", allow_pickle=False) as output:
                return self._point_output(output["prediction"].copy(), y)
