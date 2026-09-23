"""CrossLinear forecasting port from mumiao2000/CrossLinear.

Copyright (c) 2025 LINKE Lab. MIT; see docs/exogenous_model_licenses.md.
Adapted from models/CrossLinear.py, blob 594ad517e8e3f188d77ce09d995fc6f3062827aa.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ._exogenous import ExogenousModel

__all__ = ["CrossLinear"]


class CrossLinear(ExogenousModel):
    """Cross-correlation embedding with patch-wise MLP forecasting.

    A native port of the official MS (multivariate-to-single-target) path.
    Historical covariates are placed before the target, as in the source.
    Known-future/static covariates are not supported by this architecture.

    Args:
        h (int): Forecast horizon.
        input_size (int): Complete historical context length, at least two.
        patch_len (int): Nonoverlapping patch length (right-padded internally).
        hidden_size (int): Patch embedding width.
        d_ff (int): MLP width.
        alpha (float): Initial trainable target/correlation mixing weight.
        beta (float): Initial trainable value/position mixing weight.
        **kwargs: NeuralForecast options, including hist_exog_list, MAE/MSE.

    References:
        https://github.com/mumiao2000/CrossLinear
    """

    EXOGENOUS_FUTR = False

    def __init__(
        self, h, input_size, patch_len=16, hidden_size=64, d_ff=128,
        alpha=0.5, beta=0.5, **kwargs,
    ):
        if any(not isinstance(v, int) or v < 1 for v in (patch_len, hidden_size, d_ff)):
            raise ValueError("patch_len, hidden_size and d_ff must be positive integers.")
        super().__init__(h=h, input_size=input_size, **kwargs)
        patches = math.ceil(input_size / patch_len)
        self.patch_len = patch_len
        self.pad_size = patches * patch_len - input_size
        self.alpha = nn.Parameter(torch.tensor([float(alpha)]))
        self.beta = nn.Parameter(torch.tensor([float(beta)]))
        self.correlation_embedding = nn.Conv1d(1 + self.hist_exog_size, 1, 3, padding=1)
        self.value_embedding = nn.Sequential(
            nn.LayerNorm([1, patches, patch_len]), nn.Linear(patch_len, d_ff),
            nn.LayerNorm([1, patches, d_ff]), nn.ReLU(),
            nn.Linear(d_ff, hidden_size), nn.LayerNorm([1, patches, hidden_size]),
            nn.ReLU(),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, patches, hidden_size))
        self.head = nn.Sequential(
            nn.Flatten(2), nn.Linear(patches * hidden_size, d_ff),
            nn.LayerNorm([1, d_ff]), nn.ReLU(), nn.Linear(d_ff, h),
        )

    def forward(self, windows_batch):
        y, mask, hist, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        x = torch.cat((hist, y), dim=-1) if hist is not None else y
        x = x.transpose(1, 2)
        mean, std = x.mean(-1, keepdim=True), x.std(-1, keepdim=True)
        x = (x - mean) / (std + 1e-5)
        target = self.alpha * x[:, -1:] + (1 - self.alpha) * self.correlation_embedding(x)
        patches = F.pad(target, (0, self.pad_size)).unfold(-1, self.patch_len, self.patch_len)
        embedded = self.beta * self.value_embedding(patches) + (1 - self.beta) * self.pos_embedding
        prediction = self.head(embedded) * std[:, -1:] + mean[:, -1:]
        return self._point_output(prediction.transpose(1, 2), y)
