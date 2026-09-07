"""Timer-XL covariate architecture adapted from thuml/OpenLTM.

Copyright (c) 2022 THUML @ Tsinghua University. MIT.
See docs/exogenous_model_licenses.md for attribution and source blob IDs.
"""

import math

import torch
from torch import nn

from ._exogenous import ExogenousModel

__all__ = ["TimerXL"]


class _TimeAttentionBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, d_ff, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.rotary_width = self.head_dim // 2
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.var_bias = nn.Embedding(2, n_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, hidden_size), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        theta = 10000.0 ** (-torch.arange(0, self.rotary_width, 2).float() / self.rotary_width)
        self.register_buffer("theta", theta, persistent=False)

    def _rotate(self, x):
        position = torch.arange(x.shape[-2], device=x.device, dtype=self.theta.dtype)
        angle = torch.outer(position, self.theta).repeat_interleave(2, dim=-1).to(x.dtype)
        left, right = x[..., :self.rotary_width], x[..., self.rotary_width:]
        paired = left.reshape(*left.shape[:-1], -1, 2)
        rotated = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1).flatten(-2)
        return torch.cat((left * angle.cos() + rotated * angle.sin(), right), dim=-1)

    def forward(self, x, allowed, same_var):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self._rotate(q), self._rotate(k)
        bias = self.var_bias(same_var.long()).permute(2, 0, 1)
        # Match OpenLTM's non-flash path: scale both QK and binary bias.
        scores = (q @ k.transpose(-1, -2) + bias) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allowed, float("-inf"))
        attended = self.dropout(scores.softmax(-1)) @ v
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        x = self.norm1(x + self.dropout(self.projection(attended)))
        return self.norm2(x + self.ff(x))


class TimerXL(ExogenousModel):
    """Trainable Timer-XL covariate path with a next-horizon token head.

    Uses channel-major patch tokens, half-head rotary projection, a learned
    same/different-variate attention bias and the official covariate mask.
    The final target token predicts h points; unlike upstream pretraining, NF
    optimizes this final horizon rather than every next token. This is a
    from-scratch forecasting port, NOT a pretrained Timer-XL checkpoint loader.

    Args:
        h (int): Forecast horizon / output-token width.
        input_size (int): Complete context, divisible by patch_len.
        patch_len (int): Historical token length.
        hidden_size (int): Embedding width; hidden_size/n_heads divisible by four.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of Timer layers.
        d_ff (int): Feed-forward width.
        dropout (float): Dropout probability.
        use_norm (bool): Apply source-style per-channel normalization.
        **kwargs: NeuralForecast options, including hist_exog_list.

    References:
        https://github.com/thuml/Timer-XL
        https://github.com/thuml/OpenLTM
    """

    EXOGENOUS_FUTR = False

    def __init__(
        self, h, input_size, patch_len=16, hidden_size=128, n_heads=4,
        n_layers=2, d_ff=256, dropout=0.1, use_norm=True, **kwargs,
    ):
        if any(not isinstance(v, int) or v < 1 for v in (patch_len, hidden_size, n_heads, n_layers, d_ff)):
            raise ValueError("Architecture sizes must be positive integers.")
        if input_size % patch_len or hidden_size % (4 * n_heads):
            raise ValueError("input_size must divide into patches; hidden_size/n_heads must be divisible by 4.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")
        super().__init__(h=h, input_size=input_size, **kwargs)
        self.patch_len = patch_len
        self.use_norm = use_norm
        self.embedding = nn.Linear(patch_len, hidden_size)
        self.blocks = nn.ModuleList([
            _TimeAttentionBlock(hidden_size, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, h)
        channels, patches = 1 + self.hist_exog_size, input_size // patch_len
        var_id = torch.arange(channels).repeat_interleave(patches)
        time_id = torch.arange(patches).repeat(channels)
        same = var_id[:, None] == var_id[None, :]
        allowed = same & (time_id[:, None] >= time_id[None, :])
        allowed[-patches:, :-patches] = True
        self.register_buffer("allowed", allowed, persistent=False)
        self.register_buffer("same_var", same, persistent=False)

    def forward(self, windows_batch):
        y, mask, hist, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        x = torch.cat((hist, y), dim=-1) if hist is not None else y
        mean = x.mean(1, keepdim=True).detach() if self.use_norm else torch.zeros_like(x[:, :1])
        std = (x.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt() if self.use_norm else torch.ones_like(mean)
        patches = ((x - mean) / std).transpose(1, 2).unfold(-1, self.patch_len, self.patch_len)
        x = self.embedding(patches).flatten(1, 2)
        for block in self.blocks:
            x = block(x, self.allowed, self.same_var)
        prediction = self.head(self.norm(x)[:, -1]).unsqueeze(-1)
        return self._point_output(prediction * std[:, :, -1:] + mean[:, :, -1:], y)
