


__all__ = ['Mamba']


import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common._base_model import BaseModel
from ..losses.pytorch import MAE


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        return (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        )


class MambaBlock(nn.Module):
    """Selective state-space block from the Mamba paper (Section 3.4)."""

    def __init__(self, hidden_size, d_inner, d_state, d_conv, dt_rank):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(hidden_size, d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
        )

        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize dt_proj.bias so softplus(bias) is log-uniform in
        # [dt_min, dt_max] (Mamba paper Sec 3.6). PyTorch's default Linear bias
        # init gives initial delta ~0.5, which combined with A in [-d_state, -1]
        # makes cumsum(delta*A) overflow exp(-.) in fp32 and produces NaNs in
        # the parallel selective scan. Small initial delta avoids that.
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = (
            torch.arange(1, d_state + 1, dtype=torch.float)
            .unsqueeze(0)
            .expand(d_inner, d_state)
            .contiguous()
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, hidden_size, bias=False)

    def forward(self, x):
        l = x.shape[1]

        x_and_res = self.in_proj(x)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :l]
        x = x.transpose(1, 2)

        x = F.silu(x)

        y = self._ssm(x)
        y = y * F.silu(res)

        return self.out_proj(y)

    def _ssm(self, x):
        # Algorithm 2 in Section 3.2 of the paper.
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        D = self.D.float()  # (d_inner,)

        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        delta, B, C = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)

        return self._selective_scan(x, delta, A, B, C, D)

    def _selective_scan(self, u, delta, A, B, C, D):
        # Chunked parallel selective scan: within each chunk of `CHUNK_SIZE`
        # steps we run the closed-form cumsum scan in parallel; across chunks
        # we compose the (A_chunk, h_chunk_end) summaries sequentially. This
        # keeps |cumsum(delta*A)| bounded inside a chunk (so exp(-log_A) stays
        # in fp32 range even as delta and A drift during training) while still
        # collapsing L per-step kernel launches into L/CHUNK_SIZE cross-chunk
        # ops — much faster than a pure per-step Python loop, especially on
        # MPS, and numerically stable.
        CHUNK_SIZE = 8
        bsz, l, d_inner = u.shape
        n = A.shape[1]

        pad = (-l) % CHUNK_SIZE
        if pad:
            u = F.pad(u, (0, 0, 0, pad))
            delta = F.pad(delta, (0, 0, 0, pad))
            B = F.pad(B, (0, 0, 0, pad))
            C = F.pad(C, (0, 0, 0, pad))
        l_padded = l + pad
        n_chunks = l_padded // CHUNK_SIZE

        u_c = u.view(bsz, n_chunks, CHUNK_SIZE, d_inner)
        delta_c = delta.view(bsz, n_chunks, CHUNK_SIZE, d_inner)
        B_c = B.view(bsz, n_chunks, CHUNK_SIZE, n)

        # Within-chunk parallel scan. The clamp on log_a is a stability floor:
        # the SSM is allowed to decay at most by exp(-8) per step (effectively
        # a state reset). Without it, an extreme |delta*A| during training
        # would overflow exp(-log_A) in fp32. The official Mamba avoids this
        # via its fused CUDA kernel; in pure PyTorch we need the clamp.
        log_a = (delta_c.unsqueeze(-1) * A).clamp(min=-8.0)
        log_A_w = torch.cumsum(log_a, dim=2)
        b_signed = delta_c.unsqueeze(-1) * B_c.unsqueeze(-2) * u_c.unsqueeze(-1)
        v = b_signed * torch.exp(-log_A_w)
        V_w = torch.cumsum(v, dim=2)
        exp_log_A_w = torch.exp(log_A_w)
        h_w = exp_log_A_w * V_w  # state at each within-chunk pos, starting from 0

        # Chunk summaries: A_end = prod of a within the chunk, h_end = state at
        # chunk end when starting from zero.
        A_end = exp_log_A_w[:, :, -1]
        h_end = h_w[:, :, -1]

        # Sequential scan across chunks to get the starting state for each chunk.
        chunk_starts = torch.empty(
            bsz, n_chunks, d_inner, n, device=u.device, dtype=u.dtype
        )
        state = torch.zeros(bsz, d_inner, n, device=u.device, dtype=u.dtype)
        for c in range(n_chunks):
            chunk_starts[:, c] = state
            state = A_end[:, c] * state + h_end[:, c]

        # Combine within-chunk accumulation with the propagated chunk start.
        h_full = h_w + exp_log_A_w * chunk_starts.unsqueeze(2)
        h_full = h_full.view(bsz, l_padded, d_inner, n)[:, :l]

        ys = (h_full * C[:, :l].unsqueeze(-2)).sum(dim=-1)
        return ys + u[:, :l] * D


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, d_inner, d_state, d_conv, dt_rank):
        super().__init__()
        self.mixer = MambaBlock(hidden_size, d_inner, d_state, d_conv, dt_rank)
        self.norm = RMSNorm(hidden_size)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x


class Mamba(BaseModel):
    """Mamba

    Linear-time sequence model with selective state spaces. Mamba replaces
    transformer self-attention with an input-dependent state-space recurrence,
    achieving O(L) scaling in sequence length while remaining competitive with
    attention-based models on long-range tasks. The neuralforecast adaptation
    follows the encoder/projection pattern used by other univariate sequence
    models: the encoder input is built by concatenating the
    target series with historic, static, and the past portion of future
    exogenous features, then projected to `hidden_size` and passed through a
    stack of Mamba residual blocks. The hidden states are projected from the
    input length to the forecast horizon; the future portion of the
    future-exogenous features is then concatenated to each horizon step before
    the final output projection, so known-future covariates inform every
    forecast step.

    The selective scan uses a chunked parallel scan: a closed-form cumsum
    formulation within fixed-size chunks (parallel), composed sequentially
    across chunks. This keeps the cumulative exponent bounded inside a chunk
    — so it stays numerically stable in fp32 even as `delta` and `A` drift
    during training — while still collapsing the per-step recurrence into
    a small number of cross-chunk ops. Runs efficiently on CPU/GPU/MPS with
    no dependency on the `mamba_ssm` CUDA kernels.

    Args:
        h (int): forecast horizon.
        input_size (int): considered autorregresive inputs (lags), y=[1,2,3,4] input_size=2 -> lags=[1,2].
        stat_exog_list (str list): static exogenous columns.
        hist_exog_list (str list): historic exogenous columns.
        futr_exog_list (str list): future exogenous columns.
        exclude_insample_y (bool): the model skips the autoregressive features y[t-input_size:t] if True.
        hidden_size (int): dimension of the model embedding.
        d_state (int): SSM state dimension (N in the paper).
        d_conv (int): kernel size of the causal 1D convolution before the SSM.
        expand (int): expansion factor for the inner dimension (d_inner = hidden_size * expand).
        e_layers (int): number of stacked Mamba residual blocks.
        dropout (float): dropout applied to the input feature projection.
        loss (PyTorch module): instantiated train loss class from [losses collection](./losses.pytorch.html).
        valid_loss (PyTorch module): instantiated valid loss class from [losses collection](./losses.pytorch.html).
        max_steps (int): maximum number of training steps.
        learning_rate (float): Learning rate between (0, 1).
        num_lr_decays (int): Number of learning rate decays, evenly distributed across max_steps.
        early_stop_patience_steps (int): Number of validation iterations before early stopping.
        val_monitor (str): metric to monitor for early stopping. Valid options: "ptl/val_loss", "valid_loss", "train_loss". Default: "ptl/val_loss".
        val_check_steps (int): Number of training steps between every validation loss check.
        batch_size (int): number of different series in each batch.
        valid_batch_size (int): number of different series in each validation and test batch, if None uses batch_size.
        windows_batch_size (int): number of windows to sample in each training batch, default uses all.
        inference_windows_batch_size (int): number of windows to sample in each inference batch, -1 uses all.
        start_padding_enabled (bool): if True, the model will pad the time series with zeros at the beginning, by input size.
        training_data_availability_threshold (Union[float, List[float]]): minimum fraction of valid data points required for training windows. Single float applies to both insample and outsample; list of two floats specifies [insample_fraction, outsample_fraction]. Default 0.0 allows windows with only 1 valid data point (current behavior).
        step_size (int): step size between each window of temporal data.
        scaler_type (str): type of scaler for temporal inputs normalization see [temporal scalers](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/common/_scalers.py).
        random_seed (int): random_seed for pytorch initializer and numpy generators.
        drop_last_loader (bool): if True `TimeSeriesDataLoader` drops last non-full batch.
        alias (str): optional,  Custom name of the model.
        optimizer (Subclass of 'torch.optim.Optimizer'): optional, user specified optimizer instead of the default choice (Adam).
        optimizer_kwargs (dict): optional, list of parameters used by the user specified `optimizer`.
        lr_scheduler (Subclass of 'torch.optim.lr_scheduler.LRScheduler'): optional, user specified lr_scheduler instead of the default choice (StepLR).
        lr_scheduler_kwargs (dict): optional, list of parameters used by the user specified `lr_scheduler`.
        dataloader_kwargs (dict): optional, list of parameters passed into the PyTorch Lightning dataloader by the `TimeSeriesDataLoader`.
        **trainer_kwargs (int):  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).

    References:
        - [Albert Gu, Tri Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"](https://arxiv.org/abs/2312.00752)
    """

    # Class attributes
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True
    MULTIVARIATE = False
    RECURRENT = False

    def __init__(
        self,
        h,
        input_size,
        stat_exog_list=None,
        hist_exog_list=None,
        futr_exog_list=None,
        exclude_insample_y=False,
        hidden_size: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        e_layers: int = 2,
        dropout: float = 0.1,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_monitor: str = "ptl/val_loss",
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size=1024,
        inference_windows_batch_size=-1,
        start_padding_enabled=False,
        training_data_availability_threshold=0.0,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        drop_last_loader: bool = False,
        alias: Optional[str] = None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs
    ):

        super(Mamba, self).__init__(
            h=h,
            input_size=input_size,
            stat_exog_list=stat_exog_list,
            hist_exog_list=hist_exog_list,
            futr_exog_list=futr_exog_list,
            exclude_insample_y=exclude_insample_y,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_monitor=val_monitor,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            training_data_availability_threshold=training_data_availability_threshold,
            step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            drop_last_loader=drop_last_loader,
            alias=alias,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs
        )

        self.hidden_size = hidden_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.e_layers = e_layers

        d_inner = hidden_size * expand
        dt_rank = math.ceil(hidden_size / 16)

        n_features = (
            1 + self.hist_exog_size + self.stat_exog_size + self.futr_exog_size
        )
        self.feature_projection = nn.Linear(n_features, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                ResidualBlock(hidden_size, d_inner, d_state, d_conv, dt_rank)
                for _ in range(e_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size)

        self.temporal_projection = nn.Linear(input_size, h)
        self.out_proj = nn.Linear(
            hidden_size + self.futr_exog_size, self.loss.outputsize_multiplier
        )

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]  # [B, L, 1]
        futr_exog = windows_batch["futr_exog"]  # [B, L+h, F]
        hist_exog = windows_batch["hist_exog"]  # [B, L, X]
        stat_exog = windows_batch["stat_exog"]  # [B, S]

        seq_len = insample_y.shape[1]
        encoder_input = insample_y
        if self.hist_exog_size > 0:
            encoder_input = torch.cat((encoder_input, hist_exog), dim=2)
        if self.stat_exog_size > 0:
            stat_exog_expanded = stat_exog.unsqueeze(1).expand(-1, seq_len, -1)
            encoder_input = torch.cat((encoder_input, stat_exog_expanded), dim=2)
        if self.futr_exog_size > 0:
            encoder_input = torch.cat(
                (encoder_input, futr_exog[:, :seq_len]), dim=2
            )

        x = self.feature_projection(encoder_input)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # Project sequence dimension from input_size to horizon h.
        x = x.transpose(1, 2)  # [B, hidden_size, L]
        x = self.temporal_projection(x)  # [B, hidden_size, h]
        x = x.transpose(1, 2)  # [B, h, hidden_size]

        if self.futr_exog_size > 0:
            x = torch.cat((x, futr_exog[:, -self.h:]), dim=-1)

        return self.out_proj(x)
