__all__ = ['GatingBlock', 'XLinear']

from typing import Optional

import torch
import torch.nn as nn

from ..common._base_model import BaseModel
from ..losses.pytorch import MAE


class GatingBlock(nn.Module):
    """Gating block with sigmoid weighting for XLinear.

    Computes element-wise gating: output = x * sigmoid(MLP(x))
    """
    def __init__(self, d_model, hidden_ff, dropout=0.0):
        super().__init__()
        self.weight = nn.Sequential(
            nn.Linear(d_model, hidden_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_ff, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.weight(x)
        return x * weight


class XLinear(BaseModel):
    """XLinear

    XLinear is a linear-based model for multivariate time series forecasting
    that uses gating mechanisms for temporal and cross-channel interactions.
    The architecture consists of temporal gating with a global token to capture
    global temporal patterns, followed by cross-channel gating to model
    dependencies between different time series.

    Args:
        h (int): Forecast horizon.
        input_size (int): Input size, y=[1,2,3,4] input_size=2 -> lags=[1,2].
        n_series (int): Number of time series.
        stat_exog_list (str list): Static exogenous columns.
        hist_exog_list (str list): Historic exogenous columns.
        futr_exog_list (str list): Future exogenous columns.
        hidden_size (int): Dimension of the model embedding.
        temporal_ff (int): Dimension of temporal feedforward layer in gating block.
        channel_ff (int): Dimension of cross-channel feedforward layer in gating block.
        temporal_dropout (float): Dropout rate for temporal gating.
        channel_dropout (float): Dropout rate for cross-channel gating.
        embed_dropout (float): Dropout rate for embedding projection.
        head_dropout (float): Dropout rate for output head.
        use_norm (bool): Whether to use RevIN normalization.
        loss (PyTorch module): Instantiated train loss class from [losses collection](./losses.pytorch).
        valid_loss (PyTorch module): Instantiated valid loss class from [losses collection](./losses.pytorch).
        max_steps (int): Maximum number of training steps.
        learning_rate (float): Learning rate between (0, 1).
        num_lr_decays (int): Number of learning rate decays, evenly distributed across max_steps.
        early_stop_patience_steps (int): Number of validation iterations before early stopping.
        val_check_steps (int): Number of training steps between every validation loss check.
        batch_size (int): Number of different series in each batch.
        valid_batch_size (int): Number of different series in each validation and test batch, if None uses batch_size.
        windows_batch_size (int): Number of windows to sample in each training batch.
        inference_windows_batch_size (int): Number of windows to sample in each inference batch, -1 uses all.
        start_padding_enabled (bool): If True, the model will pad the time series with zeros at the beginning.
        training_data_availability_threshold (Union[float, List[float]]): minimum fraction of valid data points required for training windows. Single float applies to both insample and outsample; list of two floats specifies [insample_fraction, outsample_fraction]. Default 0.0 allows windows with only 1 valid data point (current behavior).
        step_size (int): Step size between each window of temporal data.
        scaler_type (str): type of scaler for temporal inputs normalization see [temporal scalers](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/common/_scalers.py).
        random_seed (int): Random seed for pytorch initializer and numpy generators.
        drop_last_loader (bool): If True, TimeSeriesDataLoader drops last non-full batch.
        alias (str): Optional custom name of the model.
        optimizer (Subclass of 'torch.optim.Optimizer'): Optional user specified optimizer.
        optimizer_kwargs (dict): Optional list of parameters used by the user specified optimizer.
        lr_scheduler (Subclass of 'torch.optim.lr_scheduler.LRScheduler'): Optional user specified lr_scheduler.
        lr_scheduler_kwargs (dict): Optional list of parameters used by the user specified lr_scheduler.
        dataloader_kwargs (dict): optional, list of parameters passed into the PyTorch Lightning dataloader by the `TimeSeriesDataLoader`.
        **trainer_kwargs (keyword): trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).

    References:
        - [Xinyang, C., et al. "XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs"](https://arxiv.org/abs/2601.09237)
    """

    # Class attributes
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True
    MULTIVARIATE = True
    RECURRENT = False

    def __init__(
        self,
        h,
        input_size,
        n_series,
        stat_exog_list=None,
        hist_exog_list=None,
        futr_exog_list=None,
        exclude_insample_y=False,
        hidden_size: int = 128,
        temporal_ff: int = 256,
        channel_ff: int = 256,
        temporal_dropout: float = 0.0,
        channel_dropout: float = 0.0,
        embed_dropout: float = 0.0,
        head_dropout: float = 0.0,
        use_norm: bool = True,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size=32,
        inference_windows_batch_size=32,
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
        super(XLinear, self).__init__(
            h=h,
            input_size=input_size,
            n_series=n_series,
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

        # Architecture parameters
        self.hidden_size = hidden_size
        self.use_norm = use_norm

        # Projection from input_size to hidden_size
        self.projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Dropout(embed_dropout)
        )

        # Global token for temporal attention
        self.glob_token = nn.Parameter(torch.randn(1, n_series, hidden_size))

        # Temporal gating block (operates on 2*hidden_size: embedding + global token)
        self.temporal_gating = GatingBlock(2 * hidden_size, temporal_ff, temporal_dropout)

        # Cross-channel gating block (operates on 2*n_series: original + global)
        self.channel_gating = GatingBlock(2 * n_series, channel_ff, channel_dropout)

        # Exogenous feature projections
        if self.hist_exog_size > 0:
            self.hist_exog_projection = nn.Linear(
                self.hist_exog_size * input_size, hidden_size
            )

        if self.futr_exog_size > 0:
            self.futr_exog_projection = nn.Linear(
                self.futr_exog_size * (input_size + h), hidden_size
            )

        if self.stat_exog_size > 0:
            self.stat_exog_projection = nn.Linear(self.stat_exog_size, hidden_size)

        # Calculate head input size
        head_input_size = 2 * hidden_size
        if self.hist_exog_size > 0:
            head_input_size += hidden_size
        if self.futr_exog_size > 0:
            head_input_size += hidden_size
        if self.stat_exog_size > 0:
            head_input_size += hidden_size

        # Output head
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(head_input_size, h * self.loss.outputsize_multiplier)
        )

    def forward(self, windows_batch):
        # Parse windows_batch
        insample_y = windows_batch["insample_y"]  # [B, L, N]
        hist_exog = windows_batch["hist_exog"]    # [B, X, L, N]
        futr_exog = windows_batch["futr_exog"]    # [B, F, L+h, N]
        stat_exog = windows_batch["stat_exog"]    # [N, S]

        batch_size = insample_y.shape[0]

        # RevIN normalization
        if self.use_norm:
            means = insample_y.mean(dim=1, keepdim=True).detach()
            insample_y = insample_y - means
            stdev = torch.sqrt(
                torch.var(insample_y, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            insample_y = insample_y / stdev

        # Transpose: [B, L, N] -> [B, N, L]
        x = insample_y.permute(0, 2, 1)

        # Project: [B, N, L] -> [B, N, hidden_size]
        emb = self.projection(x)

        # Expand global token: [1, N, hidden_size] -> [B, N, hidden_size]
        glob_token = self.glob_token.expand(batch_size, -1, -1)

        # Temporal gating: concatenate embedding and global token
        # [B, N, 2*hidden_size]
        en_emb = torch.cat([emb, glob_token], dim=-1)
        en_atten = self.temporal_gating(en_emb)

        # Split back
        origin_atten = en_atten[:, :, :self.hidden_size]  # [B, N, hidden_size]
        glob_atten = en_atten[:, :, self.hidden_size:]    # [B, N, hidden_size]

        # Cross-channel gating
        # Concatenate along channel dimension: [B, 2N, hidden_size]
        ex_emb = torch.cat([emb, glob_atten], dim=1)
        # Permute for channel gating: [B, hidden_size, 2N]
        ex_atten = self.channel_gating(ex_emb.permute(0, 2, 1))

        # Extract global component: [B, hidden_size, N] (second half of channels)
        glob = ex_atten[:, :, self.n_series:]

        # Final endogenous representation: [B, N, 2*hidden_size]
        en = torch.cat([origin_atten, glob.permute(0, 2, 1)], dim=-1)

        # Process exogenous features
        exog_features = []

        if self.hist_exog_size > 0:
            # hist_exog: [B, X, L, N] -> [B, N, X*L]
            hist_exog_flat = hist_exog.permute(0, 3, 1, 2)  # [B, N, X, L]
            hist_exog_flat = hist_exog_flat.reshape(batch_size, self.n_series, -1)
            hist_exog_emb = self.hist_exog_projection(hist_exog_flat)  # [B, N, hidden_size]
            exog_features.append(hist_exog_emb)

        if self.futr_exog_size > 0:
            # futr_exog: [B, F, L+h, N] -> [B, N, F*(L+h)]
            futr_exog_flat = futr_exog.permute(0, 3, 1, 2)  # [B, N, F, L+h]
            futr_exog_flat = futr_exog_flat.reshape(batch_size, self.n_series, -1)
            futr_exog_emb = self.futr_exog_projection(futr_exog_flat)  # [B, N, hidden_size]
            exog_features.append(futr_exog_emb)

        if self.stat_exog_size > 0:
            # stat_exog: [N, S] -> [B, N, hidden_size]
            stat_exog_emb = self.stat_exog_projection(stat_exog)  # [N, hidden_size]
            stat_exog_emb = stat_exog_emb.unsqueeze(0).expand(batch_size, -1, -1)
            exog_features.append(stat_exog_emb)

        # Concatenate all features
        if exog_features:
            exog_combined = torch.cat(exog_features, dim=-1)  # [B, N, exog_hidden]
            en = torch.cat([en, exog_combined], dim=-1)  # [B, N, head_input_size]

        # Output head: [B, N, head_input_size] -> [B, N, h * output_multiplier]
        dec_out = self.head(en)

        # Permute: [B, N, h * output_multiplier] -> [B, h * output_multiplier, N]
        dec_out = dec_out.permute(0, 2, 1)

        # Reverse normalization
        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.h * self.loss.outputsize_multiplier, 1
            )
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(
                1, self.h * self.loss.outputsize_multiplier, 1
            )

        # Reshape to expected output: [B, h, N * output_multiplier]
        dec_out = dec_out.reshape(batch_size, self.h, -1)

        return dec_out
