__all__ = ['PositionalEmbedding', 'STADSharp', 'SOFTSSharp']


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common._base_model import BaseModel
from ..common._modules import TransEncoder, TransEncoderLayer
from ..losses.pytorch import MAE
from .softs import DataEmbedding_inverted


class PositionalEmbedding(nn.Module):
    def __init__(self, d_series, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.register_buffer("position_embedding", torch.zeros(1, max_len, d_series))

        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_series, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / d_series)
        )

        self.position_embedding[:, :, 0::2] = torch.sin(position * div_term)
        self.position_embedding[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x, scale=1.0):
        return x + scale * self.position_embedding[:, : x.size(1)]


class STADSharp(nn.Module):
    """
    STar Aggregate Dispatch Module with stochastic variable-position encoding.
    """

    def __init__(self, d_series, d_core, dropout_rate=0.1, pe_keep_prob=0.5):
        super(STADSharp, self).__init__()

        self.positional_embedding = PositionalEmbedding(d_series)

        self.pe_scale = nn.Parameter(torch.tensor(1.0))
        self.pe_keep_prob = pe_keep_prob

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def add_positional_embedding(self, input):
        if self.training:
            if torch.rand(1, device=input.device).item() < self.pe_keep_prob:
                return self.positional_embedding(input, scale=self.pe_scale)
            return input

        return self.positional_embedding(
            input,
            scale=self.pe_keep_prob * self.pe_scale,
        )

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape

        input = self.add_positional_embedding(input)

        combined_mean = F.gelu(self.gen1(input))
        combined_mean = self.dropout1(combined_mean)
        combined_mean = self.gen2(combined_mean)

        if self.training:
            ratio = F.softmax(torch.nan_to_num(combined_mean), dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, channels, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(
                combined_mean * weight, dim=1, keepdim=True
            ).repeat(1, channels, 1)

        combined_mean = self.dropout2(combined_mean)

        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.dropout3(combined_mean_cat)
        combined_mean_cat = self.gen4(combined_mean_cat)

        return combined_mean_cat, None


class SOFTSSharp(BaseModel):
    """SOFTSSharp

    SOFTS# (SOFTSSharp) extends SOFTS by stochastically adding
    variable-position embeddings and multiple dropout layers inside the STAD
    component.

    Args:
        h (int): Forecast horizon.
        input_size (int): Autoregressive inputs size.
        n_series (int): Number of time-series.
        hidden_size (int): Dimension of the model.
        d_core (int): Dimension of core in STADSharp.
        e_layers (int): Number of encoder layers.
        d_ff (int): Dimension of fully-connected layer.
        dropout (float): Dropout rate.
        pe_keep_prob (float): probability of applying variable-position encoding
            during training. During inference, the positional encoding is scaled
            by this value.
        use_norm (bool): Whether to normalize or not.
        loss (PyTorch module): Instantiated train loss class from [losses collection](./losses.pytorch.html).
        valid_loss (PyTorch module): Instantiated valid loss class from [losses collection](./losses.pytorch.html).
        max_steps (int): Maximum number of training steps.
        learning_rate (float): Learning rate between (0, 1).
        num_lr_decays (int): Number of learning rate decays, evenly distributed across max_steps.
        early_stop_patience_steps (int): Number of validation iterations before early stopping.
        val_monitor (str): Metric to monitor for early stopping.
        val_check_steps (int): Number of training steps between every validation loss check.
        batch_size (int): Number of different series in each batch.
        valid_batch_size (int): Number of different series in each validation and test batch, if None uses batch_size.
        windows_batch_size (int): Number of windows to sample in each training batch, default uses all.
        inference_windows_batch_size (int): Number of windows to sample in each inference batch, -1 uses all.
        start_padding_enabled (bool): If True, the model will pad the time series with zeros at the beginning, by input size.
        step_size (int): Step size between each window of temporal data.
        scaler_type (str): Type of scaler for temporal inputs normalization.
        random_seed (int): Random seed for pytorch initializer and numpy generators.
        drop_last_loader (bool): If True `TimeSeriesDataLoader` drops last non-full batch.
        alias (str): Optional custom name of the model.

    References:
        - [Lu Han, Xu-Yang Chen, Han-Jia Ye, De-Chuan Zhan. "SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion"](https://arxiv.org/pdf/2404.14197)
    """

    # Class attributes
    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = True
    RECURRENT = False

    def __init__(
        self,
        h,
        input_size,
        n_series,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        exclude_insample_y=False,
        hidden_size: int = 512,
        d_core: int = 512,
        e_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pe_keep_prob: float = 0.5,
        use_norm: bool = True,
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

        super(SOFTSSharp, self).__init__(
            h=h,
            input_size=input_size,
            n_series=n_series,
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
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

        self.h = h
        self.enc_in = n_series
        self.dec_in = n_series
        self.c_out = n_series
        self.use_norm = use_norm
        self.pe_keep_prob = pe_keep_prob

        # Architecture
        self.enc_embedding = DataEmbedding_inverted(input_size, hidden_size, dropout)

        self.encoder = TransEncoder(
            [
                TransEncoderLayer(
                    STADSharp(hidden_size, d_core, dropout, pe_keep_prob),
                    hidden_size,
                    d_ff,
                    dropout=dropout,
                    activation=F.gelu,
                )
                for l in range(e_layers)
            ]
        )

        self.projection = nn.Linear(
            hidden_size, self.h * self.loss.outputsize_multiplier, bias=True
        )

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (
                stdev[:, 0, :]
                .unsqueeze(1)
                .repeat(1, self.h * self.loss.outputsize_multiplier, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :]
                .unsqueeze(1)
                .repeat(1, self.h * self.loss.outputsize_multiplier, 1)
            )
        return dec_out

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]

        y_pred = self.forecast(insample_y)
        y_pred = y_pred.reshape(insample_y.shape[0], self.h, -1)

        return y_pred
