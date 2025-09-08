


__all__ = ['DeepNPTS']


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import neuralforecast.losses.pytorch as losses

from ..common._base_model import BaseModel
from ..losses.pytorch import MAE


class DeepNPTS(BaseModel):
    """DeepNPTS

    Deep Non-Parametric Time Series Forecaster (`DeepNPTS`) is a baseline model for time-series forecasting. This model generates predictions by (weighted) sampling from the empirical distribution according to a learnable strategy. The strategy is learned by exploiting the information across multiple related time series.

    Args:
        h (int): Forecast horizon.
        input_size (int): autorregresive inputs size, y=[1,2,3,4] input_size=2 -> y_[t-2:t]=[1,2].
        hidden_size (int): hidden size of dense layers.
        batch_norm (bool): if True, applies Batch Normalization after each dense layer in the network.
        dropout (float): dropout.
        n_layers (int): number of dense layers.
        stat_exog_list (list): static exogenous columns.
        hist_exog_list (list): historic exogenous columns.
        futr_exog_list (list): future exogenous columns.
        exclude_insample_y (bool): the model skips the autoregressive features y[t-input_size:t] if True.
        loss (PyTorch module): instantiated train loss class from [losses collection](./losses.pytorch).
        valid_loss (PyTorch module): instantiated valid loss class from [losses collection](./losses.pytorch).
        max_steps (int): maximum number of training steps.
        learning_rate (float): Learning rate between (0, 1).
        num_lr_decays (int): Number of learning rate decays, evenly distributed across max_steps.
        early_stop_patience_steps (int): Number of validation iterations before early stopping.
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
        - [Rangapuram, Syama Sundar, Jan Gasthaus, Lorenzo Stella, Valentin Flunkert, David Salinas, Yuyang Wang, and Tim Januschowski (2023). "Deep Non-Parametric Time Series Forecaster". arXiv.](https://arxiv.org/abs/2312.14657)

    """

    # Class attributes
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True
    MULTIVARIATE = False  # If the model produces multivariate forecasts (True) or univariate (False)
    RECURRENT = (
        False  # If the model produces forecasts recursively (True) or direct (False)
    )

    def __init__(
        self,
        h,
        input_size: int,
        hidden_size: int = 32,
        batch_norm: bool = True,
        dropout: float = 0.1,
        n_layers: int = 2,
        stat_exog_list=None,
        hist_exog_list=None,
        futr_exog_list=None,
        exclude_insample_y=False,
        loss=MAE(),
        valid_loss=MAE(),
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = 3,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = 1024,
        start_padding_enabled=False,
        training_data_availability_threshold=0.0,
        step_size: int = 1,
        scaler_type: str = "standard",
        random_seed: int = 1,
        drop_last_loader=False,
        alias: Optional[str] = None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs
    ):

        if exclude_insample_y:
            raise Exception("DeepNPTS has no possibility for excluding y.")

        if loss.outputsize_multiplier > 1:
            raise Exception(
                "DeepNPTS only supports point loss functions (MAE, MSE, etc) as loss function."
            )

        if valid_loss is not None and not isinstance(valid_loss, losses.BasePointLoss):
            raise Exception(
                "DeepNPTS only supports point loss functions (MAE, MSE, etc) as valid loss function."
            )

        # Inherit BaseWindows class
        super(DeepNPTS, self).__init__(
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
        self.hidden_size = hidden_size
        self.dropout = dropout

        input_dim = (
            input_size * (1 + self.futr_exog_size + self.hist_exog_size)
            + self.stat_exog_size
            + self.h * self.futr_exog_size
        )

        # Create DeepNPTSNetwork
        modules = []
        for i in range(n_layers):
            modules.append(nn.Linear(input_dim if i == 0 else hidden_size, hidden_size))
            modules.append(nn.ReLU())
            if batch_norm:
                modules.append(nn.BatchNorm1d(hidden_size))
            if dropout > 0.0:
                modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(hidden_size, input_size * self.h))
        self.deepnptsnetwork = nn.Sequential(*modules)

    def forward(self, windows_batch):
        # Parse windows_batch
        x = windows_batch["insample_y"]  #   [B, L, 1]
        hist_exog = windows_batch["hist_exog"]  #   [B, L, X]
        futr_exog = windows_batch["futr_exog"]  #   [B, L + h, F]
        stat_exog = windows_batch["stat_exog"]  #   [B, S]

        batch_size, seq_len = x.shape[:2]  #   B = batch_size, L = seq_len
        insample_y = windows_batch["insample_y"]

        # Concatenate x_t with future exogenous of input
        if self.futr_exog_size > 0:
            x = torch.cat(
                (x, futr_exog[:, :seq_len]), dim=2
            )  #   [B, L, 1] + [B, L, F] -> [B, L, 1 + F]

        # Concatenate x_t with historic exogenous
        if self.hist_exog_size > 0:
            x = torch.cat(
                (x, hist_exog), dim=2
            )  #   [B, L, 1 + F] + [B, L, X] -> [B, L, 1 + F + X]

        x = x.reshape(batch_size, -1)  #   [B, L, 1 + F + X] -> [B, L * (1 + F + X)]

        # Concatenate x with static exogenous
        if self.stat_exog_size > 0:
            x = torch.cat(
                (x, stat_exog), dim=1
            )  #   [B, L * (1 + F + X)] + [B, S] -> [B, L * (1 + F + X) + S]

        # Concatenate x_t with future exogenous of horizon
        if self.futr_exog_size > 0:
            futr_exog = futr_exog[:, seq_len:]  #   [B, L + h, F] -> [B, h, F]
            futr_exog = futr_exog.reshape(
                batch_size, -1
            )  #   [B, L + h, F] -> [B, h * F]
            x = torch.cat(
                (x, futr_exog), dim=1
            )  #   [B, L * (1 + F + X) + S] + [B, h * F] -> [B, L * (1 + F + X) + S + h * F]

        # Run through DeepNPTSNetwork
        weights = self.deepnptsnetwork(
            x
        )  #   [B, L * (1 + F + X) + S + h * F]  -> [B, L * h]

        # Apply softmax for weighted input predictions
        weights = weights.reshape(batch_size, seq_len, -1)  #   [B, L * h] -> [B, L, h]
        x = (
            F.softmax(weights, dim=1) * insample_y
        )  #   [B, L, h] * [B, L, 1] = [B, L, h]
        forecast = torch.sum(x, dim=1).unsqueeze(-1)  #   [B, L, h] -> [B, h, 1]

        return forecast
