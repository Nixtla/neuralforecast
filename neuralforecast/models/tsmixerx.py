# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/models.tsmixerx.ipynb.

# %% auto 0
__all__ = ['TemporalMixing', 'FeatureMixing', 'MixingLayer', 'MixingLayerWithStaticExogenous', 'TSMixerx']

# %% ../../nbs/models.tsmixerx.ipynb 5
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from ..losses.pytorch import MAE
from ..common._base_model import BaseModel
from ..common._modules import RevINMultivariate

# %% ../../nbs/models.tsmixerx.ipynb 8
class TemporalMixing(nn.Module):
    """
    TemporalMixing
    """

    def __init__(self, num_features, h, dropout):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(normalized_shape=(h, num_features))
        self.temporal_lin = nn.Linear(h, h)
        self.temporal_drop = nn.Dropout(dropout)

    def forward(self, input):
        x = input.permute(0, 2, 1)  # [B, h, C] -> [B, C, h]
        x = F.relu(self.temporal_lin(x))  # [B, C, h] -> [B, C, h]
        x = x.permute(0, 2, 1)  # [B, C, h] -> [B, h, C]
        x = self.temporal_drop(x)  # [B, h, C] -> [B, h, C]

        return self.temporal_norm(x + input)


class FeatureMixing(nn.Module):
    """
    FeatureMixing
    """

    def __init__(self, in_features, out_features, h, dropout, ff_dim):
        super().__init__()
        self.feature_lin_1 = nn.Linear(in_features=in_features, out_features=ff_dim)
        self.feature_lin_2 = nn.Linear(in_features=ff_dim, out_features=out_features)
        self.feature_drop_1 = nn.Dropout(p=dropout)
        self.feature_drop_2 = nn.Dropout(p=dropout)
        self.linear_project_residual = False
        if in_features != out_features:
            self.project_residual = nn.Linear(
                in_features=in_features, out_features=out_features
            )
            self.linear_project_residual = True

        self.feature_norm = nn.LayerNorm(normalized_shape=(h, out_features))

    def forward(self, input):
        x = F.relu(self.feature_lin_1(input))  # [B, h, C_in] -> [B, h, ff_dim]
        x = self.feature_drop_1(x)  # [B, h, ff_dim] -> [B, h, ff_dim]
        x = self.feature_lin_2(x)  # [B, h, ff_dim] -> [B, h, C_out]
        x = self.feature_drop_2(x)  # [B, h, C_out] -> [B, h, C_out]
        if self.linear_project_residual:
            input = self.project_residual(input)  # [B, h, C_in] -> [B, h, C_out]

        return self.feature_norm(x + input)


class MixingLayer(nn.Module):
    """
    MixingLayer
    """

    def __init__(self, in_features, out_features, h, dropout, ff_dim):
        super().__init__()
        # Mixing layer consists of a temporal and feature mixer
        self.temporal_mixer = TemporalMixing(
            num_features=in_features, h=h, dropout=dropout
        )
        self.feature_mixer = FeatureMixing(
            in_features=in_features,
            out_features=out_features,
            h=h,
            dropout=dropout,
            ff_dim=ff_dim,
        )

    def forward(self, input):
        x = self.temporal_mixer(input)  # [B, h, C_in] -> [B, h, C_in]
        x = self.feature_mixer(x)  # [B, h, C_in] -> [B, h, C_out]
        return x


class MixingLayerWithStaticExogenous(nn.Module):
    """
    MixingLayerWithStaticExogenous
    """

    def __init__(self, h, dropout, ff_dim, stat_input_size):
        super().__init__()
        # Feature mixer for the static exogenous variables
        self.feature_mixer_stat = FeatureMixing(
            in_features=stat_input_size,
            out_features=ff_dim,
            h=h,
            dropout=dropout,
            ff_dim=ff_dim,
        )
        # Mixing layer consists of a temporal and feature mixer
        self.temporal_mixer = TemporalMixing(
            num_features=2 * ff_dim, h=h, dropout=dropout
        )
        self.feature_mixer = FeatureMixing(
            in_features=2 * ff_dim,
            out_features=ff_dim,
            h=h,
            dropout=dropout,
            ff_dim=ff_dim,
        )

    def forward(self, inputs):
        input, stat_exog = inputs
        x_stat = self.feature_mixer_stat(stat_exog)  # [B, h, S] -> [B, h, ff_dim]
        x = torch.cat(
            (input, x_stat), dim=2
        )  # [B, h, ff_dim] + [B, h, ff_dim] -> [B, h, 2 * ff_dim]
        x = self.temporal_mixer(x)  # [B, h, 2 * ff_dim] -> [B, h, 2 * ff_dim]
        x = self.feature_mixer(x)  # [B, h, 2 * ff_dim] -> [B, h, ff_dim]
        return (x, stat_exog)

# %% ../../nbs/models.tsmixerx.ipynb 10
class ReversibleInstanceNorm1d(nn.Module):
    def __init__(self, n_series, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((1, 1, 1, n_series)))
        self.bias = nn.Parameter(torch.zeros((1, 1, 1, n_series)))
        self.eps = eps

    def forward(self, x):
        # Batch statistics
        self.batch_mean = torch.mean(x, axis=2, keepdim=True).detach()
        self.batch_std = torch.sqrt(
            torch.var(x, axis=2, keepdim=True, unbiased=False) + self.eps
        ).detach()

        # Instance normalization
        x = x - self.batch_mean
        x = x / self.batch_std
        x = x * self.weight
        x = x + self.bias

        return x

    def reverse(self, x):
        # Reverse the normalization
        x = x - self.bias
        x = x / self.weight
        x = x * self.batch_std
        x = x + self.batch_mean

        return x

# %% ../../nbs/models.tsmixerx.ipynb 12
class TSMixerx(BaseModel):
    """TSMixerx

    Time-Series Mixer exogenous (`TSMixerx`) is a MLP-based multivariate time-series forecasting model, with capability for additional exogenous inputs. `TSMixerx` jointly learns temporal and cross-sectional representations of the time-series by repeatedly combining time- and feature information using stacked mixing layers. A mixing layer consists of a sequential time- and feature Multi Layer Perceptron (`MLP`).

    **Parameters:**<br>
    `h`: int, forecast horizon.<br>
    `input_size`: int, considered autorregresive inputs (lags), y=[1,2,3,4] input_size=2 -> lags=[1,2].<br>
    `n_series`: int, number of time-series.<br>
    `futr_exog_list`: str list, future exogenous columns.<br>
    `hist_exog_list`: str list, historic exogenous columns.<br>
    `stat_exog_list`: str list, static exogenous columns.<br>
    `exclude_insample_y`: bool=False, if True excludes insample_y from the model.<br>
    `n_block`: int=2, number of mixing layers in the model.<br>
    `ff_dim`: int=64, number of units for the second feed-forward layer in the feature MLP.<br>
    `dropout`: float=0.0, dropout rate between (0, 1) .<br>
    `revin`: bool=True, if True uses Reverse Instance Normalization on `insample_y` and applies it to the outputs.<br>
    `loss`: PyTorch module, instantiated train loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `valid_loss`: PyTorch module=`loss`, instantiated valid loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `max_steps`: int=1000, maximum number of training steps.<br>
    `learning_rate`: float=1e-3, Learning rate between (0, 1).<br>
    `num_lr_decays`: int=-1, Number of learning rate decays, evenly distributed across max_steps.<br>
    `early_stop_patience_steps`: int=-1, Number of validation iterations before early stopping.<br>
    `val_check_steps`: int=100, Number of training steps between every validation loss check.<br>
    `batch_size`: int=32, number of different series in each batch.<br>
    `valid_batch_size`: int=None, number of different series in each validation and test batch, if None uses batch_size.<br>
    `windows_batch_size`: int=32, number of windows to sample in each training batch. <br>
    `inference_windows_batch_size`: int=32, number of windows to sample in each inference batch, -1 uses all.<br>
    `start_padding_enabled`: bool=False, if True, the model will pad the time series with zeros at the beginning, by input size.<br>
    `step_size`: int=1, step size between each window of temporal data.<br>
    `scaler_type`: str='identity', type of scaler for temporal inputs normalization see [temporal scalers](https://nixtla.github.io/neuralforecast/common.scalers.html).<br>
    `random_seed`: int=1, random_seed for pytorch initializer and numpy generators.<br>
    `drop_last_loader`: bool=False, if True `TimeSeriesDataLoader` drops last non-full batch.<br>
    `alias`: str, optional,  Custom name of the model.<br>
    `optimizer`: Subclass of 'torch.optim.Optimizer', optional, user specified optimizer instead of the default choice (Adam).<br>
    `optimizer_kwargs`: dict, optional, list of parameters used by the user specified `optimizer`.<br>
    `lr_scheduler`: Subclass of 'torch.optim.lr_scheduler.LRScheduler', optional, user specified lr_scheduler instead of the default choice (StepLR).<br>
    `lr_scheduler_kwargs`: dict, optional, list of parameters used by the user specified `lr_scheduler`.<br>
    `dataloader_kwargs`: dict, optional, list of parameters passed into the PyTorch Lightning dataloader by the `TimeSeriesDataLoader`. <br>
    `**trainer_kwargs`: int,  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).<br>

    **References:**<br>
    - [Chen, Si-An, Chun-Liang Li, Nate Yoder, Sercan O. Arik, and Tomas Pfister (2023). "TSMixer: An All-MLP Architecture for Time Series Forecasting."](http://arxiv.org/abs/2303.06053)

    """

    # Class attributes
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True
    MULTIVARIATE = True  # If the model produces multivariate forecasts (True) or univariate (False)
    RECURRENT = (
        False  # If the model produces forecasts recursively (True) or direct (False)
    )

    def __init__(
        self,
        h,
        input_size,
        n_series,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        exclude_insample_y=False,
        n_block=2,
        ff_dim=64,
        dropout=0.0,
        revin=True,
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

        # Inherit BaseMultvariate class
        super(TSMixerx, self).__init__(
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
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
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
        # Reversible InstanceNormalization layer
        self.revin = revin
        if self.revin:
            self.norm = RevINMultivariate(num_features=n_series, affine=True)

        # Forecast horizon
        self.h = h

        # Temporal projection and feature mixing of historical variables
        self.temporal_projection = nn.Linear(in_features=input_size, out_features=h)

        self.feature_mixer_hist = FeatureMixing(
            in_features=n_series * (1 + self.hist_exog_size + self.futr_exog_size),
            out_features=ff_dim,
            h=h,
            dropout=dropout,
            ff_dim=ff_dim,
        )
        first_mixing_ff_dim_multiplier = 1

        # Feature mixing of future variables
        if self.futr_exog_size > 0:
            self.feature_mixer_futr = FeatureMixing(
                in_features=n_series * self.futr_exog_size,
                out_features=ff_dim,
                h=h,
                dropout=dropout,
                ff_dim=ff_dim,
            )
            first_mixing_ff_dim_multiplier += 1

        # Feature mixing of static variables
        if self.stat_exog_size > 0:
            self.feature_mixer_stat = FeatureMixing(
                in_features=self.stat_exog_size * n_series,
                out_features=ff_dim,
                h=h,
                dropout=dropout,
                ff_dim=ff_dim,
            )
            first_mixing_ff_dim_multiplier += 1

        # First mixing layer
        self.first_mixing = MixingLayer(
            in_features=first_mixing_ff_dim_multiplier * ff_dim,
            out_features=ff_dim,
            h=h,
            dropout=dropout,
            ff_dim=ff_dim,
        )

        # Mixing layer block
        if self.stat_exog_size > 0:
            mixing_layers = [
                MixingLayerWithStaticExogenous(
                    h=h,
                    dropout=dropout,
                    ff_dim=ff_dim,
                    stat_input_size=self.stat_exog_size * n_series,
                )
                for _ in range(n_block)
            ]
        else:
            mixing_layers = [
                MixingLayer(
                    in_features=ff_dim,
                    out_features=ff_dim,
                    h=h,
                    dropout=dropout,
                    ff_dim=ff_dim,
                )
                for _ in range(n_block)
            ]

        self.mixing_block = nn.Sequential(*mixing_layers)

        # Linear output with Loss dependent dimensions
        self.out = nn.Linear(
            in_features=ff_dim, out_features=self.loss.outputsize_multiplier * n_series
        )

    def forward(self, windows_batch):
        # Parse batch
        x = windows_batch[
            "insample_y"
        ]  #   [batch_size (B), input_size (L), n_series (N)]
        hist_exog = windows_batch["hist_exog"]  #   [B, hist_exog_size (X), L, N]
        futr_exog = windows_batch["futr_exog"]  #   [B, futr_exog_size (F), L + h, N]
        stat_exog = windows_batch["stat_exog"]  #   [N, stat_exog_size (S)]
        batch_size, input_size = x.shape[:2]

        # Apply revin to x
        if self.revin:
            x = self.norm(x, mode="norm")  #   [B, L, N] -> [B, L, N]

        # Add channel dimension to x
        x = x.unsqueeze(1)  #   [B, L, N] -> [B, 1, L, N]

        # Concatenate x with historical exogenous
        if self.hist_exog_size > 0:
            x = torch.cat(
                (x, hist_exog), dim=1
            )  #   [B, 1, L, N] + [B, X, L, N] -> [B, 1 + X, L, N]

        # Concatenate x with future exogenous of input sequence
        if self.futr_exog_size > 0:
            futr_exog_hist = futr_exog[
                :, :, :input_size
            ]  #   [B, F, L + h, N] -> [B, F, L, N]
            x = torch.cat(
                (x, futr_exog_hist), dim=1
            )  #   [B, 1 + X, L, N] + [B, F, L, N] -> [B, 1 + X + F, L, N]

        # Temporal projection & feature mixing of x
        x = x.permute(0, 1, 3, 2)  #   [B, 1 + X + F, L, N] -> [B, 1 + X + F, N, L]
        x = self.temporal_projection(
            x
        )  #   [B, 1 + X + F, N, L] -> [B, 1 + X + F, N, h]
        x = x.permute(0, 3, 1, 2)  #   [B, 1 + X + F, N, h] -> [B, h, 1 + X + F, N]
        x = x.reshape(
            batch_size, self.h, -1
        )  #   [B, h, 1 + X + F, N] -> [B, h, (1 + X + F) * N]
        x = self.feature_mixer_hist(x)  #   [B, h, (1 + X + F) * N] -> [B, h, ff_dim]

        # Concatenate x with future exogenous of output horizon
        if self.futr_exog_size > 0:
            x_futr = futr_exog[:, :, input_size:]  #   [B, F, L + h, N] -> [B, F, h, N]
            x_futr = x_futr.permute(0, 2, 1, 3)  #   [B, F, h, N] -> [B, h, F, N]
            x_futr = x_futr.reshape(
                batch_size, self.h, -1
            )  #   [B, h, N, F] -> [B, h, N * F]
            x_futr = self.feature_mixer_futr(
                x_futr
            )  #   [B, h, N * F] -> [B, h, ff_dim]
            x = torch.cat(
                (x, x_futr), dim=2
            )  #   [B, h, ff_dim] + [B, h, ff_dim] -> [B, h, 2 * ff_dim]

        # Concatenate x with static exogenous
        if self.stat_exog_size > 0:
            stat_exog = stat_exog.reshape(-1)  #   [N, S] -> [N * S]
            stat_exog = (
                stat_exog.unsqueeze(0).unsqueeze(1).repeat(batch_size, self.h, 1)
            )  #   [N * S] -> [B, h, N * S]
            x_stat = self.feature_mixer_stat(
                stat_exog
            )  #   [B, h, N * S] -> [B, h, ff_dim]
            x = torch.cat(
                (x, x_stat), dim=2
            )  #   [B, h, 2 * ff_dim] + [B, h, ff_dim] -> [B, h, 3 * ff_dim]

        # First mixing layer
        x = self.first_mixing(x)  #   [B, h, 3 * ff_dim] -> [B, h, ff_dim]

        # N blocks of mixing layers
        if self.stat_exog_size > 0:
            x, _ = self.mixing_block(
                (x, stat_exog)
            )  #   [B, h, ff_dim], [B, h, N * S] -> [B, h, ff_dim]
        else:
            x = self.mixing_block(x)  #   [B, h, ff_dim] -> [B, h, ff_dim]

        # Fully connected output layer
        forecast = self.out(x)  #   [B, h, ff_dim] -> [B, h, N * n_outputs]

        # Reverse Instance Normalization on output
        if self.revin:
            forecast = forecast.reshape(
                batch_size, self.h * self.loss.outputsize_multiplier, -1
            )  #   [B, h, N * n_outputs] -> [B, h * n_outputs, N]
            forecast = self.norm(forecast, "denorm")
            forecast = forecast.reshape(
                batch_size, self.h, -1
            )  #   [B, h * n_outputs, N] -> [B, h, n_outputs * N]

        return forecast
