


__all__ = ['MLPMultivariate']


from typing import Optional

import torch
import torch.nn as nn

from ..common._base_model import BaseModel
from ..losses.pytorch import MAE


class MLPMultivariate(BaseModel):
    """MLPMultivariate

    Simple Multi Layer Perceptron architecture (MLP) for multivariate forecasting.
    This deep neural network has constant units through its layers, each with
    ReLU non-linearities, it is trained using ADAM stochastic gradient descent.
    The network accepts static, historic and future exogenous data, flattens
    the inputs and learns fully connected relationships against the target variables.

    Args:
        h (int): forecast horizon.
        input_size (int): considered autorregresive inputs (lags), y=[1,2,3,4] input_size=2 -> lags=[1,2].
        n_series (int): number of time-series.
        stat_exog_list (str list): static exogenous columns.
        hist_exog_list (str list): historic exogenous columns.
        futr_exog_list (str list): future exogenous columns.
        num_layers (int): number of layers for the MLP.
        hidden_size (int): number of units for each layer of the MLP.
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
        stat_exog_list=None,
        hist_exog_list=None,
        futr_exog_list=None,
        exclude_insample_y=False,
        num_layers=2,
        hidden_size=1024,
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

        # Inherit BaseMultivariate class
        super(MLPMultivariate, self).__init__(
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

        # Architecture
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        input_size_first_layer = n_series * (
            input_size
            + self.hist_exog_size * input_size
            + self.futr_exog_size * (input_size + h)
            + self.stat_exog_size
        )

        # MultiLayer Perceptron
        layers = [
            nn.Linear(in_features=input_size_first_layer, out_features=hidden_size)
        ]
        for i in range(num_layers - 1):
            layers += [nn.Linear(in_features=hidden_size, out_features=hidden_size)]
        self.mlp = nn.ModuleList(layers)

        # Adapter with Loss dependent dimensions
        self.out = nn.Linear(
            in_features=hidden_size,
            out_features=h * self.loss.outputsize_multiplier * n_series,
        )

    def forward(self, windows_batch):

        # Parse windows_batch
        x = windows_batch[
            "insample_y"
        ]  #   [batch_size (B), input_size (L), n_series (N)]
        hist_exog = windows_batch["hist_exog"]  #   [B, hist_exog_size (X), L, N]
        futr_exog = windows_batch["futr_exog"]  #   [B, futr_exog_size (F), L + h, N]
        stat_exog = windows_batch["stat_exog"]  #   [N, stat_exog_size (S)]

        # Flatten MLP inputs [B, C, L+H, N] -> [B, C * (L+H) * N]
        # Contatenate [ Y^1_t, ..., Y^N_t | X^1_{t-L},..., X^1_{t}, ..., X^N_{t} | F^1_{t-L},..., F^1_{t+H}, ...., F^N_{t+H} | S^1, ..., S^N ]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        if self.hist_exog_size > 0:
            x = torch.cat((x, hist_exog.reshape(batch_size, -1)), dim=1)

        if self.futr_exog_size > 0:
            x = torch.cat((x, futr_exog.reshape(batch_size, -1)), dim=1)

        if self.stat_exog_size > 0:
            stat_exog = stat_exog.reshape(-1)  #   [N, S] -> [N * S]
            stat_exog = stat_exog.unsqueeze(0).repeat(
                batch_size, 1
            )  #   [N * S] -> [B, N * S]
            x = torch.cat((x, stat_exog), dim=1)

        for layer in self.mlp:
            x = torch.relu(layer(x))
        x = self.out(x)

        forecast = x.reshape(batch_size, self.h, -1)

        return forecast
