# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/models.mlpmultivariate.ipynb.

# %% auto 0
__all__ = ['MLPMultivariate']

# %% ../../nbs/models.mlpmultivariate.ipynb 5
import torch
import torch.nn as nn

from ..losses.pytorch import MAE
from ..common._base_multivariate import BaseMultivariate

# %% ../../nbs/models.mlpmultivariate.ipynb 6
class MLPMultivariate(BaseMultivariate):
    """MLPMultivariate

    Simple Multi Layer Perceptron architecture (MLP) for multivariate forecasting.
    This deep neural network has constant units through its layers, each with
    ReLU non-linearities, it is trained using ADAM stochastic gradient descent.
    The network accepts static, historic and future exogenous data, flattens
    the inputs and learns fully connected relationships against the target variables.

    **Parameters:**<br>
    `h`: int, forecast horizon.<br>
    `input_size`: int, considered autorregresive inputs (lags), y=[1,2,3,4] input_size=2 -> lags=[1,2].<br>
    `n_series`: int, number of time-series.<br>
    `stat_exog_list`: str list, static exogenous columns.<br>
    `hist_exog_list`: str list, historic exogenous columns.<br>
    `futr_exog_list`: str list, future exogenous columns.<br>
    `n_layers`: int, number of layers for the MLP.<br>
    `hidden_size`: int, number of units for each layer of the MLP.<br>
    `loss`: PyTorch module, instantiated train loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `valid_loss`: PyTorch module=`loss`, instantiated valid loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `max_steps`: int=1000, maximum number of training steps.<br>
    `learning_rate`: float=1e-3, Learning rate between (0, 1).<br>
    `num_lr_decays`: int=-1, Number of learning rate decays, evenly distributed across max_steps.<br>
    `early_stop_patience_steps`: int=-1, Number of validation iterations before early stopping.<br>
    `val_check_steps`: int=100, Number of training steps between every validation loss check.<br>
    `batch_size`: int=32, number of different series in each batch.<br>
    `step_size`: int=1, step size between each window of temporal data.<br>
    `scaler_type`: str='identity', type of scaler for temporal inputs normalization see [temporal scalers](https://nixtla.github.io/neuralforecast/common.scalers.html).<br>
    `random_seed`: int=1, random_seed for pytorch initializer and numpy generators.<br>
    `drop_last_loader`: bool=False, if True `TimeSeriesDataLoader` drops last non-full batch.<br>
    `alias`: str, optional,  Custom name of the model.<br>
    `dataloader_kwargs`: dict, optional, list of parameters passed into the PyTorch Lightning dataloader by the `TimeSeriesDataLoader`. <br>
    `**trainer_kwargs`: int,  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).<br>
    """

    # Class attributes
    SAMPLING_TYPE = "multivariate"
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True

    def __init__(
        self,
        h,
        input_size,
        n_series,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
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
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        drop_last_loader: bool = False,
        dataloader_kwargs=None,
        **trainer_kwargs
    ):

        # Inherit BaseMultivariate class
        super(MLPMultivariate, self).__init__(
            h=h,
            input_size=input_size,
            n_series=n_series,
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            step_size=step_size,
            scaler_type=scaler_type,
            drop_last_loader=drop_last_loader,
            random_seed=random_seed,
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

        x = x.reshape(batch_size, self.h, -1)
        forecast = self.loss.domain_map(x)

        # domain_map might have squeezed the last dimension in case n_series == 1
        # Note that this fails in case of a tuple loss, but Multivariate does not support tuple losses yet.
        if forecast.ndim == 2:
            return forecast.unsqueeze(-1)
        else:
            return forecast
