# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/models.gru.ipynb.

# %% auto 0
__all__ = ['GRU']

# %% ../../nbs/models.gru.ipynb 7
import warnings
from typing import Optional

import torch
import torch.nn as nn

from ..losses.pytorch import MAE
from ..common._base_model import BaseModel
from ..common._modules import MLP

# %% ../../nbs/models.gru.ipynb 8
class GRU(BaseModel):
    """GRU

    Multi Layer Recurrent Network with Gated Units (GRU), and
    MLP decoder. The network has non-linear activation functions, it is trained
    using ADAM stochastic gradient descent. The network accepts static, historic
    and future exogenous data, flattens the inputs.

    **Parameters:**<br>
    `h`: int, forecast horizon.<br>
    `input_size`: int, maximum sequence length for truncated train backpropagation. Default -1 uses 3 * horizon <br>
    `inference_input_size`: int, maximum sequence length for truncated inference. Default None uses input_size history.<br>
    `encoder_n_layers`: int=2, number of layers for the GRU.<br>
    `encoder_hidden_size`: int=200, units for the GRU's hidden state size.<br>
    `encoder_activation`: Optional[str]=None, Deprecated. Activation function in GRU is frozen in PyTorch.<br>
    `encoder_bias`: bool=True, whether or not to use biases b_ih, b_hh within GRU units.<br>
    `encoder_dropout`: float=0., dropout regularization applied to GRU outputs.<br>
    `context_size`: deprecated.<br>
    `decoder_hidden_size`: int=200, size of hidden layer for the MLP decoder.<br>
    `decoder_layers`: int=2, number of layers for the MLP decoder.<br>
    `futr_exog_list`: str list, future exogenous columns.<br>
    `hist_exog_list`: str list, historic exogenous columns.<br>
    `stat_exog_list`: str list, static exogenous columns.<br>
    `exclude_insample_y`: bool=False, whether to exclude the target variable from the input.<br>
    `recurrent`: bool=False, whether to produce forecasts recursively (True) or direct (False).<br>
    `loss`: PyTorch module, instantiated train loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `valid_loss`: PyTorch module=`loss`, instantiated valid loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `max_steps`: int=1000, maximum number of training steps.<br>
    `learning_rate`: float=1e-3, Learning rate between (0, 1).<br>
    `num_lr_decays`: int=-1, Number of learning rate decays, evenly distributed across max_steps.<br>
    `early_stop_patience_steps`: int=-1, Number of validation iterations before early stopping.<br>
    `val_check_steps`: int=100, Number of training steps between every validation loss check.<br>
    `batch_size`: int=32, number of differentseries in each batch.<br>
    `valid_batch_size`: int=None, number of different series in each validation and test batch.<br>
    `windows_batch_size`: int=128, number of windows to sample in each training batch, default uses all.<br>
    `inference_windows_batch_size`: int=1024, number of windows to sample in each inference batch, -1 uses all.<br>
    `start_padding_enabled`: bool=False, if True, the model will pad the time series with zeros at the beginning, by input size.<br>
    `step_size`: int=1, step size between each window of temporal data.<br>
    `scaler_type`: str='robust', type of scaler for temporal inputs normalization see [temporal scalers](https://nixtla.github.io/neuralforecast/common.scalers.html).<br>
    `random_seed`: int=1, random_seed for pytorch initializer and numpy generators.<br>
    `drop_last_loader`: bool=False, if True `TimeSeriesDataLoader` drops last non-full batch.<br>
    `alias`: str, optional,  Custom name of the model.<br>
    `optimizer`: Subclass of 'torch.optim.Optimizer', optional, user specified optimizer instead of the default choice (Adam).<br>
    `optimizer_kwargs`: dict, optional, list of parameters used by the user specified `optimizer`.<br>
    `lr_scheduler`: Subclass of 'torch.optim.lr_scheduler.LRScheduler', optional, user specified lr_scheduler instead of the default choice (StepLR).<br>
    `lr_scheduler_kwargs`: dict, optional, list of parameters used by the user specified `lr_scheduler`.<br>
    `dataloader_kwargs`: dict, optional, list of parameters passed into the PyTorch Lightning dataloader by the `TimeSeriesDataLoader`. <br>
    `**trainer_kwargs`: int,  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).<br>
    """

    # Class attributes
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True
    MULTIVARIATE = False  # If the model produces multivariate forecasts (True) or univariate (False)
    RECURRENT = (
        True  # If the model produces forecasts recursively (True) or direct (False)
    )

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        inference_input_size: Optional[int] = None,
        encoder_n_layers: int = 2,
        encoder_hidden_size: int = 200,
        encoder_activation: Optional[str] = None,
        encoder_bias: bool = True,
        encoder_dropout: float = 0.0,
        context_size: Optional[int] = None,
        decoder_hidden_size: int = 128,
        decoder_layers: int = 2,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        exclude_insample_y=False,
        recurrent=False,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size=32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size=128,
        inference_windows_batch_size=1024,
        start_padding_enabled=False,
        step_size: int = 1,
        scaler_type: str = "robust",
        random_seed=1,
        drop_last_loader=False,
        alias: Optional[str] = None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs
    ):

        self.RECURRENT = recurrent

        super(GRU, self).__init__(
            h=h,
            input_size=input_size,
            inference_input_size=inference_input_size,
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

        if encoder_activation is not None:
            warnings.warn(
                "The 'encoder_activation' argument is deprecated and will be removed in "
                "future versions. The activation function in GRU is frozen in PyTorch and "
                "it cannot be modified.",
                DeprecationWarning,
            )

        # RNN
        self.encoder_n_layers = encoder_n_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_bias = encoder_bias
        self.encoder_dropout = encoder_dropout

        # Context adapter
        if context_size is not None:
            warnings.warn(
                "context_size is deprecated and will be removed in future versions."
            )

        # MLP decoder
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers

        # RNN input size (1 for target variable y)
        input_encoder = (
            1 + self.hist_exog_size + self.stat_exog_size + self.futr_exog_size
        )

        # Instantiate model
        self.rnn_state = None
        self.maintain_state = False
        self.hist_encoder = nn.GRU(
            input_size=input_encoder,
            hidden_size=self.encoder_hidden_size,
            num_layers=self.encoder_n_layers,
            bias=self.encoder_bias,
            dropout=self.encoder_dropout,
            batch_first=True,
        )

        # Decoder MLP
        if self.RECURRENT:
            self.proj = nn.Linear(
                self.encoder_hidden_size, self.loss.outputsize_multiplier
            )
        else:
            self.mlp_decoder = MLP(
                in_features=self.encoder_hidden_size + self.futr_exog_size,
                out_features=self.loss.outputsize_multiplier,
                hidden_size=self.decoder_hidden_size,
                num_layers=self.decoder_layers,
                activation="ReLU",
                dropout=0.0,
            )

    def forward(self, windows_batch):

        # Parse windows_batch
        encoder_input = windows_batch["insample_y"]  # [B, seq_len, 1]
        futr_exog = windows_batch["futr_exog"]  # [B, seq_len, F]
        hist_exog = windows_batch["hist_exog"]  # [B, seq_len, X]
        stat_exog = windows_batch["stat_exog"]  # [B, S]

        # Concatenate y, historic and static inputs
        batch_size, seq_len = encoder_input.shape[:2]
        if self.hist_exog_size > 0:
            encoder_input = torch.cat(
                (encoder_input, hist_exog), dim=2
            )  # [B, seq_len, 1] + [B, seq_len, X] -> [B, seq_len, 1 + X]

        if self.stat_exog_size > 0:
            # print(encoder_input.shape)
            stat_exog = stat_exog.unsqueeze(1).repeat(
                1, seq_len, 1
            )  # [B, S] -> [B, seq_len, S]
            encoder_input = torch.cat(
                (encoder_input, stat_exog), dim=2
            )  # [B, seq_len, 1 + X] + [B, seq_len, S] -> [B, seq_len, 1 + X + S]

        if self.futr_exog_size > 0:
            encoder_input = torch.cat(
                (encoder_input, futr_exog[:, :seq_len]), dim=2
            )  # [B, seq_len, 1 + X + S] + [B, seq_len, F] -> [B, seq_len, 1 + X + S + F]

        if self.RECURRENT:
            if self.maintain_state:
                rnn_state = self.rnn_state
            else:
                rnn_state = None

            output, rnn_state = self.hist_encoder(
                encoder_input, rnn_state
            )  # [B, seq_len, rnn_hidden_state]
            output = self.proj(
                output
            )  # [B, seq_len, rnn_hidden_state] -> [B, seq_len, n_output]
            if self.maintain_state:
                self.rnn_state = rnn_state
        else:
            hidden_state, _ = self.hist_encoder(
                encoder_input, None
            )  # [B, seq_len, rnn_hidden_state]
            hidden_state = hidden_state[
                :, -self.h :
            ]  # [B, seq_len, rnn_hidden_state] -> [B, h, rnn_hidden_state]

            if self.futr_exog_size > 0:
                futr_exog_futr = futr_exog[:, -self.h :]  # [B, h, F]
                hidden_state = torch.cat(
                    (hidden_state, futr_exog_futr), dim=-1
                )  # [B, h, rnn_hidden_state] + [B, h, F] -> [B, h, rnn_hidden_state + F]

            output = self.mlp_decoder(
                hidden_state
            )  # [B, h, rnn_hidden_state + F] -> [B, seq_len, n_output]

        return output[:, -self.h :]
