


__all__ = ['xLSTM']


import warnings
from typing import Optional

import torch
import torch.nn as nn

from ..common._base_model import BaseModel
from ..common._modules import MLP
from ..losses.pytorch import MAE

try:
    from xlstm.blocks.mlstm.block import mLSTMBlockConfig
    from xlstm.blocks.slstm.block import sLSTMBlockConfig
    from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig

    IS_XLSTM_INSTALLED = True
except ImportError:
    IS_XLSTM_INSTALLED = False


class xLSTM(BaseModel):
    """xLSTM

    xLSTM encoder, with MLP decoder.

    Args:
        h (int): forecast horizon.
        input_size (int): considered autorregresive inputs (lags), y=[1,2,3,4] input_size=2 -> lags=[1,2].
        encoder_n_blocks (int): number of blocks for the xLSTM.
        encoder_hidden_size (int): units for the xLSTM's hidden state size.
        encoder_bias (bool): whether or not to use biases within xLSTM blocks.
        encoder_dropout (float): dropout regularization applied within xLSTM blocks.
        decoder_hidden_size (int): size of hidden layer for the MLP decoder.
        decoder_layers (int): number of layers for the MLP decoder.
        decoder_dropout (float): dropout regularization applied within the MLP decoder.
        decoder_activation (str): activation function for the MLP decoder, see [activations collection](https://docs.pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity).
        backbone (str): backbone for the xLSTM, either 'sLSTM' or 'mLSTM'.
        futr_exog_list (List[str]): future exogenous columns.
        hist_exog_list (list): historic exogenous columns.
        stat_exog_list (list): static exogenous columns.
        exclude_insample_y (bool): whether to exclude the target variable from the input.
        recurrent (bool): whether to produce forecasts recursively (True) or direct (False).
        loss (nn.Module): instantiated train loss class from [losses collection](./losses.pytorch).
        valid_loss (nn.Module): instantiated valid loss class from [losses collection](./losses.pytorch).
        max_steps (int): maximum number of training steps.
        learning_rate (float): Learning rate between (0, 1).
        num_lr_decays (int): Number of learning rate decays, evenly distributed across max_steps.
        early_stop_patience_steps (int): Number of validation iterations before early stopping.
        val_check_steps (int): Number of training steps between every validation loss check.
        batch_size (int): number of differentseries in each batch.
        valid_batch_size (int): number of different series in each validation and test batch.
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
        **trainer_kwargs (int): keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).

    References:
        - [Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, Sepp Hochreiter (2024). "xLSTM: Extended Long Short-Term Memory"](https://arxiv.org/abs/2405.04517)

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
        h: int,
        input_size: int = -1,
        inference_input_size: Optional[int] = None,
        h_train: int = 1,
        encoder_n_blocks: int = 2,
        encoder_hidden_size: int = 128,
        encoder_bias: bool = True,
        encoder_dropout: float = 0.1,
        decoder_hidden_size: int = 128,
        decoder_layers: int = 1,
        decoder_dropout: float = 0.0,
        decoder_activation: str = "GELU",
        backbone: str = "mLSTM",
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
        training_data_availability_threshold=0.0,
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

        if recurrent:
            warnings.warn(
                "xLSTM does not support recurrent forecasts. Setting recurrent=False."
            )
            recurrent = False

        super(xLSTM, self).__init__(
            h=h,
            input_size=input_size,
            inference_input_size=inference_input_size,
            h_train=h_train,
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

        if not IS_XLSTM_INSTALLED:
            raise ImportError(
                "Please install `xlstm`. You also need to install `mlstm_kernels` for backend='mLSTM' and `ninja` for backend='sLSTM'."
            )

        # xLSTM input size (1 for target variable y)
        input_encoder = (
            1 + self.hist_exog_size + self.stat_exog_size + self.futr_exog_size
        )

        # Architecture
        self.feature_projection = nn.Linear(
            in_features=input_encoder, out_features=encoder_hidden_size
        )
        if backbone == "sLSTM":
            block_stack_config = xLSTMBlockStackConfig(
                slstm_block=sLSTMBlockConfig(),
                context_length=input_size,
                num_blocks=encoder_n_blocks,
                embedding_dim=encoder_hidden_size,
                bias=encoder_bias,
                dropout=encoder_dropout,
            )
        elif backbone == "mLSTM":
            block_stack_config = xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(),
                context_length=input_size,
                num_blocks=encoder_n_blocks,
                embedding_dim=encoder_hidden_size,
                bias=encoder_bias,
                dropout=encoder_dropout,
            )
        self.hist_encoder = xLSTMBlockStack(block_stack_config)

        # Decoder MLP
        self.mlp_decoder = MLP(
            in_features=encoder_hidden_size + self.futr_exog_size,
            out_features=self.loss.outputsize_multiplier,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_layers,
            activation=decoder_activation,
            dropout=decoder_dropout,
        )
        self.temporal_projection = nn.Linear(self.input_size, self.h)

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

        encoder_input = self.feature_projection(
            encoder_input
        )  # [B, seq_len, 1 + X + S + F] -> [B, seq_len, rnn_hidden_state]
        hidden_state = self.hist_encoder(
            encoder_input
        )  # [B, seq_len, rnn_hidden_state]
        hidden_state = hidden_state.permute(
            0, 2, 1
        )  # [B, seq_len, rnn_hidden_state] -> [B, rnn_hidden_state, seq_len]
        hidden_state = self.temporal_projection(
            hidden_state
        )  # [B, rnn_hidden_state, seq_len] -> [B, rnn_hidden_state, h]
        hidden_state = hidden_state.permute(
            0, 2, 1
        )  # [B, rnn_hidden_state, h] -> [B, h, rnn_hidden_state]

        if self.futr_exog_size > 0:
            futr_exog_futr = futr_exog[:, -self.h :]  # [B, h, F]
            hidden_state = torch.cat(
                (hidden_state, futr_exog_futr), dim=-1
            )  # [B, h, rnn_hidden_state] + [B, h, F] -> [B, h, rnn_hidden_state + F]

        output = self.mlp_decoder(
            hidden_state
        )  # [B, h, rnn_hidden_state + F] -> [B, h, n_output]

        return output
