# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/models.tcn.ipynb.

# %% auto 0
__all__ = ['TCN']

# %% ../../nbs/models.tcn.ipynb 5
from typing import List, Optional

import torch
import torch.nn as nn

from ..losses.pytorch import MAE
from ..common._base_model import BaseModel
from ..common._modules import MLP, TemporalConvolutionEncoder

# %% ../../nbs/models.tcn.ipynb 7
class TCN(BaseModel):
    """TCN

    Temporal Convolution Network (TCN), with MLP decoder.
    The historical encoder uses dilated skip connections to obtain efficient long memory,
    while the rest of the architecture allows for future exogenous alignment.

    **Parameters:**<br>
    `h`: int, forecast horizon.<br>
    `input_size`: int, maximum sequence length for truncated train backpropagation. Default -1 uses all history.<br>
    `inference_input_size`: int, maximum sequence length for truncated inference. Default -1 uses all history.<br>
    `kernel_size`: int, size of the convolving kernel.<br>
    `dilations`: int list, ontrols the temporal spacing between the kernel points; also known as the à trous algorithm.<br>
    `encoder_hidden_size`: int=200, units for the TCN's hidden state size.<br>
    `encoder_activation`: str=`tanh`, type of TCN activation from `tanh` or `relu`.<br>
    `context_size`: int=10, size of context vector for each timestamp on the forecasting window.<br>
    `decoder_hidden_size`: int=200, size of hidden layer for the MLP decoder.<br>
    `decoder_layers`: int=2, number of layers for the MLP decoder.<br>
    `futr_exog_list`: str list, future exogenous columns.<br>
    `hist_exog_list`: str list, historic exogenous columns.<br>
    `stat_exog_list`: str list, static exogenous columns.<br>
    `loss`: PyTorch module, instantiated train loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `max_steps`: int=1000, maximum number of training steps.<br>
    `learning_rate`: float=1e-3, Learning rate between (0, 1).<br>
    `valid_batch_size`: int=None, number of different series in each validation and test batch.<br>
    `num_lr_decays`: int=-1, Number of learning rate decays, evenly distributed across max_steps.<br>
    `early_stop_patience_steps`: int=-1, Number of validation iterations before early stopping.<br>
    `val_check_steps`: int=100, Number of training steps between every validation loss check.<br>    `batch_size`: int=32, number of differentseries in each batch.<br>
    `scaler_type`: str='robust', type of scaler for temporal inputs normalization see [temporal scalers](https://nixtla.github.io/neuralforecast/common.scalers.html).<br>
    `random_seed`: int=1, random_seed for pytorch initializer and numpy generators.<br>
    `num_workers_loader`: int=os.cpu_count(), workers to be used by `TimeSeriesDataLoader`.<br>
    `drop_last_loader`: bool=False, if True `TimeSeriesDataLoader` drops last non-full batch.<br>
    `alias`: str, optional,  Custom name of the model.<br>
    `optimizer`: Subclass of 'torch.optim.Optimizer', optional, user specified optimizer instead of the default choice (Adam).<br>
    `optimizer_kwargs`: dict, optional, list of parameters used by the user specified `optimizer`.<br>
    `lr_scheduler`: Subclass of 'torch.optim.lr_scheduler.LRScheduler', optional, user specified lr_scheduler instead of the default choice (StepLR).<br>
    `lr_scheduler_kwargs`: dict, optional, list of parameters used by the user specified `lr_scheduler`.<br>
    `**trainer_kwargs`: int,  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).<br>
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
        inference_input_size: int = -1,
        kernel_size: int = 2,
        dilations: List[int] = [1, 2, 4, 8, 16],
        encoder_hidden_size: int = 128,
        encoder_activation: str = "ReLU",
        context_size: int = 10,
        decoder_hidden_size: int = 128,
        decoder_layers: int = 2,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size=128,
        inference_windows_batch_size=1024,
        start_padding_enabled=False,
        step_size: int = 1,
        scaler_type: str = "robust",
        random_seed: int = 1,
        num_workers_loader=0,
        drop_last_loader=False,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        **trainer_kwargs
    ):
        super(TCN, self).__init__(
            h=h,
            input_size=input_size,
            inference_input_size=inference_input_size,
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
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            num_workers_loader=num_workers_loader,
            drop_last_loader=drop_last_loader,
            random_seed=random_seed,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            exclude_insample_y=False,
            **trainer_kwargs
        )

        # ----------------------------------- Parse dimensions -----------------------------------#
        # TCN
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_activation = encoder_activation

        # Context adapter
        self.context_size = context_size

        # MLP decoder
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers

        # TCN input size (1 for target variable y)
        input_encoder = (
            1 + self.hist_exog_size + self.stat_exog_size + self.futr_exog_size
        )

        # ---------------------------------- Instantiate Model -----------------------------------#
        # Instantiate historic encoder
        self.hist_encoder = TemporalConvolutionEncoder(
            in_channels=input_encoder,
            out_channels=self.encoder_hidden_size,
            kernel_size=self.kernel_size,  # Almost like lags
            dilations=self.dilations,
            activation=self.encoder_activation,
        )

        # Context adapter
        self.context_adapter = nn.Linear(in_features=self.input_size, out_features=h)

        # Decoder MLP
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
        encoder_input = windows_batch["insample_y"]  # [B, L, 1]
        futr_exog = windows_batch["futr_exog"]  # [B, L + h, F]
        hist_exog = windows_batch["hist_exog"]  # [B, L, X]
        stat_exog = windows_batch["stat_exog"]  # [B, S]

        # Concatenate y, historic and static inputs
        batch_size, input_size = encoder_input.shape[:2]
        if self.hist_exog_size > 0:
            encoder_input = torch.cat(
                (encoder_input, hist_exog), dim=2
            )  # [B, L, 1] + [B, L, X] -> [B, L, 1 + X]

        if self.stat_exog_size > 0:
            # print(encoder_input.shape)
            stat_exog = stat_exog.unsqueeze(1).repeat(
                1, input_size, 1
            )  # [B, S] -> [B, L, S]
            encoder_input = torch.cat(
                (encoder_input, stat_exog), dim=2
            )  # [B, L, 1 + X] + [B, L, S] -> [B, L, 1 + X + S]

        if self.futr_exog_size > 0:
            encoder_input = torch.cat(
                (encoder_input, futr_exog[:, :input_size]), dim=2
            )  # [B, L, 1 + X + S] + [B, L, F] -> [B, L, 1 + X + S + F]

        # TCN forward
        hidden_state = self.hist_encoder(encoder_input)  # [B, L, C]

        # Context adapter
        hidden_state = hidden_state.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
        context = self.context_adapter(hidden_state)  # [B, C, L] -> [B, C, h]

        # Residual connection with futr_exog
        if self.futr_exog_size > 0:
            futr_exog_futr = futr_exog[:, input_size:].swapaxes(
                1, 2
            )  # [B, L + h, F] -> [B, F, h]
            context = torch.cat(
                (context, futr_exog_futr), dim=1
            )  # [B, C, h] + [B, F, h] = [B, C + F, h]

        context = context.swapaxes(1, 2)  # [B, C + F, h] -> [B, h, C + F]

        # Final forecast
        output = self.mlp_decoder(context)  # [B, h, C + F] -> [B, h, n_output]

        return output
