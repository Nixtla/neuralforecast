


__all__ = ['Decoder', 'DeepAR']


from typing import Optional

import torch
import torch.nn as nn

from ..common._base_model import BaseModel
from ..losses.pytorch import MAE, DistributionLoss


class Decoder(nn.Module):
    """Multi-Layer Perceptron Decoder

    Args:
        in_features (int): dimension of input.
        out_features (int): dimension of output.
        hidden_size (int): dimension of hidden layers.
        hidden_layers (int): number of hidden layers.
    """

    def __init__(self, in_features, out_features, hidden_size, hidden_layers):
        super().__init__()

        if hidden_layers == 0:
            # Input layer
            layers = [nn.Linear(in_features=in_features, out_features=out_features)]
        else:
            # Input layer
            layers = [
                nn.Linear(in_features=in_features, out_features=hidden_size),
                nn.ReLU(),
            ]
            # Hidden layers
            for i in range(hidden_layers - 2):
                layers += [
                    nn.Linear(in_features=hidden_size, out_features=hidden_size),
                    nn.ReLU(),
                ]
            # Output layer
            layers += [nn.Linear(in_features=hidden_size, out_features=out_features)]

        # Store in layers as ModuleList
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeepAR(BaseModel):
    """DeepAR

    Args:
        h (int): Forecast horizon.
        input_size (int): maximum sequence length for truncated train backpropagation. Default -1 uses 3 * horizon
        h_train (int): maximum sequence length for truncated train backpropagation. Default 1.
        lstm_n_layers (int): number of LSTM layers.
        lstm_hidden_size (int): LSTM hidden size.
        lstm_dropout (float): LSTM dropout.
        decoder_hidden_layers (int): number of decoder MLP hidden layers. Default: 0 for linear layer.
        decoder_hidden_size (int): decoder MLP hidden size. Default: 0 for linear layer.
        trajectory_samples (int): number of Monte Carlo trajectories during inference.
        stat_exog_list (str list): static exogenous columns.
        hist_exog_list (str list): historic exogenous columns.
        futr_exog_list (str list): future exogenous columns.
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
        - [David Salinas, Valentin Flunkert, Jan Gasthaus, Tim Januschowski (2020). "DeepAR: Probabilistic forecasting with autoregressive recurrent networks". International Journal of Forecasting.](https://www.sciencedirect.com/science/article/pii/S0169207019301888)<br>
        - [Alexander Alexandrov et. al (2020). "GluonTS: Probabilistic and Neural Time Series Modeling in Python". Journal of Machine Learning Research.](https://www.jmlr.org/papers/v21/19-820.html)

    """

    # Class attributes
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = True
    MULTIVARIATE = False
    RECURRENT = True

    def __init__(
        self,
        h,
        input_size: int = -1,
        h_train: int = 1,
        lstm_n_layers: int = 2,
        lstm_hidden_size: int = 128,
        lstm_dropout: float = 0.1,
        decoder_hidden_layers: int = 0,
        decoder_hidden_size: int = 0,
        trajectory_samples: int = 100,
        stat_exog_list=None,
        hist_exog_list=None,
        futr_exog_list=None,
        exclude_insample_y=False,
        loss=DistributionLoss(
            distribution="StudentT", level=[80, 90], return_params=False
        ),
        valid_loss=MAE(),
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = 3,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = -1,
        start_padding_enabled=False,
        training_data_availability_threshold=0.0,
        step_size: int = 1,
        scaler_type: str = "identity",
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
            raise Exception("DeepAR has no possibility for excluding y.")

        # Inherit BaseWindows class
        super(DeepAR, self).__init__(
            h=h,
            input_size=input_size,
            h_train=h_train,
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

        self.n_samples = trajectory_samples

        # LSTM
        self.encoder_n_layers = lstm_n_layers
        self.encoder_hidden_size = lstm_hidden_size
        self.encoder_dropout = lstm_dropout

        # LSTM input size (1 for target variable y)
        input_encoder = 1 + self.futr_exog_size + self.stat_exog_size

        # Instantiate model
        self.rnn_state = None
        self.maintain_state = False
        self.hist_encoder = nn.LSTM(
            input_size=input_encoder,
            hidden_size=self.encoder_hidden_size,
            num_layers=self.encoder_n_layers,
            dropout=self.encoder_dropout,
            batch_first=True,
        )

        # Decoder MLP
        self.decoder = Decoder(
            in_features=lstm_hidden_size,
            out_features=self.loss.outputsize_multiplier,
            hidden_size=decoder_hidden_size,
            hidden_layers=decoder_hidden_layers,
        )

    def forward(self, windows_batch):

        # Parse windows_batch
        encoder_input = windows_batch["insample_y"]  # <- [B, T, 1]
        futr_exog = windows_batch["futr_exog"]
        stat_exog = windows_batch["stat_exog"]

        _, input_size = encoder_input.shape[:2]
        if self.futr_exog_size > 0:
            encoder_input = torch.cat((encoder_input, futr_exog), dim=2)

        if self.stat_exog_size > 0:
            stat_exog = stat_exog.unsqueeze(1).repeat(
                1, input_size, 1
            )  # [B, S] -> [B, input_size-1, S]
            encoder_input = torch.cat((encoder_input, stat_exog), dim=2)

        # RNN forward
        if self.maintain_state:
            rnn_state = self.rnn_state
        else:
            rnn_state = None

        hidden_state, rnn_state = self.hist_encoder(
            encoder_input, rnn_state
        )  # [B, input_size-1, rnn_hidden_state]

        if self.maintain_state:
            self.rnn_state = rnn_state

        # Decoder forward
        output = self.decoder(hidden_state)  # [B, input_size-1, output_size]

        # Return only horizon part
        return output[:, -self.h :]
