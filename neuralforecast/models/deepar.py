# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/models.deepar.ipynb.

# %% auto 0
__all__ = ['Decoder', 'DeepAR']

# %% ../../nbs/models.deepar.ipynb 4
import numpy as np

import torch
import torch.nn as nn

import logging
import warnings

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from typing import Optional

from ..common._base_windows import BaseWindows
from ..losses.pytorch import DistributionLoss, MQLoss

# %% ../../nbs/models.deepar.ipynb 7
class Decoder(nn.Module):
    """Multi-Layer Perceptron Decoder

    **Parameters:**<br>
    `in_features`: int, dimension of input.<br>
    `out_features`: int, dimension of output.<br>
    `hidden_size`: int, dimension of hidden layers.<br>
    `num_layers`: int, number of hidden layers.<br>
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

# %% ../../nbs/models.deepar.ipynb 8
class DeepAR(BaseWindows):
    """ DeepAR

    The DeepAR model produces probabilistic forecasts based on an autoregressive recurrent neural network optimized on panel data using cross-learning. DeepAR obtains its forecast distribution uses a Markov Chain Monte Carlo sampler with the following conditional probability:
    $$\mathbb{P}(\mathbf{y}_{[t+1:t+H]}|\;\mathbf{y}_{[:t]},\; \mathbf{x}^{(f)}_{[:t+H]},\; \mathbf{x}^{(s)})$$

    where $\mathbf{x}^{(s)}$ are static exogenous inputs, $\mathbf{x}^{(f)}_{[:t+H]}$ are future exogenous available at the time of the prediction.
    The predictions are obtained by transforming the hidden states $\mathbf{h}_{t}$ into predictive distribution parameters $\theta_{t}$, and then generating samples $\mathbf{\hat{y}}_{[t+1:t+H]}$ through Monte Carlo sampling trajectories.

    \begin{align}
    \mathbf{h}_{t} &= \textrm{RNN}([\mathbf{y}_{t},\mathbf{x}^{(f)}_{t+1},\mathbf{x}^{(s)}], \mathbf{h}_{t-1})\\
    \mathbf{\theta}_{t}&=\textrm{Linear}(\mathbf{h}_{t}) \\
    \hat{y}_{t+1}&=\textrm{sample}(\;\mathrm{P}(y_{t+1}\;|\;\mathbf{\theta}_{t})\;)
    \end{align}

    **Parameters:**<br>
    `h`: int, Forecast horizon. <br>
    `input_size`: int, autorregresive inputs size, y=[1,2,3,4] input_size=2 -> y_[t-2:t]=[1,2].<br>
    `lstm_n_layers`: int=2, number of LSTM layers.<br>
    `lstm_hidden_size`: int=128, LSTM hidden size.<br>
    `lstm_dropout`: float=0.1, LSTM dropout.<br>
    `decoder_hidden_layers`: int=0, number of decoder MLP hidden layers. Default: 0 for linear layer. <br>
    `decoder_hidden_size`: int=0, decoder MLP hidden size. Default: 0 for linear layer.<br>
    `trajectory_samples`: int=100, number of Monte Carlo trajectories during inference.<br>
    `stat_exog_list`: str list, static exogenous columns.<br>
    `hist_exog_list`: str list, historic exogenous columns.<br>
    `futr_exog_list`: str list, future exogenous columns.<br>
    `exclude_insample_y`: bool=False, the model skips the autoregressive features y[t-input_size:t] if True.<br>
    `loss`: PyTorch module, instantiated train loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `valid_loss`: PyTorch module=`loss`, instantiated valid loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `max_steps`: int=1000, maximum number of training steps.<br>
    `learning_rate`: float=1e-3, Learning rate between (0, 1).<br>
    `num_lr_decays`: int=-1, Number of learning rate decays, evenly distributed across max_steps.<br>
    `early_stop_patience_steps`: int=-1, Number of validation iterations before early stopping.<br>
    `val_check_steps`: int=100, Number of training steps between every validation loss check.<br>
    `batch_size`: int=32, number of different series in each batch.<br>
    `valid_batch_size`: int=None, number of different series in each validation and test batch, if None uses batch_size.<br>
    `windows_batch_size`: int=1024, number of windows to sample in each training batch, default uses all.<br>
    `inference_windows_batch_size`: int=-1, number of windows to sample in each inference batch, -1 uses all.<br>
    `start_padding_enabled`: bool=False, if True, the model will pad the time series with zeros at the beginning, by input size.<br>
    `step_size`: int=1, step size between each window of temporal data.<br>
    `scaler_type`: str='identity', type of scaler for temporal inputs normalization see [temporal scalers](https://nixtla.github.io/neuralforecast/common.scalers.html).<br>
    `random_seed`: int, random_seed for pytorch initializer and numpy generators.<br>
    `num_workers_loader`: int=os.cpu_count(), workers to be used by `TimeSeriesDataLoader`.<br>
    `drop_last_loader`: bool=False, if True `TimeSeriesDataLoader` drops last non-full batch.<br>
    `alias`: str, optional,  Custom name of the model.<br>
    `**trainer_kwargs`: int,  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).<br>    

    **References**<br>
    - [David Salinas, Valentin Flunkert, Jan Gasthaus, Tim Januschowski (2020). "DeepAR: Probabilistic forecasting with autoregressive recurrent networks". International Journal of Forecasting.](https://www.sciencedirect.com/science/article/pii/S0169207019301888)<br>
    - [Alexander Alexandrov et. al (2020). "GluonTS: Probabilistic and Neural Time Series Modeling in Python". Journal of Machine Learning Research.](https://www.jmlr.org/papers/v21/19-820.html)<br>

    """

    # Class attributes
    SAMPLING_TYPE = "windows"

    def __init__(
        self,
        h,
        input_size: int = -1,
        lstm_n_layers: int = 2,
        lstm_hidden_size: int = 128,
        lstm_dropout: float = 0.1,
        decoder_hidden_layers: int = 0,
        decoder_hidden_size: int = 0,
        trajectory_samples: int = 100,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        exclude_insample_y=False,
        loss=DistributionLoss(
            distribution="StudentT", level=[80, 90], return_params=False
        ),
        valid_loss=MQLoss(level=[80, 90]),
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
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        num_workers_loader=0,
        drop_last_loader=False,
        **trainer_kwargs
    ):
        # DeepAR does not support historic exogenous variables
        if hist_exog_list is not None:
            raise Exception("DeepAR does not support historic exogenous variables.")

        if exclude_insample_y:
            raise Exception("DeepAR has no possibility for excluding y.")

        if not loss.is_distribution_output:
            raise Exception("DeepAR only supports distributional outputs.")

        if str(type(valid_loss)) not in [
            "<class 'neuralforecast.losses.pytorch.MQLoss'>"
        ]:
            raise Exception("DeepAR only supports MQLoss as validation loss.")

        if loss.return_params:
            raise Exception(
                "DeepAR does not return distribution parameters due to Monte Carlo sampling."
            )

        # Inherit BaseWindows class
        super(DeepAR, self).__init__(
            h=h,
            input_size=input_size,
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
            windows_batch_size=windows_batch_size,
            valid_batch_size=valid_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            scaler_type=scaler_type,
            num_workers_loader=num_workers_loader,
            drop_last_loader=drop_last_loader,
            random_seed=random_seed,
            **trainer_kwargs
        )

        self.horizon_backup = self.h  # Used because h=0 during training
        self.trajectory_samples = trajectory_samples

        # LSTM
        self.encoder_n_layers = lstm_n_layers
        self.encoder_hidden_size = lstm_hidden_size
        self.encoder_dropout = lstm_dropout

        self.futr_exog_size = len(self.futr_exog_list)
        self.hist_exog_size = 0
        self.stat_exog_size = len(self.stat_exog_list)

        # LSTM input size (1 for target variable y)
        input_encoder = 1 + self.futr_exog_size + self.stat_exog_size

        # Instantiate model
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

    # Override BaseWindows method
    def training_step(self, batch, batch_idx):
        # During training h=0
        self.h = 0
        y_idx = batch["y_idx"]

        # Create and normalize windows [Ws, L, C]
        windows = self._create_windows(batch, step="train")
        original_insample_y = windows["temporal"][
            :, :, y_idx
        ].clone()  # windows: [B, L, Feature] -> [B, L]
        original_insample_y = original_insample_y[
            :, 1:
        ]  # Remove first (shift in DeepAr, cell at t outputs t+1)
        windows = self._normalization(windows=windows, y_idx=y_idx)

        # Parse windows
        insample_y, insample_mask, _, _, _, futr_exog, stat_exog = self._parse_windows(
            batch, windows
        )

        windows_batch = dict(
            insample_y=insample_y,  # [Ws, L]
            insample_mask=insample_mask,  # [Ws, L]
            futr_exog=futr_exog,  # [Ws, L+H]
            hist_exog=None,  # None
            stat_exog=stat_exog,
            y_idx=y_idx,
        )  # [Ws, 1]

        # Model Predictions
        output = self.train_forward(windows_batch)

        if self.loss.is_distribution_output:
            _, y_loc, y_scale = self._inv_normalization(
                y_hat=original_insample_y,
                temporal_cols=batch["temporal_cols"],
                y_idx=y_idx,
            )
            outsample_y = original_insample_y
            distr_args = self.loss.scale_decouple(
                output=output, loc=y_loc, scale=y_scale
            )
            mask = insample_mask[
                :, 1:
            ].clone()  # Remove first (shift in DeepAr, cell at t outputs t+1)
            loss = self.loss(y=outsample_y, distr_args=distr_args, mask=mask)
        else:
            raise Exception("DeepAR only supports distributional outputs.")

        if torch.isnan(loss):
            print("Model Parameters", self.hparams)
            print("insample_y", torch.isnan(insample_y).sum())
            print("outsample_y", torch.isnan(outsample_y).sum())
            print("output", torch.isnan(output).sum())
            raise Exception("Loss is NaN, training stopped.")

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.train_trajectories.append((self.global_step, float(loss)))

        self.h = self.horizon_backup  # Restore horizon
        return loss

    def validation_step(self, batch, batch_idx):
        self.h == self.horizon_backup

        if self.val_size == 0:
            return np.nan

        # TODO: Hack to compute number of windows
        windows = self._create_windows(batch, step="val")
        n_windows = len(windows["temporal"])
        y_idx = batch["y_idx"]

        # Number of windows in batch
        windows_batch_size = self.inference_windows_batch_size
        if windows_batch_size < 0:
            windows_batch_size = n_windows
        n_batches = int(np.ceil(n_windows / windows_batch_size))

        valid_losses = []
        batch_sizes = []
        for i in range(n_batches):
            # Create and normalize windows [Ws, L+H, C]
            w_idxs = np.arange(
                i * windows_batch_size, min((i + 1) * windows_batch_size, n_windows)
            )
            windows = self._create_windows(batch, step="val", w_idxs=w_idxs)
            original_outsample_y = torch.clone(windows["temporal"][:, -self.h :, 0])
            windows = self._normalization(windows=windows, y_idx=y_idx)

            # Parse windows
            (
                insample_y,
                insample_mask,
                _,
                outsample_mask,
                _,
                futr_exog,
                stat_exog,
            ) = self._parse_windows(batch, windows)
            windows_batch = dict(
                insample_y=insample_y,
                insample_mask=insample_mask,
                futr_exog=futr_exog,
                hist_exog=None,
                stat_exog=stat_exog,
                temporal_cols=batch["temporal_cols"],
                y_idx=y_idx,
            )

            # Model Predictions
            output_batch = self(windows_batch)
            # Monte Carlo already returns y_hat with mean and quantiles
            output_batch = output_batch[:, :, 1:]  # Remove mean
            valid_loss_batch = self.valid_loss(
                y=original_outsample_y, y_hat=output_batch, mask=outsample_mask
            )
            valid_losses.append(valid_loss_batch)
            batch_sizes.append(len(output_batch))

        valid_loss = torch.stack(valid_losses)
        batch_sizes = torch.tensor(batch_sizes).to(valid_loss.device)
        valid_loss = torch.sum(valid_loss * batch_sizes) / torch.sum(batch_sizes)

        if torch.isnan(valid_loss):
            raise Exception("Loss is NaN, training stopped.")

        self.log("valid_loss", valid_loss, prog_bar=True, on_epoch=True)
        self.validation_step_outputs.append(valid_loss)
        return valid_loss

    def predict_step(self, batch, batch_idx):
        self.h == self.horizon_backup

        # TODO: Hack to compute number of windows
        windows = self._create_windows(batch, step="predict")
        n_windows = len(windows["temporal"])
        y_idx = batch["y_idx"]

        # Number of windows in batch
        windows_batch_size = self.inference_windows_batch_size
        if windows_batch_size < 0:
            windows_batch_size = n_windows
        n_batches = int(np.ceil(n_windows / windows_batch_size))

        y_hats = []
        for i in range(n_batches):
            # Create and normalize windows [Ws, L+H, C]
            w_idxs = np.arange(
                i * windows_batch_size, min((i + 1) * windows_batch_size, n_windows)
            )
            windows = self._create_windows(batch, step="predict", w_idxs=w_idxs)
            windows = self._normalization(windows=windows, y_idx=y_idx)

            # Parse windows
            (
                insample_y,
                insample_mask,
                _,
                _,
                _,
                futr_exog,
                stat_exog,
            ) = self._parse_windows(batch, windows)
            windows_batch = dict(
                insample_y=insample_y,  # [Ws, L]
                insample_mask=insample_mask,  # [Ws, L]
                futr_exog=futr_exog,  # [Ws, L+H]
                stat_exog=stat_exog,
                temporal_cols=batch["temporal_cols"],
                y_idx=y_idx,
            )

            # Model Predictions
            y_hat = self(windows_batch)
            # Monte Carlo already returns y_hat with mean and quantiles
            y_hats.append(y_hat)
        y_hat = torch.cat(y_hats, dim=0)
        return y_hat

    def train_forward(self, windows_batch):
        # Parse windows_batch
        encoder_input = windows_batch["insample_y"][:, :, None]  # <- [B,T,1]
        futr_exog = windows_batch["futr_exog"]
        stat_exog = windows_batch["stat_exog"]

        # [B, input_size-1, X]
        encoder_input = encoder_input[
            :, :-1, :
        ]  # Remove last (shift in DeepAr, cell at t outputs t+1)
        _, input_size = encoder_input.shape[:2]
        if self.futr_exog_size > 0:
            # Shift futr_exog (t predicts t+1, last output is outside insample_y)
            encoder_input = torch.cat((encoder_input, futr_exog[:, 1:, :]), dim=2)
        if self.stat_exog_size > 0:
            stat_exog = stat_exog.unsqueeze(1).repeat(
                1, input_size, 1
            )  # [B, S] -> [B, input_size-1, S]
            encoder_input = torch.cat((encoder_input, stat_exog), dim=2)

        # RNN forward
        hidden_state, _ = self.hist_encoder(
            encoder_input
        )  # [B, input_size-1, rnn_hidden_state]

        # Decoder forward
        output = self.decoder(hidden_state)  # [B, input_size-1, output_size]
        output = self.loss.domain_map(output)
        return output

    def forward(self, windows_batch):
        # Parse windows_batch
        encoder_input = windows_batch["insample_y"][:, :, None]  # <- [B,L,1]
        futr_exog = windows_batch["futr_exog"]  # <- [B,L+H, n_f]
        stat_exog = windows_batch["stat_exog"]
        y_idx = windows_batch["y_idx"]

        # [B, seq_len, X]
        batch_size, input_size = encoder_input.shape[:2]
        if self.futr_exog_size > 0:
            futr_exog_input_window = futr_exog[
                :, 1 : input_size + 1, :
            ]  # Align y_t with futr_exog_t+1
            encoder_input = torch.cat((encoder_input, futr_exog_input_window), dim=2)
        if self.stat_exog_size > 0:
            stat_exog_input_window = stat_exog.unsqueeze(1).repeat(
                1, input_size, 1
            )  # [B, S] -> [B, input_size, S]
            encoder_input = torch.cat((encoder_input, stat_exog_input_window), dim=2)

        # Use input_size history to predict first h of the forecasting window
        _, h_c_tuple = self.hist_encoder(encoder_input)
        h_n = h_c_tuple[0]  # [n_layers, B, lstm_hidden_state]
        c_n = h_c_tuple[1]  # [n_layers, B, lstm_hidden_state]

        # Vectorizes trajectory samples in batch dimension [1]
        h_n = torch.repeat_interleave(
            h_n, self.trajectory_samples, 1
        )  # [n_layers, B*trajectory_samples, rnn_hidden_state]
        c_n = torch.repeat_interleave(
            c_n, self.trajectory_samples, 1
        )  # [n_layers, B*trajectory_samples, rnn_hidden_state]

        # Scales for inverse normalization
        y_scale = (
            self.scaler.x_scale[:, 0, [y_idx]].squeeze(-1).to(encoder_input.device)
        )
        y_loc = self.scaler.x_shift[:, 0, [y_idx]].squeeze(-1).to(encoder_input.device)
        y_scale = torch.repeat_interleave(y_scale, self.trajectory_samples, 0)
        y_loc = torch.repeat_interleave(y_loc, self.trajectory_samples, 0)

        # Recursive strategy prediction
        quantiles = self.loss.quantiles.to(encoder_input.device)
        y_hat = torch.zeros(batch_size, self.h, len(quantiles) + 1).to(
            encoder_input.device
        )
        for tau in range(self.h):
            # Decoder forward
            last_layer_h = h_n[-1]  # [B*trajectory_samples, lstm_hidden_state]
            output = self.decoder(last_layer_h)
            output = self.loss.domain_map(output)

            # Inverse normalization
            distr_args = self.loss.scale_decouple(
                output=output, loc=y_loc, scale=y_scale
            )
            # Add horizon (1) dimension
            distr_args = list(distr_args)
            for i in range(len(distr_args)):
                distr_args[i] = distr_args[i].unsqueeze(-1)
            distr_args = tuple(distr_args)
            samples_tau, _, _ = self.loss.sample(distr_args=distr_args, num_samples=1)
            samples_tau = samples_tau.reshape(batch_size, self.trajectory_samples)
            sample_mean = torch.mean(samples_tau, dim=-1).to(encoder_input.device)
            quants = torch.quantile(input=samples_tau, q=quantiles, dim=-1).to(
                encoder_input.device
            )
            y_hat[:, tau, 0] = sample_mean
            y_hat[:, tau, 1:] = quants.permute((1, 0))  # [Q, B] -> [B, Q]

            # Stop if already in the last step (no need to predict next step)
            if tau + 1 == self.h:
                continue
            # Normalize to use as input
            encoder_input = self.scaler.scaler(
                samples_tau.flatten(), y_loc, y_scale
            )  # [B*n_samples]
            encoder_input = encoder_input[:, None, None]  # [B*n_samples, 1, 1]

            # Update input
            if self.futr_exog_size > 0:
                futr_exog_tau = futr_exog[:, [input_size + tau + 1], :]  # [B, 1, n_f]
                futr_exog_tau = torch.repeat_interleave(
                    futr_exog_tau, self.trajectory_samples, 0
                )  # [B*n_samples, 1, n_f]
                encoder_input = torch.cat(
                    (encoder_input, futr_exog_tau), dim=2
                )  # [B*n_samples, 1, 1+n_f]
            if self.stat_exog_size > 0:
                stat_exog_tau = torch.repeat_interleave(
                    stat_exog, self.trajectory_samples, 0
                )  # [B*n_samples, n_s]
                encoder_input = torch.cat(
                    (encoder_input, stat_exog_tau[:, None, :]), dim=2
                )  # [B*n_samples, 1, 1+n_f+n_s]

            _, h_c_tuple = self.hist_encoder(encoder_input, (h_n, c_n))
            h_n = h_c_tuple[0]  # [n_layers, B, rnn_hidden_state]
            c_n = h_c_tuple[1]  # [n_layers, B, rnn_hidden_state]

        return y_hat
