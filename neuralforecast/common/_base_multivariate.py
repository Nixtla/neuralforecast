# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/common.base_multivariate.ipynb.

# %% auto 0
__all__ = ['BaseMultivariate']

# %% ../../nbs/common.base_multivariate.ipynb 4
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ._scalers import TemporalNorm
from ..tsdataset import TimeSeriesDataModule

# %% ../../nbs/common.base_multivariate.ipynb 5
class BaseMultivariate(pl.LightningModule):
    def __init__(
        self,
        h,
        input_size,
        loss,
        valid_loss,
        learning_rate,
        max_steps,
        val_check_steps,
        n_series,
        batch_size,
        step_size=1,
        num_lr_decays=0,
        early_stop_patience_steps=-1,
        scaler_type="robust",
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        num_workers_loader=0,
        drop_last_loader=False,
        random_seed=1,
        alias=None,
        **trainer_kwargs,
    ):
        super(BaseMultivariate, self).__init__()

        self.save_hyperparameters()  # Allows instantiation from a checkpoint from class
        self.random_seed = random_seed
        pl.seed_everything(self.random_seed, workers=True)

        # Padder to complete train windows,
        # example y=[1,2,3,4,5] h=3 -> last y_output = [5,0,0]
        self.h = h
        self.input_size = input_size
        self.n_series = n_series
        self.padder = nn.ConstantPad1d(padding=(0, self.h), value=0)

        # Loss
        self.loss = loss
        if valid_loss is None:
            self.valid_loss = loss
        else:
            self.valid_loss = valid_loss
        self.train_trajectories = []
        self.valid_trajectories = []

        self.batch_size = batch_size

        # Optimization
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.num_lr_decays = num_lr_decays
        self.lr_decay_steps = (
            max(max_steps // self.num_lr_decays, 1) if self.num_lr_decays > 0 else 10e7
        )
        self.early_stop_patience_steps = early_stop_patience_steps
        self.val_check_steps = val_check_steps
        self.step_size = step_size

        # Scaler
        self.scaler = TemporalNorm(
            scaler_type=scaler_type, dim=2
        )  # Time dimension is in the second axis

        # Variables
        self.futr_exog_list = futr_exog_list if futr_exog_list is not None else []
        self.hist_exog_list = hist_exog_list if hist_exog_list is not None else []
        self.stat_exog_list = stat_exog_list if stat_exog_list is not None else []

        # Fit arguments
        self.val_size = 0
        self.test_size = 0

        # Model state
        self.decompose_forecast = False

        ## Trainer arguments ##
        # Max steps, validation steps and check_val_every_n_epoch
        if "max_epochs" in trainer_kwargs.keys():
            warnings.warn("max_epochs will be deprecated, use max_steps instead.")
        else:
            trainer_kwargs = {**trainer_kwargs, **{"max_steps": max_steps}}

        if "max_epochs" in trainer_kwargs.keys():
            warnings.warn("max_epochs will be deprecated, use max_steps instead.")

        # Callbacks
        if trainer_kwargs.get("callbacks", None) is None:
            callbacks = [TQDMProgressBar()]
            # Early stopping
            if self.early_stop_patience_steps > 0:
                callbacks += [
                    EarlyStopping(
                        monitor="ptl/val_loss", patience=self.early_stop_patience_steps
                    )
                ]

            trainer_kwargs["callbacks"] = callbacks

        # Add GPU accelerator if available
        if trainer_kwargs.get("accelerator", None) is None:
            if torch.cuda.is_available():
                trainer_kwargs["accelerator"] = "gpu"
        if trainer_kwargs.get("devices", None) is None:
            if torch.cuda.is_available():
                trainer_kwargs["devices"] = -1

        # Avoid saturating local memory, disabled fit model checkpoints
        if trainer_kwargs.get("enable_checkpointing", None) is None:
            trainer_kwargs["enable_checkpointing"] = False

        self.trainer_kwargs = trainer_kwargs

        # DataModule arguments
        self.num_workers_loader = num_workers_loader
        self.drop_last_loader = drop_last_loader
        # used by on_validation_epoch_end hook
        self.validation_step_outputs = []
        self.alias = alias

    def __repr__(self):
        return type(self).__name__ if self.alias is None else self.alias

    def on_fit_start(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer, step_size=self.lr_decay_steps, gamma=0.5
            ),
            "frequency": 1,
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _create_windows(self, batch, step):
        # Parse common data
        window_size = self.input_size + self.h
        temporal_cols = batch["temporal_cols"]
        temporal = batch["temporal"]

        if step == "train":
            if self.val_size + self.test_size > 0:
                cutoff = -self.val_size - self.test_size
                temporal = temporal[:, :, :cutoff]

            temporal = self.padder(temporal)
            windows = temporal.unfold(
                dimension=-1, size=window_size, step=self.step_size
            )
            # [n_series, C, Ws, L+H] 0, 1, 2, 3

            # Sample and Available conditions
            available_idx = temporal_cols.get_loc("available_mask")
            sample_condition = windows[:, available_idx, :, -self.h :]
            sample_condition = torch.sum(sample_condition, axis=2)  # Sum over time
            sample_condition = torch.sum(
                sample_condition, axis=0
            )  # Sum over time-series
            available_condition = windows[:, available_idx, :, : -self.h]
            available_condition = torch.sum(
                available_condition, axis=2
            )  # Sum over time
            available_condition = torch.sum(
                available_condition, axis=0
            )  # Sum over time-series
            final_condition = (sample_condition > 0) & (
                available_condition > 0
            )  # Of shape [Ws]
            windows = windows[:, :, final_condition, :]

            # Get Static data
            static = batch.get("static", None)
            static_cols = batch.get("static_cols", None)

            # Protection of empty windows
            if final_condition.sum() == 0:
                raise Exception("No windows available for training")

            # Sample windows
            n_windows = len(windows)
            if self.batch_size is not None:
                w_idxs = np.random.choice(
                    n_windows,
                    size=self.batch_size,
                    replace=(n_windows < self.batch_size),
                )
                windows = windows[:, :, w_idxs, :]

            windows = windows.permute(2, 1, 3, 0)  # [Ws, C, L+H, n_series]

            windows_batch = dict(
                temporal=windows,
                temporal_cols=temporal_cols,
                static=static,
                static_cols=static_cols,
            )

            return windows_batch

        elif step in ["predict", "val"]:
            if step == "predict":
                predict_step_size = self.predict_step_size
                cutoff = -self.input_size - self.test_size
                temporal = batch["temporal"][:, :, cutoff:]

            elif step == "val":
                predict_step_size = self.step_size
                cutoff = -self.input_size - self.val_size - self.test_size
                if self.test_size > 0:
                    temporal = batch["temporal"][:, :, cutoff : -self.test_size]
                else:
                    temporal = batch["temporal"][:, :, cutoff:]

            if (
                (step == "predict")
                and (self.test_size == 0)
                and (len(self.futr_exog_list) == 0)
            ):
                temporal = self.padder(temporal)

            windows = temporal.unfold(
                dimension=-1, size=window_size, step=predict_step_size
            )
            # [n_series, C, Ws, L+H] -> [Ws, C, L+H, n_series]
            windows = windows.permute(2, 1, 3, 0)

            # Get Static data
            static = batch.get("static", None)
            static_cols = batch.get("static_cols", None)

            windows_batch = dict(
                temporal=windows,
                temporal_cols=temporal_cols,
                static=static,
                static_cols=static_cols,
            )

            return windows_batch
        else:
            raise ValueError(f"Unknown step {step}")

    def _normalization(self, windows):
        # windows are already filtered by train/validation/test
        # from the `create_windows_method` nor leakage risk
        temporal = windows["temporal"]  # [Ws, C, L+H, n_series]
        temporal_cols = windows["temporal_cols"].copy()  # [Ws, C, L+H, n_series]

        # To avoid leakage uses only the lags
        temporal_data_cols = temporal_cols.drop("available_mask").tolist()
        temporal_data = temporal[:, temporal_cols.get_indexer(temporal_data_cols), :, :]
        temporal_mask = temporal[
            :, temporal_cols.get_loc("available_mask"), :, :
        ].clone()
        temporal_mask[:, -self.h :, :] = 0.0

        # Normalize. self.scaler stores the shift and scale for inverse transform
        temporal_mask = temporal_mask.unsqueeze(
            1
        )  # Add channel dimension for scaler.transform.
        temporal_data = self.scaler.transform(x=temporal_data, mask=temporal_mask)
        # Replace values in windows dict
        temporal[:, temporal_cols.get_indexer(temporal_data_cols), :, :] = temporal_data
        windows["temporal"] = temporal

        return windows

    def _inv_normalization(self, y_hat, temporal_cols):
        # Receives window predictions [Ws, H, n_series]
        # Broadcasts outputs and inverts normalization

        # Add C dimension
        # if y_hat.ndim == 2:
        #     remove_dimension = True
        #     y_hat = y_hat.unsqueeze(-1)
        # else:
        #     remove_dimension = False

        temporal_data_cols = temporal_cols.drop("available_mask")
        y_scale = self.scaler.x_scale[
            :, temporal_data_cols.get_indexer(["y"]), :
        ].squeeze(1)
        y_loc = self.scaler.x_shift[
            :, temporal_data_cols.get_indexer(["y"]), :
        ].squeeze(1)

        # y_scale = torch.repeat_interleave(y_scale, repeats=y_hat.shape[-1], dim=-1)
        # y_loc = torch.repeat_interleave(y_loc, repeats=y_hat.shape[-1], dim=-1)

        y_hat = self.scaler.inverse_transform(z=y_hat, x_scale=y_scale, x_shift=y_loc)

        # if remove_dimension:
        #     y_hat = y_hat.squeeze(-1)
        #     y_loc = y_loc.squeeze(-1)
        #     y_scale = y_scale.squeeze(-1)

        return y_hat, y_loc, y_scale

    def _parse_windows(self, batch, windows):
        # Temporal: [Ws, C, L+H, n_series]

        # Filter insample lags from outsample horizon
        y_idx = batch["temporal_cols"].get_loc("y")
        mask_idx = batch["temporal_cols"].get_loc("available_mask")
        insample_y = windows["temporal"][:, y_idx, : -self.h, :]
        insample_mask = windows["temporal"][:, mask_idx, : -self.h, :]
        outsample_y = windows["temporal"][:, y_idx, -self.h :, :]
        outsample_mask = windows["temporal"][:, mask_idx, -self.h :, :]

        # Filter historic exogenous variables
        if len(self.hist_exog_list):
            hist_exog_idx = windows["temporal_cols"].get_indexer(self.hist_exog_list)
            hist_exog = windows["temporal"][:, hist_exog_idx, : -self.h, :]
        else:
            hist_exog = None

        # Filter future exogenous variables
        if len(self.futr_exog_list):
            futr_exog_idx = windows["temporal_cols"].get_indexer(self.futr_exog_list)
            futr_exog = windows["temporal"][:, futr_exog_idx, :, :]
        else:
            futr_exog = None

        # Filter static variables
        if len(self.stat_exog_list):
            static_idx = windows["static_cols"].get_indexer(self.stat_exog_list)
            stat_exog = windows["static"][:, static_idx]
        else:
            stat_exog = None

        return (
            insample_y,
            insample_mask,
            outsample_y,
            outsample_mask,
            hist_exog,
            futr_exog,
            stat_exog,
        )

    def training_step(self, batch, batch_idx):
        # Create and normalize windows [batch_size, n_series, C, L+H]
        windows = self._create_windows(batch, step="train")
        windows = self._normalization(windows=windows)

        # Parse windows
        (
            insample_y,
            insample_mask,
            outsample_y,
            outsample_mask,
            hist_exog,
            futr_exog,
            stat_exog,
        ) = self._parse_windows(batch, windows)

        windows_batch = dict(
            insample_y=insample_y,  # [batch_size, L, n_series]
            insample_mask=insample_mask,  # [batch_size, L, n_series]
            futr_exog=futr_exog,  # [batch_size, n_feats, L+H, n_series]
            hist_exog=hist_exog,  # [batch_size, n_feats, L, n_series]
            stat_exog=stat_exog,
        )  # [n_series, n_feats]

        # Model Predictions
        output = self(windows_batch)
        if self.loss.is_distribution_output:
            outsample_y, y_loc, y_scale = self._inv_normalization(
                y_hat=outsample_y, temporal_cols=batch["temporal_cols"]
            )
            distr_args = self.loss.scale_decouple(
                output=output, loc=y_loc, scale=y_scale
            )
            loss = self.loss(y=outsample_y, distr_args=distr_args, mask=outsample_mask)
        else:
            loss = self.loss(y=outsample_y, y_hat=output, mask=outsample_mask)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.train_trajectories.append((self.global_step, float(loss)))
        return loss

    def validation_step(self, batch, batch_idx):
        if self.val_size == 0:
            return np.nan

        # Create and normalize windows [Ws, L+H, C]
        windows = self._create_windows(batch, step="val")
        windows = self._normalization(windows=windows)

        # Parse windows
        (
            insample_y,
            insample_mask,
            outsample_y,
            outsample_mask,
            hist_exog,
            futr_exog,
            stat_exog,
        ) = self._parse_windows(batch, windows)

        windows_batch = dict(
            insample_y=insample_y,  # [Ws, L]
            insample_mask=insample_mask,  # [Ws, L]
            futr_exog=futr_exog,  # [Ws, L+H]
            hist_exog=hist_exog,  # [Ws, L]
            stat_exog=stat_exog,
        )  # [Ws, 1]

        # Model Predictions
        output = self(windows_batch)
        if self.loss.is_distribution_output:
            outsample_y, y_loc, y_scale = self._inv_normalization(
                y_hat=outsample_y, temporal_cols=batch["temporal_cols"]
            )
            distr_args = self.loss.scale_decouple(
                output=output, loc=y_loc, scale=y_scale
            )

            if str(type(self.valid_loss)) in [
                "<class 'neuralforecast.losses.pytorch.sCRPS'>",
                "<class 'neuralforecast.losses.pytorch.MQLoss'>",
            ]:
                _, output = self.loss.sample(distr_args=distr_args)

        # Validation Loss evaluation
        if self.valid_loss.is_distribution_output:
            valid_loss = self.valid_loss(
                y=outsample_y, distr_args=distr_args, mask=outsample_mask
            )
        else:
            valid_loss = self.valid_loss(
                y=outsample_y, y_hat=output, mask=outsample_mask
            )

        self.log("valid_loss", valid_loss, prog_bar=True, on_epoch=True)
        self.validation_step_outputs.append(valid_loss)
        return valid_loss

    def on_validation_epoch_end(self):
        if self.val_size == 0:
            return
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log("ptl/val_loss", avg_loss)
        self.valid_trajectories.append((self.global_step, float(avg_loss)))

    def predict_step(self, batch, batch_idx):
        # Create and normalize windows [Ws, L+H, C]
        windows = self._create_windows(batch, step="predict")
        windows = self._normalization(windows=windows)

        # Parse windows
        (
            insample_y,
            insample_mask,
            _,
            _,
            hist_exog,
            futr_exog,
            stat_exog,
        ) = self._parse_windows(batch, windows)

        windows_batch = dict(
            insample_y=insample_y,  # [Ws, L]
            insample_mask=insample_mask,  # [Ws, L]
            futr_exog=futr_exog,  # [Ws, L+H]
            hist_exog=hist_exog,  # [Ws, L]
            stat_exog=stat_exog,
        )  # [Ws, 1]

        # Model Predictions
        output = self(windows_batch)
        if self.loss.is_distribution_output:
            _, y_loc, y_scale = self._inv_normalization(
                y_hat=output[0], temporal_cols=batch["temporal_cols"]
            )
            distr_args = self.loss.scale_decouple(
                output=output, loc=y_loc, scale=y_scale
            )
            _, y_hat = self.loss.sample(distr_args=distr_args)

            if self.loss.return_params:
                distr_args = torch.stack(distr_args, dim=-1)
                distr_args = torch.reshape(
                    distr_args, (len(windows["temporal"]), self.h, -1)
                )
                y_hat = torch.concat((y_hat, distr_args), axis=2)
        else:
            y_hat, _, _ = self._inv_normalization(
                y_hat=output, temporal_cols=batch["temporal_cols"]
            )
        return y_hat

    def fit(self, dataset, val_size=0, test_size=0, random_seed=None):
        """Fit.

        The `fit` method, optimizes the neural network's weights using the
        initialization parameters (`learning_rate`, `windows_batch_size`, ...)
        and the `loss` function as defined during the initialization.
        Within `fit` we use a PyTorch Lightning `Trainer` that
        inherits the initialization's `self.trainer_kwargs`, to customize
        its inputs, see [PL's trainer arguments](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).

        The method is designed to be compatible with SKLearn-like classes
        and in particular to be compatible with the StatsForecast library.

        By default the `model` is not saving training checkpoints to protect
        disk memory, to get them change `enable_checkpointing=True` in `__init__`.

        **Parameters:**<br>
        `dataset`: NeuralForecast's `TimeSeriesDataset`, see [documentation](https://nixtla.github.io/neuralforecast/tsdataset.html).<br>
        `val_size`: int, validation size for temporal cross-validation.<br>
        `test_size`: int, test size for temporal cross-validation.<br>
        """
        # Restart random seed
        if random_seed is None:
            random_seed = self.random_seed
        torch.manual_seed(random_seed)

        self.val_size = val_size
        self.test_size = test_size
        datamodule = TimeSeriesDataModule(
            dataset=dataset,
            batch_size=self.n_series,
            num_workers=self.num_workers_loader,
            drop_last=self.drop_last_loader,
        )

        ### Check validation every steps ###
        steps_in_epoch = np.ceil(dataset.n_groups / self.n_series)

        assert steps_in_epoch == 1, "n_series must be equal to number of series"

        # In v1.6.5 of PL, val_check_interval can be used for multiple validation steps
        # within one epoch (steps_in_epoch > self.val_check_steps)
        if steps_in_epoch > self.val_check_steps:
            val_check_interval = self.val_check_steps / steps_in_epoch
            check_val_every_n_epoch = 1
        # Use check_val_every_n_epoch to check validation at end of some epochs,
        # closest to self.val_check_steps.
        else:
            val_check_interval = None
            check_val_every_n_epoch = int(
                np.round(self.val_check_steps / steps_in_epoch)
            )

        self.trainer_kwargs["val_check_interval"] = val_check_interval
        self.trainer_kwargs["check_val_every_n_epoch"] = check_val_every_n_epoch

        trainer = pl.Trainer(**self.trainer_kwargs)
        trainer.fit(self, datamodule=datamodule)

    def predict(
        self,
        dataset,
        test_size=None,
        step_size=1,
        random_seed=None,
        **data_module_kwargs,
    ):
        """Predict.

        Neural network prediction with PL's `Trainer` execution of `predict_step`.

        **Parameters:**<br>
        `dataset`: NeuralForecast's `TimeSeriesDataset`, see [documentation](https://nixtla.github.io/neuralforecast/tsdataset.html).<br>
        `test_size`: int=None, test size for temporal cross-validation.<br>
        `step_size`: int=1, Step size between each window.<br>
        `**data_module_kwargs`: PL's TimeSeriesDataModule args, see [documentation](https://pytorch-lightning.readthedocs.io/en/1.6.1/extensions/datamodules.html#using-a-datamodule).
        """
        # Restart random seed
        if random_seed is None:
            random_seed = self.random_seed
        torch.manual_seed(random_seed)

        self.predict_step_size = step_size
        self.decompose_forecast = False
        datamodule = TimeSeriesDataModule(
            dataset=dataset, batch_size=self.n_series, **data_module_kwargs
        )

        # Protect when case of multiple gpu. PL does not support return preds with multiple gpu.
        pred_trainer_kwargs = self.trainer_kwargs.copy()
        if (pred_trainer_kwargs.get("accelerator", None) == "gpu") and (
            torch.cuda.device_count() > 1
        ):
            pred_trainer_kwargs["devices"] = [0]

        trainer = pl.Trainer(**pred_trainer_kwargs)
        fcsts = trainer.predict(self, datamodule=datamodule)
        fcsts = torch.vstack(fcsts).numpy()

        fcsts = np.transpose(fcsts, (2, 0, 1))
        fcsts = fcsts.flatten()
        fcsts = fcsts.reshape(-1, len(self.loss.output_names))
        return fcsts

    def decompose(self, dataset, step_size=1, random_seed=None, **data_module_kwargs):
        raise NotImplementedError("decompose")

    def forward(self, insample_y, insample_mask):
        raise NotImplementedError("forward")

    def set_test_size(self, test_size):
        self.test_size = test_size

    def save(self, path):
        """BaseWindows.save

        Save the fitted model to disk.

        **Parameters:**<br>
        `path`: str, path to save the model.<br>
        """
        self.trainer.save_checkpoint(path)
