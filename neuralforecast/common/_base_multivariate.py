# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/common.base_multivariate.ipynb.

# %% auto 0
__all__ = ['BaseMultivariate']

# %% ../../nbs/common.base_multivariate.ipynb 5
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

import neuralforecast.losses.pytorch as losses
from ._base_model import BaseModel
from ._scalers import TemporalNorm
from ..tsdataset import TimeSeriesDataModule
from ..utils import get_indexer_raise_missing

# %% ../../nbs/common.base_multivariate.ipynb 6
class BaseMultivariate(BaseModel):
    """Base Multivariate

    Base class for all multivariate models. The forecasts for all time-series are produced simultaneously
    within each window, which are randomly sampled during training.

    This class implements the basic functionality for all windows-based models, including:
    - PyTorch Lightning's methods training_step, validation_step, predict_step.<br>
    - fit and predict methods used by NeuralForecast.core class.<br>
    - sampling and wrangling methods to generate multivariate windows.
    """

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
        optimizer=None,
        optimizer_kwargs=None,
        **trainer_kwargs,
    ):
        super().__init__(
            random_seed=random_seed,
            loss=loss,
            valid_loss=valid_loss,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            max_steps=max_steps,
            early_stop_patience_steps=early_stop_patience_steps,
            **trainer_kwargs,
        )

        # Padder to complete train windows,
        # example y=[1,2,3,4,5] h=3 -> last y_output = [5,0,0]
        self.h = h
        self.input_size = input_size
        self.n_series = n_series
        self.padder = nn.ConstantPad1d(padding=(0, self.h), value=0)

        # Multivariate models do not support these loss functions yet.
        unsupported_losses = (
            losses.sCRPS,
            losses.MQLoss,
            losses.DistributionLoss,
            losses.PMM,
            losses.GMM,
            losses.HuberMQLoss,
            losses.MASE,
            losses.relMSE,
            losses.NBMM,
        )
        if isinstance(self.loss, unsupported_losses):
            raise Exception(f"{self.loss} is not supported in a Multivariate model.")
        if isinstance(self.valid_loss, unsupported_losses):
            raise Exception(
                f"{self.valid_loss} is not supported in a Multivariate model."
            )

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

        # Fit arguments
        self.val_size = 0
        self.test_size = 0

        # Model state
        self.decompose_forecast = False

        # DataModule arguments
        self.num_workers_loader = num_workers_loader
        self.drop_last_loader = drop_last_loader
        # used by on_validation_epoch_end hook
        self.validation_step_outputs = []
        self.alias = alias

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
            n_windows = windows.shape[2]
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

    def _normalization(self, windows, y_idx):

        # windows are already filtered by train/validation/test
        # from the `create_windows_method` nor leakage risk
        temporal = windows["temporal"]  # [Ws, C, L+H, n_series]
        temporal_cols = windows["temporal_cols"].copy()  # [Ws, C, L+H, n_series]

        # To avoid leakage uses only the lags
        temporal_data_cols = self._get_temporal_exogenous_cols(
            temporal_cols=temporal_cols
        )
        temporal_idxs = get_indexer_raise_missing(temporal_cols, temporal_data_cols)
        temporal_idxs = np.append(y_idx, temporal_idxs)
        temporal_data = temporal[:, temporal_idxs, :, :]
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
        temporal[:, temporal_idxs, :, :] = temporal_data
        windows["temporal"] = temporal

        return windows

    def _inv_normalization(self, y_hat, temporal_cols, y_idx):
        # Receives window predictions [Ws, H, n_series]
        # Broadcasts outputs and inverts normalization

        # Add C dimension
        # if y_hat.ndim == 2:
        #     remove_dimension = True
        #     y_hat = y_hat.unsqueeze(-1)
        # else:
        #     remove_dimension = False

        y_scale = self.scaler.x_scale[:, [y_idx], :].squeeze(1)
        y_loc = self.scaler.x_shift[:, [y_idx], :].squeeze(1)

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
        mask_idx = batch["temporal_cols"].get_loc("available_mask")
        y_idx = batch["y_idx"]
        insample_y = windows["temporal"][:, y_idx, : -self.h, :]
        insample_mask = windows["temporal"][:, mask_idx, : -self.h, :]
        outsample_y = windows["temporal"][:, y_idx, -self.h :, :]
        outsample_mask = windows["temporal"][:, mask_idx, -self.h :, :]

        # Filter historic exogenous variables
        if len(self.hist_exog_list):
            hist_exog_idx = get_indexer_raise_missing(
                windows["temporal_cols"], self.hist_exog_list
            )
            hist_exog = windows["temporal"][:, hist_exog_idx, : -self.h, :]
        else:
            hist_exog = None

        # Filter future exogenous variables
        if len(self.futr_exog_list):
            futr_exog_idx = get_indexer_raise_missing(
                windows["temporal_cols"], self.futr_exog_list
            )
            futr_exog = windows["temporal"][:, futr_exog_idx, :, :]
        else:
            futr_exog = None

        # Filter static variables
        if len(self.stat_exog_list):
            static_idx = get_indexer_raise_missing(
                windows["static_cols"], self.stat_exog_list
            )
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
        y_idx = batch["y_idx"]
        windows = self._normalization(windows=windows, y_idx=y_idx)

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
                y_hat=outsample_y, temporal_cols=batch["temporal_cols"], y_idx=y_idx
            )
            distr_args = self.loss.scale_decouple(
                output=output, loc=y_loc, scale=y_scale
            )
            loss = self.loss(y=outsample_y, distr_args=distr_args, mask=outsample_mask)
        else:
            loss = self.loss(y=outsample_y, y_hat=output, mask=outsample_mask)

        if torch.isnan(loss):
            print("Model Parameters", self.hparams)
            print("insample_y", torch.isnan(insample_y).sum())
            print("outsample_y", torch.isnan(outsample_y).sum())
            print("output", torch.isnan(output).sum())
            raise Exception("Loss is NaN, training stopped.")

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.train_trajectories.append((self.global_step, float(loss)))
        return loss

    def validation_step(self, batch, batch_idx):
        if self.val_size == 0:
            return np.nan

        # Create and normalize windows [Ws, L+H, C]
        windows = self._create_windows(batch, step="val")
        y_idx = batch["y_idx"]
        windows = self._normalization(windows=windows, y_idx=y_idx)

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
                y_hat=outsample_y, temporal_cols=batch["temporal_cols"], y_idx=y_idx
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

        if torch.isnan(valid_loss):
            raise Exception("Loss is NaN, training stopped.")

        self.log("valid_loss", valid_loss, prog_bar=True, on_epoch=True)
        self.validation_step_outputs.append(valid_loss)
        return valid_loss

    def predict_step(self, batch, batch_idx):
        # Create and normalize windows [Ws, L+H, C]
        windows = self._create_windows(batch, step="predict")
        y_idx = batch["y_idx"]
        windows = self._normalization(windows=windows, y_idx=y_idx)

        # Parse windows
        insample_y, insample_mask, _, _, hist_exog, futr_exog, stat_exog = (
            self._parse_windows(batch, windows)
        )

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
                y_hat=output[0], temporal_cols=batch["temporal_cols"], y_idx=y_idx
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
                y_hat=output, temporal_cols=batch["temporal_cols"], y_idx=y_idx
            )
        return y_hat

    def fit(
        self,
        dataset,
        val_size=0,
        test_size=0,
        random_seed=None,
        distributed_config=None,
    ):
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
        if distributed_config is not None:
            raise ValueError(
                "multivariate models cannot be trained using distributed data parallel."
            )
        return self._fit(
            dataset=dataset,
            batch_size=self.n_series,
            val_size=val_size,
            test_size=test_size,
            random_seed=random_seed,
            shuffle_train=False,
            distributed_config=None,
        )

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
        self._check_exog(dataset)
        self._restart_seed(random_seed)

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
