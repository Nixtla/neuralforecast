__all__ = ["DummyUnivariate", "DummyMultivariate", "DummyRecurrent"]

import torch
import torch.nn as nn
from typing import List, Optional, Union

from neuralforecast.common._base_model import BaseModel
from neuralforecast.losses.pytorch import MAE


class DummyUnivariate(BaseModel):
    """DummyUnivariate - A simple dummy univariate model for testing recurrent predictions.

    This model implements a seasonal naive prediction strategy.
    It simply feeds back the seasonality-lagged (seasonal length hard-coded = forecast horizon)
    values for predictions

    Caveat: This only works when h <= input_size and h hard-coded as seasonlity size
    """

    # Class attributes
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True
    MULTIVARIATE = False
    RECURRENT = False

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        stat_exog_list: Union[List, None] = None,
        hist_exog_list: Union[List, None] = None,
        futr_exog_list: Union[List, None] = None,
        inference_input_size: Optional[int] = None,
        h_train: int = 1,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1,
        learning_rate: float = 0.0,  # No learning, just to test seasonal naive
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 2,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = -1,
        start_padding_enabled: bool = False,
        training_data_availability_threshold: float = 0.0,
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
        **trainer_kwargs,
    ):
        super().__init__(
            h=h,
            input_size=input_size,
            stat_exog_list=stat_exog_list,
            hist_exog_list=hist_exog_list,
            futr_exog_list=futr_exog_list,
            inference_input_size=inference_input_size,
            h_train=h_train,
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
            **trainer_kwargs,
        )
        assert input_size >= h, "h must be <= input_size"
        self.seasonality = h
        # just to have some learnable parameters
        self.w = nn.Parameter(torch.tensor([1.0], requires_grad=True))

    def forward(self, windows_batch):
        """Forward pass that implements seasonal naive prediction.

        This implementation assumes that we always forecast at a fixed window size as
        suggested by training h.
        """
        insample_y = windows_batch["insample_y"]  # [B, L, 1]
        futr_exog = windows_batch["futr_exog"]

        # Create predictions by shifting the input sequence
        y_pred = insample_y[:, -self.seasonality :, :] * self.w  # [B, h, 1]
        if self.futr_exog_size > 0:
            # we only consider a single futr_exog feature, that is why '0' in the last dimension
            # simply multiply with the given futr_exog
            y_pred = y_pred * futr_exog[:, -self.seasonality :, 0].unsqueeze(-1)

        # Expand output to match loss function requirements
        batch_size = y_pred.shape[0]
        if self.loss.outputsize_multiplier > 1:
            # For distribution losses, we need to output multiple parameters
            # Repeat the prediction for each required parameter
            # e.g. this differs from NeuralNetwork that we do not learn parameters such as
            # loc (location),scale for Distribution(distribution="normal")
            y_pred = y_pred.repeat(1, 1, self.loss.outputsize_multiplier)
        y_pred = y_pred.reshape(batch_size, self.h, self.loss.outputsize_multiplier)
        return y_pred


class DummyMultivariate(BaseModel):
    """DummyMultivariate - A simple dummy multivariate model for testing recurrent predictions.

    This model implements a seasonal naive prediction strategy for multivariate time series.
    It simply feeds back the seasonality-lagged (seasonal length hard-coded = forecast horizon)
    values for predictions

    Caveat: This only works when h <= input_size and h hard-coded as seasonlity size
    """

    # Class attributes
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True
    MULTIVARIATE = True
    RECURRENT = False

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        stat_exog_list: Union[List, None] = None,
        hist_exog_list: Union[List, None] = None,
        futr_exog_list: Union[List, None] = None,
        inference_input_size: Optional[int] = None,
        h_train: int = 1,
        n_series: int = 2,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1,
        learning_rate: float = 0.0,  # No learning, just to test seasonal naive
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 2,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = -1,
        start_padding_enabled: bool = False,
        training_data_availability_threshold: float = 0.0,
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
        **trainer_kwargs,
    ):
        super().__init__(
            h=h,
            input_size=input_size,
            stat_exog_list=stat_exog_list,
            hist_exog_list=hist_exog_list,
            futr_exog_list=futr_exog_list,
            inference_input_size=inference_input_size,
            h_train=h_train,
            n_series=n_series,
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
            **trainer_kwargs,
        )
        assert input_size >= h, "h must be <= input_size"
        self.seasonality = h
        # just to have some learnable parameters
        self.w = nn.Parameter(torch.tensor([1.0], requires_grad=True))

    def forward(self, windows_batch):
        """Forward pass that implements multivariate seasonal naive prediction."""
        insample_y = windows_batch["insample_y"]  # [B, L, N]
        futr_exog = windows_batch["futr_exog"]  # [B, F, L, N]

        # Create predictions by shifting the input sequence
        y_pred = insample_y[:, -self.seasonality :, :] * self.w  # [B, h, N]

        if self.futr_exog_size > 0:
            # we only consider a single futr_exog feature, that is why '0' in the dimension
            # simply multiply with the given futr_exog
            y_pred = y_pred * futr_exog[:, 0, -self.seasonality :, :].squeeze(1)

        # Expand output to match loss  function requirements
        batch_size = y_pred.shape[0] # (1024, 4, 2), I need to find similar treatment to check the impact on n.
        if self.loss.outputsize_multiplier > 1:
            # For distribution losses, we need to output multiple parameters
            # Repeat the prediction for each required parameter
            # e.g. this differs from NeuralNetwork that we do not learn parameters such as
            # loc (location),scale for Distribution(distribution="normal")
            y_pred = y_pred.repeat(1, 1, self.loss.outputsize_multiplier)
        y_pred = y_pred.reshape(batch_size, self.h, self.loss.outputsize_multiplier*self.n_series)
        return y_pred


class DummyRecurrent(BaseModel):
    """DummyRecurrent - A simple dummy recurrent model for testing recurrent predictions.

    This model implements a lagged-1 prediction strategy for time series.
    """

    # Class attributes
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True
    MULTIVARIATE = False
    RECURRENT = True

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        stat_exog_list: Union[List, None] = None,
        hist_exog_list: Union[List, None] = None,
        futr_exog_list: Union[List, None] = None,
        inference_input_size: Optional[int] = None,
        h_train: int = 1,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1,
        learning_rate: float = 0.0,  # No learning, just to naive prediction
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 2,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = -1,
        start_padding_enabled: bool = False,
        training_data_availability_threshold: float = 0.0,
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
        **trainer_kwargs,
    ):
        super().__init__(
            h=h,
            input_size=input_size,
            stat_exog_list=stat_exog_list,
            hist_exog_list=hist_exog_list,
            futr_exog_list=futr_exog_list,
            inference_input_size=inference_input_size,
            h_train=h_train,
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
            **trainer_kwargs,
        )
        self.seasonality = h
        # just to have some learnable parameters
        self.w = nn.Parameter(torch.tensor([1.0], requires_grad=True))

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]  # [B, L, N]
        futr_exog = windows_batch["futr_exog"]

        # Create predictions by shifting the input sequence
        y_pred = insample_y[:, -1, :] * self.w

        if self.futr_exog_size > 0:
            # we only consider a single futr_exog feature, that is why '0' in the dimension
            # simply multiply with the given futr_exog
            y_pred = y_pred * futr_exog[:, -1:, 0]

        y_pred = y_pred.unsqueeze(-1)  # [B, 1, N]

        # Expand output to match loss function requirements
        batch_size = y_pred.shape[0]
        if self.loss.outputsize_multiplier > 1:
            # For distribution losses, we need to output multiple parameters
            # Repeat the prediction for each required parameter
            # e.g. this differs from NeuralNetwork that we do not learn parameters such as
            # loc (location),scale for Distribution(distribution="normal")
            y_pred = y_pred.repeat(1, 1, self.loss.outputsize_multiplier)
        y_pred = y_pred.reshape(batch_size, 1, self.loss.outputsize_multiplier)
        return y_pred
