__all__ = ['DummyUnivariate']

import torch
import torch.nn as nn

from ..common._base_model import BaseModel
from ..losses.pytorch import MAE


class DummyUnivariate(BaseModel):
    """DummyUnivariate - A simple dummy univariate model for testing recurrent predictions.
    
    This model implements a simple naive prediction strategy where it just feeds back
    the last input value as the prediction. It's designed to test the univariate
    recurrent prediction infrastructure without complex model logic.
    
    **Parameters:**<br>
    `h`: int, forecast horizon.<br>
    `input_size`: int, maximum sequence length for truncated train backpropagation.<br>
    `loss`: PyTorch module, instantiated train loss class.<br>
    `valid_loss`: PyTorch module, instantiated valid loss class.<br>
    `max_steps`: int, maximum number of training steps.<br>
    `learning_rate`: float, learning rate between (0, 1).<br>
    `batch_size`: int, number of different series in each batch.<br>
    `**trainer_kwargs`: keyword trainer arguments inherited from PyTorch Lightning's trainer.
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
        inference_input_size: int = None,
        h_train: int = 1,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: int = None,
        windows_batch_size: int = 128,
        inference_windows_batch_size: int = 1024,
        start_padding_enabled: bool = False,
        training_data_availability_threshold: float = 0.0,
        step_size: int = 1,
        scaler_type: str = 'robust',
        random_seed: int = 1,
        drop_last_loader: bool = False,
        alias: str = None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs
    ):
        super().__init__(
            h=h,
            input_size=input_size,
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
            **trainer_kwargs
        )
        
        # Simple linear layer for "training" - just to have some parameters
        self.dummy_layer = nn.Linear(1, 1)
        
    def forward(self, windows_batch):
        """Forward pass that implements simple naive prediction.
        
        For each prediction step, it just feeds back the last input value.
        This is the simplest possible recurrent prediction.
        """
        insample_y = windows_batch["insample_y"].squeeze(-1)  # [B, L, N]
        batch_size, seq_len, n_series = insample_y.shape
        
        # Create predictions by repeating the last value
        last_value = insample_y[:, -1, :]  # [B, N]
        
        # Apply dummy layer to simulate some "learning"
        last_value = self.dummy_layer(last_value.unsqueeze(-1)).squeeze(-1)
        
        # Repeat the last value for all horizon steps
        # [B, 1, N] -> [B, h, N]
        y_pred = last_value.unsqueeze(1).expand(-1, self.h, -1)
        
        # Reshape to match loss output requirements
        y_pred = y_pred.reshape(batch_size, self.h, self.loss.outputsize_multiplier)
        
        return y_pred
