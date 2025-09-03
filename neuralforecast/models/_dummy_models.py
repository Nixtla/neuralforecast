"""
Dummy models for testing any horizon predictions functionality.

These models provide predictable outputs to verify that recurrent predictions
work correctly when predicting horizons longer than the trained horizon.
"""

__all__ = ['DummyRNN', 'DummyMultivariate', 'DummyUnivariate']

import torch
import torch.nn as nn
import numpy as np

from neuralforecast.losses.pytorch import MAE
from neuralforecast.common._base_model import BaseModel


class DummyRNN(BaseModel):
    """
    Dummy RNN model that mimics seasonal naive behavior for testing recurrent predictions.
    
    This model just feeds back the (lagged) input as predictions, making it easy to verify
    that the recurrence logic works correctly for longer horizons.
    """
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True
    MULTIVARIATE = False
    RECURRENT = True

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        lag: int = 12,
        loss=MAE(),
        valid_loss=None,
        learning_rate: float = 1e-3,
        max_steps: int = 1000,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: int = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = None,
        start_padding_enabled: bool = False,
        **kwargs
    ):
        """
        Initialize DummyRNN.
        
        Args:
            h: Forecasting horizon
            input_size: Input window size
            lag: Seasonal lag for naive predictions (default 12)
            **kwargs: Additional arguments passed to BaseModel
        """
        super().__init__(
            h=h,
            input_size=input_size,
            loss=loss,
            valid_loss=valid_loss,
            learning_rate=learning_rate,
            max_steps=max_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            **kwargs
        )
        self.lag = lag
        
        # Simple linear layer for compatibility with framework
        self.dummy_layer = nn.Linear(1, 1)
        
    def forward(self, windows_batch):
        insample_y = windows_batch['insample_y']  # [batch_size, seq_len, n_series]
        batch_size = insample_y.shape[0]
        n_series = insample_y.shape[2] if len(insample_y.shape) > 2 else 1
        
        # For seasonal naive: use value from lag periods ago, or last value if not enough data
        if insample_y.shape[1] >= self.lag:
            # Use seasonal lag
            seasonal_value = insample_y[:, -self.lag, :]
        else:
            # Use last available value
            seasonal_value = insample_y[:, -1, :]
            
        # Reshape for output: [batch_size, 1, n_series] for horizon=1 during training
        if len(seasonal_value.shape) == 2:
            seasonal_value = seasonal_value.unsqueeze(1)
            
        return seasonal_value


class DummyMultivariate(BaseModel):
    """
    Dummy multivariate model for testing multivariate recurrent predictions.
    
    Returns simple linear combinations of input series as predictions.
    """
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True
    MULTIVARIATE = True
    RECURRENT = True

    def __init__(
        self,
        h: int,
        n_series: int,
        input_size: int = -1,
        loss=MAE(),
        valid_loss=None,
        learning_rate: float = 1e-3,
        max_steps: int = 1000,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: int = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = None,
        start_padding_enabled: bool = False,
        **kwargs
    ):
        """
        Initialize DummyMultivariate.
        
        Args:
            h: Forecasting horizon
            n_series: Number of series
            input_size: Input window size
            **kwargs: Additional arguments passed to BaseModel
        """
        super().__init__(
            h=h,
            input_size=input_size,
            loss=loss,
            valid_loss=valid_loss,
            learning_rate=learning_rate,
            max_steps=max_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            n_series=n_series,
            **kwargs
        )
        
        # Simple mixing matrix for multivariate predictions
        self.mixing_layer = nn.Linear(n_series, n_series)
        
    def forward(self, windows_batch):
        insample_y = windows_batch['insample_y']  # [batch_size, seq_len, n_series]
        
        # Take last value and apply simple linear transformation
        last_values = insample_y[:, -1, :]  # [batch_size, n_series]
        
        # Apply mixing (simple linear transformation)
        mixed_values = self.mixing_layer(last_values)
        
        # Reshape for output: [batch_size, 1, n_series] for horizon=1 during training
        mixed_values = mixed_values.unsqueeze(1)
        
        return mixed_values


class DummyUnivariate(BaseModel):
    """
    Dummy univariate model for testing direct (non-recurrent) predictions.
    
    Returns a simple function of the input mean as prediction.
    """
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True
    MULTIVARIATE = False
    RECURRENT = False

    def __init__(
        self,
        h: int,
        input_size: int = 24,
        trend_factor: float = 1.01,
        loss=MAE(),
        valid_loss=None,
        learning_rate: float = 1e-3,
        max_steps: int = 1000,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: int = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = None,
        start_padding_enabled: bool = False,
        **kwargs
    ):
        """
        Initialize DummyUnivariate.
        
        Args:
            h: Forecasting horizon
            input_size: Input window size
            trend_factor: Factor to apply for trend (default 1.01 for 1% growth)
            **kwargs: Additional arguments passed to BaseModel
        """
        super().__init__(
            h=h,
            input_size=input_size,
            loss=loss,
            valid_loss=valid_loss,
            learning_rate=learning_rate,
            max_steps=max_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            **kwargs
        )
        self.trend_factor = trend_factor
        
        # Simple layer for framework compatibility
        self.dummy_layer = nn.Linear(1, h)
        
    def forward(self, windows_batch):
        insample_y = windows_batch['insample_y']  # [batch_size, seq_len, n_series]
        
        # Calculate mean of input
        mean_value = insample_y.mean(dim=1, keepdim=True)  # [batch_size, 1, n_series]
        
        # Apply trend factor for each horizon step
        batch_size = mean_value.shape[0]
        n_series = mean_value.shape[2] if len(mean_value.shape) > 2 else 1
        
        # Create trend multipliers for each horizon step
        trend_multipliers = torch.tensor(
            [self.trend_factor ** i for i in range(1, self.h + 1)], 
            device=mean_value.device,
            dtype=mean_value.dtype
        )
        
        # Apply trend: [batch_size, h, n_series]
        predictions = mean_value * trend_multipliers.view(1, -1, 1)
        
        return predictions