"""
UR2CUTE: Using Repetitively 2 CNNs for Unsteady Timeseries Estimation.

The model uses a dual-CNN hurdle architecture tailored for intermittent demand:
1. A classifier estimates the probability of observing non-zero demand.
2. A regressor estimates the demand magnitude when demand occurs.

References:
-----------
Mirshahi, S., Brandtner, P., & Kominkova Oplatkova, Z. (2024).
Intermittent Time Series Demand Forecasting Using Dual Convolutional Neural Networks.
MENDEL -- Soft Computing Journal, 30(1).
"""

import torch
import torch.nn as nn
from typing import Optional

from neuralforecast.common._base_model import BaseModel
from neuralforecast.losses.pytorch import MAE


class CNNClassifier(nn.Module):
    """
    PyTorch CNN model for classification (zero vs. nonzero demand occurrence)

    Parameters
    ----------
    n_features : int
        Number of input features
    forecast_horizon : int
        Number of future steps to predict
    dropout_rate : float, optional (default=0.4)
        Dropout rate for regularization
    """
    def __init__(self, n_features, forecast_horizon, dropout_rate=0.4):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()

        # Calculate size after pooling
        flattened_size = 64 * (n_features // 2)

        self.fc1 = nn.Linear(flattened_size, 32)
        self.fc2 = nn.Linear(32, forecast_horizon)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for classification

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch, features, 1]

        Returns
        -------
        torch.Tensor
            Probability of non-zero demand for each horizon step [batch, horizon]
        """
        # Input shape: (batch, 1, features)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


class CNNRegressor(nn.Module):
    """
    PyTorch CNN model for regression (demand magnitude estimation)

    Parameters
    ----------
    n_features : int
        Number of input features
    forecast_horizon : int
        Number of future steps to predict
    dropout_rate : float, optional (default=0.2)
        Dropout rate for regularization
    """
    def __init__(self, n_features, forecast_horizon, dropout_rate=0.2):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()

        # Calculate size after pooling
        flattened_size = 32 * (n_features // 2)

        self.fc1 = nn.Linear(flattened_size, 46)
        self.fc2 = nn.Linear(46, forecast_horizon)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass for regression

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch, features, 1]

        Returns
        -------
        torch.Tensor
            Predicted demand magnitude for each horizon step [batch, horizon]
        """
        # Input shape: (batch, 1, features)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class UR2CUTE(BaseModel):
    """
    UR2CUTE: Using Repetitively 2 CNNs for Unsteady Timeseries Estimation

    A two-step hurdle approach for intermittent demand forecasting:
    1. Classification CNN predicts whether demand will occur (zero vs. non-zero)
    2. Regression CNN predicts the magnitude of demand

    The final forecast combines both models: if the classification probability
    exceeds the threshold, the regression output is used; otherwise, zero is predicted.

    Parameters
    ----------
    h : int
        Forecast horizon - number of future steps to predict
    input_size : int
        Number of historical timesteps to use as input (lookback window)
    loss : pytorch module, optional (default=MAE())
        Training loss function. For the two-step approach, this primarily affects
        the regression model's training.
    valid_loss : pytorch module, optional (default=None)
        Validation loss function. If None, uses the same as loss.
    classification_threshold : float or str, optional (default=0.5)
        Probability threshold for classifying zero vs. non-zero demand.
        Can be a float between 0 and 1, or "auto" to compute from training data.
    dropout_classification : float, optional (default=0.4)
        Dropout rate for the classification CNN
    dropout_regression : float, optional (default=0.2)
        Dropout rate for the regression CNN
    classification_weight : float, optional (default=0.3)
        Weight for classification loss in combined loss (between 0 and 1).
        Total loss = classification_weight * BCE + (1 - classification_weight) * regression_loss
    learning_rate : float, optional (default=1e-3)
        Learning rate for optimization
    max_steps : int, optional (default=1000)
        Maximum number of training steps
    val_check_steps : int, optional (default=100)
        Validation check frequency
    batch_size : int, optional (default=32)
        Batch size for training
    random_seed : int, optional (default=1)
        Random seed for reproducibility
    **trainer_kwargs : additional arguments
        Additional arguments passed to PyTorch Lightning Trainer

    Examples
    --------
    >>> from neuralforecast import NeuralForecast
    >>> from neuralforecast.models import UR2CUTE
    >>> from neuralforecast.losses.pytorch import MAE
    >>>
    >>> # Create model for intermittent demand forecasting
    >>> model = UR2CUTE(
    ...     h=12,                              # Forecast 12 periods ahead
    ...     input_size=24,                     # Use last 24 periods as input
    ...     classification_threshold=0.5,       # 50% probability threshold
    ...     dropout_classification=0.4,
    ...     dropout_regression=0.2,
    ...     max_steps=1000,
    ...     learning_rate=0.001
    ... )
    >>>
    >>> # Initialize NeuralForecast with the model
    >>> nf = NeuralForecast(models=[model], freq='W')
    >>>
    >>> # Fit on training data
    >>> nf.fit(df=train_df)
    >>>
    >>> # Make predictions
    >>> forecasts = nf.predict(df=test_df)

    References
    ----------
    Mirshahi, S., Brandtner, P., & Kominkova Oplatkova, Z. (2024).
    Intermittent Time Series Demand Forecasting Using Dual Convolutional Neural Networks.
    MENDEL -- Soft Computing Journal, 30(1).
    """

    # Model configuration flags for NeuralForecast framework
    EXOGENOUS_FUTR = False   # Can be extended to support future exogenous variables
    EXOGENOUS_HIST = False   # Can be extended to support historical exogenous variables
    EXOGENOUS_STAT = False   # Can be extended to support static exogenous variables
    MULTIVARIATE = False     # Univariate forecasting
    RECURRENT = False        # Direct multi-step forecasting (not recursive)

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        loss=MAE(),
        valid_loss=None,
        classification_threshold: float = 0.5,
        dropout_classification: float = 0.4,
        dropout_regression: float = 0.2,
        classification_weight: float = 0.3,
        learning_rate: float = 1e-3,
        max_steps: int = 1000,
        val_check_steps: int = 100,
        batch_size: int = 32,
        random_seed: int = 1,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = -1,
        start_padding_enabled: bool = True,
        training_data_availability_threshold=0.0,
        step_size: int = 1,
        scaler_type: str = "identity",
        drop_last_loader: bool = False,
        alias: Optional[str] = None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs
    ):
        # Initialize BaseModel with all required parameters
        if "num_sanity_val_steps" not in trainer_kwargs:
            trainer_kwargs["num_sanity_val_steps"] = 0

        super(UR2CUTE, self).__init__(
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

        # UR2CUTE specific parameters
        self.classification_threshold = classification_threshold
        self.dropout_classification = dropout_classification
        self.dropout_regression = dropout_regression
        self.classification_weight = classification_weight

        self._return_components = False
        self._auto_threshold_enabled = (
            isinstance(self.classification_threshold, str)
            and self.classification_threshold.lower() == "auto"
        )
        if self._auto_threshold_enabled:
            self.classification_threshold_: Optional[float] = None
        else:
            self.classification_threshold_ = float(self.classification_threshold)
        self._zero_demand_count = 0.0
        self._total_demand_count = 0.0

        # Models will be initialized in setup() after we know input dimensions
        self.classifier = None
        self.regressor = None

        # Loss for classification (Binary Cross Entropy)
        self.bce_loss = nn.BCELoss(reduction="none")

        # Initialize models immediately so parameters are registered before optimizer setup
        self._build_models()

    def _build_models(self):
        """
        Build the classification and regression CNN models.
        Called after input_size is known from data.
        """
        # The input features are just the historical target values
        # since EXOGENOUS flags are all False
        n_features = self.input_size

        # Build classification CNN
        self.classifier = CNNClassifier(
            n_features=n_features,
            forecast_horizon=self.h,
            dropout_rate=self.dropout_classification
        )

        # Build regression CNN
        self.regressor = CNNRegressor(
            n_features=n_features,
            forecast_horizon=self.h,
            dropout_rate=self.dropout_regression
        )

    def _current_threshold(self) -> float:
        if self.classification_threshold_ is not None:
            return float(self.classification_threshold_)
        return 0.5

    def _update_auto_threshold_stats(self, outsample_y: torch.Tensor, mask: torch.Tensor) -> None:
        if not self._auto_threshold_enabled:
            return

        if mask is None:
            mask = torch.ones_like(outsample_y)

        valid_mask = mask > 0.0
        total = valid_mask.sum()
        if total.item() == 0:
            return

        zeros = torch.logical_and(valid_mask, torch.eq(outsample_y, 0.0)).sum()
        self._zero_demand_count += zeros.item()
        self._total_demand_count += total.item()

    def _compute_classification_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask if mask is not None else torch.ones_like(predictions)
        mask = mask.float()
        loss = self.bce_loss(predictions, targets) * mask
        valid = mask.sum()
        if valid.item() == 0:
            return torch.tensor(0.0, device=predictions.device)
        return loss.sum() / valid

    def on_train_epoch_end(self) -> None:
        if self._auto_threshold_enabled and self._total_demand_count > 0:
            ratio = self._zero_demand_count / self._total_demand_count
            self.classification_threshold_ = round(float(ratio), 2)
        return super().on_train_epoch_end()

    def forward(self, windows_batch):
        """
        Forward pass implementing the two-step hurdle approach

        Parameters
        ----------
        windows_batch : dict
            Dictionary containing:
            - insample_y : torch.Tensor [Batch, input_size, 1]
                Historical target values (normalized)
            - insample_mask : torch.Tensor [Batch, input_size, 1]
                Availability mask for historical data

        Returns
        -------
        dict
            Dictionary containing:
            - 'forecast': torch.Tensor [Batch, h, 1]
                Combined forecast (classification * regression)
            - 'classification': torch.Tensor [Batch, h]
                Raw classification probabilities
            - 'regression': torch.Tensor [Batch, h]
                Raw regression predictions
        """
        # Initialize models if not already done
        if self.classifier is None:
            self._build_models()

        # Extract historical target values and reshape for CNN
        # insample_y shape: [batch, input_size, 1]
        insample_y = windows_batch["insample_y"]
        batch_size = insample_y.shape[0]

        # Reshape for CNN: [batch, 1, input_size] (channels first)
        x = insample_y.permute(0, 2, 1)

        # Classification: predict probability of non-zero demand
        order_prob = self.classifier(x)

        # Regression: predict magnitude of demand
        quantity_pred = torch.relu(self.regressor(x))

        if self._return_components:
            return {
                "classification": order_prob,
                "regression": quantity_pred,
            }

        threshold_value = self._current_threshold()
        threshold_tensor = torch.full_like(order_prob, threshold_value)
        forecast = torch.where(
            order_prob > threshold_tensor,
            quantity_pred,
            torch.zeros_like(quantity_pred),
        )

        return forecast.unsqueeze(-1)

    def training_step(self, batch, batch_idx):
        """
        Custom training step to handle the two-step loss calculation

        Overrides BaseModel.training_step to compute combined loss:
        - BCE loss for classification (zero vs. non-zero)
        - Regression loss for magnitude prediction

        Parameters
        ----------
        batch : dict
            Batch of data from DataLoader
        batch_idx : int
            Index of the batch

        Returns
        -------
        torch.Tensor
            Combined loss value
        """
        # Extract y_idx and temporal_cols from batch
        y_idx = batch["y_idx"]
        temporal_cols = batch["temporal_cols"]

        # Create windows from batch (handled by BaseModel)
        windows_temporal, static, static_cols = self._create_windows(batch, step="train")

        # Sample windows for training
        windows = self._sample_windows(
            windows_temporal, static, static_cols, temporal_cols, step="train"
        )

        # Normalize windows
        windows = self._normalization(windows=windows, y_idx=y_idx)

        # Parse windows into components
        (
            insample_y,
            insample_mask,
            outsample_y,
            outsample_mask,
            hist_exog,
            futr_exog,
            stat_exog,
        ) = self._parse_windows(batch, windows)

        # Create windows_batch dict
        windows_batch = dict(
            insample_y=insample_y,
            insample_mask=insample_mask,
            futr_exog=futr_exog,
            hist_exog=hist_exog,
            stat_exog=stat_exog,
        )

        # Get targets - squeeze to remove last dimension
        outsample_y = outsample_y.squeeze(-1)  # [batch, h]
        outsample_mask = outsample_mask.squeeze(-1)

        # Forward pass (request both outputs)
        self._return_components = True
        try:
            outputs = self(windows_batch)
        finally:
            self._return_components = False

        # Calculate classification targets (1 if non-zero, 0 if zero)
        classification_target = (outsample_y > 0).float()  # [batch, h]
        self._update_auto_threshold_stats(outsample_y, outsample_mask)

        # Calculate classification loss (masked BCE)
        classification_loss = self._compute_classification_loss(
            outputs["classification"],
            classification_target,
            outsample_mask,
        )

        # Calculate regression loss only on non-zero samples
        nonzero_mask = (outsample_y > 0).float() * outsample_mask

        if nonzero_mask.sum() > 0:
            regression_loss = self.loss(
                outputs["regression"].unsqueeze(-1),  # [batch, h, 1]
                outsample_y.unsqueeze(-1),            # [batch, h, 1]
                mask=nonzero_mask.unsqueeze(-1),      # [batch, h, 1]
            )
        else:
            regression_loss = torch.tensor(0.0, device=self.device)

        # Combined loss
        total_loss = (
            self.classification_weight * classification_loss +
            (1 - self.classification_weight) * regression_loss
        )

        # Log metrics
        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_classification_loss", classification_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_regression_loss", regression_loss, prog_bar=False, on_step=False, on_epoch=True)

        # Calculate and log classification accuracy
        predicted_class = (outputs["classification"] > 0.5).float()
        if outsample_mask.sum() > 0:
            accuracy = (
                ((predicted_class == classification_target).float() * outsample_mask).sum()
                / outsample_mask.sum()
            )
        else:
            accuracy = torch.tensor(0.0, device=self.device)
        self.log(
            "train_classification_accuracy",
            accuracy,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Custom validation step matching the training step logic

        Parameters
        ----------
        batch : dict
            Batch of validation data
        batch_idx : int
            Index of the batch

        Returns
        -------
        torch.Tensor
            Combined validation loss
        """
        if self.val_size == 0:
            return torch.tensor(0.0, device=self.device)

        # Extract y_idx and temporal_cols from batch
        y_idx = batch["y_idx"]
        temporal_cols = batch["temporal_cols"]

        # Create and process windows
        windows_temporal, static, static_cols = self._create_windows(batch, step="validation")
        windows = self._sample_windows(
            windows_temporal, static, static_cols, temporal_cols, step="validation"
        )

        windows = self._normalization(windows=windows, y_idx=y_idx)

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
            insample_y=insample_y,
            insample_mask=insample_mask,
            futr_exog=futr_exog,
            hist_exog=hist_exog,
            stat_exog=stat_exog,
        )

        outsample_y = outsample_y.squeeze(-1)
        outsample_mask = outsample_mask.squeeze(-1)

        # Forward pass
        self._return_components = True
        try:
            outputs = self(windows_batch)
        finally:
            self._return_components = False

        # Calculate losses (same as training)
        classification_target = (outsample_y > 0).float()
        classification_loss = self._compute_classification_loss(
            outputs["classification"],
            classification_target,
            outsample_mask,
        )

        nonzero_mask = (outsample_y > 0).float() * outsample_mask
        if nonzero_mask.sum() > 0:
            regression_loss = self.loss(
                outputs["regression"].unsqueeze(-1),
                outsample_y.unsqueeze(-1),
                mask=nonzero_mask.unsqueeze(-1),
            )
        else:
            regression_loss = torch.tensor(0.0, device=self.device)

        total_loss = (
            self.classification_weight * classification_loss +
            (1 - self.classification_weight) * regression_loss
        )

        # Log validation metrics
        self.log("val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_classification_loss", classification_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_regression_loss", regression_loss, prog_bar=False, on_step=False, on_epoch=True)

        # Calculate and log classification accuracy
        predicted_class = (outputs["classification"] > 0.5).float()
        if outsample_mask.sum() > 0:
            accuracy = (
                ((predicted_class == classification_target).float() * outsample_mask).sum()
                / outsample_mask.sum()
            )
        else:
            accuracy = torch.tensor(0.0, device=self.device)
        self.log(
            "val_classification_accuracy",
            accuracy,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

        return total_loss
