import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import math


def _combined_loss(alpha=0.5):
    """
    Custom loss function combining MSE and MAE.
    """
    def loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        return alpha * mse + (1 - alpha) * mae
    return loss


def _generate_lag_features(df, column_name, n_lags=1):
    """
    Generate lag features for a given column in the dataframe.
    """
    df = df.copy()
    for i in range(1, n_lags + 1):
        df[f"{column_name}_Lag{i}"] = df[column_name].shift(i)
    return df


def _create_multistep_data(df, target_name, external_features, n_steps_lag, forecast_horizon):
    """
    Build multi-step training samples. For each possible row i (up to len(df)-forecast_horizon):
      - The input vector is: [external features] + [lag features from row i]
      - The target is the next forecast_horizon values of target_name (rows i+1 .. i+forecast_horizon).
    """
    X_list = []
    y_list = []
    for i in range(len(df) - forecast_horizon):
        # Lags from current row i
        lag_vals = df.iloc[i][[f"{target_name}_Lag{j}" for j in range(1, n_steps_lag + 1)]].values

        # External features from current row i (if any)
        ext_vals = df.loc[i, external_features].values if external_features else []
        
        X_list.append(np.concatenate([ext_vals, lag_vals]))
        # Next forecast_horizon steps for the target
        y_seq = df.loc[i+1 : i+forecast_horizon, target_name].values
        y_list.append(y_seq)
    return np.array(X_list), np.array(y_list)


class UR2CUTE(BaseEstimator):
    """
    UR2CUTE: Using Repetitively 2 CNNs for Unsteady Timeseries Estimation (two-step/hurdle approach).

    This estimator does direct multi-step forecasting with:
      - A CNN-based classification model to predict zero vs. nonzero for each future step.
      - A CNN-based regression model to predict the quantity (only trained on sequences that have
        at least one nonzero step in the horizon).

    Parameters
    ----------
    n_steps_lag : int
        Number of lag features to generate.
    forecast_horizon : int
        Number of future steps to predict in one pass.
    external_features : list of str or None
        Column names for external features (if any).
    epochs : int
        Training epochs for both CNN models.
    batch_size : int
        Batch size for training.
    threshold : float
        Probability threshold for classifying zero vs. nonzero demand.
    patience : int
        Patience for EarlyStopping.
    random_seed : int
        Random seed for reproducibility.
    classification_lr : float
        Learning rate for classification model.
    regression_lr : float
        Learning rate for regression model.
    dropout_classification : float
        Dropout rate for the classification model.
    dropout_regression : float
        Dropout rate for the regression model.
    """

    def __init__(
        self,
        n_steps_lag=3,
        forecast_horizon=8,
        external_features=None,
        epochs=100,
        batch_size=32,
        threshold=0.5,
        patience=10,
        random_seed=42,
        classification_lr=0.0021,
        regression_lr=0.0021,
        dropout_classification=0.4,
        dropout_regression=0.2
    ):
        self.n_steps_lag = n_steps_lag
        self.forecast_horizon = forecast_horizon
        self.external_features = external_features if external_features is not None else []
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.patience = patience
        self.random_seed = random_seed
        self.classification_lr = classification_lr
        self.regression_lr = regression_lr
        self.dropout_classification = dropout_classification
        self.dropout_regression = dropout_regression

        # Models will be created in fit()
        self.classifier_ = None
        self.regressor_ = None
        # Scalers
        self.scaler_X_ = None
        self.scaler_y_ = None
        # Fitted dims
        self.n_features_ = None

    def _set_random_seeds(self):
        """
        Force reproducible behavior by setting seeds.
        Note: On GPU, some ops may still be non-deterministic.
        """
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

    def fit(self, df, target_col):
        """
        Fit the UR2CUTE model on a time-series dataframe `df`.

        Expected columns:
          - `target_col`: The main target to forecast.
          - If external_features is not empty, those columns must exist in df.
          - We'll generate lag features for `target_col`.

        Parameters
        ----------
        df : pd.DataFrame
            Time-series data with at least the target column. Must be sorted by time in advance
            (or you can ensure we do it here).
        target_col : str
            The name of the column to forecast.

        Returns
        -------
        self : UR2CUTE
            Fitted estimator.
        """
        self._set_random_seeds()

        # 1) Generate lag features & drop NaNs
        df_lagged = _generate_lag_features(df, target_col, n_lags=self.n_steps_lag)
        df_lagged.dropna(inplace=True)
        df_lagged.reset_index(drop=True, inplace=True)


        # 2) Create multi-step training data
        X_all, y_all = _create_multistep_data(
            df_lagged,
            target_col,
            self.external_features,
            self.n_steps_lag,
            self.forecast_horizon
        )
        # shape: X_all -> (samples, features), y_all -> (samples, forecast_horizon)

        # 3) Scale inputs
        self.scaler_X_ = MinMaxScaler()
        X_scaled = self.scaler_X_.fit_transform(X_all)

        self.scaler_y_ = MinMaxScaler()
        y_flat = y_all.flatten().reshape(-1, 1)
        self.scaler_y_.fit(y_flat)
        y_scaled = self.scaler_y_.transform(y_flat).reshape(y_all.shape)

        # For CNN, we want (samples, features, 1)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        self.n_features_ = X_reshaped.shape[1]

        # 4) Time-based split for validation (10%)
        val_split_idx = int(len(X_reshaped) * 0.9)
        X_train = X_reshaped[:val_split_idx]
        y_train = y_all[:val_split_idx]
        X_val = X_reshaped[val_split_idx:]
        y_val = y_all[val_split_idx:]

        y_train_scaled = y_scaled[:val_split_idx]
        y_val_scaled = y_scaled[val_split_idx:]

        # Classification target: zero vs. nonzero
        y_train_binary = (y_train > 0).astype(float)  # shape: (samples, horizon)
        y_val_binary = (y_val > 0).astype(float)

        # --------------------------
        # Build Classification Model
        # --------------------------
        self.classifier_ = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu',
                   input_shape=(self.n_features_, 1), padding='same'),
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Dropout(self.dropout_classification),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(self.forecast_horizon, activation='sigmoid')  # multi-step probabilities
        ])

        self.classifier_.compile(
            optimizer=Adam(learning_rate=self.classification_lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        early_stop = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)

        self.classifier_.fit(
            X_train, y_train_binary,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val_binary),
            verbose=1,
            callbacks=[early_stop]
        )

        # -----------------------
        # Build Regression Model
        # Train only on samples that have at least one nonzero step in the horizon
        # OR you can filter for sum > 0, or for any > 0, etc.
        # We'll use sum > 0 here.
        # -----------------------
        nonzero_mask_train = (y_train.sum(axis=1) > 0)
        nonzero_mask_val = (y_val.sum(axis=1) > 0)

        X_train_reg = X_train[nonzero_mask_train]
        y_train_reg = y_train_scaled[nonzero_mask_train]

        X_val_reg = X_val[nonzero_mask_val]
        y_val_reg = y_val_scaled[nonzero_mask_val]

        self.regressor_ = Sequential([
            Conv1D(filters=32, kernel_size=3, activation='relu',
                   input_shape=(self.n_features_, 1), padding='same'),
            Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Dropout(self.dropout_regression),
            Flatten(),
            Dense(46, activation='relu'),
            Dense(self.forecast_horizon)  # multi-step quantity
        ])

        self.regressor_.compile(
            optimizer=Adam(learning_rate=self.regression_lr),
            loss='mean_squared_error'
        )

        self.regressor_.fit(
            X_train_reg, y_train_reg,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val_reg, y_val_reg),
            verbose=1,
            callbacks=[early_stop]
        )

        return self

    def predict(self, df):
        """
        Predict the next self.forecast_horizon steps from the *last* row of the input DataFrame.

        We'll:
          1) Generate lag features for df.
          2) Take the final row (post-lag) as input.
          3) Predict classification (zero vs. nonzero) for each horizon step.
          4) Predict regression quantity, but only if classification > threshold.

        Parameters
        ----------
        df : pd.DataFrame
            The time-series DataFrame (sorted by time). Must have the same columns as in fit().

        Returns
        -------
        forecast : np.ndarray of shape (forecast_horizon,)
            The integer predictions for each step in the horizon.
        """
        # Build lag features
        # We'll assume the same target column name is used as in fit. 
        # You could store target_col_ in self if you want.
        # For simplicity, we do:
        target_col = "target"  # Or store the name in self during fit.

        # If you used the same approach as in fit, you must ensure you do it consistently.
        # We'll assume the user knows to pass a DF with the same column name used in training.
        # (If you want, store that in self at the end of fit: self.target_col_ = target_col.)

        # This is just a demonstration. Real usage might differ.
        if not hasattr(self, 'target_col_'):
            # The user might store it in self in fit. 
            # For demonstration, let's assume it's "target"
            target_col = "target"
        else:
            target_col = self.target_col_

        df_lagged = _generate_lag_features(df, target_col, n_lags=self.n_steps_lag)
        df_lagged.dropna(inplace=True)

        # Take the final row to forecast from
        last_idx = df_lagged.index[-1]
        lag_vals = df_lagged.loc[last_idx, [f"{target_col}_Lag{j}" for j in range(1, self.n_steps_lag + 1)]].values
        
        if self.external_features:
            ext_vals = df_lagged.loc[last_idx, self.external_features].values
        else:
            ext_vals = []

        x_input = np.concatenate([ext_vals, lag_vals]).reshape(1, -1)
        x_input_scaled = self.scaler_X_.transform(x_input)
        x_input_reshaped = x_input_scaled.reshape((1, x_input_scaled.shape[1], 1))

        # Classification (probabilities for each step)
        order_prob = self.classifier_.predict(x_input_reshaped, verbose=0)[0]  # shape: (forecast_horizon,)

        # Regression (quantity for each step)
        quantity_pred_scaled = self.regressor_.predict(x_input_reshaped, verbose=0)[0]  # shape: (forecast_horizon,)
        quantity_pred = self.scaler_y_.inverse_transform(quantity_pred_scaled.reshape(-1, 1)).flatten()

        # Combine
        final_preds = []
        for prob, qty in zip(order_prob, quantity_pred):
            pred = qty if prob > self.threshold else 0
            final_preds.append(max(0, round(pred)))

        return np.array(final_preds)

    def get_params(self, deep=True):
        """
        For sklearn compatibility: returns the hyperparameters as a dict.
        """
        return {
            'n_steps_lag': self.n_steps_lag,
            'forecast_horizon': self.forecast_horizon,
            'external_features': self.external_features,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'threshold': self.threshold,
            'patience': self.patience,
            'random_seed': self.random_seed,
            'classification_lr': self.classification_lr,
            'regression_lr': self.regression_lr,
            'dropout_classification': self.dropout_classification,
            'dropout_regression': self.dropout_regression
        }

    def set_params(self, **params):
        """
        For sklearn compatibility: sets hyperparameters from a dict.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ---------------
# Example usage:
# ---------------
if __name__ == "__main__":
    # Simple synthetic example
    # Let's say we have a small time series with columns: date, target, plus optional features
    df_example = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=50, freq='W'),
        'target': np.random.randint(0, 20, 50),
        'feat1': np.random.randn(50) * 10,
        'feat2': np.random.randn(50) * 5
    }).sort_values('date').reset_index(drop=True)

    # Build and fit UR2CUTE
    # We'll pretend 'feat1' and 'feat2' are external_features
    model = UR2CUTE(
        n_steps_lag=3,
        forecast_horizon=4,
        external_features=['feat1', 'feat2'],
        epochs=5,  # just to keep it quick
        batch_size=8,
        threshold=0.6
    )
    # Fit on the entire data (here we call the target column "target")
    model.fit(df_example, target_col='target')

    # Predict next 4 steps from the final row
    preds = model.predict(df_example)
    print("Predicted horizon:", preds)
