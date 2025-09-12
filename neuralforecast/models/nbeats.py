


__all__ = ['NBEATS']


import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import BSpline

from ..common._base_model import BaseModel
from ..losses.pytorch import MAE


def generate_legendre_basis(length, n_basis):
    """
    Generates Legendre polynomial basis functions.

    Args:
        length (int): Number of data points.
        n_basis (int): Number of basis functions to generate.

    Returns:
        legendre_basis (ndarray): An array of Legendre basis functions.
    """
    x = np.linspace(-1, 1, length)  # Legendre polynomials are defined on [-1, 1]
    legendre_basis = np.zeros((length, n_basis))
    for i in range(n_basis):
        # Legendre polynomial of degree i
        P_i = Legendre.basis(i)
        legendre_basis[:, i] = P_i(x)
    return legendre_basis


def generate_polynomial_basis(length, n_basis):
    """
    Generates standard polynomial basis functions.

    Args:
        length (int): Number of data points.
        n_basis (int): Number of polynomial functions to generate.

    Returns:
        poly_basis (ndarray): An array of polynomial basis functions.
    """
    return np.concatenate(
        [
            np.power(np.arange(length, dtype=float) / length, i)[None, :]
            for i in range(n_basis)
        ]
    ).T


def generate_changepoint_basis(length, n_basis):
    """
    Generates changepoint basis functions with automatically spaced changepoints.

    Args:
        length (int): Number of data points.
        n_basis (int): Number of changepoint functions to generate.

    Returns:
        changepoint_basis (ndarray): An array of changepoint basis functions.
    """
    x = np.linspace(0, 1, length)[:, None]  # Shape: (length, 1)
    changepoint_locations = np.linspace(0, 1, n_basis + 1)[1:][
        None, :
    ]  # Shape: (1, n_basis)
    return np.maximum(0, x - changepoint_locations)


def generate_piecewise_linear_basis(length, n_basis):
    """
    Generates piecewise linear basis functions (linear splines).

    Args:
        length (int): Number of data points.
        n_basis (int): Number of piecewise linear basis functions to generate.

    Returns:
        pw_linear_basis (ndarray): An array of piecewise linear basis functions.
    """
    x = np.linspace(0, 1, length)
    knots = np.linspace(0, 1, n_basis + 1)
    pw_linear_basis = np.zeros((length, n_basis))
    for i in range(1, n_basis):
        pw_linear_basis[:, i] = np.maximum(
            0,
            np.minimum(
                (x - knots[i - 1]) / (knots[i] - knots[i - 1]),
                (knots[i + 1] - x) / (knots[i + 1] - knots[i]),
            ),
        )
    return pw_linear_basis


def generate_linear_hat_basis(length, n_basis):
    x = np.linspace(0, 1, length)[:, None]  # Shape: (length, 1)
    centers = np.linspace(0, 1, n_basis)[None, :]  # Shape: (1, n_basis)
    width = 1.0 / (n_basis - 1)

    # Create triangular functions using piecewise linear equations
    return np.maximum(0, 1 - np.abs(x - centers) / width)


def generate_spline_basis(length, n_basis):
    """
    Generates cubic spline basis functions.

    Args:
        length (int): Number of data points.
        n_basis (int): Number of basis functions.

    Returns:
        spline_basis (ndarray): An array of cubic spline basis functions.
    """
    if n_basis < 4:
        raise ValueError(
            f"To use the spline basis, n_basis must be set to 4 or more. Current value is {n_basis}"
        )
    x = np.linspace(0, 1, length)
    knots = np.linspace(0, 1, n_basis - 2)
    t = np.concatenate(([0, 0, 0], knots, [1, 1, 1]))
    degree = 3
    # Create basis coefficient matrix once
    coefficients = np.eye(n_basis)
    # Create single BSpline object with all coefficients
    spline = BSpline(t, coefficients.T, degree)
    return spline(x)


def generate_chebyshev_basis(length, n_basis):
    """
    Generates Chebyshev polynomial basis functions.

    Args:
        length (int): Number of data points.
        n_basis (int): Number of Chebyshev polynomials to generate.

    Returns:
        chebyshev_basis (ndarray): An array of Chebyshev polynomial basis functions.
    """
    x = np.linspace(-1, 1, length)
    chebyshev_basis = np.zeros((length, n_basis))
    for i in range(n_basis):
        T_i = Chebyshev.basis(i)
        chebyshev_basis[:, i] = T_i(x)
    return chebyshev_basis


def get_basis(length, n_basis, basis):
    basis_dict = {
        "legendre": generate_legendre_basis,
        "polynomial": generate_polynomial_basis,
        "changepoint": generate_changepoint_basis,
        "piecewise_linear": generate_piecewise_linear_basis,
        "linear_hat": generate_linear_hat_basis,
        "spline": generate_spline_basis,
        "chebyshev": generate_chebyshev_basis,
    }
    return basis_dict[basis](length, n_basis + 1)


class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, out_features: int = 1):
        super().__init__()
        self.out_features = out_features
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        backcast = theta[:, : self.backcast_size]
        forecast = theta[:, self.backcast_size :]
        forecast = forecast.reshape(len(forecast), -1, self.out_features)
        return backcast, forecast


class TrendBasis(nn.Module):
    def __init__(
        self,
        n_basis: int,
        backcast_size: int,
        forecast_size: int,
        out_features: int = 1,
        basis="polynomial",
    ):
        super().__init__()
        self.out_features = out_features
        self.backcast_basis = nn.Parameter(
            torch.tensor(
                get_basis(backcast_size, n_basis, basis).T, dtype=torch.float32
            ),
            requires_grad=False,
        )
        self.forecast_basis = nn.Parameter(
            torch.tensor(
                get_basis(forecast_size, n_basis, basis).T, dtype=torch.float32
            ),
            requires_grad=False,
        )

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        polynomial_size = self.forecast_basis.shape[0]  # [polynomial_size, L+H]
        backcast_theta = theta[:, :polynomial_size]
        forecast_theta = theta[:, polynomial_size:]
        forecast_theta = forecast_theta.reshape(
            len(forecast_theta), polynomial_size, -1
        )
        backcast = torch.einsum("bp,pt->bt", backcast_theta, self.backcast_basis)
        forecast = torch.einsum("bpq,pt->btq", forecast_theta, self.forecast_basis)
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    def __init__(
        self,
        harmonics: int,
        backcast_size: int,
        forecast_size: int,
        out_features: int = 1,
    ):
        super().__init__()
        self.out_features = out_features
        frequency = np.append(
            np.zeros(1, dtype=float),
            np.arange(harmonics, harmonics / 2 * forecast_size, dtype=float)
            / harmonics,
        )[None, :]
        backcast_grid = (
            -2
            * np.pi
            * (np.arange(backcast_size, dtype=float)[:, None] / forecast_size)
            * frequency
        )
        forecast_grid = (
            2
            * np.pi
            * (np.arange(forecast_size, dtype=float)[:, None] / forecast_size)
            * frequency
        )

        backcast_cos_template = torch.tensor(
            np.transpose(np.cos(backcast_grid)), dtype=torch.float32
        )
        backcast_sin_template = torch.tensor(
            np.transpose(np.sin(backcast_grid)), dtype=torch.float32
        )
        backcast_template = torch.cat(
            [backcast_cos_template, backcast_sin_template], dim=0
        )

        forecast_cos_template = torch.tensor(
            np.transpose(np.cos(forecast_grid)), dtype=torch.float32
        )
        forecast_sin_template = torch.tensor(
            np.transpose(np.sin(forecast_grid)), dtype=torch.float32
        )
        forecast_template = torch.cat(
            [forecast_cos_template, forecast_sin_template], dim=0
        )

        self.backcast_basis = nn.Parameter(backcast_template, requires_grad=False)
        self.forecast_basis = nn.Parameter(forecast_template, requires_grad=False)

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        harmonic_size = self.forecast_basis.shape[0]  # [harmonic_size, L+H]
        backcast_theta = theta[:, :harmonic_size]
        forecast_theta = theta[:, harmonic_size:]
        forecast_theta = forecast_theta.reshape(len(forecast_theta), harmonic_size, -1)
        backcast = torch.einsum("bp,pt->bt", backcast_theta, self.backcast_basis)
        forecast = torch.einsum("bpq,pt->btq", forecast_theta, self.forecast_basis)
        return backcast, forecast


ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]


class NBEATSBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """

    def __init__(
        self,
        input_size: int,
        n_theta: int,
        mlp_units: list,
        basis: nn.Module,
        dropout_prob: float,
        activation: str,
    ):
        super().__init__()

        self.dropout_prob = dropout_prob

        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        activ = getattr(nn, activation)()

        hidden_layers = [
            nn.Linear(in_features=input_size, out_features=mlp_units[0][0])
        ]
        for layer in mlp_units:
            hidden_layers.append(nn.Linear(in_features=layer[0], out_features=layer[1]))
            hidden_layers.append(activ)

            if self.dropout_prob > 0:
                raise NotImplementedError("dropout")

        output_layer = [nn.Linear(in_features=mlp_units[-1][1], out_features=n_theta)]
        layers = hidden_layers + output_layer
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, insample_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta)
        return backcast, forecast


class NBEATS(BaseModel):
    """NBEATS

    The Neural Basis Expansion Analysis for Time Series (NBEATS), is a simple and yet
    effective architecture, it is built with a deep stack of MLPs with the doubly
    residual connections. It has a generic and interpretable architecture depending
    on the blocks it uses. Its interpretable architecture is recommended for scarce
    data settings, as it regularizes its predictions through projections unto harmonic
    and trend basis well-suited for most forecasting tasks.

    **Parameters:**<br>
    `h`: int, forecast horizon.<br>
    `input_size`: int, considered autorregresive inputs (lags), y=[1,2,3,4] input_size=2 -> lags=[1,2].<br>
    `n_harmonics`: int, Number of harmonic terms for seasonality stack type. Note that len(n_harmonics) = len(stack_types). Note that it will only be used if a seasonality stack is used.<br>
    `n_polynomials`: int, DEPRECATED - polynomial degree for trend stack. Note that len(n_polynomials) = len(stack_types). Note that it will only be used if a trend stack is used.<br>
    `basis`: str, Type of basis function to use in the trend stack. Choose one from ['legendre', 'polynomial', 'changepoint', 'piecewise_linear', 'linear_hat', 'spline', 'chebyshev']<br>
    `n_basis`: int, the degree of the basis function for the trend stack. Note that it will only be used if a trend stack is used.<br>
    `stack_types`: List[str], List of stack types. Subset from ['seasonality', 'trend', 'identity'].<br>
    `n_blocks`: List[int], Number of blocks for each stack. Note that len(n_blocks) = len(stack_types).<br>
    `mlp_units`: List[List[int]], Structure of hidden layers for each stack type. Each internal list should contain the number of units of each hidden layer. Note that len(n_hidden) = len(stack_types).<br>
    `dropout_prob_theta`: float, Float between (0, 1). Dropout for N-BEATS basis.<br>
    `activation`: str, activation from ['ReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid'].<br>
    `shared_weights`: bool, If True, all blocks within each stack will share parameters. <br>
    `loss`: PyTorch module, instantiated train loss class from [losses collection](./losses.pytorch).<br>
    `valid_loss`: PyTorch module=`loss`, instantiated valid loss class from [losses collection](./losses.pytorch).<br>
    `max_steps`: int=1000, maximum number of training steps.<br>
    `learning_rate`: float=1e-3, Learning rate between (0, 1).<br>
    `num_lr_decays`: int=3, Number of learning rate decays, evenly distributed across max_steps.<br>
    `early_stop_patience_steps`: int=-1, Number of validation iterations before early stopping.<br>
    `val_check_steps`: int=100, Number of training steps between every validation loss check.<br>
    `batch_size`: int=32, number of different series in each batch.<br>
    `valid_batch_size`: int=None, number of different series in each validation and test batch, if None uses batch_size.<br>
    `windows_batch_size`: int=1024, number of windows to sample in each training batch, default uses all.<br>
    `inference_windows_batch_size`: int=-1, number of windows to sample in each inference batch, -1 uses all.<br>
    `start_padding_enabled`: bool=False, if True, the model will pad the time series with zeros at the beginning, by input size.<br>
    `training_data_availability_threshold`: Union[float, List[float]]=0.0, minimum fraction of valid data points required for training windows. Single float applies to both insample and outsample; list of two floats specifies [insample_fraction, outsample_fraction]. Default 0.0 allows windows with only 1 valid data point (current behavior).<br>
    `step_size`: int=1, step size between each window of temporal data.<br>
    `scaler_type`: str='identity', type of scaler for temporal inputs normalization see [temporal scalers](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/common/_scalers.py).<br>
    `random_seed`: int, random_seed for pytorch initializer and numpy generators.<br>
    `drop_last_loader`: bool=False, if True `TimeSeriesDataLoader` drops last non-full batch.<br>
    `alias`: str, optional,  Custom name of the model.<br>
    `optimizer`: Subclass of 'torch.optim.Optimizer', optional, user specified optimizer instead of the default choice (Adam).<br>
    `optimizer_kwargs`: dict, optional, list of parameters used by the user specified `optimizer`.<br>
    `lr_scheduler`: Subclass of 'torch.optim.lr_scheduler.LRScheduler', optional, user specified lr_scheduler instead of the default choice (StepLR).<br>
    `lr_scheduler_kwargs`: dict, optional, list of parameters used by the user specified `lr_scheduler`.<br>
    `dataloader_kwargs`: dict, optional, list of parameters passed into the PyTorch Lightning dataloader by the `TimeSeriesDataLoader`. <br>
    `**trainer_kwargs`: int,  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).<br>

    **References:**<br>
    -[Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados, Yoshua Bengio (2019).
    "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting".](https://arxiv.org/abs/1905.10437)
    """

    # Class attributes
    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = False  # If the model produces multivariate forecasts (True) or univariate (False)
    RECURRENT = (
        False  # If the model produces forecasts recursively (True) or direct (False)
    )

    def __init__(
        self,
        h,
        input_size,
        n_harmonics: int = 2,
        n_polynomials: Optional[int] = None,
        n_basis: int = 2,
        basis: str = "polynomial",
        stack_types: list = ["identity", "trend", "seasonality"],
        n_blocks: list = [1, 1, 1],
        mlp_units: list = 3 * [[512, 512]],
        dropout_prob_theta: float = 0.0,
        activation: str = "ReLU",
        shared_weights: bool = False,
        loss=MAE(),
        valid_loss=None,
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
        drop_last_loader: bool = False,
        alias: Optional[str] = None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs,
    ):

        # Protect horizon collapsed seasonality and trend NBEATSx-i basis
        if h == 1 and (("seasonality" in stack_types) or ("trend" in stack_types)):
            raise Exception(
                "Horizon `h=1` incompatible with `seasonality` or `trend` in stacks"
            )

        # Inherit BaseWindows class
        super(NBEATS, self).__init__(
            h=h,
            input_size=input_size,
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
            training_data_availability_threshold=training_data_availability_threshold,
            step_size=step_size,
            scaler_type=scaler_type,
            drop_last_loader=drop_last_loader,
            alias=alias,
            random_seed=random_seed,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs,
        )

        # Raise deprecation warning
        if n_polynomials is not None:
            warnings.warn(
                "The parameter n_polynomials will be deprecated in favor of n_basis and basis and it is currently ignored.\n"
                "The basis parameter defines the basis function to be used in the trend stack.\n"
                "The n_basis defines the degree of the basis function used in the trend stack.",
                DeprecationWarning,
            )

        # Architecture
        blocks = self.create_stack(
            h=h,
            input_size=input_size,
            stack_types=stack_types,
            n_blocks=n_blocks,
            mlp_units=mlp_units,
            dropout_prob_theta=dropout_prob_theta,
            activation=activation,
            shared_weights=shared_weights,
            n_harmonics=n_harmonics,
            n_basis=n_basis,
            basis_type=basis,
        )
        self.blocks = torch.nn.ModuleList(blocks)

    def create_stack(
        self,
        stack_types,
        n_blocks,
        input_size,
        h,
        mlp_units,
        dropout_prob_theta,
        activation,
        shared_weights,
        n_harmonics,
        n_basis,
        basis_type,
    ):

        block_list = []
        for i in range(len(stack_types)):
            for block_id in range(n_blocks[i]):

                # Shared weights
                if shared_weights and block_id > 0:
                    nbeats_block = block_list[-1]
                else:
                    if stack_types[i] == "seasonality":
                        n_theta = (
                            2
                            * (self.loss.outputsize_multiplier + 1)
                            * int(np.ceil(n_harmonics / 2 * h) - (n_harmonics - 1))
                        )
                        basis = SeasonalityBasis(
                            harmonics=n_harmonics,
                            backcast_size=input_size,
                            forecast_size=h,
                            out_features=self.loss.outputsize_multiplier,
                        )

                    elif stack_types[i] == "trend":
                        n_theta = (self.loss.outputsize_multiplier + 1) * (n_basis + 1)
                        basis = TrendBasis(
                            n_basis=n_basis,
                            backcast_size=input_size,
                            forecast_size=h,
                            out_features=self.loss.outputsize_multiplier,
                            basis=basis_type,
                        )

                    elif stack_types[i] == "identity":
                        n_theta = input_size + self.loss.outputsize_multiplier * h
                        basis = IdentityBasis(
                            backcast_size=input_size,
                            forecast_size=h,
                            out_features=self.loss.outputsize_multiplier,
                        )
                    else:
                        raise ValueError(f"Block type {stack_types[i]} not found!")

                    nbeats_block = NBEATSBlock(
                        input_size=input_size,
                        n_theta=n_theta,
                        mlp_units=mlp_units,
                        basis=basis,
                        dropout_prob=dropout_prob_theta,
                        activation=activation,
                    )

                # Select type of evaluation and apply it to all layers of block
                block_list.append(nbeats_block)

        return block_list

    def forward(self, windows_batch):

        # Parse windows_batch
        insample_y = windows_batch["insample_y"].squeeze(-1)
        insample_mask = windows_batch["insample_mask"].squeeze(-1)

        # NBEATS' forward
        residuals = insample_y.flip(dims=(-1,))  # backcast init
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:, None]  # Level with Naive1
        block_forecasts = [forecast.repeat(1, self.h, 1)]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast

            if self.decompose_forecast:
                block_forecasts.append(block_forecast)

        if self.decompose_forecast:
            # (n_batch, n_blocks, h, out_features)
            block_forecasts = torch.stack(block_forecasts)
            block_forecasts = block_forecasts.permute(1, 0, 2, 3)
            block_forecasts = block_forecasts.squeeze(-1)  # univariate output
            return block_forecasts
        else:
            return forecast
