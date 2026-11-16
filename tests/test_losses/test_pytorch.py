import warnings

import torch

import pytest

from neuralforecast.losses.pytorch import (
    MAE,
    MAPE,
    MASE,
    MQLoss,
    MSE,
    PMM,
    RMSE,
    SMAPE,
    DistributionLoss,
    FreDF,
    HuberMQLoss,
    HuberIQLoss,
    IQLoss,
    QuantileLoss,
    sCRPS,
    _divide_no_nan,
    _weighted_mean,
    weighted_average,
)


# Unit tests to check MQLoss' stored quantiles
# attribute is correctly instantiated
def test_MQLoss_level():