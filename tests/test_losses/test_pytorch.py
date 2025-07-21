import matplotlib.pyplot as plt
import torch
from fastcore.test import test_eq as _test_eq

from neuralforecast.losses.pytorch import (
    GMM,
    MAE,
    NBMM,
    PMM,
    DistributionLoss,
    HuberIQLoss,
    IQLoss,
    MQLoss,
)


# Unit tests to check MQLoss' stored quantiles
# attribute is correctly instantiated
def test_MQLoss_level():
    check = MQLoss(level=[80, 90])
    _test_eq(len(check.quantiles), 5)

def test_MQLoss_quantiles():
    check = MQLoss(quantiles=[0.0100, 0.1000, 0.5, 0.9000, 0.9900])
    print(check.output_names)
    print(check.quantiles)
    _test_eq(len(check.quantiles), 5)

    check = MQLoss(quantiles=[0.0100, 0.1000, 0.9000, 0.9900])
    _test_eq(len(check.quantiles), 4)


# Unit tests
# Check that default quantile is set to 0.5 at initialization
def test_IQLoss_default_quantile():
    check = IQLoss()
    _test_eq(check.q, 0.5)

# Check that quantiles are correctly updated - prediction
def test_IQLoss_update_quantile():
    check = IQLoss()
    check.update_quantile([0.7])
    _test_eq(check.q, 0.7)


# Unit tests to check DistributionLoss' stored quantiles
# attribute is correctly instantiated
def test_DistributionLoss_level():
    check = DistributionLoss(distribution="Normal", level=[80, 90])
    _test_eq(len(check.quantiles), 5)

def test_DistributionLoss_quantiles():
    check = DistributionLoss(
        distribution="Normal", quantiles=[0.0100, 0.1000, 0.5, 0.9000, 0.9900]
    )
    print(check.output_names)
    print(check.quantiles)
    _test_eq(len(check.quantiles), 5)

    check = DistributionLoss(
        distribution="Normal", quantiles=[0.0100, 0.1000, 0.9000, 0.9900]
    )
    _test_eq(len(check.quantiles), 4)

# Unit tests to check DistributionLoss' horizon weight
def test_DistributionLoss_horizon_weight():
    batch_size, horizon, n_outputs = 10, 3, 2
    y_hat = torch.rand(batch_size, horizon, n_outputs).chunk(2, dim=-1)
    y = torch.rand(batch_size, horizon, 1)
    y_loc = torch.rand(batch_size, 1, 1)
    y_scale = torch.rand(batch_size, 1, 1)

    loss = DistributionLoss(distribution="Normal", level=[80, 90])
    loss_with_hweights = DistributionLoss(
        distribution="Normal", level=[80, 90], horizon_weight=torch.ones(horizon)
    )

    distr_args = loss.scale_decouple(y_hat, y_loc, y_scale)
    distr_args_weighted = loss_with_hweights.scale_decouple(y_hat, y_loc, y_scale)

    _test_eq(loss(y, distr_args), loss_with_hweights(y, distr_args_weighted))


# Unit tests to check PMM's stored quantiles
# attribute is correctly instantiated
def test_PMM_level():
    check = PMM(n_components=2, level=[80, 90])
    _test_eq(len(check.quantiles), 5)

def test_PMM_quantiles():
    check = PMM(n_components=2, quantiles=[0.0100, 0.1000, 0.5, 0.9000, 0.9900])
    print(check.output_names)
    print(check.quantiles)
    _test_eq(len(check.quantiles), 5)

    check = PMM(n_components=2, quantiles=[0.0100, 0.1000, 0.9000, 0.9900])
    _test_eq(len(check.quantiles), 4)

# Unit tests
# Check that default quantile is set to 0.5 at initialization
def test_HuberIQLoss_init():
    check = HuberIQLoss()
    _test_eq(check.q, 0.5)

# Check that quantiles are correctly updated - prediction
def test_HuberIQLoss_update_quantile():
    check = HuberIQLoss()
    check.update_quantile([0.7])
    _test_eq(check.q, 0.7)


# Each 1 is an error, there are 6 datapoints.
def test_MAE_complete_mask():
    y = torch.Tensor([[0, 0, 0], [0, 0, 0]]).unsqueeze(-1)
    y_hat = torch.Tensor([[0, 0, 1], [1, 0, 1]]).unsqueeze(-1)

    # Complete mask and horizon_weight
    mask = torch.Tensor([[1, 1, 1], [1, 1, 1]]).unsqueeze(-1)
    horizon_weight = torch.Tensor([1, 1, 1])

    mae = MAE(horizon_weight=horizon_weight)
    loss = mae(y=y, y_hat=y_hat, mask=mask)
    assert loss == (3 / 6), "Should be 3/6"


# Incomplete mask and complete horizon_weight
def test_MAE_incomplete_mask():
    # Only 1 error and points is masked.
    y = torch.Tensor([[0, 0, 0], [0, 0, 0]]).unsqueeze(-1)
    y_hat = torch.Tensor([[0, 0, 1], [1, 0, 1]]).unsqueeze(-1)

    mask = torch.Tensor([[1, 1, 1], [0, 1, 1]]).unsqueeze(-1)
    horizon_weight = torch.Tensor([1, 1, 1])
    mae = MAE(horizon_weight=horizon_weight)
    loss = mae(y=y, y_hat=y_hat, mask=mask)
    assert loss == (2 / 5), "Should be 2/5"

    # Complete mask and incomplete horizon_weight
    mask = torch.Tensor([[1, 1, 1], [1, 1, 1]]).unsqueeze(-1)
    horizon_weight = torch.Tensor([1, 1, 0])  # 2 errors and points are masked.
    mae = MAE(horizon_weight=horizon_weight)
    loss = mae(y=y, y_hat=y_hat, mask=mask)
    assert loss == (1 / 4), "Should be 1/4"

    # Incomplete mask and incomplete horizon_weight
    mask = torch.Tensor([[0, 1, 1], [1, 1, 1]]).unsqueeze(-1)
    horizon_weight = torch.Tensor([1, 1, 0])  # 2 errors are masked, and 3 points.
    mae = MAE(horizon_weight=horizon_weight)
    loss = mae(y=y, y_hat=y_hat, mask=mask)
    assert loss == (1 / 3), "Should be 1/3"
