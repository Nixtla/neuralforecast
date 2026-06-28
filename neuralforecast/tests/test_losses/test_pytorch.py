import warnings

import torch

from neuralforecast.losses.pytorch import (
    MAE,
    PMM,
    DistributionLoss,
    HuberIQLoss,
    IQLoss,
    MQLoss,
    FreDF,
)


# Unit tests to check MQLoss' stored quantiles
# attribute is correctly instantiated
def test_MQLoss_level():
    check = MQLoss(level=[80, 90])
    assert len(check.quantiles) == 5

def test_MQLoss_quantiles():
    check = MQLoss(quantiles=[0.0100, 0.1000, 0.5, 0.9000, 0.9900])
    assert len(check.quantiles) == 5

    check = MQLoss(quantiles=[0.0100, 0.1000, 0.9000, 0.9900])
    assert len(check.quantiles) == 4


# Unit tests
# Check that default quantile is set to 0.5 at initialization
def test_IQLoss_default_and_update_quantile():
    check = IQLoss()
    assert check.q == 0.5

    check.update_quantile([0.7])
    assert check.q == 0.7


# Unit tests to check DistributionLoss' stored quantiles
# attribute is correctly instantiated
def test_DistributionLoss_level():
    check = DistributionLoss(distribution="Normal", level=[80, 90])
    assert len(check.quantiles), 5

def test_DistributionLoss_quantiles():
    check = DistributionLoss(
        distribution="Normal", quantiles=[0.0100, 0.1000, 0.5, 0.9000, 0.9900]
    )
    assert len(check.quantiles) == 5

    check = DistributionLoss(
        distribution="Normal", quantiles=[0.0100, 0.1000, 0.9000, 0.9900]
    )
    assert len(check.quantiles) == 4

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

    assert loss(y, distr_args) == loss_with_hweights(y, distr_args_weighted)


# Unit tests to check PMM's stored quantiles
# attribute is correctly instantiated
def test_PMM_level():
    check = PMM(n_components=2, level=[80, 90])
    assert len(check.quantiles) == 5

def test_PMM_quantiles():
    check = PMM(n_components=2, quantiles=[0.0100, 0.1000, 0.5, 0.9000, 0.9900])
    assert len(check.quantiles) == 5

    check = PMM(n_components=2, quantiles=[0.0100, 0.1000, 0.9000, 0.9900])
    assert len(check.quantiles) == 4

# Unit tests
# Check that default quantile is set to 0.5 at initialization
def test_HuberIQLoss_init_and_update():
    check = HuberIQLoss()
    assert check.q == 0.5

    check.update_quantile([0.7])
    assert check.q == 0.7


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


def test_duplicate_level_and_quantiles_dedup():
    # Duplicate levels should be deduplicated with a warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check = MQLoss(level=[80, 80])
        assert len(w) == 1
        assert "Duplicate levels" in str(w[0].message)
    # [80] produces lo-80 and hi-80, plus median -> 3 quantiles
    assert len(check.quantiles) == 3

    # Duplicate quantiles should be deduplicated with a warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check = MQLoss(quantiles=[0.1, 0.1, 0.5, 0.9])
        assert len(w) == 1
        assert "Duplicate quantiles" in str(w[0].message)
    assert len(check.quantiles) == 3

    # No duplicates should produce no warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check = MQLoss(level=[80, 90])
        assert len(w) == 0
    assert len(check.quantiles) == 5


def test_fredf_alpha0_equals_mse():
    """When alpha=0, FreDF should reduce to pure time-domain MSE."""
    loss_fn = FreDF(alpha=0.0)

    y     = torch.tensor([[[1.0], [2.0], [3.0]]])  # [1, 3, 1]
    y_hat = torch.tensor([[[2.0], [2.0], [2.0]]])

    loss = loss_fn(y, y_hat)

    # MSE = mean((1-2)^2, (2-2)^2, (3-2)^2) = mean(1, 0, 1) = 2/3
    expected = torch.tensor(2.0 / 3.0)
    assert torch.isclose(loss, expected, atol=1e-5), \
        f"Expected {expected.item():.6f}, got {loss.item():.6f}"


def test_fredf_alpha1_equals_freq_mae():
    """When alpha=1, FreDF should reduce to pure frequency-domain MAE."""
    loss_fn = FreDF(alpha=1.0)

    y     = torch.tensor([[[1.0], [2.0], [3.0]]])  # [1, 3, 1]
    y_hat = torch.tensor([[[2.0], [2.0], [2.0]]])

    loss = loss_fn(y, y_hat)

    # rfft(y)     = [6+0j, -1.5+0.866j]
    # rfft(y_hat) = [6+0j, 0+0j]
    # |diff|      = [0.0, 1.732]
    # MAE         = mean(0.0, 1.732) = 0.866
    expected = torch.tensor(0.866025)
    assert torch.isclose(loss, expected, atol=1e-4), \
        f"Expected {expected.item():.6f}, got {loss.item():.6f}"


def test_fredf_perfect_forecast_zero_loss():
    """Perfect forecast should give zero loss for any alpha."""
    y = torch.tensor([[[1.0], [2.0], [3.0]]])

    for alpha in [0.0, 0.5, 1.0]:
        loss_fn = FreDF(alpha=alpha)
        loss = loss_fn(y, y)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), \
            f"alpha={alpha}: expected 0.0, got {loss.item():.6f}"
