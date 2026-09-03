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
    HuberIQLoss,
    IQLoss,
    QuantileLoss,
    _divide_no_nan,
    _weighted_mean,
    weighted_average,
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


# ---------------------------------------------------------------------------
# Non-finite losses must survive the reduction (issue #1601).
#
# `_divide_no_nan` used to sanitize the quotient rather than guard the
# denominator, so any inf/nan loss reduced to 0.0 -- the lowest possible value.
# A diverged trial then won every Auto search it took part in.
# ---------------------------------------------------------------------------


def test_divide_no_nan_guards_denominator_only():
    a = torch.tensor([1.0, 1.0, float("inf"), float("-inf"), float("nan")])
    b = torch.tensor([0.0, 2.0, 2.0, 2.0, 2.0])

    out = _divide_no_nan(a, b)

    assert out[0] == 0.0, "division by zero should yield 0"
    assert out[1] == 0.5
    assert torch.isposinf(out[2]), "an infinite numerator must be preserved"
    assert torch.isneginf(out[3]), "a -inf numerator must be preserved"
    assert torch.isnan(out[4]), "a nan numerator must be preserved"


def test_divide_no_nan_no_nan_gradient_on_zero_denominator():
    a = torch.tensor([1.0, 1.0], requires_grad=True)
    b = torch.tensor([0.0, 2.0])

    _divide_no_nan(a, b).sum().backward()

    assert torch.isfinite(a.grad).all(), "zero denominator must not poison backward"
    assert a.grad.tolist() == [0.0, 0.5]


@pytest.mark.parametrize(
    "loss_cls", [MAE, MSE, RMSE, MAPE, lambda: QuantileLoss(q=0.5)]
)
@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf"), float("nan")])
def test_losses_do_not_hide_diverged_predictions(loss_cls, bad_value):
    y = torch.tensor([[[10.0], [20.0]]])
    y_hat = torch.full_like(y, bad_value)

    loss = loss_cls()(y=y, y_hat=y_hat)

    assert not torch.isfinite(loss), f"{loss_cls} reported {loss.item()}"


def test_masked_out_infinity_does_not_leak():
    # The masked-out point is +inf; inf * 0 = nan would contaminate the whole
    # reduction, so zero-weight entries have to be dropped before multiplying.
    y = torch.tensor([[[10.0], [20.0]]])
    y_hat = torch.tensor([[[11.0], [float("inf")]]])
    mask = torch.tensor([[[1.0], [0.0]]])

    loss = MAE()(y=y, y_hat=y_hat, mask=mask)

    assert loss == 1.0, "only the unmasked point should count"


def test_fully_masked_batch_is_zero():
    y = torch.tensor([[[10.0], [20.0]]])
    y_hat = torch.tensor([[[11.0], [21.0]]])

    loss = MAE()(y=y, y_hat=y_hat, mask=torch.zeros_like(y))

    assert loss == 0.0, "no weighted points means no loss, not a division by zero"


def test_weighted_mean_normalizes_by_total_weight():
    losses = torch.tensor([1.0, 3.0])
    weights = torch.tensor([0.25, 0.25])

    # Total weight is 0.5, so the mean is 2.0 -- not 1.0, which is what
    # clamping the denominator to a minimum of 1 would give.
    assert _weighted_mean(losses=losses, weights=weights) == 2.0


def test_weighted_average_normalizes_by_total_weight():
    x = torch.tensor([1.0, 3.0])
    weights = torch.tensor([0.25, 0.25])

    assert weighted_average(x, weights=weights) == 2.0


def test_weighted_average_honors_dim_zero():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weights = torch.ones_like(x)

    # dim=0 is falsy; a truthiness check silently reduced over everything.
    assert weighted_average(x, weights=weights, dim=0).tolist() == [2.0, 3.0]


def test_weighted_average_zero_weights_is_zero():
    x = torch.tensor([float("nan"), float("inf")])

    assert weighted_average(x, weights=torch.zeros(2)) == 0.0


# ---------------------------------------------------------------------------
# ...while the deliberate divide-by-zero skips stay intact. These guard the
# regressions that a naive fix to `_divide_no_nan` introduces.
# ---------------------------------------------------------------------------


def test_mape_skips_zero_targets():
    # scale = 1 / |y|, so y == 0 contributes nothing rather than inf.
    y = torch.tensor([[[0.0], [20.0]]])
    y_hat = torch.tensor([[[1.0], [22.0]]])

    loss = MAPE()(y=y, y_hat=y_hat)

    assert torch.isclose(loss, torch.tensor(0.05)), "a zero in y must not give inf"


def test_smape_handles_zero_targets():
    y = torch.tensor([[[0.0], [20.0]]])
    y_hat = torch.tensor([[[1.0], [22.0]]])

    loss = SMAPE()(y=y, y_hat=y_hat)

    assert torch.isfinite(loss)
    assert torch.isclose(loss, torch.tensor(1.047619), atol=1e-5)


def test_mase_skips_zero_scale():
    # A constant insample series has seasonal scale 0.
    y = torch.tensor([[[10.0], [20.0]]])
    y_hat = torch.tensor([[[11.0], [21.0]]])
    y_insample = torch.tensor([[5.0, 5.0, 5.0]])

    loss = MASE(seasonality=1)(y=y, y_hat=y_hat, y_insample=y_insample)

    assert loss == 0.0, "a flat insample series must not give inf"


def test_mape_gradient_is_finite_with_zero_targets():
    y = torch.tensor([[[0.0], [20.0]]])
    y_hat = torch.tensor([[[1.0], [22.0]]], requires_grad=True)

    MAPE()(y=y, y_hat=y_hat).backward()

    assert torch.isfinite(y_hat.grad).all()
    assert y_hat.grad.flatten()[0] == 0.0, "the zero-target point gets no gradient"
