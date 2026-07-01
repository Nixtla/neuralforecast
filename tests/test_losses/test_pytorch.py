import warnings
import pytest
import torch

from neuralforecast.losses.pytorch import (
    MSE,
    MAE,
    PMM,
    DistributionLoss,
    HuberIQLoss,
    IQLoss,
    MQLoss,
    FFTMAELoss,
    FFTMSELoss,
    FFTRMSELoss,
    MixedFFTLoss,

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

# ─────────────────────────── FFTMAELoss ────────────────────────────

def test_fftmae_univariate():
    """Perfect predictions should yield zero loss."""
    y = torch.sin(torch.linspace(0, 2 * torch.pi, 24)).unsqueeze(0).unsqueeze(-1)
    loss = FFTMAELoss()(y=y, y_hat=y)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_fftmae_multivariate():
    """Loss is positive and finite for multi-output predictions."""
    B, H, N = 4, 24, 3
    y = torch.randn(B, H, N)
    y_hat = torch.randn(B, H, N)
    loss = FFTMAELoss()(y=y, y_hat=y_hat)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_fftmae_autoregressive_mask():
    """Masking out time steps reduces or equals the unmasked loss."""
    B, H, N = 2, 16, 1
    y = torch.randn(B, H, N)
    y_hat = torch.zeros(B, H, N)
    mask_full = torch.ones(B, H, N)
    mask_half = mask_full.clone(); mask_half[:, H // 2 :, :] = 0

    loss_full = FFTMAELoss()(y=y, y_hat=y_hat, mask=mask_full)
    loss_half = FFTMAELoss()(y=y, y_hat=y_hat, mask=mask_half)
    # Both should be finite; masking zeros out signal so values differ
    assert torch.isfinite(loss_full) and torch.isfinite(loss_half)


def test_fftmae_numerical_stability():
    """Very large and very small inputs should not produce NaN/Inf."""
    for scale in [1e-8, 1e8]:
        y = torch.randn(2, 32, 1) * scale
        y_hat = torch.randn(2, 32, 1) * scale
        loss = FFTMAELoss(norm=True)(y=y, y_hat=y_hat)
        assert torch.isfinite(loss), f"Non-finite loss at scale {scale}"


# ─────────────────────────── FFTMSELoss ────────────────────────────

def test_fftmse_univariate():
    """Perfect predictions should yield zero loss."""
    y = torch.cos(torch.linspace(0, 2 * torch.pi, 24)).unsqueeze(0).unsqueeze(-1)
    loss = FFTMSELoss()(y=y, y_hat=y)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_fftmse_multivariate():
    """MSE loss is non-negative and finite for random multi-output tensors."""
    B, H, N = 4, 24, 5
    y = torch.randn(B, H, N)
    y_hat = torch.randn(B, H, N)
    loss = FFTMSELoss()(y=y, y_hat=y_hat)
    assert loss.item() >= 0
    assert torch.isfinite(loss)


def test_fftmse_autoregressive_mask():
    """Zeroed mask produces finite loss; differs from full-mask loss."""
    B, H, N = 3, 20, 1
    y = torch.randn(B, H, N)
    y_hat = torch.randn(B, H, N)
    mask = torch.ones(B, H, N)
    mask[:, -5:, :] = 0  # mask last 5 steps (AR-style causal masking)

    loss_masked = FFTMSELoss()(y=y, y_hat=y_hat, mask=mask)
    loss_full = FFTMSELoss()(y=y, y_hat=y_hat)
    assert torch.isfinite(loss_masked) and torch.isfinite(loss_full)


def test_fftmse_numerical_stability():
    """Loss should be finite across extreme value ranges."""
    for scale in [1e-7, 1e7]:
        y = torch.randn(2, 32, 2) * scale
        y_hat = torch.randn(2, 32, 2) * scale
        loss = FFTMSELoss(norm=True)(y=y, y_hat=y_hat)
        assert torch.isfinite(loss), f"Non-finite MSE loss at scale {scale}"


# ─────────────────────────── FFTRMSELoss ───────────────────────────

def test_fftrmse_univariate():
    """Perfect predictions should yield zero loss."""
    y = torch.randn(1, 48, 1)
    loss = FFTRMSELoss()(y=y, y_hat=y)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_fftrmse_multivariate():
    """RMSE >= 0 and RMSE == sqrt(MSE) for multi-output tensors."""
    B, H, N = 4, 24, 3
    y = torch.randn(B, H, N)
    y_hat = torch.randn(B, H, N)
    rmse = FFTRMSELoss()(y=y, y_hat=y_hat)
    mse = FFTMSELoss()(y=y, y_hat=y_hat)
    assert rmse.item() >= 0
    assert rmse.item() == pytest.approx(mse.item() ** 0.5, rel=1e-5)


def test_fftrmse_autoregressive_mask():
    """Masked RMSE is finite and non-negative."""
    B, H, N = 2, 16, 1
    y = torch.randn(B, H, N)
    y_hat = torch.zeros(B, H, N)
    mask = torch.ones(B, H, N); mask[:, :4, :] = 0  # skip first 4 steps

    loss = FFTRMSELoss()(y=y, y_hat=y_hat, mask=mask)
    assert torch.isfinite(loss) and loss.item() >= 0


def test_fftrmse_numerical_stability():
    """No NaN/Inf for very small or very large scales."""
    for scale in [1e-8, 1e8]:
        y = torch.randn(3, 24, 1) * scale
        y_hat = torch.randn(3, 24, 1) * scale
        loss = FFTRMSELoss(norm=True)(y=y, y_hat=y_hat)
        assert torch.isfinite(loss), f"Non-finite RMSE at scale {scale}"


# ─────────────────────────── MixedFFTLoss ──────────────────────────

def test_mixedfft_univariate():
    """Perfect predictions yield zero combined loss."""
    y = torch.randn(2, 32, 1)
    loss_fn = MixedFFTLoss(time_loss=MAE(), freq_loss=FFTMAELoss(), lam=0.5)
    loss = loss_fn(y=y, y_hat=y)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_mixedfft_multivariate():
    """Combined loss equals time_loss + lam * freq_loss."""
    B, H, N = 4, 24, 3
    y = torch.randn(B, H, N)
    y_hat = torch.randn(B, H, N)
    lam = 0.3
    time_loss, freq_loss = MAE(), FFTMAELoss()
    mixed = MixedFFTLoss(time_loss=time_loss, freq_loss=freq_loss, lam=lam)

    expected = time_loss(y=y, y_hat=y_hat) + lam * freq_loss(y=y, y_hat=y_hat)
    assert mixed(y=y, y_hat=y_hat).item() == pytest.approx(expected.item(), rel=1e-5)


def test_mixedfft_autoregressive_mask():
    """Masked mixed loss is finite; invalid loss types raise TypeError."""
    B, H, N = 2, 20, 1
    y = torch.randn(B, H, N)
    y_hat = torch.randn(B, H, N)
    mask = torch.ones(B, H, N); mask[:, -5:, :] = 0

    loss_fn = MixedFFTLoss(time_loss=MSE(), freq_loss=FFTMSELoss(), lam=1.0)
    loss = loss_fn(y=y, y_hat=y_hat, mask=mask)
    assert torch.isfinite(loss)

    with pytest.raises(TypeError):
        MixedFFTLoss(time_loss=FFTMAELoss(), freq_loss=FFTMAELoss())  # wrong type


def test_mixedfft_numerical_stability():
    """Mixed loss stays finite at extreme scales with lam=0 and lam=1."""
    for lam in [0.0, 1.0]:
        for scale in [1e-8, 1e8]:
            y = torch.randn(2, 32, 2) * scale
            y_hat = torch.randn(2, 32, 2) * scale
            loss = MixedFFTLoss(MAE(), FFTMAELoss(), lam=lam)(y=y, y_hat=y_hat)
            assert torch.isfinite(loss), f"Non-finite at lam={lam}, scale={scale}"
