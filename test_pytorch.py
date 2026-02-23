import warnings

import torch

from neuralforecast.losses.pytorch import (
    MAE,
    PMM,
    DistributionLoss,
    HuberIQLoss,
    IQLoss,
    MQLoss,
    FFTMAELoss,
    FFTMSELoss,
    FFTRMSELoss,
    MixedFFTLoss
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

def test_fft_mae_loss():
    criterion = FFTMAELoss(norm=True)
    batch_size, horizon, channels = 16, 24, 5
    
    # Autoregressive [B, H]
    y_ar = torch.randn(batch_size, horizon)
    y_hat_ar = torch.randn(batch_size, horizon)
    loss_ar = criterion(y_ar, y_hat_ar)
    assert isinstance(loss_ar, torch.Tensor), "FFTMAELoss AR output must be a tensor."
    assert loss_ar.dim() == 0, f"FFTMAELoss AR output must be a scalar, but got dim {loss_ar.dim()}"

    # Univariate [B, H, 1]
    y_uni = torch.randn(batch_size, horizon, 1)
    y_hat_uni = torch.randn(batch_size, horizon, 1)
    loss_uni = criterion(y_uni, y_hat_uni)
    assert isinstance(loss_uni, torch.Tensor), "FFTMAELoss Univariate output must be a tensor."
    assert loss_uni.dim() == 0, f"FFTMAELoss Univariate output must be a scalar, but got dim {loss_uni.dim()}"

    # 3. Multivariate [B, H, N]
    y_multi = torch.randn(batch_size, horizon, channels)
    y_hat_multi = torch.randn(batch_size, horizon, channels)
    mask_multi = torch.ones_like(y_multi)
    loss_multi = criterion(y_multi, y_hat_multi, mask=mask_multi)
    assert isinstance(loss_multi, torch.Tensor), "FFTMAELoss Multivariate output must be a tensor."
    assert loss_multi.dim() == 0, f"FFTMAELoss Multivariate output must be a scalar, but got dim {loss_multi.dim()}"
    
    print("test_fft_mae_loss() passed.")

def test_fft_mse_loss():
    criterion = FFTMSELoss(norm=True)
    batch_size, horizon, channels = 16, 24, 5
    
    # 1. Autoregressive [B, H]
    y_ar = torch.randn(batch_size, horizon)
    y_hat_ar = torch.randn(batch_size, horizon)
    loss_ar = criterion(y_ar, y_hat_ar)
    assert isinstance(loss_ar, torch.Tensor), "FFTMSELoss AR output must be a tensor."
    assert loss_ar.dim() == 0, f"FFTMSELoss AR output must be a scalar, but got dim {loss_ar.dim()}"

    # 2. Univariate [B, H, 1]
    y_uni = torch.randn(batch_size, horizon, 1)
    y_hat_uni = torch.randn(batch_size, horizon, 1)
    loss_uni = criterion(y_uni, y_hat_uni)
    assert isinstance(loss_uni, torch.Tensor), "FFTMSELoss Univariate output must be a tensor."
    assert loss_uni.dim() == 0, f"FFTMSELoss Univariate output must be a scalar, but got dim {loss_uni.dim()}"

    # 3. Multivariate [B, H, N]
    y_multi = torch.randn(batch_size, horizon, channels)
    y_hat_multi = torch.randn(batch_size, horizon, channels)
    mask_multi = torch.ones_like(y_multi)
    loss_multi = criterion(y_multi, y_hat_multi, mask=mask_multi)
    assert isinstance(loss_multi, torch.Tensor), "FFTMSELoss Multivariate output must be a tensor."
    assert loss_multi.dim() == 0, f"FFTMSELoss Multivariate output must be a scalar, but got dim {loss_multi.dim()}"
    
    print("test_fft_mse_loss() passed.")

def test_fft_rmse_loss():
    criterion = FFTRMSELoss(norm=True)
    batch_size, horizon, channels = 16, 24, 5
    
    # 1. Autoregressive [B, H]
    y_ar = torch.randn(batch_size, horizon)
    y_hat_ar = torch.randn(batch_size, horizon)
    loss_ar = criterion(y_ar, y_hat_ar)
    assert isinstance(loss_ar, torch.Tensor), "FFTRMSELoss AR output must be a tensor."
    assert loss_ar.dim() == 0, f"FFTRMSELoss AR output must be a scalar, but got dim {loss_ar.dim()}"
    assert not torch.isnan(loss_ar), "FFTRMSELoss resulted in NaN."

    # 2. Univariate [B, H, 1]
    y_uni = torch.randn(batch_size, horizon, 1)
    y_hat_uni = torch.randn(batch_size, horizon, 1)
    loss_uni = criterion(y_uni, y_hat_uni)
    assert isinstance(loss_uni, torch.Tensor), "FFTRMSELoss Univariate output must be a tensor."
    assert loss_uni.dim() == 0, f"FFTRMSELoss Univariate output must be a scalar, but got dim {loss_uni.dim()}"
    assert not torch.isnan(loss_uni), "FFTRMSELoss resulted in NaN."

    # 3. Multivariate [B, H, N]
    y_multi = torch.randn(batch_size, horizon, channels)
    y_hat_multi = torch.randn(batch_size, horizon, channels)
    loss_multi = criterion(y_multi, y_hat_multi)
    assert isinstance(loss_multi, torch.Tensor), "FFTRMSELoss Multivariate output must be a tensor."
    assert loss_multi.dim() == 0, f"FFTRMSELoss Multivariate output must be a scalar, but got dim {loss_multi.dim()}"
    assert not torch.isnan(loss_multi), "FFTRMSELoss resulted in NaN."
    
    print("test_fft_rmse_loss() passed.")


def test_mixed_fft_loss():
    # Instantiating required loss dependencies natively within the test
    time_loss = MAE() 
    freq_loss = FFTMAELoss()
    criterion = MixedFFTLoss(time_loss=time_loss, freq_loss=freq_loss, lam=0.5)
    
    batch_size, horizon, channels = 16, 24, 5
    
    # 1. Autoregressive [B, H]
    y_ar = torch.randn(batch_size, horizon)
    y_hat_ar = torch.randn(batch_size, horizon)
    loss_ar = criterion(y_ar, y_hat_ar)
    assert isinstance(loss_ar, torch.Tensor), "MixedFFTLoss AR output must be a tensor."
    assert loss_ar.dim() == 0, f"MixedFFTLoss AR output must be a scalar, but got dim {loss_ar.dim()}"

    # 2. Univariate [B, H, 1]
    y_uni = torch.randn(batch_size, horizon, 1)
    y_hat_uni = torch.randn(batch_size, horizon, 1)
    loss_uni = criterion(y_uni, y_hat_uni)
    assert isinstance(loss_uni, torch.Tensor), "MixedFFTLoss Univariate output must be a tensor."
    assert loss_uni.dim() == 0, f"MixedFFTLoss Univariate output must be a scalar, but got dim {loss_uni.dim()}"

    # 3. Multivariate [B, H, N]
    y_multi = torch.randn(batch_size, horizon, channels)
    y_hat_multi = torch.randn(batch_size, horizon, channels)
    mask_multi = torch.randint(0, 2, (batch_size, horizon, channels)).float() 
    loss_multi = criterion(y_multi, y_hat_multi, mask=mask_multi)
    assert isinstance(loss_multi, torch.Tensor), "MixedFFTLoss Multivariate output must be a tensor."
    assert loss_multi.dim() == 0, f"MixedFFTLoss Multivariate output must be a scalar, but got dim {loss_multi.dim()}"

    print("test_mixed_fft_loss() passed.")

if __name__ == '__main__':
    test_fft_mae_loss()
    test_fft_mse_loss()
    test_fft_rmse_loss()
    test_mixed_fft_loss()