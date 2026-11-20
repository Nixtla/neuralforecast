from unittest.mock import patch

import pytest
import torch
from torch.distributions import Gamma

from neuralforecast.losses.pytorch import Tweedie, est_alpha, est_beta


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sample_shape", [(), (4,), (2, 3)])
def test_zero_poisson_counts_produce_exact_zero(dtype, sample_shape):
    distribution = Tweedie(torch.log(torch.full((3,), 0.5, dtype=dtype)), rho=1.5)
    counts = torch.tensor([0.0, 1.0, 3.0], dtype=dtype).expand(sample_shape + (3,))
    with patch("torch.poisson", return_value=counts):
        samples = distribution.sample(sample_shape)

    assert samples.shape == sample_shape + (3,)
    assert samples.dtype == dtype
    assert torch.equal(samples[..., 0], torch.zeros_like(samples[..., 0]))
    assert (samples[..., 1:] > 0).all()
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("rho", [1.1, 1.5, 1.9])
def test_positive_counts_use_the_unperturbed_conditional_gamma(rho):
    mu = torch.tensor([0.1, 0.5, 2.0], dtype=torch.float64)
    counts = torch.tensor([1.0, 2.0, 4.0], dtype=mu.dtype)
    distribution = Tweedie(mu.log(), rho=rho)
    mu = distribution.mean
    torch.manual_seed(0)
    expected = Gamma(counts * est_alpha(rho), est_beta(mu, rho)).sample()
    torch.manual_seed(0)
    with patch("torch.poisson", return_value=counts):
        actual = distribution.sample()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=0)


def test_empty_sample_shape():
    distribution = Tweedie(torch.zeros(2, 3), rho=1.5)
    assert distribution.sample((0,)).shape == (0, 2, 3)
