from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Gamma

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss, Tweedie, est_alpha, est_beta
from neuralforecast.models import MLP
from neuralforecast.utils import generate_series


def _distr_args(loss, x, loc, scale):
    loc, scale = torch.full((1, 1, 1), loc), torch.full((1, 1, 1), scale)
    return loss.scale_decouple(output=loss.domain_map(x), loc=loc, scale=scale)


def test_mean_follows_the_scaled_output():
    loss = DistributionLoss(distribution="Tweedie", rho=1.5)
    x = torch.tensor([-4.9, -2.0, 0.0, 3.0, 200.0]).reshape(1, 5, 1).requires_grad_()
    loc, scale, y = 10.0, 2.0, 3.0

    distr_args = _distr_args(loss, x, loc, scale)
    mean = loss.get_distribution(distr_args).mean
    torch.testing.assert_close(mean, F.softplus(x * scale + loc))
    assert (mean < loc).any()

    loss(y=torch.full_like(mean, y), distr_args=distr_args).backward()
    assert torch.equal(x.grad.sign(), (mean - y).sign())


def test_very_negative_outputs_stay_finite():
    loss = DistributionLoss(distribution="Tweedie", rho=1.5)
    x = torch.tensor([-1e4, -100.0, -20.0]).reshape(1, 3, 1).requires_grad_()

    distr_args = _distr_args(loss, x, loc=0.0, scale=1.0)
    assert (loss.get_distribution(distr_args).mean > 0).all()
    value = loss(y=torch.full_like(x, 5.0), distr_args=distr_args)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(x.grad).all()


def test_identity_scaler_forecasts_are_not_constant():
    df = generate_series(n_series=2, freq="D", min_length=100, max_length=100, seed=0)
    model = MLP(
        h=7,
        input_size=14,
        loss=DistributionLoss(distribution="Tweedie", rho=1.5),
        scaler_type="identity",
        max_steps=20,
        random_seed=0,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df)
    assert nf.predict()["MLP"].std() > 0.5 * df["y"].std()


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
