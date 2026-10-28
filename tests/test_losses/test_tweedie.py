import torch

from neuralforecast.losses.pytorch import DistributionLoss, poisson_scale_decouple


# The mean must follow the network output. `tweedie_scale_decouple` took a
# softplus of the log mean and clamped it at 1e-9, so the mean could not go
# below 1. With loc = 0 (the identity scaler) log(loc) = -inf was clamped as
# well: the mean was exactly 1 and the loss had no gradient.


def _decouple(loss, output, loc, scale):
    return loss.scale_decouple(output=loss.domain_map(output), loc=loc, scale=scale)


def test_mean_below_one_with_the_identity_scaler():
    loss = DistributionLoss(distribution="Tweedie", rho=1.5)
    mu = torch.tensor([0.05, 0.5, 5.0]).reshape(1, 3, 1)
    output = torch.log(torch.expm1(mu)).requires_grad_()  # softplus(output) = mu

    distr_args = _decouple(loss, output, torch.zeros(1, 1, 1), torch.ones(1, 1, 1))
    torch.testing.assert_close(loss.get_distribution(distr_args).mean, mu)

    loss(y=torch.zeros_like(mu), distr_args=distr_args).backward()
    assert (output.grad > 0).all(), "the loss must train the mean"


def test_mean_is_mapped_like_the_poisson_rate():
    loss = DistributionLoss(distribution="Tweedie", rho=1.5)
    torch.manual_seed(0)
    output = torch.randn(4, 3, 1)
    loc, scale = 3 * torch.rand(4, 1, 1), 5 * torch.rand(4, 1, 1) + 0.1

    distr_args = _decouple(loss, output, loc, scale)
    (rate,) = poisson_scale_decouple((output,), loc=loc, scale=scale)
    torch.testing.assert_close(loss.get_distribution(distr_args).mean, rate)


def test_mean_grows_linearly_with_the_output():
    # An exponential of the output makes the mean exponential in the inputs;
    # on M5 sales that blew some forecasts up by orders of magnitude.
    loss = DistributionLoss(distribution="Tweedie", rho=1.5)
    output = torch.tensor([20.0, 50.0, 200.0]).reshape(1, 3, 1)

    distr_args = _decouple(loss, output, torch.zeros(1, 1, 1), torch.ones(1, 1, 1))
    torch.testing.assert_close(loss.get_distribution(distr_args).mean, output)
