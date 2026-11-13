import pytest
import torch

from neuralforecast.losses.pytorch import NBMM


def _mean_and_grad(loss, loc):
    n_params = 3 if loss.weighted else 2
    out = torch.full((1, 3, n_params), 3.0, requires_grad=True)
    distr_args = loss.scale_decouple(
        loss.domain_map(out),
        loc=torch.full((1, 3, 1), loc),
        scale=torch.ones(1, 3, 1),
    )
    mean = loss.get_distribution(distr_args).mean
    (grad,) = torch.autograd.grad(
        loss(y=torch.full((1, 3, 1), 50.0), distr_args=distr_args), out
    )
    return mean.flatten()[0].item(), grad[0, 0, 0].item()


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("loc", [0.0, -5.0])
def test_nbmm_mean_stays_learnable_for_non_positive_loc(loc, weighted):
    # loc = 0 comes from the identity scaler, negative loc from a standard
    # scaler on negative data.
    loss = NBMM(n_components=1, weighted=weighted)

    mean, grad_mu = _mean_and_grad(loss, loc)

    assert mean > 1e-3
    assert grad_mu != 0.0


@pytest.mark.parametrize("weighted", [False, True])
def test_nbmm_mean_is_anchored_on_loc(weighted):
    loss = NBMM(n_components=1, weighted=weighted)

    mean, _ = _mean_and_grad(loss, loc=100.0)

    # softplus(3) * scale + loc, as in nbinomial_scale_decouple
    expected = torch.nn.functional.softplus(torch.tensor(3.0)).item() + 100.0
    assert mean == pytest.approx(expected, rel=1e-3)
