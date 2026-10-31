import torch
import torch.nn.functional as F

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss
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
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df)
    assert nf.predict()["MLP"].std() > 0.5 * df["y"].std()
