import pytest
import torch
import torch.nn.functional as F

from neuralforecast.losses.pytorch import MQLoss, HuberMQLoss


@pytest.mark.parametrize("quantiles", [
    [0.5],
    [0.1, 0.5, 0.9],
    [0.05, 0.1, 0.5, 0.9, 0.95],
])
def test_mqloss_matches_average_pinball_loss(quantiles):
    y = torch.zeros(2, 4, 1)
    y_hat = torch.ones(2, 4, 1, len(quantiles))

    actual = MQLoss(quantiles=quantiles)(y=y, y_hat=y_hat)
    expected = torch.tensor(
        sum(1 - q for q in quantiles) / len(quantiles)
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("quantiles", [
    [0.5],
    [0.1, 0.5, 0.9],
    [0.05, 0.1, 0.5, 0.9, 0.95],
])
def test_huber_mqloss_matches_average_quantile_loss(quantiles):
    y = torch.zeros(2, 4, 1)
    y_hat = torch.ones(2, 4, 1, len(quantiles))

    actual = HuberMQLoss(
        quantiles=quantiles, delta=1.0
    )(y=y, y_hat=y_hat)

    expected = torch.stack([
        F.huber_loss(
            torch.tensor(1 - q),
            torch.tensor(0.0),
            delta=1.0,
        )
        for q in quantiles
    ]).mean()

    torch.testing.assert_close(actual, expected)
