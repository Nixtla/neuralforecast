import pytest
import torch

from neuralforecast.losses.pytorch import TukeyLoss


def test_tukey_values_and_gradients():
    residuals = torch.tensor(
        [-100.0, -4.685, -1.0, -1e-8, 0.0, 1e-8, 1.0, 4.685, 100.0],
        dtype=torch.float64,
    ).reshape(-1, 1, 1)
    prediction = residuals.clone().requires_grad_()
    loss = TukeyLoss(normalize=False)
    value = loss(torch.zeros_like(prediction), prediction)
    gradient, = torch.autograd.grad(value, prediction)
    # Integrate psi(r) = r * (1 - (r/c)**2)**2 inside the cutoff.
    r = residuals.clamp(min=-loss.c, max=loss.c)
    expected = r.square() / 2 - r.pow(4) / (2 * loss.c**2) + r.pow(6) / (6 * loss.c**4)
    expected_gradient = residuals * (1 - (residuals / loss.c).square()).square()
    expected_gradient[residuals.abs() >= loss.c] = 0
    torch.testing.assert_close(value, expected.mean())
    torch.testing.assert_close(gradient, expected_gradient / prediction.numel())
    assert loss(torch.zeros_like(prediction), torch.zeros_like(prediction)) == 0
    assert gradient[5] > 0  # Preserve the gradient even for tiny residuals.


@pytest.mark.parametrize("normalize", [False, True])
def test_tukey_mask_and_normalization(normalize):
    y = torch.tensor([1.0, 2.0, 4.0, 100.0], dtype=torch.float64).reshape(1, 4, 1)
    prediction = (y + 0.5).requires_grad_()
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64).reshape_as(y)
    loss = TukeyLoss(normalize=normalize)
    actual = loss(y, prediction, mask=mask)
    cropped = loss(y[:, :3], prediction[:, :3])
    torch.testing.assert_close(actual, cropped)
    scale = y[:, :3].std(dim=1, correction=0, keepdim=True) + 1e-2 if normalize else 1
    reference = TukeyLoss(normalize=False)(y[:, :3] / scale, prediction[:, :3] / scale)
    torch.testing.assert_close(actual, reference)
    gradient, = torch.autograd.grad(actual, prediction)
    assert (gradient[:, :3] > 0).all()
    assert gradient[:, 3] == 0


@pytest.mark.parametrize("normalize", [False, True])
def test_tukey_series_independence(normalize):
    y = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64).reshape(1, 3, 1)
    y = torch.cat([y, 100 * y + 10], dim=-1)
    prediction = y + torch.tensor([0.2, 5.0])
    loss = TukeyLoss(normalize=normalize)
    expected = (loss(y[..., :1], prediction[..., :1]) + loss(y[..., 1:], prediction[..., 1:])) / 2
    torch.testing.assert_close(loss(y, prediction), expected)


@pytest.mark.parametrize("normalize", [False, True])
def test_tukey_masked_nonfinite_and_empty(normalize):
    y = torch.tensor([1.0, 2.0, 4.0, float("nan")]).reshape(1, 4, 1)
    prediction = torch.tensor([1.5, 2.5, 4.5, float("nan")]).reshape_as(y).requires_grad_()
    mask = torch.tensor([1.0, 0.5, 2.0, 0.0]).reshape_as(y)
    loss = TukeyLoss(normalize=normalize)
    actual = loss(y, prediction, mask=mask)
    scale = y[:, :3].std(dim=1, correction=0, keepdim=True) + 1e-2 if normalize else 1
    expected = TukeyLoss(normalize=False)(y[:, :3] / scale, prediction[:, :3] / scale)
    torch.testing.assert_close(actual, expected)
    gradient, = torch.autograd.grad(actual, prediction)
    assert torch.isfinite(gradient).all()
    assert gradient[:, -1] == 0
    empty = loss(y, prediction, mask=torch.zeros_like(mask))
    assert empty == 0
    gradient, = torch.autograd.grad(empty, prediction)
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_tukey_large_outlier_has_finite_zero_gradient():
    prediction = torch.full((1, 3, 1), 1e30, requires_grad=True)
    value = TukeyLoss(normalize=False)(torch.zeros_like(prediction), prediction)
    gradient, = torch.autograd.grad(value, prediction)
    torch.testing.assert_close(value, torch.tensor(4.685**2 / 6))
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_tukey_normalization_accepts_two_dimensional_inputs():
    y = torch.tensor([[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]])
    prediction = y + 0.5
    loss = TukeyLoss()
    torch.testing.assert_close(loss(y, prediction), loss(y.unsqueeze(-1), prediction.unsqueeze(-1)))
