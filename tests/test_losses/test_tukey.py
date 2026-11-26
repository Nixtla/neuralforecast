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
    std = y[:, :3].std(dim=1, correction=0, keepdim=True)
    scale = std.clamp(min=1) if normalize else 1
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
    std = y[:, :3].std(dim=1, correction=0, keepdim=True)
    scale = std.clamp(min=1) if normalize else 1
    expected = TukeyLoss(normalize=False)(y[:, :3] / scale, prediction[:, :3] / scale)
    torch.testing.assert_close(actual, expected)
    gradient, = torch.autograd.grad(actual, prediction)
    assert torch.isfinite(gradient).all()
    assert gradient[:, -1] == 0
    empty = loss(y, prediction, mask=torch.zeros_like(mask))
    assert empty == 0
    gradient, = torch.autograd.grad(empty, prediction)
    assert torch.equal(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("y", [[[3.0]], [[2.0], [2.0], [2.0]]], ids=["h1", "flat"])
def test_tukey_zero_spread_horizon_keeps_gradient(y):
    # A single-step horizon or a flat target has no spread to normalize by.
    y = torch.tensor(y).unsqueeze(0)
    prediction = (y + 0.5).requires_grad_()
    actual = TukeyLoss()(y, prediction)
    torch.testing.assert_close(actual, TukeyLoss(normalize=False)(y, prediction))
    gradient, = torch.autograd.grad(actual, prediction)
    assert (gradient > 0).all()


@pytest.mark.parametrize("normalize", [False, True])
def test_tukey_infinite_prediction_gives_infinite_loss(normalize):
    # A large finite outlier saturates, but an overflowed prediction must not.
    y = torch.zeros(1, 3, 1)
    prediction = torch.tensor([0.5, 0.1, float("inf")]).reshape(1, 3, 1)
    loss = TukeyLoss(normalize=normalize)
    assert torch.isinf(loss(y, prediction))
    mask = torch.tensor([1.0, 1.0, 0.0]).reshape_as(y)
    torch.testing.assert_close(
        loss(y, prediction, mask=mask), loss(y[:, :2], prediction[:, :2])
    )


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tukey_normalization_in_half_precision(dtype):
    # A deviation of 500 squares past the float16 range; the residuals of 3.2
    # standard deviations are inside the cutoff and must keep their gradient.
    y = torch.tensor([0.0, 1000.0]).reshape(1, 2, 1)
    expected_prediction = (y + 1600).requires_grad_()
    expected = TukeyLoss()(y, expected_prediction)
    expected_gradient, = torch.autograd.grad(expected, expected_prediction)
    prediction = (y + 1600).to(dtype).requires_grad_()
    actual = TukeyLoss()(y.to(dtype), prediction)
    gradient, = torch.autograd.grad(actual, prediction)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=0)
    torch.testing.assert_close(gradient.float(), expected_gradient, rtol=2e-2, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tukey_half_precision_reduction_does_not_overflow(dtype):
    # 1024 * 64 unit weights sum past the float16 range.
    y = torch.zeros(1024, 64, 1)
    expected = TukeyLoss(normalize=False)(y, torch.ones_like(y))
    prediction = torch.ones_like(y).to(dtype).requires_grad_()
    actual = TukeyLoss(normalize=False)(y.to(dtype), prediction)
    gradient, = torch.autograd.grad(actual, prediction)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=0)
    assert (gradient > 0).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tukey_half_precision_gradient_just_inside_cutoff(dtype):
    # The gradient vanishes at c. Expanded in half precision, the polynomial's
    # backward cancels to the wrong size or sign for residuals just inside c.
    prediction = torch.tensor([4.5, 4.625, 4.65625]).reshape(1, 3, 1)
    y = torch.zeros_like(prediction)
    expected_prediction = prediction.clone().requires_grad_()
    expected = TukeyLoss(normalize=False)(y, expected_prediction)
    expected_gradient, = torch.autograd.grad(expected, expected_prediction)
    prediction = prediction.to(dtype).requires_grad_()
    actual = TukeyLoss(normalize=False)(y.to(dtype), prediction)
    gradient, = torch.autograd.grad(actual, prediction)
    assert (gradient > 0).all()
    torch.testing.assert_close(gradient.float(), expected_gradient, rtol=1e-2, atol=0)


@pytest.mark.parametrize("normalize", [False, True])
def test_tukey_integer_inputs_give_floating_loss(normalize):
    y = torch.tensor([0, 2, 5]).reshape(1, 3, 1)
    prediction = torch.tensor([1, 2, 3]).reshape(1, 3, 1)
    actual = TukeyLoss(normalize=normalize)(y, prediction)
    assert actual.dtype == torch.get_default_dtype()
    torch.testing.assert_close(
        actual, TukeyLoss(normalize=normalize)(y.float(), prediction.float())
    )
