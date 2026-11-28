import pytest
import torch
import torch.nn.functional as F

from neuralforecast.models.nhits import _IdentityBasis


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize(
    "knots_size,horizon", [(1, 1), (1, 7), (2, 3), (3, 12), (12, 96), (7, 5)]
)
def test_cubic_basis_uses_single_interpolation_row(
    monkeypatch, dtype, batch_size, knots_size, horizon
):
    torch.manual_seed(42)
    theta = torch.randn(batch_size, 8 + knots_size, dtype=dtype, requires_grad=True)
    reference_theta = theta.detach().clone().requires_grad_(True)
    knots = reference_theta[:, 8:].reshape(batch_size, 1, 1, knots_size)
    # Match the existing output buffer dtype, independently of theta's dtype.
    expected = F.interpolate(knots, size=horizon, mode="bicubic")[:, 0, 0, :].to(
        torch.get_default_dtype()
    )
    interpolate = F.interpolate
    output_shapes = []

    def record_interpolation(*args, **kwargs):
        result = interpolate(*args, **kwargs)
        output_shapes.append(result.shape)
        return result

    monkeypatch.setattr(F, "interpolate", record_interpolation)
    backcast, forecast = _IdentityBasis(8, horizon, "cubic")(theta)
    assert output_shapes == [(batch_size, 1, 1, horizon)]
    torch.testing.assert_close(backcast, theta[:, :8])
    torch.testing.assert_close(forecast[..., 0], expected, atol=3e-6, rtol=3e-6)
    weight = torch.randn_like(expected)
    actual_grad = torch.autograd.grad((forecast[..., 0] * weight).sum(), theta)[0]
    expected_grad = torch.autograd.grad((expected * weight).sum(), reference_theta)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=3e-6, rtol=3e-6)
