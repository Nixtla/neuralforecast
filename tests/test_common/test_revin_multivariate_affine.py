import pytest
import torch

from neuralforecast.common._modules import RevINMultivariate


@pytest.mark.parametrize(
    "weight",
    [-1.0, -1e-5, -1e-10, -1e-12, 0.0, 1e-12, 1e-10, 1.0],
)
def test_affine_denorm_remains_finite(weight):
    torch.manual_seed(42)
    model = RevINMultivariate(num_features=3, affine=True)

    with torch.no_grad():
        model.affine_weight.fill_(weight)

    x = torch.randn(2, 10, 3)
    normalized = model(x, "norm")
    restored = model(normalized, "denorm")

    assert torch.isfinite(restored).all()


@pytest.mark.parametrize("affine", [True, False])
def test_revin_multivariate_roundtrip(affine):
    torch.manual_seed(42)
    model = RevINMultivariate(num_features=3, affine=affine)

    x = torch.randn(2, 10, 3)
    restored = model(model(x, "norm"), "denorm")

    torch.testing.assert_close(restored, x, atol=1e-5, rtol=1e-5)


def test_zero_weight_backward_is_finite():
    torch.manual_seed(42)
    model = RevINMultivariate(num_features=3, affine=True)

    with torch.no_grad():
        model.affine_weight.zero_()

    x = torch.randn(2, 10, 3)
    normalized = model(x, "norm")
    restored = model(normalized, "denorm")

    loss = restored.square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(model.affine_weight.grad).all()


@pytest.mark.parametrize("weight", [-1e-10, 0.0, 1e-10])
def test_singular_weights_have_finite_gradients(weight):
    torch.manual_seed(42)
    model = RevINMultivariate(num_features=3, affine=True)

    with torch.no_grad():
        model.affine_weight.fill_(weight)

    x = torch.randn(2, 10, 3)
    y = model(model(x, "norm"), "denorm")

    loss = y.square().mean()
    loss.backward()

    assert torch.isfinite(y).all()
    assert torch.isfinite(model.affine_weight.grad).all()


@pytest.mark.parametrize("weight", [-1.0, -0.5, 0.5, 1.0])
def test_regular_affine_weights_preserve_roundtrip(weight):
    torch.manual_seed(42)
    model = RevINMultivariate(num_features=3, affine=True)

    with torch.no_grad():
        model.affine_weight.fill_(weight)
        model.affine_bias.fill_(0.2)

    x = torch.randn(2, 10, 3)
    restored = model(model(x, "norm"), "denorm")

    torch.testing.assert_close(restored, x, atol=1e-5, rtol=1e-5)
