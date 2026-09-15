import pytest
import torch

from neuralforecast.common._modules import MLP


@pytest.mark.parametrize("num_layers", [2, 3])
@pytest.mark.parametrize("hidden_size", [0, -1])
def test_mlp_rejects_nonpositive_hidden_size(num_layers, hidden_size):
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        MLP(3, 2, "Tanh", hidden_size, num_layers, 0.0)


@pytest.mark.parametrize("num_layers, hidden_size", [(1, 0), (1, -1), (2, 4), (3, 4)])
def test_mlp_valid_width_preserves_input_gradients(num_layers, hidden_size):
    torch.manual_seed(0)
    model = MLP(3, 2, "Tanh", hidden_size, num_layers, 0.0)
    inputs = torch.randn(2, 5, 3, requires_grad=True)

    output = model(inputs)
    assert output.shape == (2, 5, 2)
    output.square().mean().backward()

    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad.abs().sum() > 0
    for parameter in model.parameters():
        assert torch.isfinite(parameter.grad).all()
