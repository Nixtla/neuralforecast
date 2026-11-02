import pytest
import torch

from neuralforecast.common._modules import FullAttention, MLP, TriangularCausalMask


@pytest.mark.parametrize("num_layers", [2, 3])
@pytest.mark.parametrize("hidden_size", [0, -1])
def test_mlp_rejects_nonpositive_hidden_size(num_layers, hidden_size):
    with pytest.raises(
        ValueError,
        match=rf"hidden_size must be positive when num_layers>=2, got {hidden_size}",
    ):
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


def _attn_inputs(seed=0, batch=1, length=5, heads=2, dim=4):
    torch.manual_seed(seed)
    return [torch.randn(batch, length, heads, dim) for _ in range(3)]


def test_full_attention_sdpa_matches_explicit_causal_path():
    """The two FullAttention paths must agree when mask_flag is set.

    The efficient path is taken whenever output_attention=False, which is the default used
    by the VanillaTransformer decoder. It used to pass no mask at all in that case, so it
    attended to future steps while the explicit path applied TriangularCausalMask.
    """
    q, k, v = _attn_inputs()
    explicit = FullAttention(
        mask_flag=True, attention_dropout=0.0, output_attention=True
    ).eval()
    efficient = FullAttention(
        mask_flag=True, attention_dropout=0.0, output_attention=False
    ).eval()

    ref, _ = explicit(q, k, v, attn_mask=None)
    out, _ = efficient(q, k, v, attn_mask=None)

    torch.testing.assert_close(out, ref)


def test_full_attention_sdpa_does_not_attend_to_future_steps():
    q, k, v = _attn_inputs()
    attentive = FullAttention(
        mask_flag=True, attention_dropout=0.0, output_attention=False
    ).eval()

    out, _ = attentive(q, k, v, attn_mask=None)
    perturbed = v.clone()
    perturbed[:, -1] += 10.0
    out_perturbed, _ = attentive(q, k, perturbed, attn_mask=None)

    # The first step comes before the perturbed one, so it cannot depend on it.
    torch.testing.assert_close(out[:, 0], out_perturbed[:, 0])


def test_full_attention_sdpa_honours_an_explicit_causal_mask():
    """TriangularCausalMask marks positions to drop, SDPA's boolean mask marks those to keep.

    Passing the mask through unchanged inverted it, so every position attended *only* to the
    future.
    """
    batch, length = 1, 5
    q, k, v = _attn_inputs(batch=batch, length=length)
    attentive = FullAttention(
        mask_flag=True, attention_dropout=0.0, output_attention=False
    ).eval()
    mask = TriangularCausalMask(batch, length, device=q.device)

    out, _ = attentive(q, k, v, attn_mask=mask)
    perturbed = v.clone()
    perturbed[:, 0] += 10.0
    out_perturbed, _ = attentive(q, k, perturbed, attn_mask=mask)

    # Step 1 is allowed to look at step 0, so changing step 0 must change it.
    assert not torch.allclose(out[:, 1], out_perturbed[:, 1])
