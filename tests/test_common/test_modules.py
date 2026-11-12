import pytest
import torch
import torch.nn as nn

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


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("explicit_mask", [False, True])
def test_full_attention_sdpa_matches_explicit_causal_path(batch, explicit_mask):
    """The two FullAttention paths must agree when mask_flag is set.

    The efficient path is taken whenever output_attention=False, which is the default used by
    the VanillaTransformer decoder. It used to pass no mask at all in that case, so it attended
    to future steps while the explicit path applied TriangularCausalMask.

    `batch` is parametrized because the mask used to be sliced to `[B, L, S]`: at `B=1` that
    still broadcasts over the head axis by accident, so a batch of one would not catch it.
    """
    q, k, v = _attn_inputs(batch=batch)
    explicit = FullAttention(
        mask_flag=True, attention_dropout=0.0, output_attention=True
    ).eval()
    efficient = FullAttention(
        mask_flag=True, attention_dropout=0.0, output_attention=False
    ).eval()
    mask = (
        TriangularCausalMask(batch, q.shape[1], device=q.device)
        if explicit_mask
        else None
    )

    ref, _ = explicit(q, k, v, attn_mask=mask)
    out, _ = efficient(q, k, v, attn_mask=mask)

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("batch", [1, 4])
def test_full_attention_sdpa_does_not_attend_to_future_steps(batch):
    q, k, v = _attn_inputs(batch=batch)
    attentive = FullAttention(
        mask_flag=True, attention_dropout=0.0, output_attention=False
    ).eval()

    out, _ = attentive(q, k, v, attn_mask=None)
    perturbed = v.clone()
    perturbed[:, -1] += 10.0
    out_perturbed, _ = attentive(q, k, perturbed, attn_mask=None)

    # The first step comes before the perturbed one, so it cannot depend on it.
    torch.testing.assert_close(out[:, 0], out_perturbed[:, 0])


@pytest.mark.parametrize("batch", [1, 4])
def test_full_attention_sdpa_honours_an_explicit_causal_mask(batch):
    """TriangularCausalMask marks positions to drop, SDPA's boolean mask marks those to keep.

    Passing the mask through unchanged inverted it, so every position attended *only* to the
    future.
    """
    length = 5
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


def test_full_attention_does_not_drop_a_caller_mask_of_wrong_shape():
    """A caller-supplied mask is applied, not dropped, when L != S.

    The first revision of this fix guarded every mask with `L == S`, which silently discarded a
    caller's mask in that case while the explicit path raised. Both paths must fail loudly.
    """
    batch, length = 2, 5
    q, k, v = _attn_inputs(batch=batch, length=length)
    keys = torch.randn(batch, length + 2, q.shape[2], q.shape[3])
    mask = TriangularCausalMask(batch, length, device=q.device)
    attentive = FullAttention(
        mask_flag=True, attention_dropout=0.0, output_attention=False
    ).eval()

    with pytest.raises(RuntimeError):
        attentive(q, keys, keys, attn_mask=mask)


def test_mlp_prelu_has_independent_parameters():
    mlp = MLP(
        in_features=8,
        out_features=1,
        activation="PReLU",
        hidden_size=16,
        num_layers=4,
        dropout=0.0,
    )
    prelus = [
        module
        for module in mlp.layers
        if isinstance(module, nn.PReLU)
    ]
    assert len(prelus) == 3
    assert len({id(module) for module in prelus}) == 3
    assert len({id(module.weight) for module in prelus}) == 3
