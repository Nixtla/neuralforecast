import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralforecast.common._modules import (
    FullAttention,
    MLP,
    RevIN,
    RevINMultivariate,
    TransDecoderLayer,
    TransEncoderLayer,
    TriangularCausalMask,
)
from neuralforecast.models import autoformer, fedformer, timexer


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


@pytest.mark.parametrize("revin_cls", [RevIN, RevINMultivariate])
@pytest.mark.parametrize(
    "weight", [-1.0, -1e-5, -1e-10, -1e-12, 0.0, 1e-12, 1e-10, 1e-5, 1.0]
)
def test_revin_affine_denorm_remains_finite(revin_cls, weight):
    torch.manual_seed(42)
    model = revin_cls(num_features=3, affine=True)
    with torch.no_grad():
        model.affine_weight.fill_(weight)

    x = torch.randn(2, 10, 3)
    restored = model(model(x, "norm"), "denorm")

    assert torch.isfinite(restored).all()


@pytest.mark.parametrize("revin_cls", [RevIN, RevINMultivariate])
@pytest.mark.parametrize("weight", [-1e-5, -5e-6, 0.0, 5e-6, 1e-5])
def test_revin_affine_denorm_uses_signed_floor(revin_cls, weight):
    eps = 1e-5
    model = revin_cls(num_features=3, eps=eps, affine=True)
    with torch.no_grad():
        model.affine_weight.fill_(weight)
        model.affine_bias.fill_(0.2)

    x = torch.randn(2, 10, 3)
    model(x, "norm")
    z = torch.randn_like(x)
    actual = model(z, "denorm")

    if isinstance(model, RevIN):
        mean, std = model.mean, model.stdev
    else:
        mean, std = model.batch_mean, model.batch_std
    denominator = -eps if weight < 0 else eps
    expected = ((z - model.affine_bias) / denominator) * std + mean

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("revin_cls", [RevIN, RevINMultivariate])
@pytest.mark.parametrize("weight", [-1e-10, 0.0, 1e-10])
def test_revin_singular_affine_weights_have_finite_gradients(revin_cls, weight):
    torch.manual_seed(42)
    model = revin_cls(num_features=3, affine=True)
    with torch.no_grad():
        model.affine_weight.fill_(weight)

    x = torch.randn(2, 10, 3)
    y = model(model(x, "norm"), "denorm")
    y.square().mean().backward()

    assert torch.isfinite(y).all()
    assert torch.isfinite(model.affine_weight.grad).all()


@pytest.mark.parametrize("revin_cls", [RevIN, RevINMultivariate])
@pytest.mark.parametrize("weight", [-1.0, -0.5, 0.5, 1.0])
def test_revin_regular_affine_weights_preserve_roundtrip(revin_cls, weight):
    torch.manual_seed(42)
    model = revin_cls(num_features=3, affine=True)
    with torch.no_grad():
        model.affine_weight.fill_(weight)
        model.affine_bias.fill_(0.2)

    x = torch.randn(2, 10, 3)
    restored = model(model(x, "norm"), "denorm")

    torch.testing.assert_close(restored, x, atol=1e-5, rtol=1e-5)


def test_revin_multivariate_non_affine_roundtrip():
    torch.manual_seed(42)
    model = RevINMultivariate(num_features=3, affine=False)
    x = torch.randn(2, 10, 3)

    restored = model(model(x, "norm"), "denorm")

    torch.testing.assert_close(restored, x, atol=1e-5, rtol=1e-5)


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


@pytest.mark.parametrize("revin_cls", [RevIN, RevINMultivariate])
@pytest.mark.parametrize("weight", [-1e-5, 0.0, 1e-5])
def test_revin_clamped_denorm_has_no_affine_weight_gradient(revin_cls, weight):
    model = revin_cls(num_features=3, eps=1e-5, affine=True)
    with torch.no_grad():
        model.affine_weight.fill_(weight)

    x = torch.randn(2, 10, 3)
    model(x, "norm")
    model(torch.randn_like(x), "denorm").sum().backward()

    torch.testing.assert_close(
        model.affine_weight.grad, torch.zeros_like(model.affine_weight)
    )


_FFN_LAYERS = {
    "TransEncoderLayer": lambda: TransEncoderLayer(None, 8, 16),
    "TransDecoderLayer": lambda: TransDecoderLayer(None, None, 8, 16),
    "Autoformer.EncoderLayer": lambda: autoformer.EncoderLayer(None, 8, 16),
    "Autoformer.DecoderLayer": lambda: autoformer.DecoderLayer(None, None, 8, 1, 16),
    "FEDformer.EncoderLayer": lambda: fedformer.EncoderLayer(None, 8, 16),
    "FEDformer.DecoderLayer": lambda: fedformer.DecoderLayer(None, None, 8, 1, 16),
    "TimeXer.EncoderLayer": lambda: timexer.EncoderLayer(None, None, 8, 16),
}


def _as_conv1x1_state_dict(module):
    """State dict as saved when the FFN used `nn.Conv1d(kernel_size=1)`."""
    return {
        k: v.unsqueeze(-1) if k.endswith(("conv1.weight", "conv2.weight")) else v
        for k, v in module.state_dict().items()
    }


@pytest.mark.parametrize("make_layer", _FFN_LAYERS.values(), ids=_FFN_LAYERS.keys())
def test_transformer_ffn_loads_conv1x1_checkpoints(make_layer):
    torch.manual_seed(0)
    source = nn.Sequential(make_layer())
    legacy = _as_conv1x1_state_dict(source)
    assert legacy["0.conv1.weight"].dim() == 3

    target = nn.Sequential(make_layer())
    target.load_state_dict(legacy, strict=True)

    for name in ("conv1", "conv2"):
        assert isinstance(getattr(target[0], name), nn.Linear)
        torch.testing.assert_close(
            getattr(target[0], name).weight, getattr(source[0], name).weight
        )


class _ZeroAttention(nn.Module):
    def forward(self, queries, keys, values, attn_mask=None):
        return torch.zeros_like(queries), None


@pytest.mark.parametrize("training", [False, True])
def test_trans_encoder_layer_matches_conv1x1_ffn(training):
    """Same output as the former Conv1d FFN, including seeded dropout masks."""
    p = 0.3
    torch.manual_seed(0)
    layer = TransEncoderLayer(_ZeroAttention(), 8, 16, dropout=p).train(training)
    legacy = _as_conv1x1_state_dict(layer)
    inputs = torch.randn(4, 6, 8)

    torch.manual_seed(1)
    x = inputs + F.dropout(torch.zeros_like(inputs), p, training)
    x = layer.norm1(x)
    y = F.conv1d(x.transpose(-1, 1), legacy["conv1.weight"], legacy["conv1.bias"])
    y = F.dropout(F.relu(y), p, training)
    y = F.conv1d(y, legacy["conv2.weight"], legacy["conv2.bias"]).transpose(-1, 1)
    expected = layer.norm2(x + F.dropout(y, p, training))

    torch.manual_seed(1)
    out, _ = layer(inputs)
    torch.testing.assert_close(out, expected)
