"""Regression coverage for finite mLSTM gate normalization."""

import pytest
import torch

from neuralforecast.models.xlstm import _finite_mlstm_parallel, xLSTM


def inputs(dtype=torch.float64, gate=None):
    generator = torch.Generator().manual_seed(42)
    tensors = [
        torch.randn(1, 2, 4, 3, dtype=dtype, generator=generator) for _ in range(3)
    ]
    tensors += [
        torch.randn(1, 2, 4, 1, dtype=dtype, generator=generator) for _ in range(2)
    ]
    if gate is not None:
        tensors[3].fill_(gate)
    return [value.requires_grad_() for value in tensors]


@pytest.mark.parametrize("rowwise", [True, False])
def test_matches_reference_outputs_and_gradients(rowwise):
    backend = pytest.importorskip("xlstm.blocks.mlstm.backends")
    original = inputs()
    stabilized = [v.detach().clone().requires_grad_() for v in original]
    expected = backend.parallel_stabilized_simple(*original, stabilize_rowwise=rowwise)
    actual = _finite_mlstm_parallel(*stabilized, stabilize_rowwise=rowwise)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for old, new in zip(original, stabilized):
        torch.testing.assert_close(new.grad, old.grad, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("gate", [-1000.0, -100.0, 1000.0])
def test_extreme_gates_have_finite_forward_and_backward(gate):
    values = inputs(torch.float32, gate)
    output = _finite_mlstm_parallel(*values)
    output.sum().backward()
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(v.grad).all() for v in values)


def test_future_values_do_not_change_prior_outputs():
    original = inputs()
    changed = [v.detach().clone() for v in original]
    for tensor in changed:
        tensor[:, :, -1, :] += 1000
    torch.testing.assert_close(
        _finite_mlstm_parallel(*original)[:, :, :-1],
        _finite_mlstm_parallel(*changed)[:, :, :-1],
    )


def test_stability_is_local_to_model_and_legacy_is_available():
    cell = pytest.importorskip("xlstm.blocks.mlstm.cell")
    model = xLSTM(h=2, input_size=8, encoder_hidden_size=32, encoder_n_blocks=1)
    legacy = xLSTM(
        h=2,
        input_size=8,
        encoder_hidden_size=32,
        encoder_n_blocks=1,
        numerical_stability=False,
    )
    assert all(
        m.backend_fn is _finite_mlstm_parallel
        for m in model.modules()
        if isinstance(m, cell.mLSTMCell)
    )
    assert all(
        m.backend_fn is not _finite_mlstm_parallel
        for m in legacy.modules()
        if isinstance(m, cell.mLSTMCell)
    )
