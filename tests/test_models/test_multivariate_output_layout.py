import pytest
import torch

from neuralforecast.losses.pytorch import DistributionLoss, GMM, MAE, MQLoss
from neuralforecast.models import (
    RMoK,
    SOFTS,
    SOFTSSharp,
    TSMixer,
    TSMixerx,
    TimeXer,
    XLinear,
    iTransformer,
)


MODEL_CONFIGS = [
    (TSMixer, dict(n_block=1, ff_dim=8), "out"),
    (TSMixerx, dict(n_block=1, ff_dim=8), "out"),
    (iTransformer, dict(hidden_size=8, n_heads=2, e_layers=1, d_ff=8), "projector"),
    (SOFTS, dict(hidden_size=8, d_core=4, e_layers=1, d_ff=8), "projection"),
    (SOFTSSharp, dict(hidden_size=8, d_core=4, e_layers=1, d_ff=8), "projection"),
    (TimeXer, dict(patch_len=2, hidden_size=8, n_heads=2, e_layers=1, d_ff=8), "head"),
    (XLinear, dict(hidden_size=8, temporal_ff=8, channel_ff=4), "head"),
    (RMoK, dict(taylor_order=2, jacobi_degree=2), "rev"),
]


@pytest.mark.parametrize("model_cls,config,head_name", MODEL_CONFIGS)
@pytest.mark.parametrize(
    "loss_cls",
    [
        MAE,
        MQLoss,
        GMM,
        pytest.param(lambda: DistributionLoss(distribution="Normal"), id="Normal"),
        pytest.param(lambda: DistributionLoss(distribution="StudentT"), id="StudentT"),
    ],
)
@pytest.mark.parametrize("normalize", [True, False])
def test_multivariate_output_layout(model_cls, config, head_name, loss_cls, normalize):
    # Different batch statistics and N != Q expose swaps hidden by one series,
    # point losses or equally scaled series. Use the real model forward path.
    batch_size, h, n_series = 2, 2, 2
    loss = loss_cls(n_components=2) if loss_cls is GMM else loss_cls()
    q = loss.outputsize_multiplier
    config = config.copy()
    if model_cls is RMoK:
        if not normalize:
            pytest.skip("RMoK always uses normalization")
    else:
        config["revin" if model_cls in (TSMixer, TSMixerx) else "use_norm"] = normalize
    model = model_cls(h=h, input_size=4, n_series=n_series, loss=loss, **config)
    model = model.float().eval()
    insample = torch.tensor(
        [
            [[1.0, 100.0], [2.0, 120.0], [3.0, 140.0], [4.0, 160.0]],
            [[10.0, -50.0], [12.0, -40.0], [14.0, -30.0], [16.0, -20.0]],
        ],
        dtype=torch.float32,
    )
    mean = insample.mean(1)
    std = (insample.var(1, unbiased=False) + 1e-5).sqrt()

    # Inject known outputs at the learned head, retaining time, series and
    # output identities. Expected values below use scalar indexing, independent
    # of the model's output reshapes.
    if model_cls is TSMixerx:
        shape = (batch_size, h, n_series * q)
    elif model_cls is RMoK:
        shape = (batch_size, h * q, n_series)
    else:
        shape = (batch_size, n_series, h * q)
    raw = torch.arange(torch.tensor(shape).prod(), dtype=torch.float32) / 10
    raw = raw.reshape(shape).requires_grad_()

    if model_cls is RMoK:

        # Every expert returns the same known tensor, so the softmax gate
        # preserves it, including when distribution heads skip RevIN's inverse.
        handles = [
            expert.register_forward_hook(
                lambda module, args, output: raw.permute(0, 2, 1).reshape(
                    batch_size * n_series, h * q
                )
            )
            for expert in model.experts
        ]
    else:
        handles = [getattr(model, head_name).register_forward_hook(
            lambda module, args, output: raw
        )]

    norm = getattr(model, "norm", getattr(model, "rev", None))
    if not normalize:
        norm = None
    if norm is not None:
        with torch.no_grad():
            norm.affine_weight.copy_(torch.tensor([[[2.0, 3.0]]]))
            norm.affine_bias.copy_(torch.tensor([[[0.1, 0.2]]]))

    try:
        actual = model(
            dict(
                insample_y=insample.clone(),
                hist_exog=None,
                futr_exog=None,
                stat_exog=None,
            )
        )
    finally:
        for handle in handles:
            handle.remove()

    expected = torch.empty(batch_size, h, n_series, q, dtype=torch.float32)
    for b in range(batch_size):
        for t in range(h):
            for n in range(n_series):
                for k in range(q):
                    if model_cls is TSMixerx:
                        index = (
                            k * n_series + n
                            if isinstance(loss, DistributionLoss)
                            else n * q + k
                        )
                        value = raw[b, t, index]
                    elif model_cls is RMoK:
                        value = raw[b, t * q + k, n]
                    else:
                        value = raw[b, n, t * q + k]
                    if norm is not None and not loss.is_distribution_output:
                        value = (
                            value - norm.affine_bias[0, 0, n]
                        ) / norm.affine_weight[0, 0, n]
                    if normalize and not loss.is_distribution_output:
                        value = value * std[b, n] + mean[b, n]
                    expected[b, t, n, k] = value

    if isinstance(loss, DistributionLoss):
        actual = torch.stack(loss.domain_map(actual), dim=-1)
    else:
        actual = actual.reshape_as(expected)
    torch.testing.assert_close(actual, expected)
    weights = torch.arange(expected.numel(), dtype=torch.float32).reshape_as(expected)
    actual_grad = torch.autograd.grad(
        (actual.reshape_as(expected) * weights).sum(), raw
    )[0]
    expected_grad = torch.autograd.grad((expected * weights).sum(), raw)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
