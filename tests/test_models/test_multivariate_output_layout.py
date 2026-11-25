import math

import pytest
import torch
import torch.nn.functional as F

from neuralforecast.losses.pytorch import DistributionLoss, GMM, MAE, MQLoss, NBMM, PMM
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


@pytest.mark.parametrize("model_cls,config,head_name", MODEL_CONFIGS)
@pytest.mark.parametrize("loss_cls", [GMM, PMM, NBMM])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("scaler_type", ["identity", "standard", "minmax1"])
def test_distribution_window_normalization(
    model_cls, config, head_name, loss_cls, weighted, scaler_type
):
    # Two different windows and target levels expose double scaling and shifts
    # of dispersion/weight channels. Exercise the real model and scaler paths.
    torch.manual_seed(12)
    loss = loss_cls(n_components=2, weighted=weighted)
    model = (
        model_cls(
            h=2, input_size=4, n_series=2, loss=loss, scaler_type=scaler_type, **config
        )
        .double()
        .eval()
    )
    y = torch.tensor(
        [
            [[1.0, 100.0], [2.0, 120.0], [3.0, 140.0], [4.0, 160.0]],
            [[20.0, 600.0], [22.0, 610.0], [24.0, 620.0], [26.0, 630.0]],
        ],
        dtype=torch.float64,
    )
    # RMoK's Jacobi expert allocates float32 intermediates.
    if model_cls is RMoK:
        model = model.float()
        y = y.float()
    z = model.scaler.transform(y.unsqueeze(2), torch.ones_like(y.unsqueeze(2)))
    z = z[:, :, 0, :]
    q = loss.outputsize_multiplier
    if model_cls is TSMixerx:
        shape = (2, 2, 2 * q)
    elif model_cls is RMoK:
        shape = (2, 2 * q, 2)
    else:
        shape = (2, 2, 2 * q)
    raw = torch.linspace(-0.5, 0.8, math.prod(shape), dtype=y.dtype)
    raw = raw.reshape(shape).requires_grad_()
    if model_cls is RMoK:
        handles = [
            expert.register_forward_hook(
                lambda module, args, output: raw.permute(0, 2, 1).reshape(4, 2 * q)
            )
            for expert in model.experts
        ]
        # Equal, exactly representable gate weights isolate the normalization
        # from float32 softmax rounding in RMoK's four-expert reduction.
        handles.append(
            model.gate.register_forward_hook(
                lambda module, args, output: torch.zeros_like(output)
            )
        )
    else:
        handles = [
            getattr(model, head_name).register_forward_hook(
                lambda module, args, output: raw
            )
        ]
    norm = getattr(model, "norm", getattr(model, "rev", None))
    if norm is not None and norm.affine:
        with torch.no_grad():
            norm.affine_weight.copy_(torch.tensor([[[-2.0, 3.0]]]))
            norm.affine_bias.copy_(torch.tensor([[[0.1, -0.2]]]))
    try:
        output = model(
            dict(insample_y=z.clone(), hist_exog=None, futr_exog=None, stat_exog=None)
        )
    finally:
        for handle in handles:
            handle.remove()
    actual = model._distribution_args(loss.domain_map(output), y_idx=0)

    # Scalar indexing is independent of the production packing helper. Derive
    # each link directly, including the count losses' non-Gaussian conventions.
    params = torch.empty(2, 2, 2, q, dtype=y.dtype)
    for b in range(2):
        for t in range(2):
            for n in range(2):
                for k in range(q):
                    if model_cls is TSMixerx:
                        value = raw[b, t, n * q + k]
                    elif model_cls is RMoK:
                        value = raw[b, t * q + k, n]
                    else:
                        value = raw[b, n, t * q + k]
                    params[b, t, n, k] = value
    # Map the window statistics back with the scaler's own inverse function
    def inverse(values):
        return model.scaler.inverse_scaler(
            values, model.scaler.x_shift[:, :, 0, :], model.scaler.x_scale[:, :, 0, :]
        )

    # minmax1 maps x to 2 (x - x_min) / x_range - 1, so its inverse has slope x_range / 2
    slope = model.scaler.x_scale[:, :, 0, :] / (2 if scaler_type == "minmax1" else 1)
    loc = inverse(z.mean(1, keepdim=True)).unsqueeze(-1)
    scale = (slope * (z.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()).unsqueeze(-1)
    first = params[..., :2]
    if loss_cls is GMM:
        expected = (first * scale + loc, (F.softplus(params[..., 2:4]) + 0.2) * scale)
    elif loss_cls is PMM:
        expected = (F.softplus(first * scale + loc) + 0.001,)
    else:
        mu = F.softplus(first * scale + loc) + 1e-8
        alpha = (F.softplus(params[..., 2:4]) + 1e-8) / (scale + 1.0)
        expected = (1.0 / alpha, mu * alpha / (1.0 + mu * alpha) + 1e-8)
    if weighted:
        expected += (params[..., -2:].softmax(-1),)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want)

    target = y[:, -1:, :].expand(-1, 2, -1).round()
    # Evaluate both density formulas in double precision to avoid cancellation
    # of float32 log-gamma terms in the independent negative-binomial formula.
    actual_logprob = loss.get_distribution(tuple(a.double() for a in actual)).log_prob(
        target.double()
    )
    expected = tuple(a.double() for a in expected)
    x = target.double().unsqueeze(-1)
    if loss_cls is GMM:
        mean, std = expected[:2]
        component_lp = (
            -0.5 * ((x - mean) / std).square() - std.log() - 0.5 * math.log(2 * math.pi)
        )
    elif loss_cls is PMM:
        rate = expected[0]
        component_lp = x * rate.log() - rate - torch.lgamma(x + 1)
    else:
        count, prob = expected[:2]
        component_lp = (
            torch.lgamma(x + count)
            - torch.lgamma(count)
            - torch.lgamma(x + 1)
            + count * torch.log1p(-prob)
            + x * prob.log()
        )
    weights = expected[-1] if weighted else torch.full_like(component_lp, 0.5)
    reference = torch.logsumexp(component_lp + weights.log(), dim=-1)
    torch.testing.assert_close(actual_logprob, reference)
    actual_grad = torch.autograd.grad(actual_logprob.sum(), raw, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(reference.sum(), raw)[0]
    torch.testing.assert_close(actual_grad, reference_grad)


@pytest.mark.parametrize(
    "model_cls,config,head_name", [c for c in MODEL_CONFIGS if c[0] is not RMoK]
)
def test_disabled_normalization_uses_scaler_statistics(model_cls, config, head_name):
    flag = "revin" if model_cls in (TSMixer, TSMixerx) else "use_norm"
    loss = GMM(n_components=2)
    model = model_cls(
        h=2, input_size=4, n_series=2, loss=loss, **config, **{flag: False}
    ).eval()
    y = torch.tensor([[[1.0, 100.0], [2.0, 120.0], [3.0, 140.0], [4.0, 160.0]]])
    model.scaler.transform(y.unsqueeze(2), torch.ones(1, 4, 1, 2))
    output = loss.domain_map(
        model(dict(insample_y=y, hist_exog=None, futr_exog=None, stat_exog=None))
    )
    assert model._distribution_output_scale == (None, None)
    expected = loss.scale_decouple(output, *model._get_loc_scale(0))
    for got, want in zip(model._distribution_args(output, 0), expected):
        torch.testing.assert_close(got, want)


@pytest.mark.parametrize("distribution", ["Normal", "StudentT"])
@pytest.mark.parametrize("model_cls,config,head_name", MODEL_CONFIGS)
def test_single_distribution_links(model_cls, config, head_name, distribution):
    # DistributionLoss has parameter-major packing, unlike mixtures. With an
    # all-zero learned head the linked mean must be the window mean, dispersion
    # must depend on window std alone, and Student-t df must be level-independent.
    loss = DistributionLoss(distribution=distribution)
    model = model_cls(h=2, input_size=4, n_series=2, loss=loss, **config).eval()
    with torch.no_grad():
        norm = getattr(model, "norm", getattr(model, "rev", None))
        if norm is not None and norm.affine:
            norm.affine_weight.fill_(2.0)
            norm.affine_bias.fill_(0.3)
    y = torch.tensor([[[1.0, 100.0], [2.0, 120.0], [3.0, 140.0], [4.0, 160.0]]])
    model.scaler.transform(y.unsqueeze(2), torch.ones(1, 4, 1, 2))
    heads = model.experts if model_cls is RMoK else [getattr(model, head_name)]
    handles = [
        head.register_forward_hook(
            lambda module, args, output: torch.zeros_like(output)
        )
        for head in heads
    ]
    try:
        output = model(
            dict(insample_y=y, hist_exog=None, futr_exog=None, stat_exog=None)
        )
    finally:
        for handle in handles:
            handle.remove()
    args = model._distribution_args(loss.domain_map(output), 0)
    mean = y.mean(1, keepdim=True).expand(-1, 2, -1)
    std = (y.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
    eps = 0.1 if distribution == "StudentT" else 0.2
    scale = ((math.log(2.0) + eps) * std).expand_as(mean)
    torch.testing.assert_close(args[-2], mean)
    torch.testing.assert_close(args[-1], scale)
    if distribution == "StudentT":
        torch.testing.assert_close(args[0], torch.full_like(mean, 3.0 + math.log(2.0)))
