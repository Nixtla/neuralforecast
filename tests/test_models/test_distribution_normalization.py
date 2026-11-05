import math

import pytest
import torch
import torch.nn.functional as F

from neuralforecast.losses.pytorch import DistributionLoss, GMM, NBMM, PMM
from neuralforecast.models import RMoK, TSMixer, TSMixerx

from .test_multivariate_output_layout import MODEL_CONFIGS


@pytest.mark.parametrize("model_cls,config,head_name", MODEL_CONFIGS)
@pytest.mark.parametrize("loss_cls", [GMM, PMM, NBMM])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("scaler_type", ["identity", "standard"])
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
    base_loc = model.scaler.x_shift[:, :, 0, :]
    base_scale = model.scaler.x_scale[:, :, 0, :]
    loc = (base_loc + base_scale * z.mean(1, keepdim=True)).unsqueeze(-1)
    scale = (
        base_scale * (z.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
    ).unsqueeze(-1)
    first = params[..., :2]
    if loss_cls is GMM:
        expected = (first * scale + loc, (F.softplus(params[..., 2:4]) + 0.2) * scale)
    elif loss_cls is PMM:
        expected = (F.softplus(first * scale + loc) + 0.001,)
    else:
        mu = (F.softplus(first) + 1e-8) * loc
        alpha = (F.softplus(params[..., 2:4]) + 1e-8) / (loc + 1.0)
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


def test_disabled_normalization_clears_window_statistics():
    model = TSMixer(
        h=2, input_size=4, n_series=2, loss=GMM(n_components=2), n_block=1, ff_dim=8
    ).eval()
    batch = dict(
        insample_y=torch.arange(8.0).reshape(1, 4, 2),
        hist_exog=None,
        futr_exog=None,
        stat_exog=None,
    )
    model(batch)
    model.revin = False
    output = model(batch)
    model.scaler.transform(batch["insample_y"].unsqueeze(2), torch.ones(1, 4, 1, 2))
    args = model._distribution_args(model.loss.domain_map(output), 0)
    expected = model.loss.scale_decouple(
        model.loss.domain_map(output), *model._get_loc_scale(0)
    )
    for got, want in zip(args, expected):
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
