"""Numerical equivalence checks for batched and fused ES-RNN execution."""

from copy import deepcopy

import pytest
import torch


def reference_scan(y, coeff, initial):
    """Frozen scalar recurrence from the original implementation, kept independent."""
    seasons = [list(v.unbind()) for v in initial]
    product = y.new_ones(())
    for s in seasons:
        product = product * s[0]
        s.append(s[0])
    levels = [y[0] / product]
    for t in range(1, len(y)):
        product = y.new_ones(())
        for s in seasons:
            product = product * s[t]
        level = coeff[0] * y[t] / product + (1 - coeff[0]) * levels[-1]
        levels.append(level)
        for j, s in enumerate(seasons):
            other = y.new_ones(())
            for k, v in enumerate(seasons):
                if k != j:
                    other = other * v[t]
            s.append(coeff[j + 1] * y[t] / (level * other) + (1 - coeff[j + 1]) * s[t])
    return torch.stack(levels), [torch.stack(s) for s in seasons]


@pytest.mark.parametrize("periods", [(), (2,), (2, 3)])
@pytest.mark.parametrize("backend", ["torch", "native"])
def test_batched_scan_values_and_all_input_gradients(periods, backend):
    from neuralforecast.models.esrnn import _ExponentialSmoothing

    torch.manual_seed(123)
    fast = _ExponentialSmoothing(3, periods, backend=backend).double()
    with torch.no_grad():
        fast.smoothing.normal_()
        for p in fast.initial_seasons:
            p.normal_(0, 0.2)
    y = (torch.rand(3, 12, dtype=torch.double) + 1).requires_grad_()
    rows = torch.tensor(
        [2, 0, 2]
    )  # Also verify repeated local rows accumulate gradients.
    levels, seasons = fast.forward_batch(y, rows)
    weights = [torch.randn_like(v) for v in [levels, *seasons]]
    loss = sum((v * w).sum() for v, w in zip([levels, *seasons], weights))
    actual = torch.autograd.grad(loss, [y, *fast.parameters()])
    slow = deepcopy(fast)
    z = y.detach().clone().requires_grad_()
    reference_levels, reference_seasons = [], [[] for _ in periods]
    for i, row in enumerate(rows):
        coeff = slow.smoothing[row].sigmoid()
        initial = [
            m * p[row].softmax(-1) for m, p in zip(periods, slow.initial_seasons)
        ]
        l, s = reference_scan(z[i], coeff, initial)
        reference_levels.append(l)
        for j in range(len(periods)):
            reference_seasons[j].append(s[j])
    expected = [
        torch.stack(reference_levels),
        *[torch.stack(v) for v in reference_seasons],
    ]
    for a, b in zip([levels, *seasons], expected):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)
    reference_loss = sum((v * w).sum() for v, w in zip(expected, weights))
    gradients = torch.autograd.grad(reference_loss, [z, *slow.parameters()])
    for a, b in zip(actual, gradients):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)


def test_fused_scan_finite_difference_gradcheck():
    from neuralforecast.common._esrnn import native_scan

    y = torch.tensor([[1.0, 2.0, 1.5, 3.0]], dtype=torch.double, requires_grad=True)
    coeff = torch.tensor([[0.4, 0.3, 0.6]], dtype=torch.double, requires_grad=True)
    s1 = torch.tensor([[0.7, 1.3]], dtype=torch.double, requires_grad=True)
    s2 = torch.tensor([[0.8, 1.0, 1.2]], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(
        native_scan, (y, coeff, s1, s2), eps=1e-6, atol=1e-5, rtol=1e-4
    )


@pytest.mark.parametrize("periods", [(), (2,), (2, 3)])
def test_batched_network_matches_independent_serial_windows(periods):
    from neuralforecast.models import ESRNN

    torch.manual_seed(1)
    model = ESRNN(
        h=5,
        input_size=3,
        season_length=periods,
        n_series=3,
        encoder_hidden_size=4,
        dilations=((1, 2), (4,)),
        es_backend="native",
    ).double()
    y = torch.rand(3, 18, dtype=torch.double) + 2
    origins = [[7, 9, 12], [8, 13, 18], [6, 8, 10]]
    actual, scales = model._forecast_batch(y, torch.tensor([0, 1, 2]), origins)
    for i, requested in enumerate(origins):
        l, s = reference_scan(
            y[i, : max(requested)],
            model.es.smoothing[i].sigmoid(),
            [m * p[i].softmax(-1) for m, p in zip(periods, model.es.initial_seasons)],
        )
        inputs = []
        for o in range(model._first_origin, max(requested) + 1):
            values = y[i, o - model.input_size : o] / l[o - 1]
            for season in s:
                values = values / season[o - model.input_size : o]
            inputs.append(values.log())
        hidden = torch.stack(inputs).unsqueeze(0)
        for j, group in enumerate(model.encoder):
            encoded, _ = group(hidden)
            hidden = encoded if j == 0 else hidden + encoded
        prediction = (
            model.decoder(hidden[0, torch.tensor(requested) - model._first_origin])
            .clamp(-20, 20)
            .exp()
        )
        torch.testing.assert_close(
            actual[i, : len(requested)], prediction, rtol=1e-10, atol=1e-10
        )
        expected_scale = torch.stack(
            [model.es.forecast_scale(l, s, o, model.h) for o in requested]
        )
        torch.testing.assert_close(
            scales[i, : len(requested)], expected_scale, rtol=1e-10, atol=1e-10
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("hidden", [4, 9])
def test_fused_lstm_matches_torch_outputs_and_parameter_gradients(dtype, hidden):
    from neuralforecast.common._esrnn import native_lstm

    torch.manual_seed(22)
    cell = torch.nn.LSTM(3, hidden).to(dtype)
    x = torch.randn(7, 3, 3, dtype=dtype, requires_grad=True)
    params = [cell.weight_ih_l0, cell.weight_hh_l0, cell.bias_ih_l0, cell.bias_hh_l0]
    actual = native_lstm(x, *params)
    expected, _ = cell(x)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    weights = torch.randn_like(actual)
    ga = torch.autograd.grad((actual * weights).sum(), [x, *params], retain_graph=True)
    ge = torch.autograd.grad((expected * weights).sum(), [x, *params])
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, rtol=4e-5, atol=4e-6)


def test_fused_lstm_finite_difference_gradcheck():
    from neuralforecast.common._esrnn import native_lstm

    torch.manual_seed(2)
    x = torch.randn(3, 2, 2, dtype=torch.double, requires_grad=True)
    cell = torch.nn.LSTM(2, 2).double()
    assert torch.autograd.gradcheck(
        native_lstm,
        (x, cell.weight_ih_l0, cell.weight_hh_l0, cell.bias_ih_l0, cell.bias_hh_l0),
        atol=1e-5,
        rtol=1e-4,
    )


@pytest.mark.parametrize("penalty", [0.0, 50.0])
def test_training_scans_once_and_batches_each_recurrent_group(monkeypatch, penalty):
    import pandas as pd
    from neuralforecast.models import ESRNN
    from neuralforecast.tsdataset import TimeSeriesDataset, TimeSeriesLoader

    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": str(i),
                    "ds": range(16),
                    "y": [2 + i + 0.1 * t + t % 3 for t in range(16)],
                }
            )
            for i in range(4)
        ]
    )
    dataset, *_ = TimeSeriesDataset.from_df(df)
    batch = next(iter(TimeSeriesLoader(dataset, batch_size=4)))
    model = ESRNN(
        h=2,
        input_size=3,
        season_length=2,
        n_series=4,
        encoder_hidden_size=4,
        dilations=((1, 2), (4,)),
        windows_batch_size=3,
        level_variability_penalty=penalty,
        es_backend="native",
    )
    model._bind_dataset(dataset, fitting=True, tail=0)
    model.val_size = model.test_size = 0
    scans = []
    recurrent = []
    original = model.es.forward_batch

    def scan(*args, **kwargs):
        scans.append(args[0].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(model.es, "forward_batch", scan)
    for group in model.encoder:
        original_group = group.forward

        def run(*args, _original=original_group, **kwargs):
            recurrent.append(args[0].shape[0])
            return _original(*args, **kwargs)

        monkeypatch.setattr(group, "forward", run)
    if penalty == 0:

        def unexpected(*args):
            raise AssertionError("Zero-weight penalty should never be evaluated")

        monkeypatch.setattr(model.es, "level_penalty", unexpected)
    model.training_step(batch, 0).backward()
    assert scans == [torch.Size([4, 16])]
    assert recurrent == [4, 4]
    assert torch.isfinite(model.es.smoothing.grad).all()


def test_auto_backend_falls_back_without_compiler(monkeypatch):
    import neuralforecast.common._esrnn as kernels
    from neuralforecast.models import ESRNN

    monkeypatch.setattr(kernels, "_NATIVE", None)
    monkeypatch.setattr(kernels, "_NATIVE_ERROR", RuntimeError("compiler missing"))
    model = ESRNN(
        h=2,
        input_size=3,
        season_length=2,
        es_backend="auto",
        encoder_hidden_size=4,
        dilations=((1,),),
    )
    y = torch.arange(10, dtype=torch.float32) + 2
    prediction, scale = model._forecast_series(y, 0, [5, 8])
    (prediction * scale).sum().backward()
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(model.es.smoothing.grad).all()


def test_native_scan_promotes_mixed_input_parameter_dtypes():
    from neuralforecast.models.esrnn import _ExponentialSmoothing

    model = _ExponentialSmoothing(1, (2,), backend="native").double()
    y = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    levels, seasons = model(y, 0)
    expected = reference_scan(
        y.double(),
        model.smoothing[0].sigmoid(),
        [2 * model.initial_seasons[0][0].softmax(-1)],
    )
    torch.testing.assert_close(levels, expected[0])
    (levels.sum() + seasons[0].sum()).backward()
    assert y.grad.dtype == torch.float32 and torch.isfinite(y.grad).all()


def test_parallel_lstm_matches_serial_values_and_gradients():
    from neuralforecast.common._esrnn import native_lstm

    torch.manual_seed(44)
    previous = torch.get_num_threads()
    try:
        x = torch.randn(11, 4, 3, dtype=torch.double, requires_grad=True)
        cell = torch.nn.LSTM(3, 9).double()
        parameters = [
            x,
            cell.weight_ih_l0,
            cell.weight_hh_l0,
            cell.bias_ih_l0,
            cell.bias_hh_l0,
        ]
        torch.set_num_threads(1)
        serial = native_lstm(*parameters)
        weights = torch.randn_like(serial)
        gs = torch.autograd.grad((serial * weights).sum(), parameters)
        torch.set_num_threads(4)
        parallel = native_lstm(*parameters)
        gp = torch.autograd.grad((parallel * weights).sum(), parameters)
        torch.testing.assert_close(serial, parallel, rtol=1e-11, atol=1e-11)
        for a, b in zip(gs, gp):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    finally:
        torch.set_num_threads(previous)


def test_unequal_history_training_loss_and_gradients_match_serial_reference():
    import pandas as pd
    from neuralforecast.models import ESRNN
    from neuralforecast.tsdataset import TimeSeriesDataset, TimeSeriesLoader

    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": str(i),
                    "ds": range(n),
                    "y": [2 + i + 0.2 * t + t % 3 for t in range(n)],
                }
            )
            for i, n in enumerate([10, 13, 16])
        ]
    )
    ds, *_ = TimeSeriesDataset.from_df(df)
    batch = next(iter(TimeSeriesLoader(ds, batch_size=3)))
    fast = ESRNN(
        h=2,
        input_size=3,
        season_length=(2, 3),
        n_series=3,
        encoder_hidden_size=4,
        dilations=((1, 2), (4,)),
        windows_batch_size=-1,
        es_backend="native",
    ).double()
    fast._bind_dataset(ds, fitting=True, tail=0)
    fast.val_size = fast.test_size = 0
    slow = deepcopy(fast)
    for group in slow.encoder:
        group.backend = "torch"
    expected = []
    for full_y, row in slow._batch_series(batch):
        y = full_y.double()
        levels, seasons = reference_scan(
            y,
            slow.es.smoothing[row].sigmoid(),
            [
                m * p[row].softmax(-1)
                for m, p in zip(slow.season_lengths, slow.es.initial_seasons)
            ],
        )
        origins = list(range(slow._first_origin, len(y) - slow.h + 1))
        inputs = []
        for origin in origins:
            values = y[origin - slow.input_size : origin] / levels[origin - 1]
            for factors in seasons:
                values = values / factors[origin - slow.input_size : origin]
            inputs.append(values.log())
        hidden = torch.stack(inputs).unsqueeze(0)
        for i, group in enumerate(slow.encoder):
            encoded, _ = group(hidden)
            hidden = encoded if i == 0 else hidden + encoded
        normalized = slow.decoder(hidden[0]).clamp(-20, 20).exp()
        scale = torch.stack(
            [slow.es.forecast_scale(levels, seasons, o, slow.h) for o in origins]
        )
        targets = torch.stack([y[o : o + slow.h] for o in origins])
        expected.append(
            slow.loss(y=(targets / scale).unsqueeze(-1), y_hat=normalized.unsqueeze(-1))
            + slow.level_variability_penalty * slow.es.level_penalty(levels)
        )
    expected = torch.stack(expected).mean()
    actual = fast.training_step(batch, 0)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    actual.backward()
    expected.backward()
    for a, b in zip(fast.parameters(), slow.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-8, atol=1e-8)


def test_backend_option_preserves_existing_positional_constructor():
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import ESRNN

    loss = MAE()
    model = ESRNN(2, 4, (), 1, 8, ((1,),), 50.0, loss)
    assert model.loss is loss
    assert model.es_backend == "auto"
