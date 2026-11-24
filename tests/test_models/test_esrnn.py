"""Numerical and public API checks for the local/global ES-RNN hybrid."""

import pytest
import torch


@pytest.mark.parametrize("periods", [(), (2,), (2, 3)])
def test_es_constant_series_and_long_seasonal_extrapolation(periods):
    from neuralforecast.models.esrnn import _ExponentialSmoothing

    es = _ExponentialSmoothing(1, periods)
    levels, seasons = es(torch.full((12,), 10.0), 0)
    torch.testing.assert_close(levels, torch.full((12,), 10.0))
    scale = es.forecast_scale(levels, seasons, origin=12, h=9)
    torch.testing.assert_close(scale, torch.full((9,), 10.0))


def test_es_hand_computed_nonseasonal_levels_and_penalty():
    from neuralforecast.models.esrnn import _ExponentialSmoothing

    es = _ExponentialSmoothing(1, ())
    levels, _ = es(torch.tensor([10.0, 20.0, 30.0]), 0)
    torch.testing.assert_close(levels, torch.tensor([10.0, 15.0, 22.5]))
    assert es.level_penalty(levels).item() == pytest.approx(0, abs=1e-12)


def test_es_hand_computed_seasonal_update_and_phase():
    from neuralforecast.models.esrnn import _ExponentialSmoothing

    es = _ExponentialSmoothing(1, (2,))
    with torch.no_grad():
        es.initial_seasons[0][0].copy_(torch.log(torch.tensor([0.5, 1.5])))
    levels, seasons = es(torch.tensor([10.0, 20.0, 30.0]), 0)
    torch.testing.assert_close(levels, torch.tensor([20.0, 50 / 3, 115 / 3]))
    # At prefix length two the next phases use s[2]=0.5 and s[3]=1.35.
    scale = es.forecast_scale(levels, seasons, origin=2, h=5)
    torch.testing.assert_close(
        scale, torch.tensor([25 / 3, 22.5, 25 / 3, 22.5, 25 / 3])
    )


def test_es_hand_computed_two_seasonal_updates():
    from neuralforecast.models.esrnn import _ExponentialSmoothing

    es = _ExponentialSmoothing(1, (2, 3))
    levels, seasons = es(torch.tensor([10.0, 20.0, 30.0, 40.0]), 0)
    torch.testing.assert_close(levels, torch.tensor([10.0, 15.0, 22.5, 795 / 28]))
    # Both seasonal updates divide by the OTHER lagged component, not its new value.
    assert seasons[0][5].item() == pytest.approx(112 / 159 + 7 / 12)
    assert seasons[1][6].item() == pytest.approx(96 / 159 + 0.5)


def test_network_restores_seasonal_baseline_and_local_rows_are_independent():
    from neuralforecast.models import ESRNN

    model = ESRNN(
        h=3,
        input_size=2,
        season_length=2,
        n_series=2,
        encoder_hidden_size=4,
        dilations=((1,),),
    )
    with torch.no_grad():
        for parameter in model.encoder.parameters():
            parameter.zero_()
        model.decoder.weight.zero_()
        model.decoder.bias.zero_()
    y = torch.tensor([10.0, 12.0, 11.0, 14.0, 12.0, 17.0])
    normalized, scale = model._forecast_series(y, 0, [4, 5, 6])
    torch.testing.assert_close(normalized, torch.ones(3, 3))
    before = scale.detach().clone()
    with torch.no_grad():
        model.es.smoothing[1].add_(3)
        model.es.initial_seasons[0][1].copy_(torch.tensor([1.0, -1.0]))
    _, after = model._forecast_series(y, 0, [4, 5, 6])
    torch.testing.assert_close(after, before)


def test_forecast_is_causal_and_both_local_and_global_parameters_get_gradients():
    from neuralforecast.models import ESRNN

    model = ESRNN(
        h=2,
        input_size=2,
        season_length=(2, 3),
        n_series=2,
        encoder_hidden_size=4,
        dilations=((1, 2), (2,)),
    )
    y = torch.tensor([10.0, 13.0, 9.0, 15.0, 11.0, 19.0, 12.0, 21.0, 14.0])
    predicted, scale = model._forecast_series(y, 0, [5, 6, 7])
    changed = y.clone()
    changed[7:] *= 10
    other, other_scale = model._forecast_series(changed, 0, [5, 6, 7])
    torch.testing.assert_close(predicted, other)
    torch.testing.assert_close(scale, other_scale)
    (predicted * scale).sum().backward()
    for parameter in [
        model.es.smoothing,
        *model.es.initial_seasons,
        model.decoder.weight,
        next(model.encoder.parameters()),
    ]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert model.es.smoothing.grad[1].abs().sum() == 0
