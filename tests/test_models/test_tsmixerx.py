import pytest
import torch

from neuralforecast.auto import AutoTSMixerx, RayOptions, TSMixerx
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model
from neuralforecast.models.tsmixerx import MixingLayerWithStaticExogenous

from .test_helpers import check_args


def _make_model(
    *,
    input_size=12,
    horizon=6,
    n_series=5,
    stat_size=2,
    n_block=2,
    dropout=0.0,
):
    return TSMixerx(
        h=horizon,
        input_size=input_size,
        n_series=n_series,
        stat_exog_list=[f"s{i}" for i in range(stat_size)],
        n_block=n_block,
        dropout=dropout,
        max_steps=1,
    )


def _make_batch(batch_size, input_size=12, n_series=5, stat_size=2, device="cpu"):
    return {
        "insample_y": torch.randn(batch_size, input_size, n_series, device=device),
        "hist_exog": None,
        "futr_exog": None,
        "stat_exog": torch.randn(n_series, stat_size, device=device),
    }


def _register_repeat_reference_hooks(model, batch_size):
    """Make the reference model use the old repeat(B, h, 1) static path."""

    def repeat_hook(_module, args, batch_size=batch_size):
        (stat_exog,) = args
        return (stat_exog.repeat(batch_size, 1, 1),)

    hooks = []
    mixers = [model.feature_mixer_stat]
    mixers.extend(layer.feature_mixer_stat for layer in model.mixing_block)
    for mixer in mixers:
        hooks.append(mixer.register_forward_pre_hook(repeat_hook))
    return hooks


def test_ts_mixer_x(suppress_warnings):
    check_model(TSMixerx, ["airpassengers"])


@pytest.mark.parametrize("training", [False, True])
def test_tsmixerx_static_exogenous_is_computed_once(training):
    """Static covariates are batch-independent and must not be repeated by B."""
    batch_size, input_size, horizon, n_series, stat_size = 4, 12, 6, 5, 2
    model = _make_model(
        input_size=input_size,
        horizon=horizon,
        n_series=n_series,
        stat_size=stat_size,
    )
    model.train(training)

    seen_inputs = []
    seen_outputs = []
    hooks = []
    mixers = [model.feature_mixer_stat]
    mixers.extend(layer.feature_mixer_stat for layer in model.mixing_block)

    def record_shapes(_module, args, output):
        seen_inputs.append(tuple(args[0].shape))
        seen_outputs.append(tuple(output.shape))

    for mixer in mixers:
        hooks.append(mixer.register_forward_hook(record_shapes))

    batch = _make_batch(batch_size, input_size, n_series, stat_size)
    with torch.no_grad():
        forecast = model(batch)

    for hook in hooks:
        hook.remove()

    expected_input = (1, horizon, n_series * stat_size)
    assert forecast.shape == (batch_size, horizon, n_series)
    assert seen_inputs == [expected_input] * len(mixers)
    assert all(shape[0] == 1 for shape in seen_outputs)


def test_tsmixerx_static_exogenous_is_batch_independent():
    """A window's forecast must not depend on the other rows in its batch."""
    torch.manual_seed(0)
    batch_size, input_size, horizon, n_series, stat_size = 4, 12, 6, 5, 2
    model = _make_model(
        input_size=input_size,
        horizon=horizon,
        n_series=n_series,
        stat_size=stat_size,
    ).eval()

    y = torch.randn(input_size, n_series)
    stat = torch.randn(n_series, stat_size)
    batch_one = {
        "insample_y": y.unsqueeze(0),
        "hist_exog": None,
        "futr_exog": None,
        "stat_exog": stat,
    }
    batch_many = {
        "insample_y": y.unsqueeze(0).repeat(batch_size, 1, 1),
        "hist_exog": None,
        "futr_exog": None,
        "stat_exog": stat,
    }
    with torch.no_grad():
        forecast_one = model(batch_one)
        forecast_many = model(batch_many)

    torch.testing.assert_close(forecast_one[0], forecast_many[0])


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tsmixerx_static_exogenous_matches_repeated_reference(batch_size, device):
    """The broadcast form matches the old repeated computation and gradients."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)

    input_size, horizon, n_series, stat_size = 12, 6, 5, 2
    model = _make_model(
        input_size=input_size,
        horizon=horizon,
        n_series=n_series,
        stat_size=stat_size,
    ).to(device)
    reference = _make_model(
        input_size=input_size,
        horizon=horizon,
        n_series=n_series,
        stat_size=stat_size,
    ).to(device)
    reference.load_state_dict(model.state_dict())
    model.train()
    reference.train()
    hooks = _register_repeat_reference_hooks(reference, batch_size)

    stat = torch.randn(n_series, stat_size, device=device, requires_grad=True)
    reference_stat = stat.detach().clone().requires_grad_(True)
    batch = _make_batch(batch_size, input_size, n_series, stat_size, device)
    reference_batch = {**batch, "stat_exog": reference_stat}
    batch["stat_exog"] = stat

    forecast = model(batch)
    reference_forecast = reference(reference_batch)

    for hook in hooks:
        hook.remove()

    # Same weights, same input, dropout=0: this is the old computation exactly.
    torch.testing.assert_close(forecast, reference_forecast, atol=1e-5, rtol=1e-5)

    model.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    forecast.sum().backward()
    reference_forecast.sum().backward()

    for (name, parameter), (reference_name, reference_parameter) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        assert name == reference_name
        assert (parameter.grad is None) == (reference_parameter.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(
                parameter.grad, reference_parameter.grad, atol=1e-5, rtol=1e-5
            )

    assert stat.grad is not None
    assert reference_stat.grad is not None
    torch.testing.assert_close(stat.grad, reference_stat.grad, atol=1e-5, rtol=1e-5)


def test_tsmixerx_static_exogenous_training_with_dropout():
    """The shared static branch must remain trainable with dropout enabled."""
    batch_size, input_size, horizon, n_series, stat_size = 4, 12, 6, 5, 2
    model = _make_model(
        input_size=input_size,
        horizon=horizon,
        n_series=n_series,
        stat_size=stat_size,
        dropout=0.5,
    )
    seen_output_shapes = []
    hooks = []
    mixers = [model.feature_mixer_stat]
    mixers.extend(layer.feature_mixer_stat for layer in model.mixing_block)
    for mixer in mixers:
        hooks.append(
            mixer.register_forward_hook(
                lambda _module, _args, output: seen_output_shapes.append(
                    tuple(output.shape)
                )
            )
        )
    batch = _make_batch(batch_size, input_size, n_series, stat_size)

    model.train()
    forecast = model(batch)

    for hook in hooks:
        hook.remove()

    assert forecast.shape == (batch_size, horizon, n_series)
    assert all(shape[0] == 1 for shape in seen_output_shapes)


def test_tsmixerx_static_exogenous_without_mixing_blocks():
    """The static branch also works when n_block is zero."""
    batch_size, input_size, horizon, n_series, stat_size = 4, 12, 6, 5, 2
    model = _make_model(
        input_size=input_size,
        horizon=horizon,
        n_series=n_series,
        stat_size=stat_size,
        n_block=0,
    ).eval()
    batch = _make_batch(batch_size, input_size, n_series, stat_size)

    forecast = model(batch)

    assert forecast.shape == (batch_size, horizon, n_series)


def test_mixing_layer_static_exogenous_accepts_batch_shaped_static():
    """The layer keeps its old [B, h, S] contract for direct callers."""
    batch_size, horizon, ff_dim, stat_input_size = 4, 6, 8, 10
    layer = MixingLayerWithStaticExogenous(
        h=horizon, dropout=0.0, ff_dim=ff_dim, stat_input_size=stat_input_size
    ).eval()
    input_tensor = torch.randn(batch_size, horizon, ff_dim)
    stat_exog = torch.randn(batch_size, horizon, stat_input_size)

    output, returned_stat = layer((input_tensor, stat_exog))

    assert output.shape == (batch_size, horizon, ff_dim)
    assert returned_stat is stat_exog


def test_autotsmixerx(setup_dataset):
    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoTSMixerx, exclude_args=['cls_model'])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoTSMixerx.get_default_config(h=12, backend='optuna', n_series=1)
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update({'max_steps': 1, 'val_check_steps': 1, 'input_size': 12})
        return config

    model = AutoTSMixerx(h=12, n_series=1, config=my_config_new, backend='optuna', num_samples=1)
    assert model.config(MockTrial())['h'] == 12
    assert model.config(MockTrial())['n_series'] == 1
    model.fit(dataset=setup_dataset)

    # Unit test for situation: Ray with updated default config
    my_config = AutoTSMixerx.get_default_config(h=12, backend='ray', n_series=1)
    my_config['max_steps'] = 1
    my_config['val_check_steps'] = 1
    my_config['input_size'] = 12
    model = AutoTSMixerx(h=12, n_series=1, config=my_config, backend='ray', num_samples=1, ray_options=RayOptions(cpus=1))
    model.fit(dataset=setup_dataset)
