import torch

from neuralforecast.auto import AutoTSMixerx, RayOptions, TSMixerx
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model
from neuralforecast.models.tsmixerx import MixingLayerWithStaticExogenous

from .test_helpers import check_args


def test_ts_mixer_x(suppress_warnings):
    check_model(TSMixerx, ["airpassengers"])


def test_tsmixerx_static_exogenous_is_computed_once():
    """Static covariates are batch-independent and must not be repeated by B."""
    batch_size, input_size, horizon, n_series, stat_size = 4, 12, 6, 5, 2
    model = TSMixerx(
        h=horizon,
        input_size=input_size,
        n_series=n_series,
        stat_exog_list=[f"s{i}" for i in range(stat_size)],
        n_block=2,
        max_steps=1,
    ).eval()

    seen_shapes = []
    hooks = []
    stat_mixers = [model.feature_mixer_stat]
    stat_mixers.extend(layer.feature_mixer_stat for layer in model.mixing_block)
    for mixer in stat_mixers:
        hooks.append(
            mixer.register_forward_hook(
                lambda _module, args, _output: seen_shapes.append(tuple(args[0].shape))
            )
        )

    batch = {
        "insample_y": torch.randn(batch_size, input_size, n_series),
        "hist_exog": None,
        "futr_exog": None,
        "stat_exog": torch.randn(n_series, stat_size),
    }
    with torch.no_grad():
        forecast = model(batch)

    for hook in hooks:
        hook.remove()

    assert forecast.shape == (batch_size, horizon, n_series)
    assert seen_shapes == [(1, horizon, n_series * stat_size)] * len(stat_mixers)


def test_tsmixerx_static_exogenous_is_batch_independent():
    """A window's forecast must not depend on the other rows in its batch."""
    batch_size, input_size, horizon, n_series, stat_size = 4, 12, 6, 5, 2
    model = TSMixerx(
        h=horizon,
        input_size=input_size,
        n_series=n_series,
        stat_exog_list=[f"s{i}" for i in range(stat_size)],
        n_block=2,
        dropout=0.0,
        max_steps=1,
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


def test_tsmixerx_static_exogenous_gradients():
    """The broadcast must keep gradients flowing back to the static tensor."""
    batch_size, input_size, horizon, n_series, stat_size = 4, 12, 6, 5, 2
    model = TSMixerx(
        h=horizon,
        input_size=input_size,
        n_series=n_series,
        stat_exog_list=[f"s{i}" for i in range(stat_size)],
        n_block=2,
        max_steps=1,
    )
    stat = torch.randn(n_series, stat_size, requires_grad=True)
    batch = {
        "insample_y": torch.randn(batch_size, input_size, n_series),
        "hist_exog": None,
        "futr_exog": None,
        "stat_exog": stat,
    }

    model(batch).sum().backward()

    assert stat.grad is not None
    assert stat.grad.shape == stat.shape
    assert torch.isfinite(stat.grad).all()


def test_tsmixerx_static_exogenous_training_with_dropout():
    """The shared static branch must remain trainable with dropout enabled."""
    batch_size, input_size, horizon, n_series, stat_size = 4, 12, 6, 5, 2
    model = TSMixerx(
        h=horizon,
        input_size=input_size,
        n_series=n_series,
        stat_exog_list=[f"s{i}" for i in range(stat_size)],
        n_block=2,
        dropout=0.5,
        max_steps=1,
    )
    batch = {
        "insample_y": torch.randn(batch_size, input_size, n_series),
        "hist_exog": None,
        "futr_exog": None,
        "stat_exog": torch.randn(n_series, stat_size),
    }

    model.train()
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
