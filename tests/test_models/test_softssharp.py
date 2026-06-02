from neuralforecast.auto import AutoSOFTSSharp, RayOptions, SOFTSSharp
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model

from .test_helpers import check_args


def test_softssharp_model(suppress_warnings):
    check_model(SOFTSSharp, ["airpassengers"])


def test_autosoftssharp(setup_dataset):

    check_args(AutoSOFTSSharp, exclude_args=['cls_model'])

    my_config = AutoSOFTSSharp.get_default_config(h=12, n_series=1, backend='optuna')
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update({
            'max_steps': 1,
            'val_check_steps': 1,
            'input_size': 12,
            'hidden_size': 16,
            'd_core': 16,
        })
        return config

    model = AutoSOFTSSharp(h=12, n_series=1, config=my_config_new, backend='optuna', num_samples=1)
    assert model.config(MockTrial())['h'] == 12
    model.fit(dataset=setup_dataset)

    my_config = AutoSOFTSSharp.get_default_config(h=12, n_series=1, backend='ray')
    my_config['max_steps'] = 1
    my_config['val_check_steps'] = 1
    my_config['input_size'] = 12
    my_config['hidden_size'] = 16
    my_config['d_core'] = 16
    model = AutoSOFTSSharp(h=12, n_series=1, config=my_config, backend='ray', num_samples=1, ray_options=RayOptions(cpus=1))
    model.fit(dataset=setup_dataset)
