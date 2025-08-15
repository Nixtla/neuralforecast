from neuralforecast.auto import AutoRMoK, RMoK
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model

from .test_helpers import check_args


def test_rmok_model(suppress_warnings):
    check_model(RMoK, ["airpassengers"])


def test_autormok(setup_dataset):
    #| hide
    # Check Optuna

    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoRMoK, exclude_args=['cls_model'])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoRMoK.get_default_config(h=12, n_series=1, backend='optuna')
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update({'max_steps': 1, 'val_check_steps': 1, 'input_size': 12, 'learning_rate': 1e-1})
        return config

    model = AutoRMoK(h=12, n_series=1, config=my_config_new, backend='optuna', num_samples=1, cpus=1)
    assert model.config(MockTrial())['h'] == 12
    model.fit(dataset=setup_dataset)

    # Unit test for situation: Ray with updated default config
    my_config = AutoRMoK.get_default_config(h=12, n_series=1, backend='ray')
    my_config['max_steps'] = 1
    my_config['val_check_steps'] = 1
    my_config['input_size'] = 12
    my_config['learning_rate'] = 1e-1
    model = AutoRMoK(h=12, n_series=1, config=my_config, backend='ray', num_samples=1, cpus=1)
    model.fit(dataset=setup_dataset)