from neuralforecast.auto import TFT, AutoTFT
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model

from .test_helpers import check_args


def test_tft_model(suppress_warnings):
    check_model(TFT, ["airpassengers"])


def test_autotft(setup_dataset):

    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoTFT, exclude_args=['cls_model'])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoTFT.get_default_config(h=12, backend='optuna')
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update({'max_steps': 1, 'val_check_steps': 1, 'input_size': 12, 'hidden_size': 8})
        return config

    model = AutoTFT(h=12, config=my_config_new, backend='optuna', num_samples=1, cpus=1)
    assert model.config(MockTrial())['h'] == 12
    model.fit(dataset=setup_dataset)

    # Unit test for situation: Ray with updated default config
    my_config = AutoTFT.get_default_config(h=12, backend='ray')
    my_config['max_steps'] = 1
    my_config['val_check_steps'] = 1
    my_config['input_size'] = 12
    my_config['hidden_size'] = 8
    model = AutoTFT(h=12, config=my_config, backend='ray', num_samples=1, cpus=1)
    model.fit(dataset=setup_dataset)