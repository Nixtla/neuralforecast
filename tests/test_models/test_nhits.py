from neuralforecast.auto import NHITS, AutoNHITS
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model

from .test_helpers import check_args


def test_nhits_model(suppress_warnings):
    check_model(NHITS, ["airpassengers"])


def test_autonhits(setup_dataset):
    dataset = setup_dataset

    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoNHITS, exclude_args=['cls_model'])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoNHITS.get_default_config(h=12, backend='optuna')
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update({'max_steps': 2, 'val_check_steps': 1, 'input_size': 12, 'mlp_units': 3 * [[8, 8]]})
        return config

    model = AutoNHITS(h=12, config=my_config_new, backend='optuna', num_samples=1, cpus=1)
    assert model.config(MockTrial())['h'] == 12
    model.fit(dataset=dataset)

    # Unit test for situation: Ray with updated default config
    my_config = AutoNHITS.get_default_config(h=12, backend='ray')
    my_config['max_steps'] = 2
    my_config['val_check_steps'] = 1
    my_config['input_size'] = 12
    my_config['mlp_units'] = 3 * [[8, 8]]
    model = AutoNHITS(h=12, config=my_config, backend='ray', num_samples=1, cpus=1)
    model.fit(dataset=dataset)