from neuralforecast.auto import AutoLSTM
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model
from neuralforecast.models import LSTM

from .test_helpers import check_args


def test_lstm_model(suppress_warnings):
    check_model(LSTM, ["airpassengers"])


def test_autolstm_model(setup_dataset):
    dataset = setup_dataset

    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoLSTM, exclude_args=['cls_model'])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoLSTM.get_default_config(h=12, backend='optuna')
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update({'max_steps': 1, 'val_check_steps': 1, 'input_size': -1, 'encoder_hidden_size': 8})
        return config

    model = AutoLSTM(h=12, config=my_config_new, backend='optuna', num_samples=1, cpus=1)
    assert model.config(MockTrial())['h'] == 12
    model.fit(dataset=dataset)

    # Unit test for situation: Ray with updated default config
    my_config = AutoLSTM.get_default_config(h=12, backend='ray')
    my_config['max_steps'] = 1
    my_config['val_check_steps'] = 1
    my_config['input_size'] = -1
    my_config['encoder_hidden_size'] = 8
    model = AutoLSTM(h=12, config=my_config, backend='ray', num_samples=1, cpus=1)
    model.fit(dataset=dataset)