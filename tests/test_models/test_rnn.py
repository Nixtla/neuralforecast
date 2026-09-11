import pytest

from neuralforecast.auto import AutoRNN, RayOptions
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model
from neuralforecast.models import RNN

from .test_helpers import assert_no_decoder_activation_warning, check_args


def test_rnn_model(suppress_warnings):
    """Test RNN model with the shared dataset fixture."""
    check_model(RNN, ["airpassengers"])

def test_autornn_model(setup_dataset):
    dataset = setup_dataset
    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoRNN, exclude_args=['cls_model'])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoRNN.get_default_config(h=12, backend='optuna')
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update({'max_steps': 1, 'val_check_steps': 1, 'input_size': -1, 'encoder_hidden_size': 8})
        return config

    model = AutoRNN(h=12, config=my_config_new, backend='optuna', num_samples=1)
    model.fit(dataset=dataset)

    assert model.config(MockTrial())['h'] == 12
    # Unit test for situation: Ray with updated default config
    my_config = AutoRNN.get_default_config(h=12, backend='ray')
    my_config['max_steps'] = 1
    my_config['val_check_steps'] = 1
    my_config['input_size'] = -1
    my_config['encoder_hidden_size'] = 8
    model = AutoRNN(h=12, config=my_config, backend='ray', num_samples=1, ray_options=RayOptions(cpus=1))
    model.fit(dataset=dataset)


def test_rnn_decoder_activation_ignored_when_recurrent():
    kwargs = dict(h=4, input_size=8, max_steps=1)

    with pytest.warns(UserWarning, match="decoder_activation is ignored"):
        RNN(**kwargs, recurrent=True, decoder_activation="Tanh")

    # No warning when the argument is left at its default, or when it is honored
    assert_no_decoder_activation_warning(RNN, **kwargs, recurrent=True)
    assert_no_decoder_activation_warning(
        RNN, **kwargs, recurrent=False, decoder_activation="Tanh"
    )
