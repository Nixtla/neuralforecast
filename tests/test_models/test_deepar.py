import pytest

from neuralforecast.auto import AutoDeepAR, DeepAR, RayOptions
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model

from .test_helpers import assert_no_decoder_activation_warning, check_args


def test_deepar(suppress_warnings):
    check_model(DeepAR, ["airpassengers"])


def test_autodeepar(setup_dataset):
    dataset = setup_dataset

    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoDeepAR, exclude_args=['cls_model'])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoDeepAR.get_default_config(h=12, backend='optuna')
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update({'max_steps': 1, 'val_check_steps': 1, 'input_size': 12, 'lstm_hidden_size': 8})
        return config

    model = AutoDeepAR(h=12, config=my_config_new, backend='optuna', num_samples=1)
    assert model.config(MockTrial())['h'] == 12
    model.fit(dataset=dataset)

    # Unit test for situation: Ray with updated default config
    my_config = AutoDeepAR.get_default_config(h=12, backend='ray')
    my_config['max_steps'] = 1
    my_config['val_check_steps'] = 1
    my_config['input_size'] = 12
    my_config['lstm_hidden_size'] = 8
    model = AutoDeepAR(h=12, config=my_config, backend='ray', num_samples=1, ray_options=RayOptions(cpus=1))
    model.fit(dataset=dataset)


def test_deepar_decoder_activation_ignored_without_hidden_layers():
    kwargs = dict(h=4, input_size=8, max_steps=1)

    with pytest.warns(UserWarning, match="decoder_activation is ignored"):
        DeepAR(**kwargs, decoder_hidden_layers=0, decoder_activation="Tanh")

    # No warning when the argument is left at its default, or when it is honored
    assert_no_decoder_activation_warning(DeepAR, **kwargs, decoder_hidden_layers=0)
    assert_no_decoder_activation_warning(
        DeepAR,
        **kwargs,
        decoder_hidden_layers=2,
        decoder_hidden_size=8,
        decoder_activation="Tanh",
    )
