import pytest
import torch


from neuralforecast.auto import AutoDeepAR, DeepAR, RayOptions
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model

from .test_helpers import (
    assert_decoder_activation_applied,
    assert_no_decoder_activation_warning,
    check_args,
)


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


def test_deepar_decoder_activation():
    kwargs = dict(h=4, input_size=8, max_steps=1)

    # Ignored when the decoder collapses to a single linear layer (the default)
    with pytest.warns(UserWarning, match="decoder_activation is ignored"):
        DeepAR(**kwargs, decoder_hidden_layers=0, decoder_activation="Tanh")

    # Silent when left at its default
    assert_no_decoder_activation_warning(DeepAR, **kwargs, decoder_hidden_layers=0)

    assert_decoder_activation_applied(
        DeepAR,
        decoder_attr="decoder",
        **kwargs,
        decoder_hidden_layers=2,
        decoder_hidden_size=8,
    )

    # Rejected even where the decoder that would consume it is never built
    with pytest.raises(AssertionError, match="is not in"):
        DeepAR(**kwargs, decoder_hidden_layers=0, decoder_activation="NotAnActivation")


@pytest.mark.parametrize("decoder_hidden_layers", [1, 2])
def test_deepar_rejects_zero_width_decoder(decoder_hidden_layers):
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        DeepAR(h=4, input_size=8, decoder_hidden_layers=decoder_hidden_layers)


@pytest.mark.parametrize("decoder_hidden_layers, decoder_hidden_size", [(0, 0), (1, 8), (2, 8)])
def test_deepar_decoder_preserves_encoder_gradients(decoder_hidden_layers, decoder_hidden_size):
    model = DeepAR(
        h=4,
        input_size=8,
        lstm_n_layers=1,
        lstm_hidden_size=8,
        lstm_dropout=0.0,
        decoder_hidden_layers=decoder_hidden_layers,
        decoder_hidden_size=decoder_hidden_size,
    )
    inputs = torch.randn(2, 8, 1, requires_grad=True)
    output = model({"insample_y": inputs, "futr_exog": None, "stat_exog": None})
    assert output.shape == (2, model.h, model.loss.outputsize_multiplier)
    output.square().mean().backward()

    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad.abs().sum() > 0
    encoder_grad = model.hist_encoder.weight_ih_l0.grad
    assert torch.isfinite(encoder_grad).all()
    assert encoder_grad.abs().sum() > 0
