
import inspect
import warnings

import torch.nn as nn

from neuralforecast.auto import BaseAuto


def check_args(auto_model, exclude_args=None):
    if not hasattr(inspect, 'getargspec'):
        getargspec = inspect.getfullargspec
    else:
        getargspec = inspect.getargspec

    base_auto_args = getargspec(BaseAuto)[0]
    auto_model_args = getargspec(auto_model)[0]
    if exclude_args is not None:
        base_auto_args = [arg for arg in base_auto_args if arg not in exclude_args]
    args_diff = set(base_auto_args) - set(auto_model_args)
    assert not args_diff, (
        f"__init__ of {auto_model.__name__} does not contain the following required variables from BaseAuto class:\n\t\t{args_diff}"
    )


def assert_decoder_activation_applied(model_cls, decoder_attr="mlp_decoder", **kwargs):
    """Assert `decoder_activation` reaches the decoder without warning about it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = model_cls(**kwargs, decoder_activation="Tanh")
    assert not [w for w in caught if "decoder_activation" in str(w.message)]
    assert isinstance(getattr(model, decoder_attr).activation, nn.Tanh)


def assert_no_decoder_activation_warning(model_cls, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model_cls(**kwargs)
    assert not [w for w in caught if "decoder_activation" in str(w.message)]
