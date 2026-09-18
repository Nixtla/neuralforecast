import pytest
import torch.nn as nn

pytest.importorskip("xlstm")

from neuralforecast.models import xLSTM

from .test_helpers import (
    assert_decoder_activation_applied,
    assert_no_decoder_activation_warning,
)


def test_xlstm_decoder_activation():
    kwargs = dict(h=4, input_size=8, max_steps=1)

    # Ignored when the decoder collapses to a single linear layer
    with pytest.warns(UserWarning, match="decoder_activation is ignored"):
        xLSTM(**kwargs, decoder_layers=1, decoder_activation="Tanh")

    # Silent when left at its default, whether or not a decoder exists
    assert_no_decoder_activation_warning(xLSTM, **kwargs, decoder_layers=1)
    assert_no_decoder_activation_warning(xLSTM, **kwargs)

    # The default is GELU, not the 'ReLU' the helper falls back to
    assert isinstance(xLSTM(**kwargs).mlp_decoder.activation, nn.GELU)

    assert_decoder_activation_applied(xLSTM, **kwargs)

    # Rejected even where the decoder that would consume it is never built
    with pytest.raises(AssertionError, match="is not in"):
        xLSTM(**kwargs, decoder_layers=1, decoder_activation="NotAnActivation")
