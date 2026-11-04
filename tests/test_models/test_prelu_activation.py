import pytest
import torch.nn as nn

from neuralforecast.models.nhits import NHITSBlock
from neuralforecast.models.nbeats import NBEATSBlock
from neuralforecast.models.nbeatsx import NBEATSBlock as NBEATSxBlock


MLP_UNITS = [[16, 16], [16, 16]]


@pytest.mark.parametrize(
    ("block_class", "kwargs"),
    [
        pytest.param(
            NHITSBlock,
            {
                "input_size": 8,
                "h": 4,
                "n_theta": 4,
                "mlp_units": MLP_UNITS,
                "basis": nn.Identity(),
                "futr_input_size": 0,
                "hist_input_size": 0,
                "stat_input_size": 0,
                "n_pool_kernel_size": 1,
                "pooling_mode": "MaxPool1d",
                "dropout_prob": 0.0,
                "activation": "PReLU",
            },
            id="nhits",
        ),
        pytest.param(
            NBEATSBlock,
            {
                "input_size": 8,
                "n_theta": 4,
                "mlp_units": MLP_UNITS,
                "basis": nn.Identity(),
                "dropout_prob": 0.0,
                "activation": "PReLU",
            },
            id="nbeats",
        ),
        pytest.param(
            NBEATSxBlock,
            {
                "input_size": 8,
                "h": 4,
                "futr_input_size": 0,
                "hist_input_size": 0,
                "stat_input_size": 0,
                "n_theta": 4,
                "mlp_units": MLP_UNITS,
                "basis": nn.Identity(),
                "dropout_prob": 0.0,
                "activation": "PReLU",
            },
            id="nbeatsx",
        ),
    ],
)
def test_model_blocks_have_independent_prelu_parameters(block_class, kwargs):
    block = block_class(**kwargs)

    prelus = [
        module
        for module in block.layers
        if isinstance(module, nn.PReLU)
    ]

    assert len(prelus) == 2
    assert len({id(module) for module in prelus}) == 2
    assert len({id(module.weight) for module in prelus}) == 2
