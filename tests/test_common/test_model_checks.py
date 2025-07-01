import logging
import warnings

import pytest

from neuralforecast.common._model_checks import check_model
from neuralforecast.models import (
    GRU,
    KAN,
    LSTM,
    MLP,
    NBEATS,
    NHITS,
    RNN,
    SOFTS,
    TCN,
    TFT,
    Autoformer,
    BiTCN,
    DeepAR,
    DeepNPTS,
    DilatedRNN,
    DLinear,
    FEDformer,
    Informer,
    MLPMultivariate,
    NBEATSx,
    NLinear,
    RMoK,
    StemGNN,
    TiDE,
    TimeMixer,
    TimesNet,
    TSMixer,
    TSMixerx,
    VanillaTransformer,
    iTransformer,
)


@pytest.fixture
def setup_module():
    models = [
        RNN,
        GRU,
        TCN,
        LSTM,
        DeepAR,
        DilatedRNN,
        BiTCN,
        MLP,
        NBEATS,
        NBEATSx,
        NHITS,
        DLinear,
        NLinear,
        TiDE,
        DeepNPTS,
        TFT,
        VanillaTransformer,
        Informer,
        Autoformer,
        FEDformer,
        TimesNet,
        iTransformer,
        KAN,
        RMoK,
        StemGNN,
        TSMixer,
        TSMixerx,
        MLPMultivariate,
        SOFTS,
        TimeMixer,
    ]
    return models

def test_model_checks(setup_module):
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for model in setup_module:
            check_model(model, checks=["losses"])
