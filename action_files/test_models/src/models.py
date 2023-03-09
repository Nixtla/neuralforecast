import os
import time

import fire
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from neuralforecast.core import NeuralForecast

from neuralforecast.models.gru import GRU
from neuralforecast.models.rnn import RNN
from neuralforecast.models.tcn import TCN
from neuralforecast.models.lstm import LSTM
from neuralforecast.models.dilated_rnn import DilatedRNN
from neuralforecast.models.mlp import MLP
from neuralforecast.models.nhits import NHITS
from neuralforecast.models.nbeats import NBEATS
from neuralforecast.models.nbeatsx import NBEATSx
from neuralforecast.models.tft import TFT
from neuralforecast.models.informer import Informer

from neuralforecast.auto import (
    AutoMLP, AutoNHITS, AutoNBEATS, AutoDilatedRNN, AutoTFT
)

from neuralforecast.losses.pytorch import SMAPE
from ray import tune

from src.data import get_data


def main(dataset: str = 'M3', group: str = 'Other') -> None:
    train, horizon, freq, seasonality = get_data('data/', dataset, group)
    train['ds'] = pd.to_datetime(train['ds']) 
    config_nbeats = {
        "mlp_units": tune.choice([3 * [[512, 512]]]),
        "input_size": tune.choice([2 * horizon, 3 * horizon, horizon, 4 * horizon]),
        "max_epochs": 100,
        "val_check_steps": 100
    }
    config = {
        "hidden_size": tune.choice([256, 512, 1024]),
        "num_layers": tune.choice([2, 4, 6]),
        "input_size": tune.choice([2 * horizon, 3 * horizon, horizon]),
        "max_epochs": 100,
        "val_check_steps": 100
    }
    config_drnn = {'input_size': tune.choice([2 * horizon, 3 * horizon]),
                   'encoder_hidden_size': tune.choice([50]),
                   'max_epochs': 50,
                   "val_check_steps": 100}
    models = [
        DilatedRNN(h=horizon, input_size=2 * horizon, encoder_hidden_size=50, max_epochs=50),
        RNN(h=horizon, input_size=2 * horizon, encoder_hidden_size=50, max_epochs=50),
        TCN(h=horizon, input_size=2 * horizon, encoder_hidden_size=20, max_epochs=100),
        LSTM(h=horizon, input_size=2 * horizon, encoder_hidden_size=50, max_epochs=50),
        GRU(h=horizon, input_size=2 * horizon, encoder_hidden_size=50, max_epochs=50),
        AutoDilatedRNN(h=horizon, loss=SMAPE(), config=config_drnn, num_samples=2, cpus=1),
        AutoNBEATS(h=horizon, loss=SMAPE(), config=config_nbeats, num_samples=2, cpus=1),
        AutoNHITS(h=horizon, loss=SMAPE(), config=config_nbeats, num_samples=2, cpus=1),
        AutoMLP(h=horizon, loss=SMAPE(), config=config, num_samples=2, cpus=1),
        NHITS(h=horizon, input_size=2 * horizon, loss=SMAPE(), max_epochs=100),
        NBEATS(h=horizon, input_size=2 * horizon, loss=SMAPE(), max_epochs=100),
        NBEATSx(h=horizon, input_size=2 * horizon, loss=SMAPE(), max_epochs=100),
        MLP(h=horizon, input_size=2 * horizon, num_layers=2, loss=SMAPE(), max_epochs=300),
        TFT(h=horizon, input_size=2 * horizon, loss=SMAPE(), max_epochs=100),
        Informer(h=horizon, input_size=2 * horizon, loss=SMAPE(), max_epochs=300)
    ]
    for model in models:
        model_name = type(model).__name__
        start = time.time()
        fcst = NeuralForecast(models=[model], freq=freq)
        fcst.fit(train)
        forecasts = fcst.predict()
        end = time.time()
        print(end - start)

        forecasts = forecasts.reset_index()
        forecasts.to_csv(f'data/{model_name}-forecasts-{dataset}-{group}.csv', index=False)
        time_df = pd.DataFrame({'time': [end - start], 'model': [model_name]})
        time_df.to_csv(f'data/{model_name}-time-{dataset}-{group}.csv', index=False)


if __name__ == '__main__':
    fire.Fire(main)
