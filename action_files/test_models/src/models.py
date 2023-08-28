import os
import time

import fire
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from neuralforecast.core import NeuralForecast

from neuralforecast.models.gru import GRU
from neuralforecast.models.rnn import RNN
from neuralforecast.models.tcn import TCN
from neuralforecast.models.lstm import LSTM
from neuralforecast.models.dilated_rnn import DilatedRNN
from neuralforecast.models.deepar import DeepAR
from neuralforecast.models.mlp import MLP
from neuralforecast.models.nhits import NHITS
from neuralforecast.models.nbeats import NBEATS
from neuralforecast.models.nbeatsx import NBEATSx
from neuralforecast.models.tft import TFT
from neuralforecast.models.vanillatransformer import VanillaTransformer
from neuralforecast.models.informer import Informer
from neuralforecast.models.autoformer import Autoformer
from neuralforecast.models.patchtst import PatchTST

from neuralforecast.auto import (
    AutoMLP, AutoNHITS, AutoNBEATS, AutoDilatedRNN, AutoTFT
)

from neuralforecast.losses.pytorch import SMAPE
from ray import tune

from src.data import get_data


def main(dataset: str = 'M3', group: str = 'Monthly') -> None:
    train, horizon, freq, seasonality = get_data('data/', dataset, group)
    train['ds'] = pd.to_datetime(train['ds'])
    #n_steps_epoch = len(train['unique_id']) // 1024 # WindowsBased
    #n_steps_epoch = len(train['unique_id'].unique()) // 32 # RNNBased

    config_nbeats = {
        "mlp_units": tune.choice([3 * [[512, 512]]]),
        "input_size": tune.choice([2 * horizon]),
        "max_steps": 1000,
        "val_check_steps": 100
    }
    config = {
        "hidden_size": tune.choice([256, 512, 1024]),
        "num_layers": tune.choice([2, 4, 6]),
        "input_size": tune.choice([2 * horizon]),
        "max_steps": 1000,
        "val_check_steps": 100
    }
    config_drnn = {'input_size': tune.choice([2 * horizon]),
                   'encoder_hidden_size': tune.choice([50]),
                   "max_steps": 300,
                   "val_check_steps": 100}
    models = [
        DilatedRNN(h=horizon, input_size=2 * horizon, encoder_hidden_size=50, max_steps=300),
        RNN(h=horizon, input_size=2 * horizon, encoder_hidden_size=50, max_steps=300),
        TCN(h=horizon, input_size=2 * horizon, encoder_hidden_size=20, max_steps=300),
        LSTM(h=horizon, input_size=2 * horizon, encoder_hidden_size=50, max_steps=300),
        GRU(h=horizon, input_size=2 * horizon, encoder_hidden_size=50, max_steps=300),
        AutoDilatedRNN(h=horizon, loss=SMAPE(), config=config_drnn, num_samples=2, cpus=1),
        AutoNBEATS(h=horizon, loss=SMAPE(), config=config_nbeats, num_samples=2, cpus=1),
        AutoNHITS(h=horizon, loss=SMAPE(), config=config_nbeats, num_samples=2, cpus=1),
        AutoMLP(h=horizon, loss=SMAPE(), config=config, num_samples=2, cpus=1),
        NHITS(h=horizon, input_size=2 * horizon, dropout_prob_theta=0.5, loss=SMAPE(), max_steps=1000),
        #NBEATS(h=horizon, input_size=2 * horizon, loss=SMAPE(), max_steps=1000),
        NBEATSx(h=horizon, input_size=2 * horizon, loss=SMAPE(), max_steps=1000),
        #MLP(h=horizon, input_size=2 * horizon, num_layers=2, loss=SMAPE(), max_steps=2000),
        TFT(h=horizon, input_size=2 * horizon, loss=SMAPE(), max_steps=1000),
        #VanillaTransformer(h=horizon, input_size=2 * horizon, loss=SMAPE(), scaler_type='robust', max_steps=5000),
        #Informer(h=horizon, input_size=2 * horizon, loss=SMAPE(), scaler_type='robust', max_steps=5000),
        #Autoformer(h=horizon, input_size=2 * horizon, loss=SMAPE(), scaler_type='robust', max_steps=5000),
        PatchTST(h=horizon, input_size=2 * horizon, patch_len=4, stride=4, loss=SMAPE(), scaler_type='robust', max_steps=1000),
        DeepAR(h=horizon, input_size=2 * horizon, max_steps=1000),
    ]

    # Models
    for model in models[:-1]:
        model_name = type(model).__name__
        start = time.time()
        fcst = NeuralForecast(models=[model], freq=freq)
        fcst.fit(train)
        forecasts = fcst.predict()
        end = time.time()
        print(end - start)

        forecasts = forecasts.reset_index()
        forecasts.columns = ['unique_id', 'ds', model_name]
        forecasts.to_csv(f'data/{model_name}-forecasts-{dataset}-{group}.csv', index=False)
        time_df = pd.DataFrame({'time': [end - start], 'model': [model_name]})
        time_df.to_csv(f'data/{model_name}-time-{dataset}-{group}.csv', index=False)

    # DeepAR
    model_name = type(models[-1]).__name__
    start = time.time()
    fcst = NeuralForecast(models=[models[-1]], freq=freq)
    fcst.fit(train)
    forecasts = fcst.predict()
    end = time.time()
    print(end - start)

    forecasts = forecasts.reset_index()
    forecasts = forecasts[['unique_id', 'ds', 'DeepAR-median']]
    forecasts.columns = ['unique_id', 'ds', 'DeepAR']
    forecasts.to_csv(f'data/{model_name}-forecasts-{dataset}-{group}.csv', index=False)
    time_df = pd.DataFrame({'time': [end - start], 'model': [model_name]})
    time_df.to_csv(f'data/{model_name}-time-{dataset}-{group}.csv', index=False)


if __name__ == '__main__':
    fire.Fire(main)
