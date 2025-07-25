import time

import fire
import pandas as pd

from neuralforecast.core import NeuralForecast

from neuralforecast.models.gru import GRU
from neuralforecast.models.lstm import LSTM
from neuralforecast.models.dilated_rnn import DilatedRNN
from neuralforecast.models.nbeatsx import NBEATSx
from neuralforecast.models.xlstm import xLSTM

from neuralforecast.auto import (
    AutoNHITS, 
    AutoNBEATS, 
)

from neuralforecast.losses.pytorch import MAE
from ray import tune

from src.data import get_data

def main(dataset: str = 'M3', group: str = 'Monthly') -> None:
    train, horizon, freq, seasonality = get_data('data/', dataset, group)
    train['ds'] = pd.to_datetime(train['ds'])

    config_nbeats = {
        "input_size": tune.choice([2 * horizon]),
        "max_steps": 1000,
        "val_check_steps": 300,
        "scaler_type": "minmax1",
        "random_seed": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    }
    models = [
        LSTM(h=horizon, input_size=2 * horizon, encoder_hidden_size=64, max_steps=300),
        DilatedRNN(h=horizon, input_size=2 * horizon, encoder_hidden_size=64, max_steps=300),
        GRU(h=horizon, input_size=2 * horizon, encoder_hidden_size=64, max_steps=300),
        AutoNBEATS(h=horizon, loss=MAE(), config=config_nbeats, num_samples=2, cpus=1),
        AutoNHITS(h=horizon, loss=MAE(), config=config_nbeats, num_samples=2, cpus=1),
        NBEATSx(h=horizon, input_size=2 * horizon, loss=MAE(), max_steps=1000),
        xLSTM(h=horizon, input_size=2 * horizon, max_steps=300, backbone='mLSTM',),
    ]

    # Models
    for model in models:
        model_name = type(model).__name__
        print(50*'-', model_name, 50*'-')
        start = time.time()
        fcst = NeuralForecast(models=[model], freq=freq)
        fcst.fit(train)
        forecasts = fcst.predict()
        end = time.time()
        print(end - start)

        forecasts.columns = ['unique_id', 'ds', model_name]
        forecasts.to_csv(f'data/{model_name}-forecasts-{dataset}-{group}.csv', index=False)
        time_df = pd.DataFrame({'time': [end - start], 'model': [model_name]})
        time_df.to_csv(f'data/{model_name}-time-{dataset}-{group}.csv', index=False)


if __name__ == '__main__':
    fire.Fire(main)
