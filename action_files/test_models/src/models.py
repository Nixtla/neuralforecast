import time

import fire
import pandas as pd

from neuralforecast.core import NeuralForecast

from neuralforecast.models.rnn import RNN
from neuralforecast.models.tcn import TCN
from neuralforecast.models.deepar import DeepAR
from neuralforecast.models.nhits import NHITS
from neuralforecast.models.nbeats import NBEATS
from neuralforecast.models.tft import TFT
from neuralforecast.models.vanillatransformer import VanillaTransformer
from neuralforecast.models.dlinear import DLinear
from neuralforecast.models.bitcn import BiTCN   
from neuralforecast.models.tide import TiDE
from neuralforecast.models.deepnpts import DeepNPTS
from neuralforecast.models.kan import KAN

from neuralforecast.auto import (
    AutoMLP, 
    AutoDilatedRNN, 
)

from neuralforecast.losses.pytorch import SMAPE, MAE, IQLoss
from ray import tune

from src.data import get_data

def main(dataset: str = 'M3', group: str = 'Monthly') -> None:
    train, horizon, freq, seasonality = get_data('data/', dataset, group)
    train['ds'] = pd.to_datetime(train['ds'])

    config = {
        "hidden_size": tune.choice([256, 512]),
        "num_layers": tune.choice([2, 4]),
        "input_size": tune.choice([2 * horizon]),
        "max_steps": 1000,
        "val_check_steps": 300,
        "scaler_type": "minmax1",
        "random_seed": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    }
    config_drnn = {'input_size': tune.choice([2 * horizon]),
                   'encoder_hidden_size': tune.choice([16]),
                   "max_steps": 300,
                   "val_check_steps": 100,
                   "random_seed": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                   "scaler_type": "minmax1"}

    models = [
        AutoDilatedRNN(h=horizon, loss=MAE(), config=config_drnn, num_samples=2, cpus=1),
        RNN(h=horizon, input_size=2 * horizon, encoder_hidden_size=64, max_steps=300),
        TCN(h=horizon, input_size=2 * horizon, encoder_hidden_size=64, max_steps=300),
        NHITS(h=horizon, input_size=2 * horizon, dropout_prob_theta=0.5, loss=MAE(), max_steps=1000, val_check_steps=500),
        AutoMLP(h=horizon, loss=MAE(), config=config, num_samples=2, cpus=1),
        DLinear(h=horizon, input_size=2 * horizon, loss=MAE(), max_steps=2000, val_check_steps=500),
        TFT(h=horizon, input_size=2 * horizon, loss=SMAPE(), hidden_size=64, scaler_type='robust', windows_batch_size=512, max_steps=1500, val_check_steps=500),
        VanillaTransformer(h=horizon, input_size=2 * horizon, loss=MAE(), hidden_size=64, scaler_type='minmax1', windows_batch_size=512, max_steps=1500, val_check_steps=500),
        DeepAR(h=horizon, input_size=2 * horizon, scaler_type='minmax1', max_steps=500),
        BiTCN(h=horizon, input_size=2 * horizon, loss=MAE(), dropout=0.0, max_steps=1000, val_check_steps=500),
        TiDE(h=horizon, input_size=2 * horizon, loss=MAE(), max_steps=1000, val_check_steps=500),
        DeepNPTS(h=horizon, input_size=2 * horizon, loss=MAE(), max_steps=1000, val_check_steps=500),
        NBEATS(h=horizon, input_size=2 * horizon, loss=IQLoss(), max_steps=2000, val_check_steps=500),
        KAN(h=horizon, input_size= 2 * horizon, loss=MAE(), max_steps=1000, val_check_steps=500)
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

        if model_name == 'DeepAR':
            forecasts = forecasts[['unique_id', 'ds', 'DeepAR-median']]

        forecasts.columns = ['unique_id', 'ds', model_name]
        forecasts.to_csv(f'data/{model_name}-forecasts-{dataset}-{group}.csv', index=False)
        time_df = pd.DataFrame({'time': [end - start], 'model': [model_name]})
        time_df.to_csv(f'data/{model_name}-time-{dataset}-{group}.csv', index=False)

if __name__ == '__main__':
    fire.Fire(main)
