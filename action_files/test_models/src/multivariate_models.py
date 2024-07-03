import os
import time

import fire
import pandas as pd

from neuralforecast.core import NeuralForecast

from neuralforecast.models.softs import SOFTS
from neuralforecast.models.tsmixer import TSMixer
from neuralforecast.models.itransformer import iTransformer

from neuralforecast.auto import (
    AutoiTransformer, 
)

from neuralforecast.losses.pytorch import MAE
from ray import tune

from src.data import get_data

os.environ['NIXTLA_ID_AS_COL'] = '1'


def main(dataset: str = 'multivariate', group: str = 'ETTm2') -> None:
    train, horizon, freq = get_data('data/', dataset, group)
    train['ds'] = pd.to_datetime(train['ds'])

    itransformer_config = {
        "hidden_size": tune.choice([256, 512]),
        "input_size": tune.choice([2 * horizon]),
        "max_steps": 100,
        "val_check_steps": 300,
        "scaler_type": "identity",
        "random_seed": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    }

    models = [
        AutoiTransformer(h=horizon, n_series=7, loss=MAE(), config=itransformer_config, num_samples=2, cpus=1),
        SOFTS(h=horizon, n_series=7, input_size=2 * horizon, loss=MAE(), dropout=0.0, max_steps=1000, val_check_steps=500),
        TSMixer(h=horizon, n_series=7, input_size=2 * horizon, loss=MAE(), dropout=0.0, max_steps=1000, val_check_steps=500),
        iTransformer(h=horizon, n_series=7, input_size=2 * horizon, loss=MAE(), dropout=0.0, max_steps=1000, val_check_steps=100),
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
