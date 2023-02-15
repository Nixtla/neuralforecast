import time

import fire
import pandas as pd
import pytorch_lightning as pl
import torch

from neuralforecast.auto import AutoNHITS
from neuralforecast.core import NeuralForecast
from neuralforecast.models.nhits import NHITS
from neuralforecast.losses.pytorch import SMAPE
from ray import tune

from src.data import get_data


def main(dataset: str = 'M3', group: str = 'Other') -> None:
    assert torch.cuda.is_available(), 'GPU is not available'
    train, horizon, freq, seasonality = get_data('data/', dataset, group)
    train['ds'] = pd.to_datetime(train['ds']) 
    config_nhits = {
        "mlp_units": tune.choice([3 * [[512, 512]]]),
        "input_size": tune.choice([2 * horizon, 3 * horizon, horizon, 4 * horizon]),
        "max_epochs": 100,
        "val_check_steps": 100
    }
    models = [
        AutoNHITS(h=horizon, loss=SMAPE(), config=config_nhits, num_samples=2, cpus=1),
        NHITS(h=horizon, input_size=2 * horizon, loss=SMAPE(), max_epochs=100),
    ]
    start = time.time()
    fcst = NeuralForecast(models=models, freq=freq)
    fcst.fit(train)
    forecasts = fcst.predict()
    end = time.time()
    print(end - start)
    print(forecasts)


if __name__ == '__main__':
    fire.Fire(main)
