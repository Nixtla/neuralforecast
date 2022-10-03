import os
import time

import fire
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from ray import tune

from datasetsforecast.long_horizon import LongHorizon, LongHorizonInfo

from neuralforecast.core import NeuralForecast
from neuralforecast.auto import AutoNHITS
from neuralforecast.models.nhits import NHITS
from neuralforecast.losses.pytorch import SMAPE

from src.data import get_data


def main(group: str = 'ETTm2') -> None:
    info = LongHorizonInfo[group]
    freq = info.freq
    freq = freq if freq != 'W' else 'W-TUE'
    test_size = info.test_size
    val_size = info.val_size
    train, *_ = LongHorizon.load('./data', group)
    train['ds'] = pd.to_datetime(train['ds']) 
    evals = []
    for horizon in info.horizons:
        models = [
            NHITS(input_size=2 * horizon, h=horizon, loss=SMAPE(), max_epochs=100),
        ]
        start = time.time()
        fcst = NeuralForecast(models=models, freq=freq)
        forecasts = fcst.cross_validation(train, n_windows=None, val_size=val_size, test_size=test_size)
        end = time.time()
        print(end - start)
        mse = np.power(forecasts['y'] - forecasts['NHITS'], 2).mean()
        mae = np.abs(forecasts['y'] - forecasts['NHITS']).mean()
        eval_h = pd.DataFrame({
            'dataset': group, 'h': horizon, 'mse': mse, 
            'mae': mae, 'time': end - start
        }, index=[0])
        evals.append(eval_h)
    evals = pd.concat(evals)
    print(evals)
    evals.to_csv('data/cv_evaluation.csv', index=False)


if __name__ == '__main__':
    fire.Fire(main)
