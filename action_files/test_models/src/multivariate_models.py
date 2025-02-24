import os
import time

import fire
import pandas as pd

from neuralforecast.core import NeuralForecast

from neuralforecast.models.softs import SOFTS
from neuralforecast.models.tsmixer import TSMixer
from neuralforecast.models.tsmixerx import TSMixerx
from neuralforecast.models.itransformer import iTransformer
from neuralforecast.models.mlpmultivariate import MLPMultivariate
from neuralforecast.models.timemixer import TimeMixer
from neuralforecast.models.timexer import TimeXer

from neuralforecast.losses.pytorch import MAE

from src.data import get_data

os.environ['NIXTLA_ID_AS_COL'] = '1'


def main(dataset: str = 'multivariate', group: str = 'ETTm2') -> None:
    train, horizon, freq = get_data('data/', dataset, group)
    train['ds'] = pd.to_datetime(train['ds'])

    models = [
        SOFTS(h=horizon, n_series=7, input_size=2 * horizon, loss=MAE(), dropout=0.0, max_steps=500, val_check_steps=100, windows_batch_size=64, inference_windows_batch_size=64),
        TSMixer(h=horizon, n_series=7, input_size=2 * horizon, loss=MAE(), dropout=0.0, max_steps=1000, val_check_steps=100, windows_batch_size=64, inference_windows_batch_size=64),
        TSMixerx(h=horizon, n_series=7, input_size=2*horizon, loss=MAE(), dropout=0.0, max_steps=1000, val_check_steps=100, windows_batch_size=64, inference_windows_batch_size=64),
        iTransformer(h=horizon, n_series=7, input_size=2 * horizon, loss=MAE(), dropout=0.0, max_steps=500, val_check_steps=100, windows_batch_size=64, inference_windows_batch_size=64),
        MLPMultivariate(h=horizon, n_series=7, input_size=2*horizon, loss=MAE(), max_steps=1000, val_check_steps=100, windows_batch_size=64, inference_windows_batch_size=64),
        TimeMixer(h=horizon, n_series=7, input_size=2*horizon, loss=MAE(), dropout=0.0, max_steps=500, val_check_steps=100, windows_batch_size=64, inference_windows_batch_size=64),
        TimeXer(h=horizon, n_series=7, input_size=2*horizon, loss=MAE(), dropout=0.0, max_steps=500, val_check_steps=100, windows_batch_size=64, inference_windows_batch_size=64, hidden_size=128, d_ff=512)
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
