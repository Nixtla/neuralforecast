import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import pandas as pd

from ray import tune

from neuralforecast.auto import AutoNHITS
from neuralforecast.core import NeuralForecast

from neuralforecast.losses.pytorch import MAE
from neuralforecast.losses.numpy import mae, mse
from datasetsforecast.long_horizon import LongHorizon, LongHorizonInfo

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


if __name__ == '__main__':

    # Parse execution parameters
    verbose = True
    parser = argparse.ArgumentParser()
    parser.add_argument("-horizon", "--horizon", type=int)
    parser.add_argument("-dataset", "--dataset", type=str)

    args = parser.parse_args()
    horizon = args.horizon
    dataset = args.dataset

    assert horizon in [96, 192, 336, 720]

    # Load dataset
    Y_df, _, _ = LongHorizon.load(directory='./', group=dataset)
    Y_df['ds'] = pd.to_datetime(Y_df['ds'])
    freq = LongHorizonInfo[dataset].freq
    n_time = len(Y_df.ds.unique())
    val_size = int(.2 * n_time)
    test_size = int(.2 * n_time)

    nhits_config = {
        "learning_rate": tune.choice([1e-3]),                                     # Initial Learning rate
        "max_steps": tune.choice([1000]),                                         # Number of SGD steps
        "input_size": tune.choice([5 * horizon]),                                 # input_size = multiplier * horizon
        "batch_size": tune.choice([7]),                                           # Number of series in windows
        "windows_batch_size": tune.choice([256]),                                 # Number of windows in batch
        "n_pool_kernel_size": tune.choice([[2, 2, 2], [16, 8, 1]]),               # MaxPool's Kernelsize
        "n_freq_downsample": tune.choice([[168, 24, 1], [24, 12, 1], [1, 1, 1]]), # Interpolation expressivity ratios
        "activation": tune.choice(['ReLU']),                                      # Type of non-linear activation
        "n_blocks":  tune.choice([[1, 1, 1]]),                                    # Blocks per each 3 stacks
        "mlp_units":  tune.choice([[[512, 512], [512, 512], [512, 512]]]),        # 2 512-Layers per block for each stack
        "interpolation_mode": tune.choice(['linear']),                            # Type of multi-step interpolation
        "val_check_steps": tune.choice([100]),                                    # Compute validation every 100 epochs
        "random_seed": tune.randint(1, 10),
        }

    models = [AutoNHITS(h=horizon,
                        config=nhits_config, 
                        num_samples=1)]

    nf = NeuralForecast(models=models, freq='15min')

    Y_hat_df = nf.cross_validation(df=Y_df, val_size=val_size,
                                   test_size=test_size, n_windows=None)


    y_true = Y_hat_df.y.values
    y_hat = Y_hat_df['AutoNHITS'].values

    n_series = len(Y_df.unique_id.unique())

    y_true = y_true.reshape(n_series, -1, horizon)
    y_hat = y_hat.reshape(n_series, -1, horizon)

    print('Parsed results')
    print('2. y_true.shape (n_series, n_windows, n_time_out):\t', y_true.shape)
    print('2. y_hat.shape  (n_series, n_windows, n_time_out):\t', y_hat.shape)

    print('MAE: ', mae(y_hat, y_true))
    print('MSE: ', mse(y_hat, y_true))