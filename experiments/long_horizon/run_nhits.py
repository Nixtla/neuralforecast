import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import pandas as pd

from ray import tune

from neuralforecast.auto import AutoNHITS
from neuralforecast.core import NeuralForecast

from neuralforecast.losses.pytorch import MAE, HuberLoss
from neuralforecast.losses.numpy import mae, mse
#from datasetsforecast.long_horizon import LongHorizon, LongHorizonInfo
from datasetsforecast.long_horizon2 import LongHorizon2, LongHorizon2Info

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


if __name__ == '__main__':

    # Parse execution parameters
    verbose = True
    parser = argparse.ArgumentParser()
    parser.add_argument("-horizon", "--horizon", type=int)
    parser.add_argument("-dataset", "--dataset", type=str)
    parser.add_argument("-num_samples", "--num_samples", default=5, type=int)

    args = parser.parse_args()
    horizon = args.horizon
    dataset = args.dataset
    num_samples = args.num_samples

    assert horizon in [96, 192, 336, 720]

    # Load dataset
    #Y_df, _, _ = LongHorizon.load(directory='./data/', group=dataset)
    #Y_df['ds'] = pd.to_datetime(Y_df['ds'])

    Y_df = LongHorizon2.load(directory='./data/', group=dataset)
    freq = LongHorizon2Info[dataset].freq
    n_time = len(Y_df.ds.unique())
    #val_size = int(.2 * n_time)
    #test_size = int(.2 * n_time)
    val_size = LongHorizon2Info[dataset].val_size
    test_size = LongHorizon2Info[dataset].test_size

    # Adapt input_size to available data
    input_size = tune.choice([7 * horizon])
    if dataset=='ETTm1' and horizon==720:
        input_size = tune.choice([2 * horizon])

    nhits_config = {
        #"learning_rate": tune.choice([1e-3]),                                     # Initial Learning rate
        "learning_rate": tune.loguniform(1e-5, 5e-3),
        "max_steps": tune.choice([200, 1000]),                                    # Number of SGD steps
        "input_size": input_size,                                                 # input_size = multiplier * horizon
        "batch_size": tune.choice([7]),                                           # Number of series in windows
        "windows_batch_size": tune.choice([256]),                                 # Number of windows in batch
        "n_pool_kernel_size": tune.choice([[2, 2, 2], [16, 8, 1]]),               # MaxPool's Kernelsize
        "n_freq_downsample": tune.choice([[(96*7)//2, 96//2, 1],
                                          [(24*7)//2, 24//2, 1],
                                          [1, 1, 1]]),                            # Interpolation expressivity ratios
        "dropout_prob_theta": tune.choice([0.5]),                                 # Dropout regularization
        "activation": tune.choice(['ReLU']),                                      # Type of non-linear activation
        "n_blocks":  tune.choice([[1, 1, 1]]),                                    # Blocks per each 3 stacks
        "mlp_units":  tune.choice([[[512, 512], [512, 512], [512, 512]]]),        # 2 512-Layers per block for each stack
        "interpolation_mode": tune.choice(['linear']),                            # Type of multi-step interpolation
        "val_check_steps": tune.choice([100]),                                    # Compute validation every 100 epochs
        "random_seed": tune.randint(1, 10),
        }

    models = [AutoNHITS(h=horizon,
                        loss=HuberLoss(delta=0.5),
                        valid_loss=MAE(),
                        config=nhits_config, 
                        num_samples=num_samples,
                        refit_with_val=True)]

    nf = NeuralForecast(models=models, freq=freq)

    Y_hat_df = nf.cross_validation(df=Y_df, val_size=val_size,
                                   test_size=test_size, n_windows=None)


    y_true = Y_hat_df.y.values
    y_hat = Y_hat_df['AutoNHITS'].values

    n_series = len(Y_df.unique_id.unique())

    y_true = y_true.reshape(n_series, -1, horizon)
    y_hat = y_hat.reshape(n_series, -1, horizon)

    print('\n'*4)
    print('Parsed results')
    print(f'NHITS {dataset} h={horizon}')
    print('test_size', test_size)
    print('y_true.shape (n_series, n_windows, n_time_out):\t', y_true.shape)
    print('y_hat.shape  (n_series, n_windows, n_time_out):\t', y_hat.shape)

    print('MSE: ', mse(y_hat, y_true))
    print('MAE: ', mae(y_hat, y_true))

    # Save Outputs
    if not os.path.exists(f'./data/{dataset}'):
        os.makedirs(f'./data/{dataset}')
    yhat_file = f'./data/{dataset}/{horizon}_forecasts.csv'
    Y_hat_df.to_csv(yhat_file, index=False)
