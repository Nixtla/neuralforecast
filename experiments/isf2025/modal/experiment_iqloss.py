#%% Load packages
import inspect
import logging
import s3fs
import torch
import time
from s3fs.core import S3FileSystem
fs = S3FileSystem()

torch.set_float32_matmul_precision("high")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)

import pandas as pd
import polars as pl

from datasetsforecast.long_horizon import LongHorizon, LongHorizonInfo
from datasetsforecast.m5 import M5
from functools import partial
from neuralforecast.core import NeuralForecast
from neuralforecast.models import NHITS, BiTCN, TSMixer, DLinear, iTransformer, DeepNPTS, PatchTST, TFT, TiDE, MLP, LSTM, DeepAR, MLPMultivariate, NLinear, NBEATS, TimeMixer, VanillaTransformer, TimesNet
from neuralforecast.losses.pytorch import MAE, MQLoss, IQLoss, MSE, HuberLoss, HuberMQLoss, GMM, IQLoss, HuberIQLoss
from neuralforecast.utils import PredictionIntervals
from pathlib import Path
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mse, smape, mqloss, scaled_mqloss, scaled_crps, mase, coverage

polars_to_pandas_freq = {
    "D": "1d",
    "W": "1w",
    "M": "1mo",
    "Q": "1q",
    "Y": "1y",
    "H": "1h",
    "T": "1m",
    "15T": "15m",
    "30T": "30m",
    "60T": "1h",
    "10M": "10m",
}

seasonality = {
    "D": 7,
    "W": 52,
    "M": 12,
    "Q": 4,
    "Y": 1,
    "H": 24,
    "T": 1440,
    "15T": 96,
    "30T": 48,
    "60T": 24,
    "10M": 144,}
#%%
def load_models(horizon, input_size, n_series, seed):
    loss = HuberIQLoss
    valid_loss = HuberIQLoss
    level= [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99]
    early_stop_patience_steps = 5
    val_check_steps = 10
    max_steps = 1000
    scaler_type = 'robust'
    models = [    
               NHITS(h=horizon,
                    input_size=input_size,
                    scaler_type=scaler_type,
                    loss=loss(),
                    valid_loss=valid_loss(),
                    random_seed=seed,
                    # early_stop_patience_steps=early_stop_patience_steps,
                    # val_check_steps=val_check_steps,
                    max_steps=max_steps,
                    ),   
               # DeepNPTS(h=horizon,
               #      input_size=input_size,
               #      scaler_type=scaler_type,
               #      loss=loss(level=level),
               #      random_seed=seed,
               #      # early_stop_patience_steps=early_stop_patience_steps,
               #      # val_check_steps=val_check_steps,
               #      max_steps=max_steps,
               #      ),  
               DLinear(h=horizon,
                    input_size=input_size,
                    scaler_type=scaler_type,
                    loss=loss(),
                    valid_loss=valid_loss(),
                    random_seed=seed,
                    # early_stop_patience_steps=early_stop_patience_steps,
                    # val_check_steps=val_check_steps,
                    max_steps=max_steps,
                    ), 
               TiDE(h=horizon,
                    input_size=input_size,
                    scaler_type=scaler_type,
                    loss=loss(),
                    valid_loss=valid_loss(),
                    random_seed=seed,
                    # early_stop_patience_steps=early_stop_patience_steps,
                    # val_check_steps=val_check_steps,
                    max_steps=max_steps,
                    ), 
               BiTCN(h=horizon,
                    input_size=input_size,
                    scaler_type=scaler_type,
                    loss=loss(),
                    valid_loss=valid_loss(),
                    random_seed=seed,
                    # early_stop_patience_steps=early_stop_patience_steps,
                    # val_check_steps=val_check_steps,
                    max_steps=max_steps,
                    ),                  
               MLP(h=horizon,
                    input_size=input_size,
                    scaler_type=scaler_type,
                    loss=loss(),
                    valid_loss=valid_loss(),
                    random_seed=seed,
                    # early_stop_patience_steps=early_stop_patience_steps,
                    # val_check_steps=val_check_steps,
                    max_steps=max_steps,
                    ),         
               PatchTST(h=horizon,
                    input_size=input_size,
                    scaler_type=scaler_type,
                    loss=loss(),
                    valid_loss=valid_loss(),
                    random_seed=seed,
                    # early_stop_patience_steps=early_stop_patience_steps,
                    # val_check_steps=val_check_steps,
                    max_steps=max_steps,
               ),
               TFT(h=horizon,
                    input_size=input_size,
                    scaler_type=scaler_type,
                    loss=loss(),
                    valid_loss=valid_loss(),
                    random_seed=seed,
                    # early_stop_patience_steps=early_stop_patience_steps,
                    # val_check_steps=val_check_steps,
                    max_steps=max_steps,
               ),
               TSMixer(h=horizon,
                    input_size=input_size,
                    scaler_type=scaler_type,
                    n_series=n_series,
                    loss=loss(),
                    valid_loss=valid_loss(),
                    random_seed=seed,
                    # early_stop_patience_steps=early_stop_patience_steps,
                    # val_check_steps=val_check_steps,
                    max_steps=max_steps,
                    ),                                                                                         
               iTransformer(h=horizon,
                    input_size=input_size,
                    scaler_type=scaler_type,
                    n_series=n_series,
                    loss=loss(),
                    valid_loss=valid_loss(),
                    random_seed=seed,
                    # early_stop_patience_steps=early_stop_patience_steps,
                    # val_check_steps=val_check_steps,
                    max_steps=max_steps,
                    ), 
               LSTM(h=horizon,
                    input_size=input_size,
                    scaler_type=scaler_type,
                    loss=loss(),
                    valid_loss=valid_loss(),
                    random_seed=seed,
                    # early_stop_patience_steps=early_stop_patience_steps,
                    # val_check_steps=val_check_steps,
                    max_steps=max_steps,
                    ),
               DeepAR(h=horizon,
                    input_size=input_size,
                    scaler_type=scaler_type,
                    loss=loss(),
                    valid_loss=valid_loss(),
                    random_seed=seed,
                    # early_stop_patience_steps=early_stop_patience_steps,
                    # val_check_steps=val_check_steps,
                    max_steps=max_steps,
                    ),   
               MLPMultivariate(h=horizon,
                    input_size=input_size,
                    scaler_type=scaler_type,
                    n_series=n_series,
                    loss=loss(),
                    valid_loss=valid_loss(),
                    random_seed=seed,
                    # early_stop_patience_steps=early_stop_patience_steps,
                    # val_check_steps=val_check_steps,
                    max_steps=max_steps,
                    ),
               NLinear(h=horizon,
                    input_size=input_size,
                    scaler_type=scaler_type,
                    loss=loss(),
                    valid_loss=valid_loss(),
                    random_seed=seed,
                    # early_stop_patience_steps=early_stop_patience_steps,
                    # val_check_steps=val_check_steps,
                    max_steps=max_steps,
               ),                   
               NBEATS(h=horizon,
                    input_size=input_size,
                    scaler_type=scaler_type,
                    loss=loss(),
                    valid_loss=valid_loss(),
                    random_seed=seed,
                    # early_stop_patience_steps=early_stop_patience_steps,
                    # val_check_steps=val_check_steps,
                    max_steps=max_steps,
                    ),
               TimeMixer(h=horizon,
                    input_size=input_size,
                    n_series=n_series,
                    scaler_type=scaler_type,
                    loss=loss(),
                    valid_loss=valid_loss(),
                    random_seed=seed,
                    # early_stop_patience_steps=early_stop_patience_steps,
                    # val_check_steps=val_check_steps,
                    max_steps=max_steps,
                    ),
               VanillaTransformer(h=horizon,
                    input_size=input_size,
                    scaler_type=scaler_type,
                    loss=loss(),
                    valid_loss=valid_loss(),
                    random_seed=seed,
                    # early_stop_patience_steps=early_stop_patience_steps,
                    # val_check_steps=val_check_steps,
                    max_steps=max_steps,
                    ),   
               TimesNet(h=horizon,
                    input_size=input_size,
                    scaler_type=scaler_type,
                    loss=loss(),
                    valid_loss=valid_loss(),
                    random_seed=seed,
                    # early_stop_patience_steps=early_stop_patience_steps,
                    # val_check_steps=val_check_steps,
                    max_steps=max_steps,
                    ),                                                                                                                                                 
          ]

    return models
#%% Run cross-validation
def cross_validation(dataset, horizon, metrics, seed=1234567, path_prefix="s3://timenet/isf2025/iqloss"):
    
    id_col = 'unique_id'
    time_col = 'ds'
    target_col = 'y'

    # Access the frequency, validation size, test_size and n_series of the dataset
    if dataset != "M5":
       freq = LongHorizonInfo[dataset].freq
       test_size = 3 * horizon
       n_series = LongHorizonInfo[dataset].n_ts  
       # Load the dataset
       Y_df, X_df, _ = LongHorizon.load(directory=f'/isf2025/data', group=dataset)
       Y_df = Y_df.merge(X_df)
       Y_df[time_col] = pd.to_datetime(Y_df[time_col])
    else:
       freq = "D"
       test_size = 28 * 3
       Y_df, _, _ = M5.load(directory=f'/isf2025/data')
       n_series = 30490
       Y_df[time_col] = pd.to_datetime(Y_df[time_col])

    # Create the model list
    if dataset == "ILI":
       input_size = 24 
    elif dataset == "M5":
       input_size = 84
    else:
       input_size = 96
    models = load_models(horizon, input_size, n_series, seed=seed)

    # Instantiate NeuralForecast
    Y_df_pl = pl.from_pandas(Y_df)
    train_file_path = f"{path_prefix}/{dataset}/train_h{horizon}.parquet"
    exists = fs.exists(train_file_path)
    if not exists:
        print(f"File {train_file_path} does not exist. Saving the dataset.")
        Y_df.to_parquet(train_file_path)

    eval_df = None
    time_df = None
    for model in models:
        file_path_predictions = f"{path_prefix}/{dataset}/{model}/predictions_h{horizon}.parquet"
        if fs.exists(file_path_predictions):
            print(f"Skipping {model} for {dataset} with horizon {horizon} as predictions already exist.")
            continue

        # Set the model parameters
        nf = NeuralForecast(
            models=[model],
            freq=polars_to_pandas_freq[freq],)   

        # Predict the test set
        level = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99]
        start = time.perf_counter()
        Y_hat_df = nf.cross_validation(df=Y_df_pl,
                                    n_windows=None,
                                    test_size=test_size,
                                    level=level,
                                    refit=True,
                                    step_size=horizon,
                                    )       
        end = time.perf_counter()
        time_lapsed = end - start

        # Save the metric results to a dictionary
        metrics_ = []
        for metric in metrics:
            # if seasonlity is an argument of the metric, we create a new instance of the metric with the seasonality
            metric_p = metric
            sig = inspect.signature(metric)
            if 'seasonality' in sig.parameters:
                metric_p = partial(metric, seasonality=seasonality[freq])
            
            metrics_.append(metric_p)
        
        train_df = Y_df_pl.filter(pl.col(time_col) < Y_hat_df[time_col].min())
        Y_hat_df = Y_hat_df.rename({f"{model.__class__.__name__}-median": f"{model.__class__.__name__}"})
        eval_df = evaluate(df=Y_hat_df.drop(["cutoff"]),
                        train_df=train_df,
                        metrics=metrics_,
                        level=level[1:],
                        agg_fn="mean",
                        )
        eval_df = eval_df.with_columns([pl.lit(dataset).alias("dataset"),
                                        pl.lit(horizon).alias("horizon"),
                                        ])
        time_df = pl.DataFrame({"time_lapsed": [time_lapsed]})
        time_df = time_df.with_columns([pl.lit(dataset).alias("dataset"),
                                        pl.lit(horizon).alias("horizon"),
                                        ])

        # Store predictions and data
        Y_hat_df.to_pandas().to_parquet(f"{path_prefix}/{dataset}/{model}/predictions_h{horizon}.parquet")
        eval_df.to_pandas().to_parquet(f"{path_prefix}/{dataset}/{model}/eval_h{horizon}.parquet")
        time_df.to_pandas().to_parquet(f"{path_prefix}/{dataset}/{model}/time_h{horizon}.parquet")