import os
import argparse
import time

import pandas as pd
import numpy as np
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch

from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import HuberLoss
from neuralforecast.auto import AutoNHITS, AutoTFT

from config import config_nhits, config_tft

MODEL_DICT = {'nhits': AutoNHITS,
              'tft': AutoTFT}
CONFIG_DICT = {'nhits': config_nhits,
                'tft': config_tft}

def load_model(args):
    model_name = args.model

    # Config, add exogenous features
    config = CONFIG_DICT[model_name]
    config['hist_exog_list'] = args.hist_exog_list
    config['stat_exog_list'] = args.stat_exog_list

    # Load model
    model_func = MODEL_DICT[model_name]
    if model_name == 'nhits':
        model = model_func(h=args.horizon,
                        config=config,
                        loss=HuberLoss(),
                        search_alg=HyperOptSearch(),
                        num_samples=args.num_samples)
    if model_name == 'tft':
        model = model_func(h=args.horizon,
                           n_series=12,
                           config=config,
                           loss=HuberLoss(),
                           search_alg=HyperOptSearch(),
                           num_samples=args.num_samples)
    return model


def main(args):

    Y_df = pd.read_csv('data_glucose/ohiot1dm_exog_9_day_test.csv')
    Y_df['ds'] = pd.to_datetime(Y_df['ds'])

    static_df = pd.read_csv('data_glucose/ohiot1dm_static.csv')

    # days = 10
    # Y_df = Y_df.groupby('unique_id').tail(days*288+2691+2691)
    # print('Y_df.shape: ', Y_df.shape)
    # print(Y_df.unique_id.value_counts())

    # Train model
    model = load_model(args)
    nf = NeuralForecast(models=[model],
                        freq='5min')
    start = time.time()
    Y_hat_df = nf.cross_validation(df=Y_df,
                                   static_df=static_df,
                                   val_size=2691,
                                   test_size=2691,
                                   step_size=1,
                                   n_windows=None)
    end = time.time()
    print(f'Time: {end-start} seconds')
    # Save forecasts
    forecasts_dir = f'./results_glucose/h_{args.horizon}/exogenous/'
    os.makedirs(forecasts_dir, exist_ok=True)
    Y_hat_df.to_csv(forecasts_dir + f'{args.model}_{args.experiment_id}.csv', index=False)
    

def parse_args():
    parser = argparse.ArgumentParser(description="script arguments")
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--experiment_id', type=str, help='identify experiment')
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    
    # Other params
    args.num_samples = 20
    args.horizon = 6
    args.hist_exog_list = ['CHO', 'bolus_insulin', 'basal_insulin']
    args.stat_exog_list = ['#559', '#563', '#570', '#575', '#588', 
                           '#591', '#540', '#544', '#552', '#567',
                           '#584', 'insulin_type_novalog', 'female',
                           'age_20_40', 'age_40_60', 'pump_model_630G']

    main(args)

# CUDA_VISIBLE_DEVICES=0 python run_exogenous.py --model "nhits" --experiment_id "20230901_10_1"
# CUDA_VISIBLE_DEVICES=0 python run_exogenous.py --model "tft" --experiment_id "20230829"