import os
import argparse
import time
import pandas as pd
import numpy as np

from datasetsforecast.m3 import M3
from datasetsforecast.losses import mae
from statsforecast.utils import AirPassengersDF

from statsforecast import StatsForecast
from statsforecast.models import Naive, AutoARIMA

HORIZON_DICT = {'yearly': 6,
                'quarterly': 8,
                'monthly': 18,
                'daily': 14}

def main(args):

    if (args.target_dataset == 'M3'):
        if args.frequency == 'yearly':
            Y_df_target, *_ = M3.load(directory='./', group='Yearly')
            frequency = 'Y'
            season_length = 1
        elif args.frequency == 'quarterly':
            Y_df_target, *_ = M3.load(directory='./', group='Quarterly')
            frequency = 'Q'
            season_length = 4
        elif args.frequency == 'monthly':
            Y_df_target, *_ = M3.load(directory='./', group='Monthly')
            frequency = 'M'
            season_length = 12
        elif args.frequency == 'daily':
            Y_df_target, *_ = M3.load(directory='./', group='Other')
            frequency = 'D'
            season_length = 7

    if model == 'naive':
        models = [Naive()]
    elif model == 'autoarima':
        models = [AutoARIMA(season_length=season_length)]

    horizon = HORIZON_DICT[args.frequency]

    sf = StatsForecast(df=Y_df_target,
                       models=models,
                       freq=frequency,
                       n_jobs=-1)

    Y_hat_df = sf.cross_validation(df=Y_df_target,
                                   h=horizon,
                                   step_size=1,
                                   n_windows=1)
    
    # Store forecasts results, also check if this folder exists/create it if its done
    forecasts_dir = f'./results/transferability/forecasts/{args.target_dataset}/{args.frequency}/'
    os.makedirs(forecasts_dir, exist_ok=True)

    forecasts_dir = f'{forecasts_dir}/{args.model}_{args.experiment_id}.csv'
    Y_hat_df.to_csv(forecasts_dir, index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="script arguments")
    parser.add_argument('--target_dataset', type=str, help='run model on this dataset')
    parser.add_argument('--experiment_id', type=str, help='identify experiment')
    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    for frequency in ['yearly', 'quarterly','monthly', 'daily']:
        for model in ['autoarima']: # ,'autoarima' naive
            args.model = model
            args.frequency = frequency
            print(f'Running {frequency} {model}!!')
            main(args)

# CUDA_VISIBLE_DEVICES=0 python run_baselines.py --target_dataset "M3" --experiment_id "20230816"
