import os
import numpy as np
import pickle
import time
import argparse
import pandas as pd

from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.auto import AutoPatchTST
from datasetsforecast.long_horizon import LongHorizon


def get_random_occlusion_mask(dataset, n_intervals, occlusion_prob):
    n_features, len_dataset = dataset.shape

    interval_size = int(np.ceil(len_dataset/n_intervals))
    mask = np.ones(dataset.shape)
    for i in range(n_intervals):
        u = np.random.rand(n_features)
        mask_interval = (u>occlusion_prob)*1
        mask[:, i*interval_size:(i+1)*interval_size] = mask[:, i*interval_size:(i+1)*interval_size]*mask_interval[:,None]

    # Add one random interval for complete missing features 
    feature_sum = mask.sum(axis=1)
    missing_features = np.where(feature_sum==0)[0]
    for feature in missing_features:
        i = np.random.randint(0, n_intervals)
        mask[feature, i*interval_size:(i+1)*interval_size] = 1

    return mask

def get_experiment_space(args):

    if args.model == 'PatchTST':
        config = {# Architecture parameters
            "input_size": tune.choice([104]),
            "hidden_size": tune.choice([16, 128, 256]),
            "n_head": tune.choice([4]),
            "patch_len": tune.choice([24]),
            "stride": tune.choice([2]),
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "scaler_type": tune.choice([None, "robust", "standard"]),
            "revin": tune.choice([False, True]),
            "max_steps": tune.choice([500, 1000, 5000]),
            "batch_size": tune.choice([32, 64, 128, 256]),
            "windows_batch_size": tune.choice([128, 256, 512, 1024]),
            "random_seed": tune.randint(1, 10)
        }
    return config

def main(args):

    #----------------------------------------------- Load Data -----------------------------------------------#
    if args.dataset in ['simulated7_long', 'simulated7_long_trend', 'simulated7_long_amplitude']:
        Y_df = pd.read_csv(f'./data/{args.dataset}/{args.dataset}.csv')
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        X_df = Y_df.copy()
        X_df = X_df[['unique_id','ds']]

    elif args.dataset == 'solar':
        Y_df = pd.read_csv('./data/solar/solar.csv')
        Y_df['hour'] = pd.to_datetime(Y_df['hour'])
        Y_df = Y_df.rename(columns={'hour':'ds'})
        Y = Y_df['y'].values.reshape(32,-1)
        Y = Y/Y.max(axis=1, keepdims=1)
        Y_df['y'] = Y.flatten()

        X_df = Y_df.copy()
        X_df = X_df[['unique_id','ds']]

    else:
        Y_df, X_df, S_df = LongHorizon.load(directory='./data', group=args.dataset)
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])

    S_df = None

    print('Y_df: ', Y_df.head())

    if args.dataset in ['simulated7_long', 'simulated7_long_trend', 'simulated7_long_amplitude']:
        len_val = args.horizon*(1992//args.horizon)
        len_test = args.horizon*(1992//args.horizon)
        args.n_y = 7
    if args.dataset == 'solar':
        len_val = args.horizon*(1440//args.horizon)
        len_test = args.horizon*(1440//args.horizon)
        args.n_y = 32
    if args.dataset == 'ETTm2':
        len_val = args.horizon*(11520//args.horizon)
        len_test = args.horizon*(11520//args.horizon)
        args.n_y = 7
        freq = '15T'
    if args.dataset == 'Exchange':
        len_val = args.horizon*(744//args.horizon)
        len_test = args.horizon*(1512//args.horizon)
        args.n_y = 8
        freq = 'D'
    if args.dataset == 'Weather':
        len_val = args.horizon*(5270//args.horizon)
        len_test = args.horizon*(10536//args.horizon)
        args.n_y = 21
        freq = '10M'
    if args.dataset == 'ILI':
        len_val =  args.horizon*(96//args.horizon)
        len_test = args.horizon*(192//args.horizon)
        args.n_y = 7
        freq = 'W'

    Y_df = Y_df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
    Y_df['ds'] = np.tile(np.array(range(len(Y_df)//args.n_y)), args.n_y)

    # Masks
    n_series = args.n_y
    n_time = len(Y_df)//n_series # asssuming square data

    args.occlusion_intervals = int(np.ceil(n_time/args.occlusion_size))

    if args.dataset in ['simulated7_long', 'simulated7_long_trend', 'simulated7_long_amplitude', 'solar']:
        mask_filename = f'./data/{args.dataset}/mask_{args.occlusion_size}_{args.occlusion_prob}.p'
    else:
        if args.dataset=='Weather':
            dataset = 'weather'
        elif args.dataset=='ILI':
            dataset = 'ili'
        else:
            dataset = args.dataset
        mask_filename = f'./data/longhorizon/datasets/{dataset}/M/mask_{args.occlusion_size}_{args.occlusion_prob}.p'
        
    if os.path.exists(mask_filename):
        print(f'Train mask {mask_filename} loaded!')
        mask = pickle.load(open(mask_filename,'rb'))
    else:
        print('Train mask not found, creating new one')
        mask = get_random_occlusion_mask(dataset=np.ones((n_series, n_time)), n_intervals=args.occlusion_intervals, occlusion_prob=args.occlusion_prob)
        with open(mask_filename,'wb') as f:
            pickle.dump(mask, f)
        print(f'Train mask {mask_filename} created!')

    # Hide data with 0s
    Y_df['y'] = Y_df['y']*mask.flatten()

    #---------------------------------------------- Directories ----------------------------------------------#
    output_dir = f'./results/{args.dataset}_{args.occlusion_size}_{args.occlusion_prob}_{args.horizon}/{args.model}/'

    os.makedirs(output_dir, exist_ok = True)
    assert os.path.exists(output_dir), f'Output dir {output_dir} does not exist'

    results_file = output_dir + f'Y_hat_{args.experiment_id}.csv'

    if not os.path.isfile(results_file):
        print('Hyperparameter optimization')
        config = get_experiment_space(args)
        if args.model=='PatchTST':
            model = AutoPatchTST(h=args.horizon,
                                loss=MAE(),
                                config=config,
                                search_alg=HyperOptSearch(),
                                num_samples=args.hyperopt_max_evals
                                )
        
        nf = NeuralForecast(models=[model],
                            freq=freq)

        Y_hat_df = nf.cross_validation(df=Y_df,                                    
                                       val_size=len_val,
                                       test_size=len_test,
                                       step_size=args.horizon,
                                       n_windows=None)

        Y_hat_df.to_csv(results_file, index=False)
    else:
        print('Hyperparameter optimization already done!')

def parse_args():
    desc = "Example of hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--hyperopt_max_evals', type=int, help='hyperopt_max_evals')
    parser.add_argument('--occlusion_size', type=int, help='occlusion_intervals')
    parser.add_argument('--experiment_id', default=None, required=False, type=str, help='string to identify experiment')
    return parser.parse_args()

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    occlusion_probs = [0.0] # [0.0, 0.2, 0.4, 0.6, 0.8]
    horizons = [24] # [192, 336]
    ILI_horizons = [24] # 36, 48, 60
    datasets = ['ILI'] #[''ILI', 'Exchange', 'ETTm2', 'Weather']

    # Dataset loop
    for dataset in datasets:
        # Horizon loop
        if dataset == 'ILI':
            horizons_dataset = ILI_horizons
        else:
            horizons_dataset = horizons
        for horizon in horizons_dataset:
            # Occlusion prob loop
            for occlusion_prob in occlusion_probs:
                print(50*'-', dataset, 50*'-')
                print(50*'-', horizon, 50*'-')
                print(50*'-', occlusion_prob, 50*'-')
                start = time.time()
                args.dataset = dataset
                args.horizon = horizon
                args.occlusion_prob = occlusion_prob
                main(args)
                print('Time: ', time.time() - start)

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate neuralforecast
# CUDA_VISIBLE_DEVICES=0 python run_transformers.py --model 'PatchTST' --hyperopt_max_evals 1 --occlusion_size 10 --experiment_id "test"