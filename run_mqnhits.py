import os
import pickle
import time
import argparse
import pandas as pd

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from neuralforecast.data.datasets.long_horizon import LongHorizon
from neuralforecast.losses.numpy import mqloss
from neuralforecast.experiments.utils import hyperopt_tunning

def get_experiment_space(args):
    space= {# Architecture parameters
            'model':'mqnhits',
            'mode': 'simple',
            'n_time_in': hp.choice('n_time_in', [5*args.horizon]),
            'n_time_out': hp.choice('n_time_out', [args.horizon]),
            'quantiles': hp.choice('quantiles', [ [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] ]),
            'n_x_hidden': hp.choice('n_x_hidden', [0]),
            'n_s_hidden': hp.choice('n_s_hidden', [0]),
            'shared_weights': hp.choice('shared_weights', [False]),
            'activation': hp.choice('activation', ['ReLU']),
            'initialization':  hp.choice('initialization', ['lecun_normal']),
            'stack_types': hp.choice('stack_types', [ 3*['identity'] ]),
            'constant_n_blocks': hp.choice('constant_n_blocks', [ 1 ]),
            'constant_n_layers': hp.choice('constant_n_layers', [ 2 ]),
            'constant_n_mlp_units': hp.choice('constant_n_mlp_units', [ 512 ]),
            'n_pool_kernel_size': hp.choice('n_pool_kernel_size', [ 3*[1], 3*[2], 3*[4], 3*[8], [8, 4, 1], [16, 8, 1] ]),
            'n_freq_downsample': hp.choice('n_freq_downsample', [ [168, 24, 1], [24, 12, 1],
                                                                  [180, 60, 1], [60, 8, 1],
                                                                  [40, 20, 1]
                                                                ]),
            'pooling_mode': hp.choice('pooling_mode', [ 'max' ]),
            'interpolation_mode': hp.choice('interpolation_mode', ['linear']),
            # Regularization and optimization parameters
            'batch_normalization': hp.choice('batch_normalization', [False]),
            'dropout_prob_theta': hp.choice('dropout_prob_theta', [ 0 ]),
            'dropout_prob_exogenous': hp.choice('dropout_prob_exogenous', [0]),
            'learning_rate': hp.choice('learning_rate', [0.001]),
            'lr_decay': hp.choice('lr_decay', [0.5] ),
            'n_lr_decays': hp.choice('n_lr_decays', [3]), 
            'weight_decay': hp.choice('weight_decay', [0] ),
            'max_epochs': hp.choice('max_epochs', [None]),
            'max_steps': hp.choice('max_steps', [5]), # 1_000
            'early_stop_patience': hp.choice('early_stop_patience', [10]),
            'eval_freq': hp.choice('eval_freq', [50]),
            'loss_train': hp.choice('loss', ['MQ']),
            'loss_hypar': hp.choice('loss_hypar', [0.5]),                
            'loss_valid': hp.choice('loss_valid', ['MQ']),
            'l1_theta': hp.choice('l1_theta', [0]),
            # Data parameters
            'scaler': hp.choice('scaler', [None]),
            'complete_windows':  hp.choice('complete_windows', [True]),
            'frequency': hp.choice('frequency', ['H']),
            'seasonality': hp.choice('seasonality', [24]),      
            'idx_to_sample_freq': hp.choice('idx_to_sample_freq', [1]),
            'val_idx_to_sample_freq': hp.choice('val_idx_to_sample_freq', [1]),
            'batch_size': hp.choice('batch_size', [1]),
            'n_windows': hp.choice('n_windows', [256]),
            'random_seed': hp.quniform('random_seed', 1, 10, 1)}
    return space

def main(args):

    #----------------------------------------------- Load Data -----------------------------------------------#
    Y_df, _, _ = LongHorizon.load(group=args.dataset, directory='./data')

    X_df = None
    S_df = None

    print('Y_df: ', Y_df.head())
    if args.dataset == 'ETTm2':
        len_val = 11520
        len_test = 11520
    if args.dataset == 'Exchange':
        len_val = 760
        len_test = 1517
    if args.dataset == 'ECL':
        len_val = 2632
        len_test = 5260
    if args.dataset == 'TrafficL':
        len_val = 1756
        len_test = 3508
    if args.dataset == 'Weather':
        len_val = 5270
        len_test = 10539
    if args.dataset == 'ILI':
        len_val = 97
        len_test = 193

    space = get_experiment_space(args)

    #---------------------------------------------- Directories ----------------------------------------------#
    results_dir = f'./results/multivariate/{args.dataset}_{args.horizon}/NHITS/{args.experiment_id}/'

    os.makedirs(results_dir, exist_ok = True)
    assert os.path.exists(results_dir), f'Output dir {results_dir} does not exist'

    #----------------------------------------------- Hyperopt -----------------------------------------------#
    hyperopt_tunning(space=space, hyperopt_max_evals=args.hyperopt_max_evals, loss_function_val=mqloss,
                     loss_functions_test={'MQ':mqloss},
                     Y_df=Y_df, X_df=X_df, S_df=S_df, f_cols=[],
                     ds_in_val=len_val, ds_in_test=len_test,
                     return_forecasts=False,
                     return_model=False,
                     save_trials=True,
                     results_dir=results_dir,
                     step_save_progress=5,
                     verbose=True,
                     loss_kwargs={})

def parse_args():
    desc = "Example of hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--hyperopt_max_evals', type=int, help='hyperopt_max_evals')
    parser.add_argument('--experiment_id', default=None, required=False, type=str, help='string to identify experiment')
    return parser.parse_args()

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    horizons = [96, 192, 336, 720]
    ILI_horizons = [24, 36, 48, 60]
    datasets = ['ETTm2', 'Exchange', 'Weather', 'ILI', 'ECL', 'TrafficL']

    for dataset in datasets:
        # Horizon
        if dataset == 'ili':
            horizons_dataset = ILI_horizons
        else:
            horizons_dataset = horizons
        for horizon in horizons_dataset:
            print(50*'-', dataset, 50*'-')
            print(50*'-', horizon, 50*'-')
            start = time.time()
            args.dataset = dataset
            args.horizon = horizon
            main(args)
            print('Time: ', time.time() - start)

    main(args)

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate nixtla
# CUDA_VISIBLE_DEVICES=0 python run_mqnhits.py --hyperopt_max_evals 10 --experiment_id "debug"