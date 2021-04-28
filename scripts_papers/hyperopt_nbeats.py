import os
import pickle
import glob
import time
import numpy as np
import pandas as pd
import argparse
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from nixtla.losses.numpy import mae, mape, smape, rmse, pinball_loss
from nixtla.experiments.utils import hyperopt_tunning, model_fit_predict
from nixtla.data.datasets.epf import EPF, EPFInfo

TEST_DATE = {'NP': '2016-12-27',
             'PJM':'2016-12-27',
             'BE':'2015-01-04',
             'FR': '2015-01-04',
             'DE':'2016-01-04'}

def get_experiment_space(args):
    """
    Defines the search space for hyperopt. The space depends on the type of model specified.
    For more information of each hyperparameter, refer to NBEATS model comments.
    """
    if args.space == 'nbeats_x':
        if args.data_augmentation:
            idx_to_sample_freq = 24
        else:
            idx_to_sample_freq = 1
        space = {# Architecture parameters
                 'model':'nbeats',
                 'input_size_multiplier': hp.choice('input_size_multiplier', [2, 3, 7]),
                 'output_size': hp.choice('output_size', [24]),
                 'shared_weights': hp.choice('shared_weights', [False]),
                 'activation': hp.choice('activation', ['softplus', 'selu', 'prelu', 'sigmoid']),
                 'initialization':  hp.choice('initialization', ['orthogonal', 'he_uniform',
                                                                 'he_normal', 'glorot_normal']),
                 'stack_types': hp.choice('stack_types', [['identity'],
                                                          1*['identity']+['exogenous_wavenet'],
                                                          ['exogenous_wavenet']+1*['identity'],
                                                          1*['identity']+['exogenous_tcn'],
                                                          ['exogenous_tcn']+1*['identity'] ]),
                 'n_blocks': hp.choice('n_blocks', [ [1, 1] ]),
                 'n_layers': hp.choice('n_layers', [ [2, 2] ]),
                 'n_hidden': hp.quniform('n_hidden', 50, 500, 1),
                 'n_harmonics': hp.choice('n_harmonics', [0]),
                 'n_polynomials': hp.choice('n_polynomials', [0]),
                 'exogenous_n_channels': hp.quniform('exogenous_n_channels', 1, 10, 1),
                 'x_s_n_hidden': hp.choice('x_s_n_hidden', [0]),
                 # Regularization and optimization parameters
                 'batch_normalization': hp.choice('batch_normalization', [True, False]),
                 'dropout_prob_theta': hp.uniform('dropout_prob_theta', 0, 1),
                 'dropout_prob_exogenous': hp.uniform('dropout_prob_exogenous', 0, 0.5),
                 'learning_rate': hp.loguniform('learning_rate', np.log(5e-4), np.log(0.005)),
                 'lr_decay': hp.choice('lr_decay', [0.5]),
                 'n_lr_decay_steps': hp.choice('n_lr_decay_steps', [3]),
                 'weight_decay': hp.choice('weight_decay', [0]),
                 'n_iterations': hp.choice('n_iterations', [30_000]), # 
                 'early_stopping': hp.choice('early_stopping', [10]),
                 'eval_freq': hp.choice('eval_freq', [100]),
                 'n_val_weeks': hp.choice('n_val_weeks', [52]),
                 'loss': hp.choice('loss', ['MAE']),
                 'loss_hypar': hp.choice('loss_hypar', [0.5]),                
                 'val_loss': hp.choice('val_loss', ['MAE']),
                 'l1_theta': hp.choice('l1_theta', [0]),
                 # Data parameters
                 'len_sample_chunks': hp.choice('len_sample_chunks', [None]),
                 'normalizer_y': hp.choice('normalizer_y', [None, 'norm', 'std', 'median', 'invariant']),
                 'normalizer_x': hp.choice('normalizer_x', [None, 'norm', 'std', 'median', 'invariant']),
                 'window_sampling_limit': hp.choice('window_sampling_limit', [365*4*24]),
                 'complete_inputs': hp.choice('complete_inputs', [False]),
                 'complete_sample': hp.choice('complete_sample', [False]),                
                 'frequency': hp.choice('frequency', ['H']),
                 'seasonality': hp.choice('seasonality', [24]),      
                 'idx_to_sample_freq': hp.choice('idx_to_sample_freq', [idx_to_sample_freq]),
                 'val_idx_to_sample_freq': hp.choice('val_idx_to_sample_freq', [24]),
                 'batch_size': hp.choice('batch_size', [256, 512]),
                 'n_series_per_batch': hp.choice('n_series_per_batch', [1]),
                 'random_seed': hp.quniform('random_seed', 1, 1000, 1),
                 'device': hp.choice('device', [None])}
    if args.space == 'nbeats_x_i':
    	if args.data_augmentation:
            idx_to_sample_freq = 24
        else:
            idx_to_sample_freq = 1
        space = {# Architecture parameters
                 'model':'nbeats',
                 'input_size_multiplier': hp.choice('input_size_multiplier', [2, 3, 7]),
                 'output_size': hp.choice('output_size', [24]),
                 'shared_weights': hp.choice('shared_weights', [False]),
                 'activation': hp.choice('activation', ['softplus', 'selu', 'prelu', 'sigmoid']),
                 'initialization':  hp.choice('initialization', ['orthogonal', 'he_uniform',
                                                                 'he_normal', 'glorot_normal']),
                 'stack_types': hp.choice('stack_types', [['trend', 'seasonality', ],
                                                          ['trend', 'seasonality', 'exogenous_wavenet'],
                                                          ['exogenous_tcn', 'trend', 'seasonality'],
                                                          ['exogenous_wavenet', 'trend', 'seasonality']]),
                 'n_blocks': hp.choice('n_blocks', [ [1, 1] ]),
                 'n_layers': hp.choice('n_layers', [ [2, 2] ]),
                 'n_hidden': hp.quniform('n_hidden', 50, 500, 1),
                 'n_harmonics': hp.choice('n_harmonics', [1, 2]),
                 'n_polynomials': hp.choice('n_polynomials', [1, 2, 3]),
                 'exogenous_n_channels': hp.quniform('exogenous_n_channels', 1, 10, 1),
                 'x_s_n_hidden': hp.choice('x_s_n_hidden', [0]),
                 # Regularization and optimization parameters
                 'batch_normalization': hp.choice('batch_normalization', [True, False]),
                 'dropout_prob_theta': hp.uniform('dropout_prob_theta', 0, 1),
                 'dropout_prob_exogenous': hp.uniform('dropout_prob_exogenous', 0, 0.5),
                 'learning_rate': hp.loguniform('learning_rate', np.log(5e-4), np.log(0.005)),
                 'lr_decay': hp.choice('lr_decay', [0.5]),
                 'n_lr_decay_steps': hp.choice('n_lr_decay_steps', [3]),
                 'weight_decay': hp.choice('weight_decay', [0]),
                 'n_iterations': hp.choice('n_iterations', [30_000]), # 
                 'early_stopping': hp.choice('early_stopping', [10]),
                 'eval_freq': hp.choice('eval_freq', [100]),
                 'n_val_weeks': hp.choice('n_val_weeks', [52]),
                 'loss': hp.choice('loss', ['MAE']),
                 'loss_hypar': hp.choice('loss_hypar', [0.5]),                
                 'val_loss': hp.choice('val_loss', ['MAE']),
                 'l1_theta': hp.choice('l1_theta', [0]),
                 # Data parameters
                 'len_sample_chunks': hp.choice('len_sample_chunks', [None]),
                 'normalizer_y': hp.choice('normalizer_y', [None, 'norm', 'std', 'median', 'invariant']),
                 'normalizer_x': hp.choice('normalizer_x', [None, 'norm', 'std', 'median', 'invariant']),
                 'window_sampling_limit': hp.choice('window_sampling_limit', [365*4*24]),
                 'complete_inputs': hp.choice('complete_inputs', [False]),
                 'complete_sample': hp.choice('complete_sample', [False]),                
                 'frequency': hp.choice('frequency', ['H']),
                 'seasonality': hp.choice('seasonality', [24]),      
                 'idx_to_sample_freq': hp.choice('idx_to_sample_freq', [idx_to_sample_freq]),
                 'val_idx_to_sample_freq': hp.choice('val_idx_to_sample_freq', [24]),
                 'batch_size': hp.choice('batch_size', [256, 512]),
                 'n_series_per_batch': hp.choice('n_series_per_batch', [1]),
                 'random_seed': hp.quniform('random_seed', 1, 1000, 1),
                 'device': hp.choice('device', [None])}
    return space

def main(args):
    #----------------------------------------------- Load Data -----------------------------------------------#
    Y_df, X_df, S_df = EPF.load_groups(directory='../data', groups=[args.dataset])

    # Remove test set
    test_date = TEST_DATE[args.dataset]
    Y_train_df = Y_df[Y_df['ds']<test_date].reset_index(drop=True)
    X_train_df = X_df[X_df['ds']<test_date].reset_index(drop=True)

    space = get_experiment_space(args)

    #---------------------------------------------- Directories ----------------------------------------------#
    output_dir = f'./results/{args.dataset}/{args.space}/'
    os.makedirs(output_dir, exist_ok = True)
    assert os.path.exists(output_dir), f'Output dir {output_dir} does not exist'

    if args.experiment_id is None:
        experiment_id = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    else:
        experiment_id = args.experiment_id

    hyperopt_file = output_dir + f'hyperopt_{experiment_id}.p'
    results_file =  output_dir + f'forecasts_{experiment_id}.p'

    if not os.path.isfile(hyperopt_file):
        print('Hyperparameter optimization')
        #----------------------------------------------- Hyperopt -----------------------------------------------#
        trials = hyperopt_tunning(space=space, hyperopt_max_evals=args.hyperopt_max_evals, loss_function=mae,
                                Y_df=Y_train_df, X_df=X_train_df, S_df=S_df, f_cols=[],
                                ds_in_val=52*7*24, n_uids=1, n_val_windows=52, freq='H',
                                is_val_random=args.is_val_random, loss_kwargs={})

        with open(hyperopt_file, "wb") as f:
            pickle.dump(trials, f)
    else:
        print('Hyperparameter optimization already done, using previous results.')

    # Best mc
    trials = pickle.load(open(hyperopt_file, 'rb'))
    best_mc = trials.trials[np.argmin(trials.losses())]['result']['mc']
    best_mc['device'] = None

    y_hat = []
    y_true = []
    start_time = time.time()
    for split in range(728, 0, -1):
        print(20*'-', f'SPLIT {split}', 20*'-')
        len_train = len(Y_df) - split*best_mc['output_size']
        Y_split_df = Y_df.head(len_train)
        X_split_df = X_df.head(len_train)
        
        assert len(Y_df) % best_mc['output_size'] == 0, \
            f'len(Y_df) not compatible with output_size {len(Y_df)} not module {mc["output_size"]}'    
        
        y_true_split, y_hat_split, *_ = model_fit_predict(mc=best_mc, S_df=S_df, Y_df=Y_split_df, X_df=X_split_df, f_cols=[],
                                                          ds_in_test=best_mc['output_size'], ds_in_val=42*7*24,
                                                          n_uids=1, n_val_windows=42, freq='H',
                                                          is_val_random=True)

        y_true.append(y_true_split)
        y_hat.append(y_hat_split)
        print('y_hat', y_hat_split)

    y_true.append(y_true)
    y_hat = np.vstack(y_hat)

    run_time = time.time() - start_time
    print(10*'-', f'Time: {run_time} s', 10*'-')

    # Output evaluation
    evaluation_dict = {'mc': best_mc,
                       'y_true': y_true,
                       'y_hat': y_hat,
                       'run_time': run_time}
    
    with open(results_file, "wb") as f:
            pickle.dump(evaluation_dict, f)


def parse_args():
    desc = "ESRNN for EPF"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, required=True, help='NP')
    parser.add_argument('--space', type=str, required=True, help='Experiment hyperparameter space')
    parser.add_argument('--hyperopt_max_evals', type=int, help='hyperopt_max_evals')
    parser.add_argument('--data_augmentation', type=int, required=True, help='Data augmentation flag')
    parser.add_argument('--is_val_random', type=int, required=True, help='Random validation flag')
    parser.add_argument('--experiment_id', default=None, required=False, type=str, help='string to identify experiment')
    return parser.parse_args()

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    
    main(args)

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate riemann
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts_papers/hyperopt_nbeats.py --dataset 'NP' --space "nbeats_x" --data_augmentation 0 --is_val_random 0 --hyperopt_max_evals 1 --experiment_id "20210401_0_0"
