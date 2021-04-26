import os
import pickle
import glob
import numpy as np
import pandas as pd
import argparse
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from nixtla.losses.numpy import mae, mape, smape, rmse, pinball_loss
from nixtla.experiments.utils import hyperopt_tunning
from nixtla.data.datasets.epf import EPF, EPFInfo


def get_experiment_space(args):
    esrnn_space = {'model': hp.choice('model', ['new_rnn']),
               # Architecture parameters
               'input_size_multiplier': hp.choice('input_size_multiplier', [1,2,7]),
               'output_size': hp.choice('output_size', [24]),
               'dilations': hp.choice('dilations', [ [[1, 2]], [[1,2], [7, 14]] ]),
               'es_component': hp.choice('es_component', ['multiplicative']),
               'cell_type': hp.choice('cell_type', ['LSTM']),
               'state_hsize': hp.quniform('state_hsize', 10, 100, 10),
               'add_nl_layer': hp.choice('add_nl_layer', [True, False]),
               'seasonality': hp.choice('seasonality', [ [24] ]),
               # Regularization and optimization parameters
               'n_iterations':hp.choice('n_iterations', [1000]),
               'early_stopping':hp.choice('early_stopping', [10]),
               'eval_freq': hp.choice('eval_freq', [10]),
               'batch_size': hp.choice('batch_size', [32]),
               'learning_rate': hp.loguniform('learning_rate', np.log(5e-4), np.log(0.01)),
               'lr_scheduler_step_size': hp.choice('lr_scheduler_step_size', [100]),
               'lr_decay': hp.quniform('lr_decay', 0.5, 0.8, 0.1),
               'per_series_lr_multip': hp.choice('per_series_lr_multip', [0.5, 1.0, 1.5, 2.0, 3.0]),
               'gradient_eps': hp.choice('gradient_eps', [1e-8]),
               'gradient_clipping_threshold': hp.choice('gradient_clipping_threshold', [10, 50]),
               'rnn_weight_decay': hp.choice('rnn_weight_decay', [0, 0.0005, 0.005]),
               'noise_std': hp.loguniform('noise_std', np.log(0.0001), np.log(0.001)),
               'level_variability_penalty': hp.quniform('level_variability_penalty', 0, 100, 10),
               'testing_percentile': hp.choice('testing_percentile', [50]),
               'training_percentile': hp.choice('training_percentile', [48, 49, 50, 51]),
               'random_seed': hp.quniform('random_seed', 1, 1000, 1),
               'loss': hp.choice('loss', ['SMYL']),
               'val_loss': hp.choice('val_loss', ['MAE']),
               # Data parameters
               'len_sample_chunks': hp.choice('len_sample_chunks', [7*3*24]),
               'window_sampling_limit': hp.choice('window_sampling_limit', [500_000]),
               'complete_inputs': hp.choice('complete_inputs', [True]),
               'complete_sample': hp.choice('complete_sample', [True]),
               'idx_to_sample_freq': hp.choice('idx_to_sample_freq', [24]),
               'val_idx_to_sample_freq': hp.choice('val_idx_to_sample_freq', [24]),
               'n_series_per_batch': hp.choice('n_series_per_batch', [1]),
               'normalizer_y': hp.choice('normalizer_y', [None]),
               'normalizer_x': hp.choice('normalizer_x',  [None])}
    return esrnn_space

def main(args):
    Y_df, X_df, S_df = EPF.load_groups(directory='../data', groups=[args.dataset])

    X_df = X_df[['unique_id', 'ds', 'week_day']]
    Y_min = Y_df.y.min()
    Y_df.y = Y_df.y - Y_min + 20

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

    trials = hyperopt_tunning(space=space, hyperopt_iters=args.hyperopt_iters, loss_function=mae, Y_df=Y_df, X_df=X_df, S_df=S_df,
                              ds_in_test=728*24, shuffle_outsample=False)

    with open(hyperopt_file, "wb") as f:
        pickle.dump(trials, f)


def parse_args():
    desc = "ESRNN for EPF"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, required=True, help='NP')
    parser.add_argument('--space', type=str, required=True, help='Experiment hyperparameter space')
    parser.add_argument('--hyperopt_iters', type=int, help='hyperopt_iters')
    parser.add_argument('--experiment_id', default=None, required=False, type=str, help='string to identify experiment')
    return parser.parse_args()

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    
    main(args)

# PYTHONPATH=. python scripts_papers/hyperopt_esrnn.py --dataset 'NP' --space "esrnn_lstm" --hyperopt_iters 50 --experiment_id "20210316_1"
