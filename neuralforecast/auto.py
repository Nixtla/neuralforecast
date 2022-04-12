# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/auto.ipynb (unless otherwise specified).

__all__ = ['AutoBaseModel', 'NHITS', 'nhits_space', 'NBEATS', 'nbeats_space', 'RNN', 'rnn_space', 'auto',
           'instantiate_space']

# Cell
import numpy as np
import pandas as pd
from hyperopt import hp
import pytorch_lightning as pl

from .experiments.utils import hyperopt_tunning

# Cell
class AutoBaseModel(object):
    def __init__(self, n_time_out):
        super(AutoBaseModel, self).__init__()

        self.n_time_out = n_time_out

    def fit(self, Y_df, X_df, S_df, hyperopt_steps, loss_function_val, n_ts_val, results_dir,
            save_trials=False, loss_functions_test=None, n_ts_test=0, return_test_forecast=False, verbose=False):

        # Override space with data specific information
        self.space['n_series'] = hp.choice('n_series', [ Y_df['unique_id'].nunique() ])
        self.space['n_x'] = hp.choice('n_x', [ 0 if X_df is None else (X_df.shape[1]-2) ])
        self.space['n_s'] = hp.choice('n_s', [ 0 if S_df is None else (S_df.shape[1]-1) ])
        self.space['n_x_hidden'] = hp.choice('n_x_hidden', [ 0 if X_df is None else (X_df.shape[1]-2) ])
        self.space['n_s_hidden'] = hp.choice('n_s_hidden', [ 0 if S_df is None else (S_df.shape[1]-1) ])
        self.space['frequency'] = hp.choice('frequency', [ pd.infer_freq(Y_df['ds']) ])

        self.model, self.trials = hyperopt_tunning(space=self.space,
                                                   hyperopt_max_evals=hyperopt_steps,
                                                   loss_function_val=loss_function_val,
                                                   loss_functions_test=loss_functions_test,
                                                   S_df=S_df, Y_df=Y_df, X_df=X_df,
                                                   f_cols=[], ds_in_val=n_ts_val,
                                                   ds_in_test=n_ts_test,
                                                   return_forecasts=return_test_forecast,
                                                   return_model=True,
                                                   save_trials=save_trials,
                                                   results_dir=results_dir,
                                                   step_save_progress=5,
                                                   verbose=verbose)

        return self

    def forecast(self, Y_df: pd.DataFrame, X_df: pd.DataFrame = None, S_df: pd.DataFrame = None,
                 batch_size: int =1, trainer: pl.Trainer =None) -> pd.DataFrame:

        return self.model.forecast(Y_df=Y_df, X_df=X_df, S_df=S_df, batch_size=batch_size, trainer=trainer)

# Cell
class NHITS(AutoBaseModel):
    def __init__(self, n_time_out, space=None):
        super(NHITS, self).__init__(n_time_out)

        if space is None:
            space = nhits_space(n_time_out=n_time_out)
        self.space = space


def nhits_space(n_time_out: int, n_series: int = None, n_x: int = None,
                n_s: int = None,
                frequency: str = None) -> dict:
    """
    Suggested hyperparameters search space for tuning. To be used with hyperopt library.

    Parameters
    ----------
    n_time_out: int
        Forecasting horizon.
    n_series: int
        Number of time-series.
    n_x: int
        Number of exogenous variables.
    n_s: int
        Number of static variables.
    frequency: str
        Frequency of time-seris.

    Returns
    ----------
    space: Dict
        Dictionary with search space for hyperopt library.
    """

    space= {# Architecture parameters
        'model':'nhits',
        'mode': 'simple',
        'n_time_in': hp.choice('n_time_in', [2*n_time_out, 3*n_time_out, 5*n_time_out]),
        'n_time_out': hp.choice('n_time_out', [n_time_out]),
        'n_x': hp.choice('n_x', [n_x]),
        'n_s': hp.choice('n_s', [n_s]),
        'shared_weights': hp.choice('shared_weights', [False]),
        'activation': hp.choice('activation', ['ReLU']),
        'initialization':  hp.choice('initialization', ['lecun_normal']),
        'stack_types': hp.choice('stack_types', [ 3*['identity'] ]),
        'constant_n_blocks': hp.choice('n_blocks', [ 1, 3 ]), # Constant n_blocks across stacks
        'constant_n_layers': hp.choice('n_layers', [ 2, 3 ]), # Constant n_layers across stacks
        'constant_n_mlp_units': hp.choice('n_mlp_units', [ 128, 256, 512, 1024 ]), # Constant n_mlp_units across stacks
        'n_x_hidden': hp.choice('n_x_hidden', [n_x] ),
        'n_s_hidden': hp.choice('n_s_hidden', [n_s] ),
        'n_pool_kernel_size': hp.choice('n_pool_kernel_size', [ 3*[1], 3*[2], 3*[4], 3*[8], [8, 4, 1], [16, 8, 1] ]),
        'n_freq_downsample': hp.choice('n_freq_downsample', [ [168, 24, 1], [24, 12, 1],
                                                                [180, 60, 1], [60, 8, 1],
                                                                [40, 20, 1] ]),
        'pooling_mode': hp.choice('pooling_mode', [ 'max' ]),
        'interpolation_mode': hp.choice('interpolation_mode', [ 'linear' ]),
        # Regularization and optimization parameters
        'batch_normalization': hp.choice('batch_normalization', [False]),
        'dropout_prob_theta': hp.choice('dropout_prob_theta', [ 0 ]),
        'learning_rate': hp.choice('learning_rate', [0.0001, 0.001, 0.005, 0.01]),
        'lr_decay': hp.choice('lr_decay', [0.5] ),
        'n_lr_decays': hp.choice('n_lr_decays', [3]),
        'weight_decay': hp.choice('weight_decay', [0] ),
        'max_epochs': hp.choice('max_epochs', [None]),
        'max_steps': hp.choice('max_steps', [1_000, 3_000, 5_000]),
        'early_stop_patience': hp.choice('early_stop_patience', [10]),
        'eval_freq': hp.choice('eval_freq', [50]),
        'loss_train': hp.choice('loss', ['MAE']),
        'loss_hypar': hp.choice('loss_hypar', [0.5]),
        'loss_valid': hp.choice('loss_valid', ['MAE']),
        # Data parameters
        'normalizer_y': hp.choice('normalizer_y', [None]),
        'normalizer_x': hp.choice('normalizer_x', [None]),
        'complete_windows':  hp.choice('complete_windows', [True]),
        'frequency': hp.choice('frequency', [frequency]),
        'idx_to_sample_freq': hp.choice('idx_to_sample_freq', [1]),
        'val_idx_to_sample_freq': hp.choice('val_idx_to_sample_freq', [1]),
        'batch_size': hp.choice('batch_size', [1]),
        'n_windows': hp.choice('n_windows', [32, 64, 128, 256, 512]),
        'random_seed': hp.quniform('random_seed', 1, 20, 1)}

    return space

# Cell
class NBEATS(AutoBaseModel):
    def __init__(self, n_time_out, space=None):
        super(NBEATS, self).__init__(n_time_out)

        if space is None:
            space = nbeats_space(n_time_out=n_time_out)
        self.space = space

def nbeats_space(n_time_out: int, n_series: int = None,
                 n_x: int = None, n_s: int = None,
                 frequency: str = None) -> dict:
    """
    Suggested hyperparameters search space for tuning. To be used with hyperopt library.

    Parameters
    ----------
    n_time_out: int
        Forecasting horizon.
    n_series: int
        Number of time-series.
    n_x: int
        Number of exogenous variables.
    n_s: int
        Number of static variables.
    frequency: str
        Frequency of time-seris.

    Returns
    ----------
    space: Dict
        Dictionary with search space for hyperopt library.
    """

    space= {# Architecture parameters
        'model':'nbeats',
        'mode': 'simple',
        'n_time_in': hp.choice('n_time_in', [2*n_time_out, 3*n_time_out, 5*n_time_out]),
        'n_time_out': hp.choice('n_time_out', [n_time_out]),
        'n_x': hp.choice('n_x', [n_x]),
        'n_s': hp.choice('n_s', [n_s]),
        'shared_weights': hp.choice('shared_weights', [False]),
        'activation': hp.choice('activation', ['ReLU']),
        'initialization':  hp.choice('initialization', ['lecun_normal']),
        'stack_types': hp.choice('stack_types', [ 3*['identity'] ]),
        'constant_n_blocks': hp.choice('n_blocks', [ 1, 3 ]), # Constant n_blocks across stacks
        'constant_n_layers': hp.choice('n_layers', [ 2, 3 ]), # Constant n_layers across stacks
        'constant_n_mlp_units': hp.choice('n_mlp_units', [ 128, 256, 512, 1024 ]), # Constant n_mlp_units across stacks
        'n_x_hidden': hp.choice('n_x_hidden', [n_x] ),
        'n_s_hidden': hp.choice('n_s_hidden', [n_s] ),
        # Regularization and optimization parameters
        'batch_normalization': hp.choice('batch_normalization', [False]),
        'dropout_prob_theta': hp.choice('dropout_prob_theta', [ 0 ]),
        'learning_rate': hp.choice('learning_rate', [0.0001, 0.001, 0.005, 0.01]),
        'lr_decay': hp.choice('lr_decay', [0.5] ),
        'n_lr_decays': hp.choice('n_lr_decays', [3]),
        'weight_decay': hp.choice('weight_decay', [0] ),
        'max_epochs': hp.choice('max_epochs', [None]),
        'max_steps': hp.choice('max_steps', [1_000, 3_000, 5_000]),
        'early_stop_patience': hp.choice('early_stop_patience', [10]),
        'eval_freq': hp.choice('eval_freq', [50]),
        'loss_train': hp.choice('loss', ['MAE']),
        'loss_hypar': hp.choice('loss_hypar', [0.5]),
        'loss_valid': hp.choice('loss_valid', ['MAE']),
        # Data parameters
        'normalizer_y': hp.choice('normalizer_y', [None]),
        'normalizer_x': hp.choice('normalizer_x', [None]),
        'complete_windows':  hp.choice('complete_windows', [True]),
        'frequency': hp.choice('frequency', [frequency]),
        'idx_to_sample_freq': hp.choice('idx_to_sample_freq', [1]),
        'val_idx_to_sample_freq': hp.choice('val_idx_to_sample_freq', [1]),
        'batch_size': hp.choice('batch_size', [1]),
        'n_windows': hp.choice('n_windows', [32, 64, 128, 256, 512]),
        'random_seed': hp.quniform('random_seed', 1, 20, 1)}

    return space

# Cell
class RNN(AutoBaseModel):
    def __init__(self, n_time_out, space=None):
        super(RNN, self).__init__(n_time_out)

        if space is None:
            space = rnn_space(n_time_out=n_time_out)
        self.space = space

def rnn_space(n_time_out: int, n_series: int = None, n_x: int = None,
              n_s: int = None,
              frequency: str = None) -> dict:
    """
    Suggested hyperparameters search space for tuning. To be used with hyperopt library.

    Parameters
    ----------
    n_time_out: int
        Forecasting horizon.
    n_series: int
        Number of time-series.
    n_x: int
        Number of exogenous variables.
    n_s: int
        Number of static variables.
    frequency: str
        Frequency of time-seris.

    Returns
    ----------
    space: Dict
        Dictionary with search space for hyperopt library.
    """

    space= {# Architecture parameters
        'model':'rnn',
        'mode': 'full',
        'n_time_in': hp.choice('n_time_in', [1*n_time_out, 2*n_time_out, 3*n_time_out]),
        'n_time_out': hp.choice('n_time_out', [n_time_out]),
        'n_x': hp.choice('n_x', [n_x]),
        'n_s': hp.choice('n_s', [n_s]),
        'cell_type': hp.choice('cell_type', ['LSTM', 'GRU']),
        'state_hsize': hp.choice('state_hsize', [10, 20, 50, 100]),
        'dilations': hp.choice('dilations', [ [[1, 2]], [[1, 2, 4, 8]], [[1,2],[4,8]] ]),
        'add_nl_layer': hp.choice('add_nl_layer', [ False ]),
        'sample_freq': hp.choice('sample_freq', [n_time_out]),
        # Regularization and optimization parameters
        'learning_rate': hp.choice('learning_rate', [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]),
        'lr_decay': hp.choice('lr_decay', [0.5] ),
        'n_lr_decays': hp.choice('n_lr_decays', [3]),
        'gradient_eps': hp.choice('gradient_eps', [1e-8]),
        'gradient_clipping_threshold': hp.choice('gradient_clipping_threshold', [10]),
        'weight_decay': hp.choice('weight_decay', [0]),
        'noise_std': hp.choice('noise_std', [0.001]),
        'max_epochs': hp.choice('max_epochs', [None]),
        'max_steps': hp.choice('max_steps', [500, 1000]),
        'early_stop_patience': hp.choice('early_stop_patience', [10]),
        'eval_freq': hp.choice('eval_freq', [50]),
        'loss_train': hp.choice('loss', ['MAE']),
        'loss_hypar': hp.choice('loss_hypar', [0.5]),
        'loss_valid': hp.choice('loss_valid', ['MAE']),
        # Data parameters
        'normalizer_y': hp.choice('normalizer_y', [None]),
        'normalizer_x': hp.choice('normalizer_x', [None]),
        'complete_windows':  hp.choice('complete_windows', [True]),
        'frequency': hp.choice('frequency', [frequency]),
        'idx_to_sample_freq': hp.choice('idx_to_sample_freq', [1]),
        'val_idx_to_sample_freq': hp.choice('val_idx_to_sample_freq', [1]),
        'batch_size': hp.choice('batch_size', [16, 32, 64]),
        'n_windows': hp.choice('n_windows', [None]),
        'random_seed': hp.quniform('random_seed', 1, 20, 1)}

    return space

# Cell
def auto(config_dict: dict,
         Y_df: pd.DataFrame, X_df: pd.DataFrame, S_df: pd.DataFrame,
         loss_function_val: callable, loss_functions_test: dict,
         forecast_horizon: int, ts_in_val: int, ts_in_test: int,
         results_dir: str,
         return_forecasts: bool = False,
         test_auto: bool = False, verbose: bool = False):
    """
    Auto hyperparameter tuning function.

    Parameters
    ----------
    config_dict: Dict
        Dictionary with configuration. Keys should be name of models.
        For each model specify the hyperparameter space
        (None will use default suggested space), hyperopt steps and timeout.
    Y_df: pd.DataFrame
        Target time series with columns ['unique_id', 'ds', 'y'].
    X_df: pd.DataFrame
        Exogenous time series with columns ['unique_id', 'ds', 'y'].
    S_df: pd.DataFrame
        Static exogenous variables with columns ['unique_id', 'ds'].
        and static variables.
    loss_function_val: function
        Loss function used for validation.
    loss_functions_test: Dictionary
        Loss functions used for test evaluation.
        (function name: string, function: fun)
    forecast_horizon: int
        Forecast horizon
    ts_in_val: int
        Number of timestamps in validation.
    ts_in_test: int
        Number of timestamps in test.
    return_forecasts: bool
        If true return forecast on test.
    return_model: bool
        If true return model.
    test_auto: bool
        If true, will only run one training step and hyperopt iteration for each model.
        For testing purposes, to ensure your pipeline will finish running, without waiting.
    verbose:
        If true, will print summary of dataset, model and training.
    """

    if test_auto:
        print('WARNING: test_auto=True, MODELS WILL NOT BE TRAINED PROPERLY!')

    # Data characteristics
    n_series = Y_df['unique_id'].nunique()
    n_x = len(X_df.columns)-2 if X_df is not None else 0
    n_s = len(S_df.columns)-1 if S_df is not None else 0
    frequency = pd.infer_freq(Y_df['ds'])

    # Hyperopt
    models = config_dict.keys()
    output_dict = {}
    best_model = None
    best_loss = np.inf
    for model_str in models:
        print('MODEL:', model_str)
        model_config = config_dict[model_str]

        hyperopt_steps = model_config['hyperopt_steps'] if (test_auto==False) else 1

        if model_config['space'] is None:
            # If hyperparameter space is None, use predefined space for the model
            # Available spaces, [N-BEATS, N-HiTS, RNN]
            space = instantiate_space(model=model_str, n_time_out=forecast_horizon,
                                      n_series=n_series, n_x=n_x, n_s=n_s,
                                      frequency=frequency, test=test_auto)
        else:
            space = model_config['space']

        # Run automated hyperparameter optimization
        results_dir_model = f'{results_dir}/{model_str}'
        model, trials = hyperopt_tunning(space=space,
                                         hyperopt_max_evals=hyperopt_steps,
                                         loss_function_val=loss_function_val,
                                         loss_functions_test=loss_functions_test,
                                         S_df=S_df, Y_df=Y_df, X_df=X_df,
                                         f_cols=[], ds_in_val=ts_in_val,
                                         ds_in_test=ts_in_test,
                                         return_forecasts=return_forecasts,
                                         return_model=True,
                                         save_trials=False,
                                         results_dir=results_dir_model,
                                         step_save_progress=0,
                                         verbose=verbose)

        model_output = {'best_mc': trials.best_trial['result']['mc'],
                        'run_time': trials.best_trial['result']['run_time'],
                        'best_val_loss': trials.best_trial['result']['loss']}

        # Return model
        model_output['model'] = model

        # Return test losses
        if ts_in_test > 0:
            model_output['best_test_loss'] = trials.best_trial['result']['test_losses']

        # Return test forecasts
        if (return_forecasts) and (ts_in_test > 0):
            model_output['y_hat'] = trials.best_trial['result']['forecasts_test']['test_y_hat']
            model_output['y_true'] = trials.best_trial['result']['forecasts_test']['test_y_true']

        # Improvement
        optimization_times = [trials.trials[0]['result']['loss']]
        optimization_losses = [trials.trials[0]['result']['run_time']]
        for i in range(1, len(trials)):
            loss = trials.trials[i]['result']['loss']
            time = trials.trials[i]['result']['run_time']

            if loss > np.min(optimization_losses):
                loss = np.min(optimization_losses)
            optimization_losses.append(loss)
            optimization_times.append(np.sum(optimization_times)+time)

        model_output['optimization_losses'] = optimization_losses
        model_output['optimization_times'] = optimization_times

        # Append to dict
        output_dict[model_str] = model_output

        if trials.best_trial['result']['loss'] < best_loss:
            best_model = model
            best_loss = trials.best_trial['result']['loss']

    return best_model, output_dict

# Cell
def instantiate_space(model, n_time_out, n_series, n_x, n_s, frequency, test):
    assert model in ['nbeats', 'nhits', 'rnn'], f'Invalid model {model}'

    if model == 'nbeats':
        space = nbeats_space(n_time_out=n_time_out, n_series=n_series,
                             n_x=n_x, n_s=n_s, frequency=frequency)
    elif model == 'nhits':
        space = nhits_space(n_time_out=n_time_out, n_series=n_series,
                            n_x=n_x, n_s=n_s, frequency=frequency)
    elif model == 'rnn':
        space = rnn_space(n_time_out=n_time_out, n_series=n_series,
                          n_x=n_x, n_s=n_s, frequency=frequency)

    if test:
        space['max_steps'] = hp.choice('max_steps', [1])
        space['max_epochs'] = hp.choice('max_epochs', [None])

    return space