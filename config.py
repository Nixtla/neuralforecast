from ray import tune

#### BASELINES ####
# config_nhits = {'input_size': 120,
#                 'stack_types': 3*['identity'],
#                 'mlp_units': tune.choice([ 3*[[512,512]], 3*[[1024,1024]] ]),
#                 'n_pool_kernel_size': [1,1,1],
#                 'n_freq_downsample': [1,1,1],
#                 'dropout_prob_theta': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#                 'max_steps': tune.choice([1000, 2000]), # 1000
#                 'batch_size': 4,
#                 'windows_batch_size': 256,
#                 'scaler_type': None, #'minmax_treatment',
#                 'early_stop_patience_steps': 5,
#                 'val_check_steps': 100,
#                 'learning_rate': tune.loguniform(1e-4, 1e-1),
#                 'random_seed': tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
#                 }

config_nhits =  {'input_size': 120,
                'stack_types': ['identity', 'identity', 'identity'],
                'mlp_units': tune.choice([3*[[1024,1024]] ]),
                'n_pool_kernel_size': [1,1,1],
                'n_freq_downsample': [1,1,1],
                'dropout_prob_theta': tune.choice([0.0]),
                'max_steps': tune.choice([2000]),
                'batch_size': 4,
                'windows_batch_size': 256,
                'scaler_type':  None,
                'early_stop_patience_steps': 5,
                'val_check_steps': 100,
                'learning_rate': tune.loguniform(1e-4, 1e-2),
                'random_seed': tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                }

config_tft = {'input_size': 120,
                'max_steps': tune.choice([1000, 2000]), # 1000
                'hidden_size': tune.choice([128, 256]), # 256
                'dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]), # 0.2
                'attn_dropout': 0.0,
                'batch_size': 4,
                'windows_batch_size': 256,
                'scaler_type': 'minmax_treatment',
                'early_stop_patience_steps': 5,
                'val_check_steps': 100,
                'learning_rate': tune.loguniform(1e-4, 1e-2),
                'random_seed': tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                }

#### TREAT ####
# config_nhits_treat = {'input_size': 120,
#                 'stack_types': ['identity', 'identity', 'concentrator'],
#                 'mlp_units': tune.choice([ 3*[[512,512]], 3*[[1024,1024]] ]),
#                 'n_pool_kernel_size': [1,1,1],
#                 'n_freq_downsample': [1,1,1],
#                 'dropout_prob_theta': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#                 'max_steps': tune.choice([1000, 2000]), # 1000
#                 'batch_size': 4,
#                 'windows_batch_size': 256,
#                 'scaler_type':  None, #'minmax_treatment',
#                 'early_stop_patience_steps': 5,
#                 'val_check_steps': 100,
#                 'learning_rate': tune.loguniform(1e-4, 1e-1),
#                 'random_seed': tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
#                 'concentrator_type': 'log_normal',
#                 'init_ka1': tune.choice([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]),
#                 'init_ka2': tune.choice([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]),
#                 'freq': 5
#                 }

config_nhits_treat = {'input_size': 120,
                'stack_types': ['identity', 'identity', 'concentrator'],
                'mlp_units': tune.choice([3*[[1024,1024]] ]),
                'n_pool_kernel_size': [1,1,1],
                'n_freq_downsample': [1,1,1],
                'dropout_prob_theta': tune.choice([0.0]),
                'max_steps': tune.choice([2000]),
                'batch_size': 4,
                'windows_batch_size': 256,
                'scaler_type':  None,
                'early_stop_patience_steps': 5,
                'val_check_steps': 100,
                'learning_rate': tune.loguniform(1e-4, 1e-2),
                'random_seed': tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                'concentrator_type': 'log_normal',
                'init_ka1': tune.choice([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]),
                'init_ka2': tune.choice([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]),
                'freq': 5
                }

config_tft_treat = {'input_size': 120,
                'max_steps': tune.choice([1000, 2000]), # 1000
                'hidden_size': tune.choice([128, 256]), # 256
                'dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]), # 0.2
                'attn_dropout': 0.0,
                'batch_size': 4,
                'windows_batch_size': 256,
                'scaler_type': 'minmax_treatment',
                'early_stop_patience_steps': 5,
                'val_check_steps': 100,
                'learning_rate': tune.loguniform(1e-4, 1e-2),
                'random_seed': tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                'use_concentrator': True,
                'concentrator_type': 'log_normal',
                'init_ka1': tune.choice([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]),
                'init_ka2': tune.choice([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]),
                'freq': 5
                }