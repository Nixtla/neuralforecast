from neuralforecast.models import *
from neuralforecast.losses.pytorch import MAE, MQLoss, HuberMQLoss

# GLOBAL parameters

LOSS = HuberMQLoss(quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

MODEL_LIST = [
              'nhits_30_1024_10minutely',
              'nhits_30_1024_15minutely',
              'nhits_30_1024_30minutely',
              'nhits_30_1024_hourly',
              'nhits_30_1024_daily',
              'nhits_30_1024_weekly',
              'nhits_30_1024_monthly',
              'nhits_30_1024_quarterly',
              'nhits_30_1024_yearly',
              'nhits_30_1024_minutely',
              'tft_1024_daily',
              'tft_1024_weekly',
              'tft_1024_monthly',
              ]

def load_model(model_name):
    assert model_name in MODEL_LIST, f"Model {model_name} not in list of models"

    # NHITS
    if model_name == 'nhits_30_1024_minutely':
        horizon = 60
        model = NHITS(h=horizon,
                        input_size=5*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[4, 2, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=100,
                        scaler_type='minmax1',
                        max_steps=1000,
                        batch_size=32,
                        windows_batch_size=128,
                        random_seed=1)
    if model_name == 'nhits_30_1024_10minutely':
        horizon = 144
        model = NHITS(h=horizon,
                        input_size=5*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[24, 2, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=100,
                        scaler_type='minmax1',
                        max_steps=1000,
                        batch_size=32,
                        windows_batch_size=128,
                        random_seed=1)
    if model_name == 'nhits_30_1024_15minutely':
        horizon = 96
        model = NHITS(h=horizon,
                        input_size=5*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[24, 2, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=100,
                        scaler_type='minmax1',
                        max_steps=1000,
                        batch_size=32,
                        windows_batch_size=128,
                        random_seed=1)
    if model_name == 'nhits_30_1024_30minutely':
        horizon = 48
        model = NHITS(h=horizon,
                        input_size=5*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[24, 2, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=100,
                        scaler_type='minmax1',
                        max_steps=1000,
                        batch_size=32,
                        windows_batch_size=128,
                        random_seed=1)
    if model_name == 'nhits_30_1024_hourly':
        horizon = 24
        model = NHITS(h=horizon,
                        input_size=3*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[12, 2, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=200,
                        scaler_type='minmax1',
                        max_steps=5000,
                        batch_size=128,
                        windows_batch_size=1024,
                        random_seed=1)
    if model_name == 'nhits_30_1024_daily':
        horizon = 7
        model = NHITS(h=horizon,
                        input_size=3*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[4, 2, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=-1,
                        val_check_steps=1000,
                        scaler_type='minmax1',
                        max_steps=200,
                        batch_size=128,
                        windows_batch_size=1024,
                        random_seed=1)
    if model_name == 'nhits_30_1024_weekly':
        horizon = 1
        model = NHITS(h=horizon,
                        input_size=52*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[1, 1, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=-1,
                        val_check_steps=1000,
                        scaler_type='minmax1',
                        max_steps=200,
                        batch_size=128,
                        windows_batch_size=1024,
                        random_seed=1)

    if model_name == 'nhits_30_1024_monthly':
        horizon = 12
        model = NHITS(h=horizon,
                        input_size=3*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[6, 2, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=-1,
                        val_check_steps=1000,
                        scaler_type='minmax1',
                        max_steps=200,
                        batch_size=128,
                        windows_batch_size=1024,
                        random_seed=1)
    if model_name == 'nhits_30_1024_quarterly':
        horizon = 4
        model = NHITS(h=horizon,
                        input_size=3*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[1, 1, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=200,
                        scaler_type='minmax1',
                        max_steps=5000,
                        batch_size=128,
                        windows_batch_size=1024,
                        random_seed=1)
    if model_name == 'nhits_30_1024_yearly':
        horizon = 1
        model = NHITS(h=horizon,
                        input_size=3*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[1, 1, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=200,
                        scaler_type='minmax1',
                        max_steps=5000,
                        batch_size=128,
                        windows_batch_size=1024,
                        random_seed=1)

    if model_name == 'tft_1024_daily':
        horizon = 7
        model = TFT(h=horizon,
                    input_size=3*horizon,
                    hidden_size=1024,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=-1,
                    val_check_steps=1000,
                    scaler_type='minmax1',
                    max_steps=200,
                    batch_size=128,
                    windows_batch_size=1024,
                    random_seed=1)

    if model_name == 'tft_1024_weekly':
        horizon = 1
        model = TFT(h=horizon,
                    input_size=3*horizon,
                    hidden_size=1024,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=-1,
                    val_check_steps=1000,
                    scaler_type='minmax1',
                    max_steps=200,
                    batch_size=128,
                    windows_batch_size=1024,
                    random_seed=1)
        
    if model_name == 'tft_1024_monthly':
        horizon = 12
        model = TFT(h=horizon,
                    input_size=3*horizon,
                    hidden_size=1024,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=-1,
                    val_check_steps=1000,
                    scaler_type='minmax1',
                    max_steps=200,
                    batch_size=128,
                    windows_batch_size=1024,
                    random_seed=1)
    return model