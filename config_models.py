from neuralforecast.models import *
from neuralforecast.losses.pytorch import MAE

# GLOBAL parameters
HORIZON = 12
LOSS = MAE()

MODEL_LIST = ['nhits_15_512',
              'nhits_30_1024',
              'patchtst_128_3',
              'patchtst_512_6',
              'tft_128',
              'tft_512',
              'mlp_512_8',
              'mlp_2048_32',
              ]

def load_model(model_name):
    assert model_name in MODEL_LIST, f"Model {model_name} not in list of models"

    # NHITS
    if model_name == 'nhits_15_512':
        model = NHITS(h=HORIZON,
                        input_size=HORIZON,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[512, 512, 512, 512]],
                        n_blocks=3*[5],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[6, 2, 1],
                        loss=LOSS,
                        learning_rate=1e-3,
                        early_stop_patience_steps=5,
                        val_check_steps=500,
                        scaler_type='minmax1',
                        max_steps=15000,
                        batch_size=256,
                        windows_batch_size=1024,
                        random_seed=1)

    elif model_name == 'nhits_30_1024':
        model = NHITS(h=HORIZON,
                        input_size=HORIZON,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[6, 2, 1],
                        loss=LOSS,
                        learning_rate=1e-3,
                        early_stop_patience_steps=5,
                        val_check_steps=500,
                        scaler_type='minmax1',
                        max_steps=15000,
                        batch_size=256,
                        windows_batch_size=1024,
                        random_seed=1)

    # PatchTST
    elif model_name == 'patchtst_128_3': 
        model = PatchTST(h=HORIZON,
                            input_size=HORIZON,
                            encoder_layers=3,
                            hidden_size=128,
                            linear_hidden_size=128,
                            patch_len=4,
                            stride=4,
                            revin=False,
                            loss=LOSS,
                            learning_rate=1e-3,
                            early_stop_patience_steps=5,
                            val_check_steps=500,
                            scaler_type='minmax1',
                            max_steps=15000,
                            batch_size=256,
                            windows_batch_size=1024,
                            random_seed=1)

    elif model_name == 'patchtst_512_6':
        model = PatchTST(h=HORIZON,
                            input_size=HORIZON,
                            encoder_layers=6,
                            hidden_size=512,
                            linear_hidden_size=512,
                            patch_len=4,
                            stride=4,
                            revin=False,
                            loss=LOSS,
                            learning_rate=1e-3,
                            early_stop_patience_steps=5,
                            val_check_steps=500,
                            scaler_type='minmax1',
                            max_steps=15000,
                            batch_size=256,
                            windows_batch_size=1024,
                            random_seed=1)

    # TFT
    elif model_name == 'tft_128':
        model = TFT(h=HORIZON,
                        input_size=HORIZON,
                        hidden_size=128,
                        loss=LOSS,
                        learning_rate=1e-3,
                        early_stop_patience_steps=5,
                        val_check_steps=500,
                        scaler_type='minmax1',
                        max_steps=15000,
                        batch_size=256,
                        windows_batch_size=1024,
                        random_seed=1)

    elif model_name == 'tft_512':
        model = TFT(h=HORIZON,
                    input_size=HORIZON,
                    hidden_size=512,
                    loss=LOSS,
                    learning_rate=1e-3,
                    early_stop_patience_steps=5,
                    val_check_steps=500,
                    scaler_type='minmax1',
                    max_steps=15000,
                    batch_size=256,
                    windows_batch_size=1024,
                    random_seed=1)

    # MLP
    elif model_name == 'mlp_512_8':
        model = MLP(h=HORIZON,
                    input_size=HORIZON,
                    num_layers=8,
                    hidden_size=512,
                    loss=LOSS,
                    learning_rate=1e-3,
                    early_stop_patience_steps=5,
                    val_check_steps=500,
                    scaler_type='minmax1',
                    max_steps=15000,
                    batch_size=256,
                    windows_batch_size=1024,
                    random_seed=1)

    elif model_name == 'mlp_2048_32':
        model = MLP(h=HORIZON,
                    input_size=HORIZON,
                    num_layers=32,
                    hidden_size=2048,
                    loss=LOSS,
                    learning_rate=1e-3,
                    early_stop_patience_steps=5,
                    val_check_steps=500,
                    scaler_type='minmax1',
                    max_steps=15000,
                    batch_size=256,
                    windows_batch_size=1024,
                    random_seed=1)
        
    return model