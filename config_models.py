from neuralforecast.models import *
from neuralforecast.losses.pytorch import MAE, MQLoss

# GLOBAL parameters
HORIZON = 12
#LOSS = MAE()

LOSS = MQLoss(quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

MODEL_LIST = ['nhits_15_512',
              'nhits_30_1024',
              'nhits_30_2048',
              'patchtst_128_3',
              'patchtst_512_6',
              #'patchtst_1024_6',
              'tft_128',
              'tft_512',
              'tft_1024',
              'mlp_512_8',
              'mlp_2048_32',
              'tcn_128_3',
              'tcn_512_5',
              'lstm_128_3',
              'lstm_512_5',
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
                        early_stop_patience_steps=10,
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
                        learning_rate=1e-4,
                        early_stop_patience_steps=10,
                        val_check_steps=500,
                        scaler_type='minmax1',
                        max_steps=15000,
                        batch_size=256,
                        windows_batch_size=1024,
                        random_seed=1)
        
    elif model_name == 'nhits_30_2048':
        model = NHITS(h=HORIZON,
                        input_size=HORIZON,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[2048, 2048, 2048, 2048]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[6, 2, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=10,
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
                            early_stop_patience_steps=10,
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
                            learning_rate=1e-4,
                            early_stop_patience_steps=10,
                            val_check_steps=500,
                            scaler_type='minmax1',
                            max_steps=15000,
                            batch_size=256,
                            windows_batch_size=1024,
                            random_seed=1)
        
    # elif model_name == 'patchtst_1024_6':
    #     model = PatchTST(h=HORIZON,
    #                         input_size=HORIZON,
    #                         encoder_layers=6,
    #                         hidden_size=1024,
    #                         linear_hidden_size=1024,
    #                         patch_len=3,
    #                         stride=3,
    #                         revin=False,
    #                         loss=LOSS,
    #                         learning_rate=1e-4,
    #                         early_stop_patience_steps=10,
    #                         val_check_steps=500,
    #                         scaler_type='minmax1',
    #                         max_steps=15000,
    #                         batch_size=256,
    #                         windows_batch_size=1024,
    #                         random_seed=1)

    # TFT
    elif model_name == 'tft_128':
        model = TFT(h=HORIZON,
                        input_size=HORIZON,
                        hidden_size=128,
                        loss=LOSS,
                        learning_rate=1e-3,
                        early_stop_patience_steps=10,
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
                    learning_rate=1e-4,
                    early_stop_patience_steps=10,
                    val_check_steps=500,
                    scaler_type='minmax1',
                    max_steps=15000,
                    batch_size=256,
                    windows_batch_size=1024,
                    random_seed=1)
        
    elif model_name == 'tft_1024':
        model = TFT(h=HORIZON,
                    input_size=HORIZON,
                    hidden_size=1024,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=10,
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
                    early_stop_patience_steps=10,
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
                    early_stop_patience_steps=10,
                    val_check_steps=500,
                    scaler_type='minmax1',
                    max_steps=15000,
                    batch_size=256,
                    windows_batch_size=1024,
                    random_seed=1)
        
    # TCN
    elif model_name == 'tcn_128_3':
        model = TCN(h=HORIZON,
                    input_size=3*HORIZON,
                    inference_input_size=3*HORIZON,
                    dilations=[1, 2, 4],
                    encoder_hidden_size=128,
                    context_size=16,
                    decoder_hidden_size=128,
                    decoder_layers=2,
                    loss=LOSS,
                    learning_rate=1e-3,
                    early_stop_patience_steps=10,
                    val_check_steps=500,
                    scaler_type='minmax1',
                    max_steps=15000,
                    batch_size=256,
                    random_seed=1)
        
    elif model_name == 'tcn_512_5':
        model = TCN(h=HORIZON,
                    input_size=3*HORIZON,
                    inference_input_size=3*HORIZON,
                    dilations= [1, 2, 4, 8, 16],
                    encoder_hidden_size=512,
                    context_size=16,
                    decoder_hidden_size=128,
                    decoder_layers=2,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=10,
                    val_check_steps=500,
                    scaler_type='minmax1',
                    max_steps=15000,
                    batch_size=256,
                    random_seed=1)

    # LSTM
    elif model_name == 'lstm_128_3':
        model = LSTM(h=HORIZON,
                    input_size=3*HORIZON,
                    inference_input_size=3*HORIZON,
                    encoder_n_layers=3,
                    encoder_hidden_size=128,
                    context_size=16,
                    decoder_hidden_size=128,
                    decoder_layers=2,
                    loss=LOSS,
                    learning_rate=1e-3,
                    early_stop_patience_steps=10,
                    val_check_steps=500,
                    scaler_type='minmax1',
                    max_steps=15000,
                    batch_size=256,
                    random_seed=1)
        
    elif model_name == 'lstm_512_5':
        model = LSTM(h=HORIZON,
                    input_size=3*HORIZON,
                    inference_input_size=3*HORIZON,
                    encoder_n_layers=5,
                    encoder_hidden_size=512,
                    context_size=16,
                    decoder_hidden_size=128,
                    decoder_layers=2,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=10,
                    val_check_steps=500,
                    scaler_type='minmax1',
                    max_steps=15000,
                    batch_size=256,
                    random_seed=1) 

    return model