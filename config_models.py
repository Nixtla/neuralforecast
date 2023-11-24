from neuralforecast.models import *
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss, HuberLoss

# GLOBAL parameters
HORIZON_DICT = {'yearly': 6,
                'quarterly': 8,
                'monthly': 12,
                'weekly': 4,
                'daily': 14}

HORIZON_DICT_SHORT = {'monthly': 6}

LOSS = MQLoss(quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
#LOSS = HuberLoss()
LOSS_PROBA = DistributionLoss(distribution="StudentT", quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], return_params=False)


MODEL_LIST = ['nhits_15_512',
              'nhits_30_1024',
              'tft_128',
              'tft_1024',
              'mlp_512_8',
              'mlp_2048_32',
              'tcn_128_3',
              'tcn_512_5',
              'lstm_128_3',
              'lstm_512_5',
              'vanillatransformer_256_3']

# MODEL_LIST = ['deepar_64_2', 'patchtst_5']

# MODEL_LIST = ['vanillatransformer_256_3',
#               'vanillatransformer_2',
#               'vanillatransformer_3',
#               'vanillatransformer_4',
#               'vanillatransformer_5',
#               'vanillatransformer_6'
#               ]

# MODEL_LIST = ['patchtst_128_3',
#               'patchtst_2',
#               'patchtst_3',
#               'patchtst_4',
#               'patchtst_5',
#               'patchtst_6'
#               ]

def load_model(model_name, frequency, short=False, random_seed=1):
    assert model_name in MODEL_LIST, f"Model {model_name} not in list of models"
    
    if short:
        horizon = HORIZON_DICT_SHORT[frequency]
    else:
        horizon = HORIZON_DICT[frequency]
    max_steps = 15000

    # NHITS
    if model_name == 'nhits_15_512':
        model = NHITS(h=horizon,
                        input_size=2*horizon,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[512, 512, 512, 512]],
                        n_blocks=3*[5],
                        n_pool_kernel_size=3*[1],
                        #n_freq_downsample=[6, 2, 1],
                        n_freq_downsample=[1, 1, 1],
                        loss=LOSS,
                        learning_rate=1e-3,
                        early_stop_patience_steps=10,
                        val_check_steps=300,
                        scaler_type='minmax1',
                        max_steps=max_steps,
                        batch_size=256,
                        windows_batch_size=1024,
                        random_seed=random_seed)

    elif model_name == 'nhits_30_1024':
        model = NHITS(h=horizon,
                        input_size=2*horizon,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        #n_freq_downsample=[6, 2, 1],
                        n_freq_downsample=[1, 1, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=10,
                        val_check_steps=300,
                        scaler_type='minmax1',
                        max_steps=max_steps,
                        batch_size=256,
                        windows_batch_size=1024,
                        random_seed=random_seed)

    # Transformer
    elif model_name == 'vanillatransformer_256_3': 
        model = VanillaTransformer(h=horizon,
                                input_size=2*horizon,
                                encoder_layers=3,
                                decoder_layers=1,
                                n_head=4,
                                hidden_size=256,
                                conv_hidden_size=64,
                                dropout=0.1,
                                loss=LOSS,
                                learning_rate=1e-4,
                                early_stop_patience_steps=10, # -1
                                val_check_steps=300,
                                scaler_type='minmax1',
                                max_steps=max_steps,
                                batch_size=256, # 256
                                windows_batch_size=1024, # 1024
                                num_sanity_val_steps=0,
                                random_seed=random_seed)
    elif model_name == 'vanillatransformer_2': 
        model = VanillaTransformer(h=horizon,
                                input_size=2*horizon,
                                encoder_layers=3,
                                decoder_layers=1,
                                n_head=4,
                                hidden_size=256,
                                conv_hidden_size=64,
                                dropout=0.1,
                                loss=LOSS,
                                learning_rate=1e-5,
                                early_stop_patience_steps=10, # -1
                                val_check_steps=300,
                                scaler_type='minmax1',
                                max_steps=max_steps,
                                batch_size=256, # 256
                                windows_batch_size=1024, # 1024
                                num_sanity_val_steps=0,
                                random_seed=random_seed)
    elif model_name == 'vanillatransformer_3': 
        model = VanillaTransformer(h=horizon,
                                input_size=2*horizon,
                                encoder_layers=3,
                                decoder_layers=1,
                                n_head=4,
                                hidden_size=128,
                                conv_hidden_size=32,
                                dropout=0.1,
                                loss=LOSS,
                                learning_rate=1e-4,
                                early_stop_patience_steps=10, # -1
                                val_check_steps=300,
                                scaler_type='minmax1',
                                max_steps=max_steps,
                                batch_size=256, # 256
                                windows_batch_size=1024, # 1024
                                num_sanity_val_steps=0,
                                random_seed=random_seed)
    elif model_name == 'vanillatransformer_4': 
        model = VanillaTransformer(h=horizon,
                                input_size=2*horizon,
                                encoder_layers=3,
                                decoder_layers=1,
                                n_head=4,
                                hidden_size=256,
                                conv_hidden_size=64,
                                dropout=0.3,
                                loss=LOSS,
                                learning_rate=1e-4,
                                early_stop_patience_steps=10, # -1
                                val_check_steps=300,
                                scaler_type='minmax1',
                                max_steps=max_steps,
                                batch_size=256, # 256
                                windows_batch_size=1024, # 1024
                                num_sanity_val_steps=0,
                                random_seed=random_seed)
    elif model_name == 'vanillatransformer_5': 
        model = VanillaTransformer(h=horizon,
                                input_size=3*horizon,
                                encoder_layers=3,
                                decoder_layers=1,
                                n_head=4,
                                hidden_size=256,
                                conv_hidden_size=64,
                                dropout=0.1,
                                loss=LOSS,
                                learning_rate=1e-4,
                                early_stop_patience_steps=10, # -1
                                val_check_steps=300,
                                scaler_type='minmax1',
                                max_steps=max_steps,
                                batch_size=256, # 256
                                windows_batch_size=1024, # 1024
                                num_sanity_val_steps=0,
                                random_seed=random_seed)
    elif model_name == 'vanillatransformer_6': 
        model = VanillaTransformer(h=horizon,
                                input_size=2*horizon,
                                encoder_layers=3,
                                decoder_layers=1,
                                n_head=4,
                                hidden_size=256,
                                conv_hidden_size=64,
                                dropout=0.1,
                                loss=LOSS,
                                learning_rate=1e-4,
                                early_stop_patience_steps=10, # -1
                                val_check_steps=300,
                                scaler_type='standard',
                                max_steps=max_steps,
                                batch_size=256, # 256
                                windows_batch_size=1024, # 1024
                                num_sanity_val_steps=0,
                                random_seed=random_seed)
    # TFT
    elif model_name == 'tft_128':
        model = TFT(h=horizon,
                        input_size=2*horizon,
                        hidden_size=128,
                        loss=LOSS,
                        learning_rate=1e-3,
                        early_stop_patience_steps=10,
                        val_check_steps=300,
                        scaler_type='minmax1',
                        max_steps=max_steps,
                        batch_size=256,
                        windows_batch_size=1024,
                        random_seed=random_seed)

    elif model_name == 'tft_1024':
        model = TFT(h=horizon,
                    input_size=2*horizon,
                    hidden_size=1024,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=10,
                    val_check_steps=300,
                    scaler_type='minmax1',
                    max_steps=max_steps,
                    batch_size=256,
                    windows_batch_size=1024,
                    random_seed=random_seed)
    
    # MLP
    elif model_name == 'mlp_512_8':
        model = MLP(h=horizon,
                    input_size=2*horizon,
                    num_layers=8,
                    hidden_size=512,
                    loss=LOSS,
                    learning_rate=1e-3,
                    early_stop_patience_steps=10,
                    val_check_steps=300,
                    scaler_type='minmax1',
                    max_steps=max_steps,
                    batch_size=256,
                    windows_batch_size=1024,
                    random_seed=random_seed)

    elif model_name == 'mlp_2048_32':
        model = MLP(h=horizon,
                    input_size=2*horizon,
                    num_layers=32,
                    hidden_size=2048,
                    loss=LOSS,
                    learning_rate=1e-3,
                    early_stop_patience_steps=10,
                    val_check_steps=300,
                    scaler_type='minmax1',
                    max_steps=max_steps,
                    batch_size=256,
                    windows_batch_size=1024,
                    random_seed=random_seed)
        
    # TCN
    elif model_name == 'tcn_128_3':
        model = TCN(h=horizon,
                    input_size=2*horizon,
                    inference_input_size=2*horizon,
                    dilations=[1, 2, 4],
                    encoder_hidden_size=128,
                    context_size=16,
                    decoder_hidden_size=128,
                    decoder_layers=2,
                    loss=LOSS,
                    learning_rate=1e-3,
                    early_stop_patience_steps=10,
                    val_check_steps=300,
                    scaler_type='minmax1',
                    max_steps=max_steps,
                    batch_size=256,
                    random_seed=1)
        
    elif model_name == 'tcn_512_5':
        model = TCN(h=horizon,
                    input_size=2*horizon,
                    inference_input_size=2*horizon,
                    dilations= [1, 2, 4, 8, 16],
                    encoder_hidden_size=512,
                    context_size=16,
                    decoder_hidden_size=128,
                    decoder_layers=2,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=10,
                    val_check_steps=300,
                    scaler_type='minmax1',
                    max_steps=max_steps,
                    batch_size=256,
                    random_seed=random_seed)

    # LSTM
    elif model_name == 'lstm_128_3':
        model = LSTM(h=horizon,
                    input_size=2*horizon,
                    inference_input_size=2*horizon,
                    encoder_n_layers=3,
                    encoder_hidden_size=128,
                    context_size=16,
                    decoder_hidden_size=128,
                    decoder_layers=2,
                    loss=LOSS,
                    learning_rate=1e-3,
                    early_stop_patience_steps=10,
                    val_check_steps=300,
                    scaler_type='minmax1',
                    max_steps=max_steps,
                    batch_size=256,
                    random_seed=random_seed)
        
    elif model_name == 'lstm_512_5':
        model = LSTM(h=horizon,
                    input_size=2*horizon,
                    inference_input_size=2*horizon,
                    encoder_n_layers=5,
                    encoder_hidden_size=512,
                    context_size=16,
                    decoder_hidden_size=128,
                    decoder_layers=2,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=10,
                    val_check_steps=300,
                    scaler_type='minmax1',
                    max_steps=max_steps,
                    batch_size=256,
                    random_seed=random_seed)

    elif model_name == 'deepar_128_4':
        model = DeepAR(h=horizon,
                        input_size=2*horizon,
                        lstm_n_layers=4,
                        lstm_hidden_size=128,
                        lstm_dropout=0.1,
                        loss=LOSS_PROBA,
                        learning_rate=1e-4,
                        early_stop_patience_steps=10,
                        val_check_steps=300,
                        scaler_type='minmax1',
                        max_steps=max_steps,
                        batch_size=256,
                        windows_batch_size=1024,
                        random_seed=random_seed)

    elif model_name == 'deepar_64_2':
        model = DeepAR(h=horizon,
                        input_size=2*horizon,
                        lstm_n_layers=2,
                        lstm_hidden_size=64,
                        lstm_dropout=0.0, # 0.1
                        loss=LOSS_PROBA,
                        valid_loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=10,
                        val_check_steps=500, # 300
                        scaler_type='robust', # minmax1
                        max_steps=max_steps,
                        batch_size=256,
                        windows_batch_size=1024,
                        random_seed=random_seed)
        
    elif model_name == 'deepar_3': # 3 layers, no dropout
        model = DeepAR(h=horizon,
                        input_size=2*horizon,
                        lstm_n_layers=3,
                        lstm_hidden_size=128,
                        lstm_dropout=0,
                        decoder_hidden_layers=0,
                        decoder_hidden_size=0,
                        loss=LOSS_PROBA,
                        learning_rate=1e-4,
                        early_stop_patience_steps=10,
                        val_check_steps=300,
                        scaler_type='standard',
                        max_steps=max_steps,
                        batch_size=256,
                        windows_batch_size=1024,
                        random_seed=random_seed)
        
    elif model_name == 'deepar_4': # 3 layers, no dropout, MLP decoder
        model = DeepAR(h=horizon,
                        input_size=2*horizon,
                        lstm_n_layers=3,
                        lstm_hidden_size=128,
                        lstm_dropout=0,
                        decoder_hidden_layers=2,
                        decoder_hidden_size=128,
                        loss=LOSS_PROBA,
                        learning_rate=1e-4,
                        early_stop_patience_steps=10,
                        val_check_steps=300,
                        scaler_type='standard',
                        max_steps=max_steps,
                        batch_size=256,
                        windows_batch_size=1024,
                        random_seed=random_seed)
        
    elif model_name == 'deepar_5': # 3 layers, no dropout, MLP decoder, 64 batch size
        model = DeepAR(h=horizon,
                        input_size=2*horizon,
                        lstm_n_layers=3,
                        lstm_hidden_size=128,
                        lstm_dropout=0,
                        decoder_hidden_layers=2,
                        decoder_hidden_size=128,
                        loss=LOSS_PROBA,
                        learning_rate=1e-4,
                        early_stop_patience_steps=10,
                        val_check_steps=300,
                        scaler_type='standard',
                        max_steps=max_steps,
                        batch_size=64,
                        windows_batch_size=64,
                        random_seed=random_seed)

    elif model_name == 'deepar_6': # 3 layers, no dropout, MLP decoder, 64 batch size, 1e-5 lr, //2 max steps
        model = DeepAR(h=horizon,
                        input_size=2*horizon,
                        lstm_n_layers=3,
                        lstm_hidden_size=128,
                        lstm_dropout=0,
                        decoder_hidden_layers=2,
                        decoder_hidden_size=128,
                        loss=LOSS_PROBA,
                        learning_rate=1e-5,
                        early_stop_patience_steps=10,
                        val_check_steps=300,
                        scaler_type='standard',
                        max_steps=max_steps//2,
                        batch_size=64,
                        windows_batch_size=64,
                        random_seed=random_seed)
        
        
    elif model_name == 'patchtst_128_3': 
        model = PatchTST(h=horizon,
                        input_size=2*horizon,
                        encoder_layers=3,
                        hidden_size=128,
                        linear_hidden_size=128,
                        patch_len=4,
                        stride=4,
                        revin=False,
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=10,
                        val_check_steps=300,
                        scaler_type='minmax1',
                        max_steps=max_steps,
                        batch_size=256,
                        windows_batch_size=1024,
                        random_seed=random_seed)
    elif model_name == 'patchtst_2': # Stride 1
        model = PatchTST(h=horizon,
                        input_size=2*horizon,
                        encoder_layers=3,
                        hidden_size=128,
                        linear_hidden_size=128,
                        patch_len=4,
                        stride=1,
                        revin=False,
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=10,
                        val_check_steps=300,
                        scaler_type='minmax1',
                        max_steps=max_steps,
                        batch_size=256,
                        windows_batch_size=1024,
                        random_seed=random_seed)
    elif model_name == 'patchtst_3': # Stride 1 and patch 2
        model = PatchTST(h=horizon,
                        input_size=2*horizon,
                        encoder_layers=3,
                        hidden_size=128,
                        linear_hidden_size=128,
                        patch_len=2,
                        stride=1,
                        revin=False,
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=10,
                        val_check_steps=300,
                        scaler_type='minmax1',
                        max_steps=max_steps,
                        batch_size=256,
                        windows_batch_size=1024,
                        random_seed=random_seed)
    elif model_name == 'patchtst_4': # Use Revin 4,4
        model = PatchTST(h=horizon,
                        input_size=2*horizon,
                        encoder_layers=3,
                        hidden_size=128,
                        linear_hidden_size=128,
                        patch_len=4,
                        stride=4,
                        revin=True,
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=10,
                        val_check_steps=300,
                        scaler_type=None,
                        max_steps=max_steps,
                        batch_size=256,
                        windows_batch_size=1024,
                        random_seed=random_seed)
    elif model_name == 'patchtst_5': # Use revin 4,1
        model = PatchTST(h=horizon,
                        input_size=2*horizon,
                        encoder_layers=3,
                        hidden_size=128,
                        linear_hidden_size=128,
                        patch_len=4,
                        stride=1,
                        revin=True,
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=10,
                        val_check_steps=300,
                        scaler_type=None,
                        max_steps=max_steps,
                        batch_size=256,
                        windows_batch_size=1024,
                        random_seed=random_seed)
    elif model_name == 'patchtst_6': # Larger model
        model = PatchTST(h=horizon,
                        input_size=2*horizon,
                        encoder_layers=3,
                        hidden_size=256,
                        linear_hidden_size=256,
                        patch_len=2,
                        stride=1,
                        revin=False,
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=10,
                        val_check_steps=300,
                        scaler_type='minmax1',
                        max_steps=max_steps,
                        batch_size=256,
                        windows_batch_size=1024,
                        random_seed=random_seed)

    return model