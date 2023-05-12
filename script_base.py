from neuralforecast.auto import NHITS, LSTM, TFT, TCN, DilatedRNN
from neuralforecast.core import NeuralForecast
from datasetsforecast.m4 import M4
from datasetsforecast.m3 import M3
from datasetsforecast.long_horizon import LongHorizon
from neuralforecast.utils import AirPassengersDF
from ray import tune

import argparse
from neuralforecast.losses.pytorch import MAE
import pandas as pd
import os
import numpy as np

"""
Pipeline:
 	1. Read source dataset using datasetsforecast (https://github.com/Nixtla/datasetsforecast).
        Specified with `source_dataset` paramater in script.
 	2. Fit Auto model on source dataset. Model specified with `model` argument.
 	3. Save model, using folder './results/stored_models/{dataset}/{model}/{experiment_id}/'.
 	4. Read target dataset using datasetsforecast. Specified with `target_dataset` argument in script.
 	5. Load model, predict on target dataset, store forecasts in './results/forecasts/{target_dataset}/{model_source_dataset_experiment_id}.csv'.
Script arguments:
	1. source_dataset
	2. target_dataset
	3. model
	4. experiment_id
------------------------------------------
Notes:
1. Use Transfer Learning tutorial notebook as starting point.
2. Use argparse (https://github.com/cchallu/n-hits/blob/main/nhits_multivariate.py)
3. Use dictionaries to select between models. First list: AutoNHITS, AutoLSTM, AutoTFT
	MODEL_DICT={'name_1': AutoNHITS, ..., 'name_n':model_n}.
	model = MODEL_DICT[args.model_name]
4. For first example define source datasets as: M3 or M4.
5. For target dataset use AirPassengers.
6. For using Auto models: https://nixtla.github.io/neuralforecast/examples/forecasting_tft.html
 ------------------------------------------
 Next steps:
 	1. k-shot learning
 	2. evaluation scripts
 	3. more datasets
"""

# GLOBAL parameters
horizon = 18
loss = MAE()

nhits = [NHITS(h=horizon,
	       input_size=2*18,
	       stack_types=3*['identity'],
		   mlp_units=3*[[512, 512]],
		   n_blocks=3*[5],
		   n_pool_kernel_size=3*[1],
		   n_freq_downsample=3*[1],
		   early_stop_patience_steps=5,
		   val_check_steps=500,
		   scaler_type='robust',
		   max_steps=10_000,
		   windows_batch_size=512,
		   random_seed=1
		   )]

tft = [TFT(h=horizon,
	       input_size=2*18,
	       hidden_size=320,
		   early_stop_patience_steps=5,
		   val_check_steps=500,
		   scaler_type='robust',
		   max_steps=5000,
		   windows_batch_size=512,
		   random_seed=1
		   )]

lstm = [LSTM(h=horizon,
	       input_size=18,
	       inference_input_size=18,
	       encoder_hidden_size=512,
	       encoder_n_layers=2,
	       context_size=50,
	       decoder_hidden_size=512,
		   early_stop_patience_steps=5,
		   val_check_steps=500,
		   scaler_type='robust',
		   max_steps=5000,
		   random_seed=1
		   )]

tcn = [TCN(h=horizon,
	       input_size=18,
	       inference_input_size=18,
	       encoder_hidden_size=512,
	       context_size=50,
	       decoder_hidden_size=128,
		   early_stop_patience_steps=5,
		   val_check_steps=500,
		   scaler_type='robust',
		   max_steps=5000,
		   random_seed=1
		   )]

dilatedrnn = [DilatedRNN(h=horizon,
	       input_size=18,
	       inference_input_size=18,
	       encoder_hidden_size=512,
	       context_size=50,
	       decoder_hidden_size=128,
		   early_stop_patience_steps=5,
		   val_check_steps=500,
		   scaler_type='robust',
		   max_steps=5000,
		   random_seed=1
		   )]

MODEL_DICT = {'nhits': nhits, 'tft':tft, 'lstm':lstm, 'tcn':tcn, 'dilatedrnn':dilatedrnn}

def main(args):

	# make sure folder exists, then check if the file exists in the folder
	model_dir = f'./results/stored_models/{args.source_dataset}/{args.model}/{args.experiment_id}/'
	os.makedirs(model_dir, exist_ok=True)
	file_exists = os.path.isfile(
		f'./results/stored_models/{args.source_dataset}/{args.model}/{args.experiment_id}/{args.model}_0.ckpt')
	
	if (not file_exists):
		# Read source data
		if (args.source_dataset == 'M4'): # add more if conditions later, expects M4 only for now
			Y_df, a, b = M4.load(directory='./', group='Monthly', cache=True)
			frequency = 'M'
		elif (args.source_dataset == 'M4-all'):
			Y_df1, *_ = M4.load(directory='./', group='Monthly', cache=True)
			Y_df2, *_ = M4.load(directory='./', group='Daily', cache=True)
			Y_df3, *_ = M4.load(directory='./', group='Weekly', cache=True)
			Y_df4, *_ = M4.load(directory='./', group='Hourly', cache=True)
			Y_df = pd.concat([Y_df1, Y_df2, Y_df3, Y_df4], axis=0).reset_index(drop=True)
			frequency = 'M'
		else:
			raise Exception("Dataset not defined")
		Y_df['ds'] = pd.to_datetime(Y_df['ds'])

		# Train model
		model = MODEL_DICT[args.model]
		if model is None: raise Exception("Model not defined")
		
		# frequency = sampling rate of data
		print('Fitting model')
		nf = NeuralForecast(models=model,freq=frequency)
		nf.fit(df=Y_df, val_size=18)
		
		# Save model
		nf.save(path=f'./results/stored_models/{args.source_dataset}/{args.model}/{args.experiment_id}/',
			overwrite=False, save_dataset=False)
	else:
		print('Hyperparameter optimization already done. Loading saved model!')
		# do i need to check if the file/path exists? shouldn't it already be checked
		nf = NeuralForecast.load(path=
			  f'./results/stored_models/{args.source_dataset}/{args.model}/{args.experiment_id}/')
		
	# Load target data
	if (args.target_dataset == 'AirPassengers'):
		Y_df_target = AirPassengersDF.copy()
		Y_df_target['ds'] = pd.to_datetime(Y_df_target['ds'])
		test_size = horizon*4
		frequency = 'M'
	elif (args.target_dataset == 'M3'):
		Y_df_target, *_ = M3.load(directory='./', group='Monthly')
		Y_df_target['ds'] = pd.to_datetime(Y_df_target['ds'])
		frequency = 'M'
		test_size = horizon*4
	elif (args.target_dataset == 'M4'):
		Y_df_target, *_ = M4.load(directory='./', group='Monthly', cache=True)
		Y_df_target['ds'] = pd.to_datetime(Y_df_target['ds'])
		frequency = 'M'
		test_size = horizon*4
	elif (args.target_dataset == 'ILI'):
		Y_df_target, _, _ = LongHorizon.load(directory='./', group='ILI')
		Y_df_target['ds'] = np.repeat(np.array(range(len(Y_df_target)//7)), 7)
		test_size = horizon
		frequency = 'W'
	elif (args.target_dataset == 'TrafficL'):
		Y_df_target, _, _ = LongHorizon.load(directory='./', group='TrafficL')
		Y_df_target['ds'] = np.repeat(np.array(range(len(Y_df_target)//862)), 862)
		test_size = horizon
		frequency = 'W'
	else:
		raise Exception("Dataset not defined")

	# Predict on the test set of the target data
	print('Predicting on target data')
	Y_hat_df = nf.cross_validation(df=Y_df_target,
								   n_windows=None, test_size=test_size,
								   fit_models=False).reset_index()
	results_dir = f'./results/forecasts/{args.target_dataset}/'
	os.makedirs(results_dir, exist_ok=True)

	# store results, also check if this folder exists/create it if its done
	Y_hat_df.to_csv(f'{results_dir}/{args.model}_{args.source_dataset}_{args.experiment_id}.csv',
		 index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="script arguments")
    parser.add_argument('--source_dataset', type=str, help='dataset to train models on')
    parser.add_argument('--target_dataset', type=str, help='run model on this dataset')
    parser.add_argument('--model', type=str, help='auto model to use')
    parser.add_argument('--experiment_id', type=str, help='identify experiment')
    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    models = ['nhits', 'tft', 'lstm', 'tcn', 'dilatedrnn']
    for model in models:
        args.model = model
        print('Running model: ', args.model)
        main(args)

# CUDA_VISIBLE_DEVICES=3 python script_base.py --source_dataset "M4" --target_dataset "M3" --model "nhits" --experiment_id "20230422"
# CUDA_VISIBLE_DEVICES=3 python script_base.py --source_dataset "M4" --target_dataset "M3" --model "tft" --experiment_id "20230422"
# CUDA_VISIBLE_DEVICES=3 python script_base.py --source_dataset "M4" --target_dataset "M3" --model "lstm" --experiment_id "20230422"


# CUDA_VISIBLE_DEVICES=3 python script_base.py --source_dataset "M4" --target_dataset "ILI" --model "all" --experiment_id "20230424"
