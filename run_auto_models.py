from neuralforecast.auto import AutoNHITS, AutoLSTM, AutoTFT, AutoPatchTST, AutoTCN, AutoMLP
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
horizon = 12
loss = MAE()
num_samples = 10  # how many configuration we try during tuning
config = None

def load_model(args):
	nhits = [AutoNHITS(h=horizon,
					loss=loss, num_samples=num_samples,
					config={
						"input_size": tune.choice([horizon]),
						"stack_types": tune.choice([3*['identity']]),
						"mlp_units": tune.choice([3 * [[512, 512, 512, 512]]]),
						"n_blocks": tune.choice([3*[10]]),
						"n_pool_kernel_size": tune.choice([3*[1],
					 									   3*[2],
														   3*[4]]),
						"n_freq_downsample": tune.choice([[1, 1, 1],
														  [6, 2, 1],
														  [6, 3, 1],
														  [12, 6, 1]]),
						"learning_rate": tune.loguniform(1e-4, 1e-2),
						"early_stop_patience_steps": tune.choice([5]),
						"val_check_steps": tune.choice([500]),
						"scaler_type": tune.choice(['minmax1','robust']),
						"max_steps": tune.choice([10000, 15000]),
						"batch_size": tune.choice([128, 256]),
						"windows_batch_size": tune.choice([128, 512, 1024]),
						"random_seed": tune.randint(1, 20),
					})]

	patchtst = [AutoPatchTST(h=horizon,
						loss=loss, num_samples=num_samples,
						config={
							"input_size": tune.choice([horizon]),
							"hidden_size": tune.choice([64, 128, 256]),
							"linear_hidden_size": tune.choice([128, 256, 512]),
							"patch_len": tune.choice([3,4,6]),
							"stride": tune.choice([3,4,6]),
							"revin": tune.choice([True, False]),
							"learning_rate": tune.loguniform(1e-4, 1e-2),
							"early_stop_patience_steps": tune.choice([5]),
							"val_check_steps": tune.choice([500]),
							"scaler_type": tune.choice(['minmax1','robust']),
							"max_steps": tune.choice([10000, 15000]),
							"batch_size": tune.choice([128, 256]),
							"windows_batch_size": tune.choice([128, 512, 1024]),
							"random_seed": tune.randint(1, 20),
					})]

	tft = [AutoTFT(h=horizon,
					loss=loss, num_samples=num_samples,
					config={
						"input_size": tune.choice([horizon]),
						"hidden_size": tune.choice([64, 128, 256]),
						"learning_rate": tune.loguniform(1e-4, 1e-2),
						"early_stop_patience_steps": tune.choice([5]),
						"val_check_steps": tune.choice([500]),
						"scaler_type": tune.choice(['minmax1','robust']),
						"max_steps": tune.choice([10000, 15000]),
						"batch_size": tune.choice([128, 256]),
						"windows_batch_size": tune.choice([128, 512, 1024]),
						"random_seed": tune.randint(1, 20),
			})]
	
	mlp = [AutoMLP(h=horizon,
				loss=loss, num_samples=num_samples,
				config={
					"input_size": tune.choice([horizon]),
					"num_layers": tune.choice([4, 8, 32]),
					"hidden_size": tune.choice([512, 1024, 2048]),
					"learning_rate": tune.loguniform(1e-4, 1e-2),
					"early_stop_patience_steps": tune.choice([5]),
					"val_check_steps": tune.choice([500]),
					"scaler_type": tune.choice(['minmax1','robust']),
					"max_steps": tune.choice([10000, 15000]),
					"batch_size": tune.choice([128, 256]),
					"windows_batch_size": tune.choice([128, 512, 1024]),
					"random_seed": tune.randint(1, 20),
				})]
	
	tcn = [AutoTCN(h=horizon,
				loss=loss, num_samples=num_samples,
				config={
					"input_size": tune.choice([horizon, 2*horizon, 3*horizon]),
					"encoder_hidden_size": tune.choice([64, 128, 256, 512]),
					"context_size": tune.choice([4, 16, 64]),
					"decoder_hidden_size": tune.choice([64, 128]),
					"learning_rate": tune.loguniform(1e-4, 1e-2),
					"early_stop_patience_steps": tune.choice([5]),
					"val_check_steps": tune.choice([500]),
					"scaler_type": tune.choice(['minmax1','robust']),
					"max_steps": tune.choice([10000, 15000]),
					"batch_size": tune.choice([128, 256]),
					"random_seed": tune.randint(1, 20),
			})]

	lstm = [AutoLSTM(h=horizon,
				loss=loss, num_samples=num_samples,
				config={
					"input_size": tune.choice([-1, horizon, 2*horizon, 3*horizon]),
					"inference_input_size": tune.choice([-1]),
					"encoder_hidden_size": tune.choice([64, 128, 256, 512]),
					"encoder_n_layers": tune.randint(4, 8),
					"context_size": tune.choice([4, 16, 64]),
					"decoder_hidden_size": tune.choice([64, 128, 256]),
					"learning_rate": tune.loguniform(1e-4, 1e-2),
					"early_stop_patience_steps": tune.choice([5]),
					"val_check_steps": tune.choice([500]),
					"scaler_type": tune.choice(['minmax1','robust']),
					"max_steps": tune.choice([10000, 15000]),
					"batch_size": tune.choice([128, 256]),
					"random_seed": tune.randint(1, 20),
			})]
	
	MODEL_DICT = {'autonhits': nhits,
	       		  'autotft': tft,
				  'autopatchtst': patchtst,
				  'automlp': mlp,
				  'autotcn': tcn,
				  'autolstm': lstm}
	model = MODEL_DICT[args.model]

	return model

def load_data(args):

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

	return Y_df, frequency

def main(args):

	# make sure folder exists, then check if the file exists in the folder
	model_dir = f'./results/stored_models/{args.source_dataset}/{args.model}/{args.experiment_id}/'
	os.makedirs(model_dir, exist_ok=True)
	file_exists = os.path.isfile(
		f'./results/stored_models/{args.source_dataset}/{args.model}/{args.experiment_id}/{args.model}_0.ckpt')
	
	if (not file_exists):
		if args.k_shot > 0:
			raise Exception('Train model before k_shot learning')

		# Load data
		Y_df, frequency = load_data(args)

		# Train model
		print('Fitting model')
		model = load_model(args)
		nf = NeuralForecast(models=model, freq=frequency)
		Y_hat_source_df = nf.cross_validation(df=Y_df, val_size=horizon, test_size=horizon, n_windows=None)
		
		# Store source forecasts
		results_dir = f'./results/forecasts/{args.source_dataset}/'
		os.makedirs(results_dir, exist_ok=True)
		Y_hat_source_df.to_csv(f'{results_dir}/{args.model}_{args.source_dataset}_{args.experiment_id}.csv',
		 					   index=False)
		
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
	### TODO: <<<<<<< ADD TIME.TIME()

	# Fit model if k_shot > 0:
	if (args.k_shot > 0):
		##### TODO: <<<<<<< UPDATE max_steps in model
		nf.models[0].max_steps = args.k_shot
		Y_hat_df = nf.cross_validation(df=Y_df_target,
								n_windows=None, test_size=test_size,
								fit_models=True).reset_index()
	else:
		Y_hat_df = nf.cross_validation(df=Y_df_target,
									n_windows=None, test_size=test_size,
									fit_models=False).reset_index()
		
	#### TODO: <<<<<<< ADD TIME.TIME()
	#### AND COMPUTE TIME DIFFERENCE
	#### PRINT TIME DIFFERENCE
	results_dir = f'./results/forecasts/{args.target_dataset}/'
	os.makedirs(results_dir, exist_ok=True)

	# store results, also check if this folder exists/create it if its done
	Y_hat_df.to_csv(f'{results_dir}/{args.model}_{args.source_dataset}_{args.experiment_id}.csv',
		 index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="script arguments")
    #### TODO: <<<<<<< ADD k_shot
    parser.add_argument('--source_dataset', type=str, help='dataset to train models on')
    parser.add_argument('--target_dataset', type=str, help='run model on this dataset')
    parser.add_argument('--model', type=str, help='auto model to use')
    parser.add_argument('--k_shot', type=int, help='number of steps to fin tune model')
    parser.add_argument('--experiment_id', type=str, help='identify experiment')
    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    main(args)

# CUDA_VISIBLE_DEVICES=0 python run_models.py --source_dataset "M4" --target_dataset "M3" --model "automlp" --k_shot 0 --experiment_id "20230606"
# CUDA_VISIBLE_DEVICES=0 python run_models.py --source_dataset "M4" --target_dataset "M3" --model "autonhits" --k_shot 0 --experiment_id "20230606"
# CUDA_VISIBLE_DEVICES=0 python run_models.py --source_dataset "M4" --target_dataset "M3" --model "autotft" --k_shot 0 --experiment_id "20230606"
# CUDA_VISIBLE_DEVICES=0 python run_models.py --source_dataset "M4" --target_dataset "M3" --model "autopatchtst" --k_shot 0 --experiment_id "20230606"
# CUDA_VISIBLE_DEVICES=0 python run_models.py --source_dataset "M4" --target_dataset "M3" --model "autotcn" --k_shot 0 --experiment_id "20230606"
# CUDA_VISIBLE_DEVICES=0 python run_models.py --source_dataset "M4" --target_dataset "M3" --model "autolstm" --k_shot 0 --experiment_id "20230606"