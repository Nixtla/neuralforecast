import os
import argparse
import pandas as pd
import numpy as np

from neuralforecast.core import NeuralForecast
from datasetsforecast.m4 import M4
from datasetsforecast.m3 import M3
from datasetsforecast.long_horizon import LongHorizon

from neuralforecast.utils import AirPassengersDF

from config_models import MODEL_LIST, load_model

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
	model_type = args.model.split('_')[0]
	# make sure folder exists, then check if the file exists in the folder
	model_dir = f'./results/stored_models/{args.source_dataset}/{args.model}/{args.experiment_id}/'
	os.makedirs(model_dir, exist_ok=True)
	file_exists = os.path.isfile(
		f'./results/stored_models/{args.source_dataset}/{args.model}/{args.experiment_id}/{model_type}_0.ckpt')
	
	if (not file_exists):
		if args.k_shot > 0:
			raise Exception('Train model before k_shot learning')

		# Load data
		Y_df, frequency = load_data(args)

		# Train model
		print('Fitting model')
		model = load_model(args.model)
		nf = NeuralForecast(models=[model], freq=frequency)
		Y_hat_source_df = nf.cross_validation(df=Y_df, val_size=model.h, test_size=model.h, n_windows=None)
		
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

	horizon = nf.models[0].h
	
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
		
    #### AND COMPUTE TIME DIFFERENCE
	#### PRINT TIME DIFFERENCE
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
    parser.add_argument('--k_shot', type=int, help='number of steps to fin tune model')
    parser.add_argument('--experiment_id', type=str, help='identify experiment')
    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    for model in MODEL_LIST:
        print(f'Running {model}!!')
        args.model = model
        main(args)


# CUDA_VISIBLE_DEVICES=0 python run_models.py --source_dataset "M4" --target_dataset "M3" --k_shot 0 --experiment_id "20230621_2"