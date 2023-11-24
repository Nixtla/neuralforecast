import os
import argparse
import time
import pandas as pd
import numpy as np

import torch
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from neuralforecast.core import NeuralForecast
from datasetsforecast.m4 import M4
from datasetsforecast.m3 import M3
from datasetsforecast.long_horizon import LongHorizon

from neuralforecast.utils import AirPassengersDF

from config_models import MODEL_LIST, load_model


DATASET_CONFIG = {'M3': ['M1', 'Tourism'], 'M4': ['M1', 'M3', 'Tourism'], 'wiki': ['M3_daily']}
TARGET_FREQ_CONFIG = {'M1': ['yearly', 'quarterly', 'monthly'], 'M3': ['yearly', 'quarterly', 'monthly','daily'], 'Tourism': ['yearly', 'quarterly', 'monthly'], 'M3_daily': ['daily']}

def load_data(args, frequency):

	# Read source data
	if (args.source_dataset == 'M4'): # add more if conditions later, expects M4 only for now
		if frequency == 'yearly':
			Y_df, *_ = M4.load(directory='./', group='Yearly')
		elif frequency == 'quarterly':
			Y_df, *_ = M4.load(directory='./', group='Quarterly')
		elif frequency == 'monthly':
			Y_df, *_ = M4.load(directory='./', group='Monthly')
		elif frequency == 'daily':
			Y_df, *_ = M4.load(directory='./', group='Daily')
			frequency = 'M'
		else:
			raise Exception("Frequency not defined")
	elif (args.source_dataset == 'wiki'):
		if frequency == 'daily':
			Y_df = pd.read_csv('wiki_daily.csv')
		elif frequency == 'weekly':
			Y_df = pd.read_csv('wiki_weekly.csv')
			frequency = 'W'
		elif frequency == 'monthly':
			Y_df = pd.read_csv('wiki_monthly.csv')
			frequency = 'M'
	elif (args.source_dataset == 'M3'):
		if args.frequency == 'yearly':
			Y_df, *_ = M3.load(directory='./', group='Yearly')
		elif args.frequency == 'quarterly':
			Y_df, *_ = M3.load(directory='./', group='Quarterly')
		elif args.frequency == 'monthly':
			Y_df, *_ = M3.load(directory='./', group='Monthly')
		else:
			raise Exception("Frequency not defined")
	else:
		raise Exception("Dataset not defined")

	Y_df['ds'] = pd.to_datetime(Y_df['ds'])

	print('Y_df', Y_df.shape)
	print('Y_df', Y_df)

	return Y_df, frequency

def set_trainer_kwargs(nf, max_steps, early_stop_patience_steps):
	## Trainer arguments ##
	# Max steps, validation steps and check_val_every_n_epoch
	trainer_kwargs = {**{'max_steps': max_steps}}

	if 'max_epochs' in trainer_kwargs.keys():
		raise Exception('max_epochs is deprecated, use max_steps instead.')

	# Callbacks
	if trainer_kwargs.get('callbacks', None) is None:
		callbacks = [TQDMProgressBar()]
		# Early stopping
		if early_stop_patience_steps > 0:
			callbacks += [EarlyStopping(monitor='ptl/val_loss',
										patience=early_stop_patience_steps)]

		trainer_kwargs['callbacks'] = callbacks

	# Add GPU accelerator if available
	if trainer_kwargs.get('accelerator', None) is None:
		if torch.cuda.is_available():
			trainer_kwargs['accelerator'] = "gpu"
	if trainer_kwargs.get('devices', None) is None:
		if torch.cuda.is_available():
			trainer_kwargs['devices'] = -1

	# Avoid saturating local memory, disabled fit model checkpoints
	if trainer_kwargs.get('enable_checkpointing', None) is None:
		trainer_kwargs['enable_checkpointing'] = False

	nf.models[0].trainer_kwargs = trainer_kwargs
	nf.models_init[0].trainer_kwargs = trainer_kwargs
	

def main(args):
	model_type = args.model.split('_')[0]
	# make sure folder exists, then check if the file exists in the folder
	model_dir = f'./results/transferability/stored_models/{args.source_dataset}/{args.frequency}/{args.model}/{args.experiment_id}'
	os.makedirs(model_dir, exist_ok=True)
	file_exists = os.path.isfile(f'{model_dir}/{model_type}_0.ckpt')
	print('model_dir', model_dir)
	if (not file_exists):
		if args.k_shot > 0:
			raise Exception('Train model before k_shot learning')

		Y_df, frequency = load_data(args, args.frequency)

		# Train model
		print('Fitting model')
		# Load data
		if (args.frequency == 'monthly') and (args.source_dataset == 'wiki'):
			short= True
		else:
			short = False
		model = load_model(args.model, args.frequency, short=short, random_seed=args.random_seed)
		nf = NeuralForecast(models=[model], freq=None) # freq is set later
		print('HORIZON: ', nf.models[0].h)
		nf.fit(df=Y_df, val_size=2*model.h)

		# Save model
		nf.save(path=model_dir, overwrite=False, save_dataset=False)
	else:
		print('Hyperparameter optimization already done. Loading saved model!')
		# do i need to check if the file/path exists? shouldn't it already be checked
		nf = NeuralForecast.load(path=model_dir)

	# UNCOMMENT FROM HERE

	horizon = nf.models[0].h
	
	# Load target data
	if (args.target_dataset == 'M3'):
		if args.frequency == 'yearly':
			Y_df_target, *_ = M3.load(directory='./', group='Yearly')
			frequency = 'Y'		
		elif args.frequency == 'quarterly':
			Y_df_target, *_ = M3.load(directory='./', group='Quarterly')
			frequency = 'Q'
		elif args.frequency == 'monthly':
			Y_df_target, *_ = M3.load(directory='./', group='Monthly')
			frequency = 'M'
		elif args.frequency == 'daily':
			Y_df_target, *_ = M3.load(directory='./', group='Other')
			frequency = 'D'
	elif (args.target_dataset == 'Tourism'):
		if args.frequency == 'yearly':
			Y_df_target = pd.read_csv('tourism/data_yearly.csv')
			Y_df_target['ds'] = pd.to_datetime(Y_df_target['ds'])
			frequency = 'Y'
		elif args.frequency == 'quarterly':
			Y_df_target = pd.read_csv('tourism/data_quarterly.csv')
			Y_df_target['ds'] = pd.to_datetime(Y_df_target['ds'])
			frequency = 'Q'
		elif args.frequency == 'monthly':
			Y_df_target = pd.read_csv('tourism/data.csv')
			Y_df_target['ds'] = pd.to_datetime(Y_df_target['ds'])
			frequency = 'MS'
	elif (args.target_dataset == 'M1'):
		if args.frequency == 'yearly':
			Y_df_target = pd.read_csv('m1/data_Yearly.csv')
			Y_df_target['ds'] = pd.to_datetime(Y_df_target['ds'])
			frequency = 'YS'
		elif args.frequency == 'quarterly':
			Y_df_target = pd.read_csv('m1/data_Quarterly.csv')
			Y_df_target['ds'] = pd.to_datetime(Y_df_target['ds'])
			frequency = 'QS'
		elif args.frequency == 'monthly':
			Y_df_target = pd.read_csv('m1/data_Monthly.csv')
			Y_df_target['ds'] = pd.to_datetime(Y_df_target['ds'])
			frequency = 'MS'
	else:
		raise Exception("Dataset not defined")

	# # if (args.target_dataset == 'AirPassengers'):
	# # 	Y_df_target = AirPassengersDF.copy()
	# # 	Y_df_target['ds'] = pd.to_datetime(Y_df_target['ds'])
	# # 	test_size = horizon
	# # 	frequency = 'M'
	# # elif (args.target_dataset == 'M3'):
	# # 	Y_df_target, *_ = M3.load(directory='./', group='Monthly')
	# # 	Y_df_target['ds'] = pd.to_datetime(Y_df_target['ds'])
	# # 	frequency = 'M'
	# # 	test_size = horizon
	# # elif (args.target_dataset == 'M4'):
	# # 	Y_df_target, *_ = M4.load(directory='./', group='Monthly', cache=True)
	# # 	Y_df_target['ds'] = pd.to_datetime(Y_df_target['ds'])
	# # 	frequency = 'M'
	# # 	test_size = horizon
	# # elif (args.target_dataset == 'ILI'):
	# # 	Y_df_target, _, _ = LongHorizon.load(directory='./', group='ILI')
	# # 	Y_df_target['ds'] = np.repeat(np.array(range(len(Y_df_target)//7)), 7)
	# # 	test_size = horizon
	# # 	frequency = 'W'
	# # elif (args.target_dataset == 'TrafficL'):
	# # 	Y_df_target, _, _ = LongHorizon.load(directory='./', group='TrafficL')
	# # 	Y_df_target['ds'] = pd.to_datetime(Y_df_target['ds'])
	# # 	test_size = horizon
	# # 	frequency = 'H'
	# # elif (args.target_dataset == 'Tourism'):
	# # 	tourism_data = pd.read_csv('tourism/data.csv')
	# # 	tourism_data = tourism_data.rename(columns={'Unnamed: 0': 'ds'})
	# # 	Y_df_target = []
	# # 	for col in tourism_data.columns[1:]:
	# # 		hola = tourism_data[['ds', col]].rename(columns={col: 'y'})
	# # 		hola['unique_id'] = col
	# # 		Y_df_target.append(hola)
	# # 	Y_df_target = pd.concat(Y_df_target).reset_index(drop=True)
	# # 	Y_df_target['ds'] = pd.to_datetime(Y_df_target['ds'])
	# # 	test_size = horizon
	# # 	frequency = 'MS'
	# # else:
	# # 	raise Exception("Dataset not defined")
	
	nf.freq = pd.tseries.frequencies.to_offset(frequency)
	test_size = horizon

	# Predict on the test set of the target data
	print('Predicting on target data')
	start = time.time()
	# Fit model if k_shot > 0:
	# Set new trainer kwargs for k_shot learning
	set_trainer_kwargs(nf, max_steps=args.k_shot, early_stop_patience_steps=0)
	if (args.k_shot > 0):
		# Fit model and predict
		Y_hat_df = nf.cross_validation(df=Y_df_target,
									   n_windows=None,
									   test_size=test_size,
									   use_init_models=False).reset_index()
	else:
		# Predict
		print('Predicting on target data')
		print(nf.models[0].trainer_kwargs)
		Y_hat_df = nf.cross_validation(df=Y_df_target,
									   n_windows=None,
									   test_size=test_size,
									   use_init_models=False).reset_index()
	
	end = time.time()
	time_df = pd.DataFrame({'time': [end-start]})

	# Store time
	time_dir = f'./results/transferability/time/{args.target_dataset}/{args.frequency}'
	os.makedirs(time_dir, exist_ok=True)
	time_dir = f'{time_dir}/{args.model}_{args.k_shot}_{args.source_dataset}_{args.experiment_id}.csv'
	time_df.to_csv(time_dir, index=False)

	# Store forecasts results, also check if this folder exists/create it if its done
	forecasts_dir = f'./results/transferability/forecasts/{args.target_dataset}/{args.frequency}/'
	os.makedirs(forecasts_dir, exist_ok=True)

	forecasts_dir = f'{forecasts_dir}/{args.model}_{args.k_shot}_{args.source_dataset}_{args.experiment_id}.csv'
	Y_hat_df.to_csv(forecasts_dir, index=False)

def parse_args():
	parser = argparse.ArgumentParser(description="script arguments")
	#parser.add_argument('--source_dataset', type=str, help='dataset to train models on')
	#parser.add_argument('--frequency', type=str, help='frequency')
	#parser.add_argument('--target_dataset', type=str, help='run model on this dataset')
	#parser.add_argument('--k_shot', type=int, help='number of steps to fin tune model')
	parser.add_argument('--experiment_id', type=str, help='identify experiment')
	return parser.parse_args()

if __name__ == '__main__':
	# parse arguments
	args = parse_args()

	for k_shot in [500]:
		for random_seed in [1]:
			for source_dataset in DATASET_CONFIG.keys():
				for target_dataset in DATASET_CONFIG[source_dataset]:
					for frequency in TARGET_FREQ_CONFIG[target_dataset]:
						for model in MODEL_LIST:
							args.source_dataset = source_dataset
							if target_dataset == 'M3_daily':
								target_dataset = 'M3'
							args.target_dataset = target_dataset
							args.k_shot = k_shot
							args.model = model
							args.frequency = frequency
							args.random_seed = random_seed
							print(f'Running {frequency} {model} {k_shot} {random_seed}!!')
							# forecasts_dir = f'./results/transferability/forecasts/{args.target_dataset}/'
							# forecasts_dir = f'{forecasts_dir}{args.model}_{args.k_shot}_{args.source_dataset}_{args.experiment_id}.csv'
							# print('forecasts_dir', forecasts_dir)
							# file_exists = os.path.isfile(forecasts_dir)
							#if (not file_exists):
							main(args)
							#else:
							print('Already done!')


# CUDA_VISIBLE_DEVICES=0 python run_transfer.py --experiment_id "20230816"
