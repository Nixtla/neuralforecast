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
        nf.models_fitted[0].trainer_kwargs = trainer_kwargs
	

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
		test_size = horizon
		frequency = 'M'
	elif (args.target_dataset == 'M3'):
		Y_df_target, *_ = M3.load(directory='./', group='Monthly')
		Y_df_target['ds'] = pd.to_datetime(Y_df_target['ds'])
		frequency = 'M'
		test_size = horizon
	elif (args.target_dataset == 'M4'):
		Y_df_target, *_ = M4.load(directory='./', group='Monthly', cache=True)
		Y_df_target['ds'] = pd.to_datetime(Y_df_target['ds'])
		frequency = 'M'
		test_size = horizon
	elif (args.target_dataset == 'ILI'):
		Y_df_target, _, _ = LongHorizon.load(directory='./', group='ILI')
		Y_df_target['ds'] = np.repeat(np.array(range(len(Y_df_target)//7)), 7)
		test_size = horizon
		frequency = 'W'
	elif (args.target_dataset == 'TrafficL'):
		Y_df_target, _, _ = LongHorizon.load(directory='./', group='TrafficL')
		Y_df_target['ds'] = pd.to_datetime(Y_df_target['ds'])
		test_size = horizon
		frequency = 'H'
	else:
		raise Exception("Dataset not defined")

	# Predict on the test set of the target data
	print('Predicting on target data')
	start = time.time()
	# Fit model if k_shot > 0:
	if (args.k_shot > 0):
		# Set new trainer kwargs for k_shot learning
		set_trainer_kwargs(nf, max_steps=args.k_shot, early_stop_patience_steps=0)
		# Fit model and predict
		Y_hat_df = nf.cross_validation(df=Y_df_target,
									   n_windows=None,
									   test_size=test_size,
								       fit_models=True,
									   use_init_models=False).reset_index()
	else:
		# Predict
		Y_hat_df = nf.cross_validation(df=Y_df_target,
									n_windows=None, test_size=test_size,
									fit_models=False).reset_index()
	
	end = time.time()
	time_df = pd.DataFrame({'time': [end-start]})

	# Store time
	time_dir = f'./results/time/{args.target_dataset}/'
	os.makedirs(time_dir, exist_ok=True)
	time_dir = f'{time_dir}{args.model}_{args.k_shot}_{args.source_dataset}_{args.experiment_id}.csv'
	time_df.to_csv(time_dir, index=False)

	# Store forecasts results, also check if this folder exists/create it if its done
	forecasts_dir = f'./results/forecasts/{args.target_dataset}/'
	os.makedirs(forecasts_dir, exist_ok=True)

	forecasts_dir = f'{forecasts_dir}{args.model}_{args.k_shot}_{args.source_dataset}_{args.experiment_id}.csv'
	Y_hat_df.to_csv(forecasts_dir, index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="script arguments")
    parser.add_argument('--source_dataset', type=str, help='dataset to train models on')
    #parser.add_argument('--target_dataset', type=str, help='run model on this dataset')
    #parser.add_argument('--model', type=str, help='auto model to use')
    #parser.add_argument('--k_shot', type=int, help='number of steps to fin tune model')
    #parser.add_argument('--experiment_id', type=str, help='identify experiment')
    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    for k_shot in [0, 50, 100, 500, 1000]:
        for dataset in ['M3', 'AirPassengers', 'ILI', 'TrafficL']:
            for experiment_id in ['20230621', '20230621_2', '20230621_3']:
                for model in MODEL_LIST:
                    args.k_shot = k_shot
                    args.target_dataset = dataset
                    args.experiment_id = experiment_id
                    args.model = model
                    print(f'Running {k_shot} {dataset} {experiment_id} {model}!!')
                    forecasts_dir = f'./results/forecasts/{args.target_dataset}/'
                    forecasts_dir = f'{forecasts_dir}{args.model}_{args.k_shot}_{args.source_dataset}_{args.experiment_id}.csv'
                    print('forecasts_dir', forecasts_dir)
                    file_exists = os.path.isfile(forecasts_dir)
                    if (not file_exists):
                        main(args)
                    else:
                        print('Already done!')


# CUDA_VISIBLE_DEVICES=0 python run_models.py --source_dataset "M4" --target_dataset "M3" --k_shot 0 --experiment_id "20230621"

# CUDA_VISIBLE_DEVICES=0 python run_models.py --source_dataset "M4"