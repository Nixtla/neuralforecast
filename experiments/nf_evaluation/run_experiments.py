import os
import argparse
import time

import pandas as pd
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.numpy import mae, mse, smape, mape

from .models import get_model
from .datasets import get_dataset


def main(args):
    
    # Load dataset and model
    Y_df, h, freq, val_size, test_size = get_dataset(args.dataset)
    model = get_model(model=args.model, horizon=h, num_samples=20)

    # create experiment directory
    start_time = time.time()
	
    # Train model
    nf = NeuralForecast(model=[model], freq=freq)
    forecasts_df = nf.cross_validation(df=Y_df, val_size=val_size, test_size=test_size, verbose=True)
    elapsed_time = time.time() - start_time
    
    # Evaluation
    metrics_df = pd.DataFrame()
    metrics_df['MAE'] = mae(y=forecasts_df['y'], y_hat=forecasts_df['model'])
    metrics_df['MSE'] = mse(y=forecasts_df['y'], y_hat=forecasts_df['model'])
    metrics_df['SMAPE'] = smape(y=forecasts_df['y'], y_hat=forecasts_df['model'])
    metrics_df['MAPE'] = mape(y=forecasts_df['y'], y_hat=forecasts_df['model'])
    metrics_df['elapsed_time'] = elapsed_time
	
    # Save results
    os.makedirs(f'./results/{args.model}_{args.dataset}_{args.experiment_id}', exist_ok=True)
	
def parse_args():
	parser = argparse.ArgumentParser(description="script arguments")
	parser.add_argument('--dataset', type=str, help='dataset to train models on')
	parser.add_argument('--model', type=int, help='number of steps to fin tune model')
	parser.add_argument('--experiment_id', type=str, help='identify experiment')
	return parser.parse_args()

if __name__ == '__main__':
	# parse arguments
	args = parse_args()