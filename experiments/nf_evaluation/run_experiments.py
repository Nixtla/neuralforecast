import os
import argparse
import time

import pandas as pd
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.numpy import mae, mse, smape, mape

from models import get_model
from datasets import get_dataset

# For compatibitlity with Mac with M chip
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def main(args):

    # Load dataset and model
    Y_df, h, freq, val_size, test_size = get_dataset(args.dataset)
    model = get_model(model_name=args.model, horizon=h, num_samples=20)

    # Start time
    start_time = time.time()

    # Train model
    nf = NeuralForecast(models=[model], freq=freq)
    forecasts_df = nf.cross_validation(df=Y_df, val_size=val_size, test_size=test_size, n_windows=None, verbose=True)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Evaluation
    model_mae = mae(y=forecasts_df['y'], y_hat=forecasts_df[f'{args.model}'])
    model_mse = mse(y=forecasts_df['y'], y_hat=forecasts_df[f'{args.model}'])
    model_smape = smape(y=forecasts_df['y'], y_hat=forecasts_df[f'{args.model}'])
    model_mape = mape(y=forecasts_df['y'], y_hat=forecasts_df[f'{args.model}'])

    metrics = {
            'Model': [args.model],
            'MAE': [model_mae],
            'MSE': [model_mse],
            'sMAPE': [model_smape],
            'MAPE': [model_mape],
            'time': [elapsed_time]
    }
        
    
    # Save results
    results_path = f'./results/{args.dataset}'
    os.makedirs(results_path, exist_ok=True)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'{results_path}/{args.model}_metrics.csv', header=True, index=False)
	
def parse_args():
	parser = argparse.ArgumentParser(description="script arguments")
	parser.add_argument('--dataset', type=str, help='dataset to train models on')
	parser.add_argument('--model', type=str, help='name of the model')
	# parser.add_argument('--experiment_id', type=str, help='identify experiment')
	return parser.parse_args()

if __name__ == '__main__':
	# parse arguments
	args = parse_args()
	
    # Run experiment
	main(args)