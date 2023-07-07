import os
import time
import argparse
import pandas as pd
import numpy as np
import gc

from neuralforecast.core import NeuralForecast
from config_timenet import MODEL_LIST, load_model


def main(args):
    frequency = args.frequency
    model_name = args.model + '_' + frequency.lower()

    model_type = args.model.split('_')[0]
    # make sure folder exists, then check if the file exists in the folder
    model_dir = f'./results/stored_models/{args.source_dataset}/{model_name}/{args.experiment_id}/'
    os.makedirs(model_dir, exist_ok=True)
    file_exists = os.path.isfile(
        f'./results/stored_models/{args.source_dataset}/{model_name}/{args.experiment_id}/{model_type}_0.ckpt')

    if (not file_exists):

        if frequency == 'Hourly':
            freq = 'H'
        if frequency == 'Quarterly':
            freq = 'Q'
        if frequency == 'Yearly':
            freq = 'Y'
        if frequency == '10Minutely':
            freq = '10T'
        if frequency == '15Minutely':
            freq = '15T'
        if frequency == 'Weekly':
            freq = 'W'
        if frequency == '30Minutely':
            freq = '30T'
        if frequency == 'Minutely':
            freq = 'T'

        partitions_df = pd.read_csv('partitions_df.csv')
        partitions_df = partitions_df[partitions_df['frequency'] == frequency]

        if frequency == '15Minutely':
            print('Removing ECL')
            print('Partitions before: ', partitions_df.shape)
            partitions_df = partitions_df[partitions_df['subdataset'] != 'ECL']
            print('Partitions before: ', partitions_df.shape)
        
        if frequency == 'Weekly':
            print('Removing Wiki')
            print('Partitions before: ', partitions_df.shape)
            partitions_df = partitions_df[partitions_df['subdataset'] != 'Mini']
            print('Partitions before: ', partitions_df.shape)
    
        urls = partitions_df['url'].tolist()
        print('PARTITIONS: ', partitions_df)

        # Read data
        print('Reading data')
        df_list = []
        for url in urls:
            df_list.append(pd.read_parquet(url))
        Y_df = pd.concat(df_list, axis=0).reset_index(drop=True)
        Y_df['ds'] = pd.to_datetime(Y_df['ds']).dt.tz_localize(None)

        if frequency == 'Minutely':
            print('Filtering last history')
            Y_df = Y_df.groupby('unique_id').tail(60*24*30).reset_index(drop=True)

        if frequency == '10Minutely':
            Y_df = Y_df.groupby('unique_id').tail(144*90).reset_index(drop=True)

        if frequency == '30Minutely':
            Y_df = Y_df.groupby('unique_id').tail(48*120).reset_index(drop=True)

        print('Y_df shape: ', Y_df.shape)

        model = load_model(model_name)
        nf = NeuralForecast(models=[model], freq=freq)

        Y_hat_df = nf.cross_validation(df=Y_df, val_size=model.h, test_size=model.h, n_windows=None)
    
        # Save model
        print('Saving model')
        nf.save(path=f'./results/stored_models/{args.source_dataset}/{model_name}/{args.experiment_id}/',
            overwrite=False, save_dataset=False)

        results_dir = f'./results/forecasts/{args.source_dataset}/'
        os.makedirs(results_dir, exist_ok=True)

        # Store results, also check if this folder exists/create it if its done
        Y_hat_df.to_csv(f'{results_dir}/{model_name}_{args.source_dataset}_{args.experiment_id}.csv',
            index=False)
    else:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="script arguments")
    parser.add_argument('--source_dataset', type=str, help='dataset to train models on')
    parser.add_argument('--model', type=str, help='auto model to use')
    parser.add_argument('--experiment_id', type=str, help='identify experiment')
    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    frequency_list = ['Minutely'] #['Hourly', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']

    for frequency in frequency_list:
        print('Frequency: ', frequency)
        args.frequency = frequency
        main(args)


# CUDA_VISIBLE_DEVICES=0 python run_timenet_small.py --model "nhits_30_1024" --source_dataset "timenet" --experiment_id "20230626"

# 4 with 17
# 3 with 10
# 2 with 5