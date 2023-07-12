import os
import time
import argparse
import pandas as pd
import numpy as np
import gc

from neuralforecast.losses.numpy import mae, mape, rmse, smape
from neuralforecast.core import NeuralForecast
from config_timenet import MODEL_LIST, load_model

HORIZON_DICT = {'Yearly': 1,
                'Quarterly': 4,
                'Monthly': 12,
                'Weekly': 1,
                'Daily': 7,
                'Hourly': 24,
                '30Minutely': 48,
                '15Minutely': 96,
                '10Minutely': 144,
                'Minutely': 60}

def read_data(partitions_dataset):
    urls = partitions_dataset['url'].values
    df_list = []
    for url in urls:
        df_list.append(pd.read_parquet(url))
    Y_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    Y_df['ds'] = pd.to_datetime(Y_df['ds']).dt.tz_localize(None)
    return Y_df

def run_inference(nf, Y_df, horizon):
    Y_hat_df = nf.cross_validation(df=Y_df,
                                   n_windows=1,
                                   fit_models=False,
                                   use_init_models=False).reset_index()
    Y_hat_df = Y_hat_df.groupby('unique_id').tail(horizon)
    return Y_hat_df

def compute_losses_by_ts(Y_hat_df, y_hat_col, model_name, dataset, subdataset, frequency):
    mae_lambda = lambda x: mae(y=x['y'], y_hat=x[y_hat_col])
    mape_lambda = lambda x: mape(y=x['y'], y_hat=x[y_hat_col])
    rmse_lambda = lambda x: rmse(y=x['y'], y_hat=x[y_hat_col])
    smape_lambda = lambda x: smape(y=x['y'], y_hat=x[y_hat_col])

    df_metric_by_id = pd.DataFrame(columns=['unique_id', 'dataset', 'subdataset','metric', 'frequency', model_name])
    for metric in [mae_lambda, mape_lambda, rmse_lambda, smape_lambda]:
        Y_metric = Y_hat_df.groupby('unique_id').apply(metric)
        if metric == mae_lambda:
            metric = 'mae'
        elif metric == mape_lambda:
            metric = 'mape'
        elif metric == rmse_lambda:
            metric = 'rmse'
        elif metric == smape_lambda:
            metric = 'smape'
        Y_metric = pd.DataFrame({'unique_id': Y_metric.index, 'dataset': dataset, 'subdataset': subdataset, 'metric': metric, 'frequency': frequency, model_name: Y_metric.values})
        df_metric_by_id = pd.concat([df_metric_by_id, Y_metric], ignore_index=True)
    return df_metric_by_id

def main(args):
    frequency = args.frequency
    model_name = args.model + '_' + frequency.lower()

    model_type = args.model.split('_')[0]
    # make sure folder exists, then check if the file exists in the folder
    model_dir = f'./results/timenet/stored_models/{model_name}/{args.experiment_id}'
    os.makedirs(model_dir, exist_ok=True)
    time_dir = f'./results/timenet/time'
    os.makedirs(time_dir, exist_ok=True)
    results_dir = f'./results/timenet/metrics/{model_name}/{args.experiment_id}'
    os.makedirs(results_dir, exist_ok=True)

    ############################### Train model ###############################
    file_exists = os.path.isfile(f'{model_dir}/{model_type}_0.ckpt')
    if (not file_exists):
        if frequency in ['Monthly', 'Weekly', 'Daily']:
            raise Exception('Frequency must be trained with run_timenet.py!!!')
        print('Training model...')
        if frequency == 'Yearly':
            freq = 'Y'
        if frequency == 'Quarterly':
            freq = 'Q'
        if frequency == 'Hourly':
            freq = 'H'
        if frequency == '30Minutely':
            freq = '30T'
        if frequency == '15Minutely':
            freq = '15T'
        if frequency == '10Minutely':
            freq = '10T'
        if frequency == 'Minutely':
            freq = 'T'

        partitions_df = pd.read_csv('partitions_df.csv')
        partitions_df = partitions_df[partitions_df['frequency'] == frequency]

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
        nf.save(path=model_dir, overwrite=False, save_dataset=False)
    else:
        print('Model already trained')
        pass

    ############################### Inference and metric ###############################
    print('Inference and metric')
    nf = NeuralForecast.load(path=model_dir)

    parts_df = pd.read_csv('partitions_df.csv')
    parts_df = parts_df[parts_df['frequency'] == frequency]

    datasets = parts_df['dataset'].unique()
    print('Datasets', datasets)
    for dataset in datasets:
        parts_dataset = parts_df[parts_df['dataset'] == dataset]
        subdatasets = parts_dataset['subdataset'].unique()
        print('Subdatasets', subdatasets)
        for subdataset in subdatasets:
            # Continue if file exists
            file_exists = os.path.isfile(f'{results_dir}/{dataset}_{subdataset}_{frequency}.parquet')
            if file_exists:
                print(f'{dataset}_{subdataset}_{frequency} already exists')
                continue

            subparts_dataset = parts_dataset[parts_dataset['subdataset'] == subdataset]

            # Read Data
            Y_df = read_data(partitions_dataset=subparts_dataset)

            # Modify frequency
            if frequency == 'Yearly':
                freq = 'AS'
                nf.freq = pd.tseries.frequencies.to_offset(freq)
            elif frequency == 'Quarterly':
                if subdataset == 'M3':
                    Y_df['ds'] = Y_df['ds'] + pd.DateOffset(months=1)
                freq = 'QS'
                nf.freq = pd.tseries.frequencies.to_offset(freq)
            elif frequency == 'Monthly':
                freq = 'MS'
                nf.freq = pd.tseries.frequencies.to_offset(freq)
            elif frequency == 'Weekly':
                if subdataset == 'ILI':
                    freq = 'W-TUE'
                elif subdataset == 'electricity':
                    freq = 'W-SUN'
                elif subdataset == 'nn5':
                    freq = 'W-MON'
                nf.freq = pd.tseries.frequencies.to_offset(freq)
            elif frequency == '30Minutely':
                freq = '30T'
                nf.freq = pd.tseries.frequencies.to_offset(freq)
            elif frequency == '15Minutely':
                freq = '15T'
                nf.freq = pd.tseries.frequencies.to_offset(freq)
            elif frequency == '10Minutely':
                freq = '10T'
                nf.freq = pd.tseries.frequencies.to_offset(freq)
            elif frequency == 'Minutely':
                freq = 'T'
                nf.freq = pd.tseries.frequencies.to_offset(freq)

            # Shorten history
            if frequency == '30Minutely':
                Y_df = Y_df.groupby('unique_id').tail(48*120).reset_index(drop=True)
            if frequency == '10Minutely':
                Y_df = Y_df.groupby('unique_id').tail(144*90).reset_index(drop=True)
            if frequency == 'Minutely':
                Y_df = Y_df.groupby('unique_id').tail(60*24*30).reset_index(drop=True)

            # Run inference 
            Y_hat_df = run_inference(nf=nf, Y_df=Y_df, horizon=HORIZON_DICT[frequency])
            print('nulls:', Y_hat_df['y'].isnull().sum())
            if Y_hat_df['y'].isnull().sum()>0:
                raise Exception('Nulls in Y_hat_df')

            # Compute metrics
            model_type = model_type.upper()
            df_metric_by_id = compute_losses_by_ts(Y_hat_df=Y_hat_df, y_hat_col=f'{model_type}-median', model_name=model_type,
                                                dataset=dataset, subdataset=subdataset, frequency=frequency)

            df_metric_by_id.to_parquet(f'{results_dir}/{dataset}_{subdataset}_{frequency}.parquet')

def parse_args():
    parser = argparse.ArgumentParser(description="script arguments")
    parser.add_argument('--model', type=str, help='auto model to use')
    parser.add_argument('--experiment_id', type=str, help='identify experiment')
    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    frequency_list = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly', '30Minutely', '15Minutely', '10Minutely', 'Minutely']

    for frequency in frequency_list:
        print('Frequency: ', frequency)
        args.frequency = frequency
        main(args)


# CUDA_VISIBLE_DEVICES=0 python run_timenet_small.py --model "nhits_30_1024"  --experiment_id "20230710"