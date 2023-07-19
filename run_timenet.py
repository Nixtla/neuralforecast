import os
import time
import argparse
import pandas as pd
import numpy as np
import gc

from neuralforecast.losses.numpy import mae, mape, mase, rmse, smape
from neuralforecast.core import NeuralForecast
from config_timenet import load_model

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

def run_inference(nf, Y_df, horizon):
    Y_hat_df = nf.cross_validation(df=Y_df,
                                   n_windows=1,
                                   fit_models=False,
                                   use_init_models=False).reset_index()
    Y_hat_df = Y_hat_df.groupby('unique_id').tail(horizon)
    return Y_hat_df

def compute_losses(Y_hat_df, y_hat_col, dataset, subdataset, frequency):
    mae_loss = mae(y=Y_hat_df['y'], y_hat=Y_hat_df[y_hat_col])
    mape_loss = mape(y=Y_hat_df['y'], y_hat=Y_hat_df[y_hat_col])
    rmse_loss = rmse(y=Y_hat_df['y'],y_hat=Y_hat_df[y_hat_col])
    smape_loss = smape(y=Y_hat_df['y'], y_hat=Y_hat_df[y_hat_col])

    row = pd.DataFrame({'dataset':[dataset], 'subdataset': [subdataset], 'frequency':frequency, 'mae': [mae_loss], 'mape':[mape_loss], 'rmse':[rmse_loss], 'smape':[smape_loss]})
    df_results = pd.concat([df_results, row], ignore_index=True)
    return df_results

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
        print('Training model...')
        start = time.time()

        # Frequency for nf
        if frequency == 'Monthly':
            freq = 'M'
        elif frequency == 'Weekly':
            freq = 'W'
        elif frequency == 'Daily':
            freq = 'D'

        model = load_model(model_name)
        nf = NeuralForecast(models=[model], freq=freq)

        # TOTAL STEPS
        if frequency in ['Monthly', 'Weekly', 'Daily']:
            total_steps = 20000

        individual_steps = model.max_steps

        partitions_df = pd.read_csv('partitions_df.csv')
        assert len(partitions_df) == 304, 'Old partitions!'
        
        partitions_df = partitions_df[partitions_df['frequency'] == frequency]
        print('PARTITIONS: ', partitions_df)

        for i in range(total_steps//individual_steps):
            start_2 = time.time()
            print('ITERATION: ', i , 'out of ', total_steps//individual_steps)

            df_list = []
            # Add WIKI
            if frequency in ['Monthly', 'Weekly', 'Daily']:
                print("Loading WIKI")

                # Limit number of partitions to control RAM usage
                if frequency == 'Monthly':
                    num_partitions = 10
                elif frequency == 'Weekly':
                    num_partitions = 7
                elif frequency == 'Daily':
                    num_partitions = 3

                wiki_urls = partitions_df[partitions_df['dataset'] == 'Wikipedia']['url'].values
                wiki_urls = np.random.choice(wiki_urls, size=num_partitions, replace=False) # Ensure no duplicates

                for wiki_url in wiki_urls:
                    df_list.append(pd.read_parquet(wiki_url))

            # OTHER
            other_urls = partitions_df[partitions_df['dataset'] != 'Wikipedia']['url'].values
            if frequency == 'Monthly':
                other_urls = np.random.choice(other_urls, size=7, replace=False) # Ensure no duplicates
            elif frequency == 'Daily':
                other_urls = np.random.choice(other_urls, size=3, replace=False) # Ensure no duplicates
            
            print('Other urls: ', other_urls)

            # Load
            print("Loading OTHER")
            for url in other_urls:
                df_list.append(pd.read_parquet(url))
                              
            # Process data
            Y_df = pd.concat(df_list, axis=0).reset_index(drop=True)
            Y_df = Y_df.drop(Y_df.groupby(['unique_id']).tail(model.h).index, axis=0).reset_index(drop=True)

            if frequency == 'Daily':
                print('Subsampling to 10_000 unique ids')
                unique_ids = Y_df['unique_id'].unique()
                if len(unique_ids) > 10_000:
                    unique_ids = np.random.choice(unique_ids, size=10_000, replace=False)
                    Y_df = Y_df[Y_df['unique_id'].isin(unique_ids)].reset_index(drop=True)
            
            nf.fit(df=Y_df, val_size=0, use_init_models=False)
            end = time.time()            
            print('Time: ', end-start_2)

        del Y_df
        del df_list
        gc.collect()

        end = time.time()
        time_df = pd.DataFrame({'time': [end-start]})
        # Store time
        time_file = f'{time_dir}/{model_name}__{args.experiment_id}.csv'
        time_df.to_csv(time_file, index=False)

        # Save model
        print('Saving model')
        nf.save(path=model_dir, overwrite=False, save_dataset=False)
    else:
        print('Model already trained')
        pass
    
    ############################### Inference and metric ###############################
    print('Inference and metric')
    nf = NeuralForecast.load(path=model_dir)

    horizon = HORIZON_DICT[frequency]

    if frequency == 'Weekly':
        freq = 'W-MON'
        nf.freq = pd.tseries.frequencies.to_offset(freq)        
    elif frequency == 'Monthly':
        freq = 'MS'
        nf.freq = pd.tseries.frequencies.to_offset(freq)

    # Read partitions
    partitions_df = pd.read_csv('partitions_df.csv')
    assert len(partitions_df) == 304, 'Old partitions!'
    partitions_df = partitions_df[partitions_df['frequency'] == frequency]
    partitions_df = partitions_df[partitions_df['dataset'] == 'Wikipedia'].reset_index(drop=True)
    print('len partitions_df:', len(partitions_df))

    for row in partitions_df.iterrows():
        print('Iteration: ', row[0])
        url = row[1]['url']
        Y_df = pd.read_parquet(url)
        Y_df['ds'] = pd.to_datetime(Y_df['ds']).dt.tz_localize(None)

        if frequency == 'Daily':
            Y_df = Y_df.groupby('unique_id').tail(5*horizon).reset_index(drop=True)
        
        print('Y_df.dtypes', Y_df.dtypes)
        print('Y_df', Y_df.head())
        Y_hat_df = run_inference(nf=nf, Y_df=Y_df, horizon=horizon)

        print('Y_hat_df.dtypes', Y_hat_df.dtypes)
        print('Y_hat_df', Y_hat_df.head())

        print('nulls:', Y_hat_df['y'].isnull().sum())
        print('len:', len(Y_hat_df))
        
        # Compute metrics
        if (model_type == 'deepar') or (model_type == 'DeepAR'):
            model_type = 'DeepAR'
        else: 
            model_type = model_type.upper()
        df_metric_by_id = compute_losses_by_ts(Y_hat_df=Y_hat_df, y_hat_col=f'{model_type}-median', model_name=model_type,
                                               dataset='Wikipedia', subdataset='Mini', frequency=frequency)

        df_metric_by_id.to_parquet(f'{results_dir}/Wikipedia_Mini_{frequency}_{row[0]}.parquet')

    data_list = []
    for i in range(len(partitions_df)):
        data = pd.read_parquet(f'{results_dir}/Wikipedia_Mini_{frequency}_{i}.parquet')
        data_list.append(data)
    data = pd.concat(data_list, ignore_index=True)
    data.to_parquet(f'{results_dir}/Wikipedia_Mini_{frequency}.parquet')


def parse_args():
    parser = argparse.ArgumentParser(description="script arguments")
    parser.add_argument('--model', type=str, help='auto model to use')
    parser.add_argument('--experiment_id', type=str, help='identify experiment')
    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    frequency_list = ['Monthly', 'Weekly', 'Daily'] #['Hourly', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']

    for frequency in frequency_list:
        print('Frequency: ', frequency)
        args.frequency = frequency
        main(args)


# CUDA_VISIBLE_DEVICES=0 python run_timenet.py --model "nhits_30_1024" --experiment_id "20230710"
# CUDA_VISIBLE_DEVICES=0 python run_timenet.py --model "tft_1024" --experiment_id "20230710"
# CUDA_VISIBLE_DEVICES=0 python run_timenet.py --model "lstm_512_4" --experiment_id "20230710"
# CUDA_VISIBLE_DEVICES=0 python run_timenet.py --model "deepar_128_4" --experiment_id "20230710"
