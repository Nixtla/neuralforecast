import os
import time
import argparse
import pandas as pd
import numpy as np

from neuralforecast.core import NeuralForecast
from neuralforecast.losses.numpy import mae, mape, mase, rmse, smape
from config_timenet import MODEL_LIST, load_model


def main():
    source_dataset = 'timenet'
    model = 'nhits_30_1024_daily'
    experiment_id = '20230626'
    horizon = 7

    nf = NeuralForecast.load(path=
            f'./results/stored_models/{source_dataset}/{model}/{experiment_id}/')

    partitions_df = pd.read_csv('partitions_df.csv')
    partitions_df = partitions_df[partitions_df['frequency'] == 'Daily']
    partitions_df = partitions_df[partitions_df['dataset'] == 'Wikipedia'].reset_index(drop=True)

    print('len partitions_df:', len(partitions_df))

    for row in partitions_df.iterrows():
        print('Iteration: ', row[0])
        url = row[1]['url']
        Y_df = pd.read_parquet(url)

        Y_df = Y_df.groupby('unique_id').tail(5*horizon).reset_index(drop=True)

        Y_hat_df = nf.cross_validation(df=Y_df,
                                    n_windows=1,
                                    fit_models=False,
                                    use_init_models=False).reset_index()
        Y_hat_df = Y_hat_df.groupby('unique_id').tail(horizon)

        print('nulls:', Y_hat_df['y'].isnull().sum())
        print('len:', len(Y_hat_df))
        
        mae_lambda = lambda x: mae(y=x['y'], y_hat=x['NHITS-median'])
        mape_lambda = lambda x: mape(y=x['y'], y_hat=x['NHITS-median'])
        rmse_lambda = lambda x: rmse(y=x['y'], y_hat=x['NHITS-median'])
        smape_lambda = lambda x: smape(y=x['y'], y_hat=x['NHITS-median'])
        
        df_metric_by_id = pd.DataFrame(columns=['unique_id', 'dataset', 'subdataset','metric', 'frequency', 'NHITS'])
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
            Y_metric = pd.DataFrame({'unique_id': Y_metric.index, 'dataset': 'Wikipedia', 'subdataset': 'Mini', 'metric': metric, 'frequency': 'Daily', 'NHITS': Y_metric.values})
            df_metric_by_id = pd.concat([df_metric_by_id, Y_metric], ignore_index=True)
            df_metric_by_id.to_parquet(f'./results/final/Wikipedia_Mini_{row[0]}.parquet')

    data_list = []
    for i in range(len(partitions_df)):
        data = pd.read_parquet('./results/final/Wikipedia_Mini_{}.parquet'.format(i))
        data_list.append(data)
    data = pd.concat(data_list, ignore_index=True)
    data.to_parquet('./results/final/Wikipedia_Mini.parquet')

if __name__ == '__main__':
    main()