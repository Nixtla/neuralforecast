from itertools import product

import numpy as np
import pandas as pd

from src.data import get_data

def mae(y, y_hat, axis):
    delta_y = np.abs(y - y_hat)
    mae = np.average(delta_y, axis=axis)
    return mae

def smape(y, y_hat, axis):
    delta_y = np.abs(y - y_hat)
    scale = np.abs(y) + np.abs(y_hat)
    smape = delta_y / scale
    smape = 200 * np.average(smape, axis=axis)
    return smape

def evaluate(model: str, dataset: str, group: str):
    try:
        forecast = pd.read_csv(f'data/{model}-forecasts-{dataset}-{group}.csv')
    except:
        return None
    y_test, horizon, freq, seasonality = get_data('data/', dataset, group, False)
    y_hat = forecast[model].values.reshape(-1, horizon)
    y_test = y_test['y'].values.reshape(-1, horizon)

    evals = {}
    for metric in (mae, smape):
        metric_name = metric.__name__
        loss = metric(y_test, y_hat, axis=1).mean()
        evals[metric_name] = loss 

    evals = pd.DataFrame(evals, index=[f'{dataset}_{group}']).rename_axis('dataset').reset_index()
    times = pd.read_csv(f'data/{model}-time-{dataset}-{group}.csv')
    evals = pd.concat([evals, times], axis=1)

    return evals


if __name__ == '__main__':
    groups = ['Monthly']
    models = ['AutoDilatedRNN', 'RNN', 
              'TCN', 
              'DeepAR',
              'NHITS', 'TFT', 'AutoMLP', 'DLinear', 'VanillaTransformer',
              'BiTCN', 'TiDE', 'DeepNPTS', 'NBEATS', 'KAN'
              ]
    datasets = ['M3']
    evaluation = [evaluate(model, dataset, group) for model, group in product(models, groups) for dataset in datasets]
    evaluation = [eval_ for eval_ in evaluation if eval_ is not None]
    df_evaluation = pd.concat(evaluation)
    df_evaluation = df_evaluation.loc[:, ['dataset', 'model', 'time', 'mae', 'smape']]
    df_evaluation['time'] /= 60 #minutes
    df_evaluation = df_evaluation.set_index(['dataset', 'model']).stack().reset_index()
    df_evaluation.columns = ['dataset', 'model', 'metric', 'val']
    df_evaluation = df_evaluation.set_index(['dataset', 'metric', 'model']).unstack().round(3)
    df_evaluation = df_evaluation.droplevel(0, 1).reset_index()
    df_evaluation['AutoARIMA'] = [666.82, 15.35, 3.000]
    df_evaluation.to_csv('data/evaluation.csv')
    print(df_evaluation.T)
