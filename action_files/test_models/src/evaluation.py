from itertools import product

import numpy as np
import pandas as pd

from src.data import get_data

def mape(y, y_hat, axis):
    delta_y = np.abs(y - y_hat)
    scale = np.abs(y)
    mape = delta_y / scale
    mape = np.average(mape, axis=axis)
    return 100 * mape

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
    for metric in (mape, smape):
        metric_name = metric.__name__
        loss = metric(y_test, y_hat, axis=1).mean()
        evals[metric_name] = loss 

    evals = pd.DataFrame(evals, index=[f'{dataset}_{group}']).rename_axis('dataset').reset_index()
    times = pd.read_csv(f'data/{model}-time-{dataset}-{group}.csv')
    evals = pd.concat([evals, times], axis=1)

    return evals


if __name__ == '__main__':
    groups = ['Other']#, 'Quarterly', 'Monthly']
    models = ['GRU', 'RNN', 'TCN', 'LSTM', 'DilatedRNN',
              'MLP', 'NHITS', 'NBEATS', 'NBEATSx', 'TFT',
              'AutoNHITS', 'AutoNBEATS', 'AutoMLP', 'AutoDilatedRNN']
    datasets = ['M3']
    evaluation = [evaluate(model, dataset, group) for model, group in product(models, groups) for dataset in datasets]
    evaluation = [eval_ for eval_ in evaluation if eval_ is not None]
    evaluation = pd.concat(evaluation)
    evaluation = evaluation[['dataset', 'model', 'time', 'mape', 'smape']]
    evaluation['time'] /= 60 #minutes
    evaluation = evaluation.set_index(['dataset', 'model']).stack().reset_index()
    evaluation.columns = ['dataset', 'model', 'metric', 'val']
    evaluation = evaluation.set_index(['dataset', 'metric', 'model']).unstack().round(3)
    evaluation = evaluation.droplevel(0, 1).reset_index()
    evaluation['ETS'] = [4.769, 4.345, 0.011]
    evaluation.to_csv('data/evaluation.csv')
    print(evaluation.T)
