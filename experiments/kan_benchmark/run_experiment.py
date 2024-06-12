import os
import time
import argparse

import pandas as pd

from datasetsforecast.m3 import M3
from datasetsforecast.m4 import M4

from utilsforecast.losses import mae, smape
from utilsforecast.evaluation import evaluate

from neuralforecast import NeuralForecast
from neuralforecast.models import KAN, MLP, NBEATS, NHITS
results = []

def get_dataset(name):
    if name == 'M3-yearly':
        Y_df, *_ = M3.load("./data", "Yearly")
        horizon = 6
        freq = 'Y'
    elif name == 'M3-quarterly':
        Y_df, *_ = M3.load("./data", "Quarterly")
        horizon = 8
        freq = 'Q'
    elif name == 'M3-monthly':
        Y_df, *_ = M3.load("./data", "Monthly")
        horizon = 18
        freq = 'M'
    elif name == 'M4-yearly':
        Y_df, *_ = M4.load("./data", "Yearly")
        Y_df['ds'] = Y_df['ds'].astype(int)
        horizon = 6
        freq = 1
    elif name == 'M4-quarterly':
        Y_df, *_ = M4.load("./data", "Quarterly")
        Y_df['ds'] = Y_df['ds'].astype(int)
        horizon = 8
        freq = 1
    elif name == 'M4-monthly':
        Y_df, *_ = M4.load("./data", "Monthly")
        Y_df['ds'] = Y_df['ds'].astype(int)
        horizon = 18
        freq = 1
    elif name == 'M4-weekly':
        Y_df, *_ = M4.load("./data", "Weekly")
        Y_df['ds'] = Y_df['ds'].astype(int)
        horizon = 13
        freq = 1
    elif name == 'M4-daily':
        Y_df, *_ = M4.load("./data", "Daily")
        Y_df['ds'] = Y_df['ds'].astype(int)
        horizon = 14
        freq = 1
    elif name == 'M4-hourly':
        Y_df, *_ = M4.load("./data", "Hourly")
        Y_df['ds'] = Y_df['ds'].astype(int)
        horizon = 48
        freq = 1

    return Y_df, horizon, freq

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--dataset", type=str)

    args = parser.parse_args()
    dataset = args.dataset

    Y_df, horizon, freq = get_dataset(dataset)

    test_df = Y_df.groupby('unique_id').tail(horizon)
    train_df = Y_df.drop(test_df.index).reset_index(drop=True)

    kan_model = KAN(input_size=2*horizon, h=horizon, scaler_type='robust', early_stop_patience_steps=3)
    mlp_model = MLP(input_size=2*horizon, h=horizon, scaler_type='robust', max_steps=1000, early_stop_patience_steps=3)
    nbeats_model = NBEATS(input_size=2*horizon, h=horizon, scaler_type='robust', max_steps=1000, early_stop_patience_steps=3)
    nhits_model = NHITS(input_size=2*horizon, h=horizon, scaler_type='robust', max_steps=1000, early_stop_patience_steps=3)

    MODELS = [kan_model, mlp_model, nbeats_model, nhits_model]
    MODEL_NAMES = ['KAN', 'MLP', 'NBEATS', 'NHITS']

    for i, model in enumerate(MODELS):
        nf = NeuralForecast(models=[model], freq=freq)
        
        start = time.time()
        
        nf.fit(train_df, val_size=horizon)
        preds = nf.predict()

        end = time.time()
        elapsed_time = round(end - start,0)

        preds = preds.reset_index()
        test_df = pd.merge(test_df, preds, 'left', ['ds', 'unique_id'])

        evaluation = evaluate(
            test_df,
            metrics=[mae, smape],
            models=[f"{MODEL_NAMES[i]}"],
            target_col="y",
        )

        evaluation = evaluation.drop(['unique_id'], axis=1).groupby('metric').mean().reset_index()

        model_mae = evaluation[f"{MODEL_NAMES[i]}"][0]
        model_smape = evaluation[f"{MODEL_NAMES[i]}"][1]

        results.append([dataset, MODEL_NAMES[i], round(model_mae, 0), round(model_smape*100,2), elapsed_time])

    results_df = pd.DataFrame(data=results, columns=['dataset', 'model', 'mae', 'smape', 'time'])
    os.makedirs('./results', exist_ok=True)
    results_df.to_csv(f'./results/{dataset}_results_KANtuned.csv', header=True, index=False)




