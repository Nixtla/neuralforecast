import pandas as pd

from datasetsforecast.m4 import M4
from datasetsforecast.m3 import M3
from datasetsforecast.long_horizon import LongHorizon


def get_dataset(name):

    # Read data and parameters
    if name == 'M3-yearly':
        Y_df, *_ = M3.load(directory='./', group='Yearly')
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        freq = 'Y'
        h = 6
        val_size = 6
        test_size = 6
    elif name == 'M3-quarterly':
        Y_df, *_ = M3.load(directory='./', group='Quarterly')
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        freq = 'Q'
        h = 8
        val_size = 8
        test_size = 8
    elif name == 'M3-monthly':
        Y_df, *_ = M3.load(directory='./', group='Monthly')
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        freq = 'M'
        h = 12
        val_size = 12
        test_size = 12
    elif name == 'M4-yearly':
        Y_df, *_ = M4.load(directory='./', group='Yearly')
        Y_df['ds'] = Y_df['ds'].astype(int)
        freq = 1
        h = 6
        val_size = 6
        test_size = 6
    elif name == 'M4-quarterly':
        Y_df, *_ = M4.load(directory='./', group='Quarterly')
        Y_df['ds'] = Y_df['ds'].astype(int)
        freq = 4
        h = 8
        val_size = 8
        test_size = 8
    elif name == 'M4-monthly':
        Y_df, *_ = M4.load(directory='./', group='Monthly')
        Y_df['ds'] = Y_df['ds'].astype(int)
        freq = 12
        h = 18
        val_size = 18
        test_size = 18
    elif name == 'M4-daily':
        Y_df, *_ = M4.load(directory='./', group='Daily')
        Y_df['ds'] = Y_df['ds'].astype(int)
        freq = 365
        h = 14
        val_size = 14
        test_size = 14
    elif name == 'M4-hourly':
        Y_df, *_ = M4.load(directory='./', group='Hourly')
        Y_df['ds'] = Y_df['ds'].astype(int)
        freq = 24
        h = 48
        val_size = 48
        test_size = 48
    elif name == 'Ettm2':
        Y_df, *_ = LongHorizon.load(directory='./', group='ETTm2')
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        freq = '15T'
        h = 720
        val_size = 11520
        test_size = 11520
    elif name == 'Ettm1':
        Y_df, *_ = LongHorizon.load(directory='./', group='ETTm1')
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        freq = '15T'
        h = 720
        val_size = 11520
        test_size = 11520
    elif name == 'Etth1':
        Y_df, *_ = LongHorizon.load(directory='./', group='ETTh1')
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        freq = 'H'
        h = 720
        val_size = 2880
        test_size = 2880
    elif name == 'Etth2':
        Y_df, *_ = LongHorizon.load(directory='./', group='ETTh2')
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        freq = 'H'
        h = 720
        val_size = 2880
        test_size = 2880
    elif name == 'Electricity':
        Y_df, *_ = LongHorizon.load(directory='./', group='ECL')
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        freq = 'H'
        h = 720
        val_size = 2632
        test_size = 5260
    elif name == 'Exchange':
        Y_df, *_ = LongHorizon.load(directory='./', group='Exchange')
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        freq = 'D'
        h = 720
        val_size = 760
        test_size = 1517
    elif name == 'Weather':
        Y_df, *_ = LongHorizon.load(directory='./', group='Weather')
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        freq = '10T'
        h = 720
        val_size = 5270
        test_size = 10539
    elif name == 'Traffic':
        Y_df, *_ = LongHorizon.load(directory='./', group='TrafficL')
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        freq = 'H'
        h = 720
        val_size = 1756
        test_size = 3508
    elif name == 'ILI':
        Y_df, *_ = LongHorizon.load(directory='./', group='ILI')
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        freq = 'W'
        h = 60
        val_size = 97
        test_size = 193
    else:
        raise Exception("Frequency not defined")

    return Y_df, h, freq, val_size, test_size