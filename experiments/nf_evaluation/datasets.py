import pandas as pd

from datasetsforecast.m4 import M4
from datasetsforecast.m3 import M3
from datasetsforecast.long_horizon import LongHorizon


def get_dataset(name):

    # Read data and parameters
    if name == 'M4-yearly':
        Y_df, *_ = M4.load(directory='./', group='Yearly')
        freq = 'Y'
        h = 6
        val_size = 6
        test_size = 6
    elif name == 'M4-quarterly':
        Y_df, *_ = M4.load(directory='./', group='Quarterly')
        freq = 'Q'
        h = 8
        val_size = 8
        test_size = 8
    elif name == 'M4-monthly':
        Y_df, *_ = M4.load(directory='./', group='Monthly')
        freq = 'M'
        h = 18
        val_size = 18
        test_size = 18
    elif name == 'M4-daily':
        Y_df, *_ = M4.load(directory='./', group='Daily')
        freq = 'D'
        h = 14
        val_size = 14
        test_size = 14
    elif name == 'ETTm2':
        Y_df, *_ = LongHorizon.load(directory='./', group='ETTm2')
        freq = '15T'
        h = 720
        val_size = 11520
        test_size = 11520
    elif name == 'Electricity':
        Y_df, *_ = LongHorizon.load(directory='./', group='ECL')
        freq = 'H'
        h = 720
        val_size = 2632
        test_size = 5260   
    elif name == 'Weather':
        Y_df, *_ = LongHorizon.load(directory='./', group='Weather')
        freq = '10T'
        h = 720
        val_size = 5270
        test_size = 10539
    elif name == 'Traffic':
        Y_df, *_ = LongHorizon.load(directory='./', group='Traffic')
        freq = 'H'
        h = 720
        val_size = 1756
        test_size = 3508
    elif name == 'ILI':
        Y_df, *_ = LongHorizon.load(directory='./', group='ILI')
        freq = 'W'
        h = 60
        val_size = 97
        test_size = 193
    else:
        raise Exception("Frequency not defined")

    return Y_df, X_df, h, freq, val_size, test_size