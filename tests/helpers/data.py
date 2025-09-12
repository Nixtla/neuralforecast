from neuralforecast.common.enums import TimeSeriesDatasetEnum
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
from neuralforecast.utils import augment_calendar_df

def air_passengers(h=12, augment_calendar=False, simple_augment=True):
    panel = AirPassengersPanel.copy()
    calendar_cols = None
    if augment_calendar:
        if simple_augment:
            calendar_col = 'quarter'
            panel[calendar_col] = panel[TimeSeriesDatasetEnum.Datetime].dt.quarter
            calendar_cols = [calendar_col]
        else:
            panel, calendar_cols = augment_calendar_df(df=panel, freq='M')

    Y_train_df = panel[panel.ds<panel[TimeSeriesDatasetEnum.Datetime].values[-h]] # 132 train
    Y_test_df = panel[panel.ds>=panel[TimeSeriesDatasetEnum.Datetime].values[-h]].reset_index(drop=True) # 12 test    
    return Y_train_df, Y_test_df, calendar_cols, AirPassengersStatic
