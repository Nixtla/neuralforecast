from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
from neuralforecast.utils import augment_calendar_df


def air_passengers(h=12, augment_calendar=False):

    AirPassengersStatic
    panel = AirPassengersPanel.copy()
    calendar_cols = None
    if augment_calendar:
        panel, calendar_cols = augment_calendar_df(df=panel, freq='M')

    Y_train_df = panel[panel.ds<panel['ds'].values[-h]] # 132 train
    Y_test_df = panel[panel.ds>=panel['ds'].values[-h]].reset_index(drop=True) # 12 test    
    return Y_train_df, Y_test_df, calendar_cols, AirPassengersStatic
