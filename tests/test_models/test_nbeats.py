import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from fastcore.test import test_eq as _test_eq

from neuralforecast.common._model_checks import check_model
from neuralforecast.models import NBEATS
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengersDF as Y_df


def test_nbeats_model():
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_model(NBEATS, ["airpassengers"])

@pytest.fixture
def setup_data():
    Y_train_df = Y_df[Y_df.ds<Y_df['ds'].values[-12]] # 132 train
    Y_test_df = Y_df[Y_df.ds>=Y_df['ds'].values[-12]]   # 12 test

    dataset, *_ = TimeSeriesDataset.from_df(df = Y_train_df)
    return dataset, Y_train_df, Y_test_df


def test_nbeats_model2(setup_data):
    dataset, Y_train_df, Y_test_df = setup_data
    nbeats = NBEATS(h=12, input_size=24, windows_batch_size=None,
                    stack_types=['identity', 'trend', 'seasonality'], max_steps=1)
    nbeats.fit(dataset=dataset)
    y_hat = nbeats.predict(dataset=dataset)
    Y_test_df['N-BEATS'] = y_hat

    pd.concat([Y_train_df, Y_test_df]).drop('unique_id', axis=1).set_index('ds').plot()
    #test we recover the same forecast
    y_hat2 = nbeats.predict(dataset=dataset)
    _test_eq(y_hat, y_hat2)


#test no leakage with test_size
def test_nbeats_model_no_leakage(setup_data):
    dataset, *_ = setup_data
    model = NBEATS(input_size=24, h=12,
                   windows_batch_size=None, max_steps=1)
    model.fit(dataset=dataset, test_size=12)

    nbeats = NBEATS(h=12, input_size=24, windows_batch_size=None,
                    stack_types=['identity', 'trend', 'seasonality'], max_steps=1)
    y_hat = nbeats.predict(dataset=dataset)

    y_hat_test = model.predict(dataset=dataset, step_size=1)
    np.testing.assert_almost_equal(y_hat, y_hat_test, decimal=4)
    #test we recover the same forecast
    y_hat_test2 = model.predict(dataset=dataset, step_size=1)
    _test_eq(y_hat_test, y_hat_test2)

# test validation step
def test_nbeats_model_with_validation(setup_data):
    dataset, Y_train_df, Y_test_df = setup_data
    model = NBEATS(input_size=24, h=12, windows_batch_size=None, max_steps=1)
    model.fit(dataset=dataset, val_size=12)
    y_hat_w_val = model.predict(dataset=dataset)
    Y_test_df['N-BEATS'] = y_hat_w_val

    # pd.concat([Y_train_df, Y_test_df]).drop('unique_id', axis=1).set_index('ds').plot()

    # test no leakage with test_size and val_size
    dataset, *_ = TimeSeriesDataset.from_df(Y_train_df)
    model = NBEATS(input_size=24, h=12, windows_batch_size=None, max_steps=1)
    model.fit(dataset=dataset, val_size=12)
    y_hat_w_val = model.predict(dataset=dataset)

    dataset, *_ = TimeSeriesDataset.from_df(Y_df)
    model = NBEATS(input_size=24, h=12, windows_batch_size=None, max_steps=1)
    model.fit(dataset=dataset, val_size=12, test_size=12)

    y_hat_test_w_val = model.predict(dataset=dataset, step_size=1)

    np.testing.assert_almost_equal(y_hat_test_w_val, y_hat_w_val, decimal=4)
