import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from neuralforecast.auto import NBEATS, AutoNBEATS
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengersDF as Y_df

from .test_helpers import check_args


def test_nbeats_model(suppress_warnings):
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

    # pd.concat([Y_train_df, Y_test_df]).drop('unique_id', axis=1).set_index('ds').plot()
    #test we recover the same forecast
    y_hat2 = nbeats.predict(dataset=dataset)
    assert (y_hat == y_hat2).all()


#test no leakage with test_size
def test_nbeats_model_no_leakage(setup_data):
    dataset, *_ = setup_data
    model = NBEATS(input_size=24, h=12,
                   windows_batch_size=None, max_steps=1)
    model.fit(dataset=dataset, test_size=12)
    y_hat_test = model.predict(dataset=dataset, step_size=1)

    #test we recover the same forecast
    y_hat_test2 = model.predict(dataset=dataset, step_size=1)
    assert (y_hat_test == y_hat_test2).all()

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

def test_autonbeats(setup_dataset):
    dataset = setup_dataset

    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoNBEATS, exclude_args=['cls_model'])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoNBEATS.get_default_config(h=12, backend='optuna')
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update({'max_steps': 2, 'val_check_steps': 1, 'input_size': 12, 'mlp_units': 3 * [[8, 8]]})
        return config

    model = AutoNBEATS(h=12, config=my_config_new, backend='optuna', num_samples=1, cpus=1)
    assert model.config(MockTrial())['h'] == 12
    model.fit(dataset=dataset)

    # Unit test for situation: Ray with updated default config
    my_config = AutoNBEATS.get_default_config(h=12, backend='ray')
    my_config['max_steps'] = 2
    my_config['val_check_steps'] = 1
    my_config['input_size'] = 12
    my_config['mlp_units'] = 3 * [[8, 8]]
    model = AutoNBEATS(h=12, config=my_config, backend='ray', num_samples=1, cpus=1)
    model.fit(dataset=dataset)