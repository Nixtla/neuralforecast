import numpy as np
import pandas as pd
import pytest

from neuralforecast.auto import MLP, AutoMLP
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengersDF as Y_df

from .test_helpers import check_args


def test_mlp_model(suppress_warnings):
    check_model(MLP, ["airpassengers"])

@pytest.fixture
def setup_module():
    Y_train_df = Y_df[Y_df.ds<='1959-12-31'] # 132 train
    Y_test_df = Y_df[Y_df.ds>'1959-12-31']   # 12 test

    dataset, *_ = TimeSeriesDataset.from_df(Y_train_df)
    return dataset, Y_train_df, Y_test_df

def test_mlp_model2(setup_module):
    dataset, Y_train_df, Y_test_df = setup_module
    model = MLP(h=12, input_size=24, max_steps=1)
    model.fit(dataset=dataset)
    y_hat = model.predict(dataset=dataset)
    Y_test_df['MLP'] = y_hat

    #test we recover the same forecast
    y_hat2 = model.predict(dataset=dataset)
    np.testing.assert_array_equal(y_hat, y_hat2)

# test no leakage with test_size
def test_no_leakage_with_test_size(setup_module):
    dataset, Y_train_df, Y_test_df = setup_module
    model = MLP(h=12, input_size=24, max_steps=1)
    model.fit(dataset=dataset, test_size=12)
    y_hat = model.predict(dataset=dataset)

    y_hat_test = model.predict(dataset=dataset, step_size=1)
    np.testing.assert_almost_equal(
        y_hat,
        y_hat_test,
        decimal=4
    )
    # test we recover the same forecast
    y_hat_test2 = model.predict(dataset=dataset, step_size=1)
    np.testing.assert_array_equal(y_hat_test, y_hat_test2)

# test validation step
def test_validation_step(setup_module):
    dataset, Y_train_df, Y_test_df = setup_module
    model = MLP(h=12, input_size=24, step_size=1,
                hidden_size=1024, num_layers=2,
                max_steps=1)
    model.fit(dataset=dataset, val_size=12)
    y_hat_w_val = model.predict(dataset=dataset)
    Y_test_df['MLP'] = y_hat_w_val

    pd.concat([Y_train_df, Y_test_df]).drop('unique_id', axis=1).set_index('ds').plot()
    # test no leakage with test_size and val_size
    dataset, *_ = TimeSeriesDataset.from_df(Y_df)
    model = MLP(h=12, input_size=24, step_size=1,
                hidden_size=1024, num_layers=2,
                max_steps=1)
    model.fit(dataset=dataset, val_size=12, test_size=12)
    y_hat_test_w_val = model.predict(dataset=dataset, step_size=1)
    np.testing.assert_almost_equal(y_hat_test_w_val,
                                y_hat_w_val, decimal=4)


def test_automlp():
    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoMLP, exclude_args=['cls_model'])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoMLP.get_default_config(h=12, backend='optuna')
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update({'max_steps': 2, 'val_check_steps': 1, 'input_size': 12, 'hidden_size': 8})
        return config

    model = AutoMLP(h=12, config=my_config_new, backend='optuna', num_samples=1, cpus=1)
    assert model.config(MockTrial())['h'] == 12
