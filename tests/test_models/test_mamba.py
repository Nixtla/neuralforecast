import numpy as np
import pandas as pd
import pytest
import torch

from neuralforecast.auto import AutoMamba, Mamba
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model
from neuralforecast.models.mamba import MambaBlock
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengersDF as Y_df

from .test_helpers import check_args


def test_mamba_model(suppress_warnings):
    check_model(Mamba, ["airpassengers"])

@pytest.fixture
def setup_module():
    Y_train_df = Y_df[Y_df.ds<='1959-12-31'] # 132 train
    Y_test_df = Y_df[Y_df.ds>'1959-12-31']   # 12 test

    dataset, *_ = TimeSeriesDataset.from_df(Y_train_df)
    return dataset, Y_train_df, Y_test_df

def test_mamba_model2(setup_module):
    dataset, Y_train_df, Y_test_df = setup_module
    model = Mamba(h=12, input_size=24, max_steps=1)
    model.fit(dataset=dataset)
    y_hat = model.predict(dataset=dataset)
    Y_test_df['Mamba'] = y_hat

    #test we recover the same forecast
    y_hat2 = model.predict(dataset=dataset)
    np.testing.assert_array_equal(y_hat, y_hat2)

# test no leakage with test_size
def test_no_leakage_with_test_size(setup_module):
    dataset, Y_train_df, Y_test_df = setup_module
    model = Mamba(h=12, input_size=24, max_steps=1)
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
    model = Mamba(h=12, input_size=24, step_size=1,
                  hidden_size=32, e_layers=1,
                  max_steps=1)
    model.fit(dataset=dataset, val_size=12)
    y_hat_w_val = model.predict(dataset=dataset)
    Y_test_df['Mamba'] = y_hat_w_val

    pd.concat([Y_train_df, Y_test_df]).drop('unique_id', axis=1).set_index('ds').plot()
    # test no leakage with test_size and val_size
    dataset, *_ = TimeSeriesDataset.from_df(Y_df)
    model = Mamba(h=12, input_size=24, step_size=1,
                  hidden_size=32, e_layers=1,
                  max_steps=1)
    model.fit(dataset=dataset, val_size=12, test_size=12)
    y_hat_test_w_val = model.predict(dataset=dataset, step_size=1)
    np.testing.assert_almost_equal(y_hat_test_w_val,
                                   y_hat_w_val, decimal=4)


@pytest.mark.parametrize("l", [11, 16, 24, 33])
def test_selective_scan_matches_sequential_reference(l):
    # The chunked parallel scan must match a textbook sequential reference
    # for arbitrary sequence lengths (including non-multiples of the chunk
    # size, which exercise the padding path).
    torch.manual_seed(0)
    b, d_inner, d_state = 2, 8, 4
    u = torch.randn(b, l, d_inner)
    delta = torch.nn.functional.softplus(torch.randn(b, l, d_inner))
    A = -torch.rand(d_inner, d_state).abs() - 0.1
    B = torch.randn(b, l, d_state)
    C = torch.randn(b, l, d_state)
    D = torch.randn(d_inner)

    block = MambaBlock(
        hidden_size=d_inner, d_inner=d_inner, d_state=d_state, d_conv=2, dt_rank=2
    )
    chunked = block._selective_scan(u, delta, A, B, C, D)

    deltaA = torch.exp(delta.unsqueeze(-1) * A)
    deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(-2) * u.unsqueeze(-1)
    state = torch.zeros(b, d_inner, d_state)
    ys_ref = torch.empty(b, l, d_inner)
    for i in range(l):
        state = deltaA[:, i] * state + deltaB_u[:, i]
        ys_ref[:, i] = (state * C[:, i].unsqueeze(1)).sum(dim=-1)
    sequential = ys_ref + u * D

    torch.testing.assert_close(chunked, sequential, rtol=1e-4, atol=1e-4)


def test_distribution_loss_training_no_nan():
    # Regression test for the NaN-during-training bug. Mirrors the user-reported
    # failure: panel data with all three exogenous types and DistributionLoss.
    # The selective scan must stay numerically stable as parameters drift, so
    # loc/scale parameters remain finite across the full training run.
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import DistributionLoss
    from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

    Y_train_df = AirPassengersPanel[
        AirPassengersPanel.ds < AirPassengersPanel['ds'].values[-12]
    ]
    Y_test_df = AirPassengersPanel[
        AirPassengersPanel.ds >= AirPassengersPanel['ds'].values[-12]
    ].reset_index(drop=True)

    model = Mamba(
        h=12,
        input_size=24,
        stat_exog_list=['airline1'],
        futr_exog_list=['trend'],
        hist_exog_list=['y_[lag12]'],
        loss=DistributionLoss(distribution='Normal', level=[80, 90]),
        scaler_type='robust',
        learning_rate=1e-3,
        max_steps=100,
        val_check_steps=10,
    )
    fcst = NeuralForecast(models=[model], freq='ME')
    fcst.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
    Y_hat_df = fcst.predict(futr_df=Y_test_df)
    preds = Y_hat_df.drop(columns=['unique_id', 'ds']).to_numpy().astype(float)
    assert np.isfinite(preds).all(), "predictions contain NaN/inf"


def test_automamba():
    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoMamba, exclude_args=['cls_model'])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoMamba.get_default_config(h=12, backend='optuna')
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update({'max_steps': 2, 'val_check_steps': 1, 'input_size': 12, 'hidden_size': 8})
        return config

    model = AutoMamba(h=12, config=my_config_new, backend='optuna', num_samples=1, cpus=1)
    assert model.config(MockTrial())['h'] == 12
