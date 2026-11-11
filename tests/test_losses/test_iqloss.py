# %% Test IQLoss for all types of architectures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import sys
import torch

from neuralforecast import NeuralForecast
from neuralforecast.common.enums import TimeSeriesDatasetEnum
from neuralforecast.models import MLP, NBEATSx, NHITS, TSMixerx, LSTM, BiTCN
from neuralforecast.losses.pytorch import HuberIQLoss, IQLoss
from neuralforecast.utils import AirPassengersPanel
from tests.helpers.data import air_passengers

@pytest.mark.skipif(
    sys.platform == "darwin",
    reason=(
        "RuntimeError: MPS backend out of memory (MPS allocated: 8.00 MB, other allocations: 16.00 KB). Tried to allocate 256 bytes on shared pool."
        "Test failed in MacOS as shown by https://github.com/Nixtla/neuralforecast/actions/runs/17818711510/job/50656840114?pr=1368"
    )
)
def test_iqloss(show_plot=False):
    Y_train_df, Y_test_df, _, AirPassengersStatic = air_passengers(h=12)

    max_steps = 10

    fcst = NeuralForecast(
        models=[
            NBEATSx(
                h=12,
                input_size=24,
                loss=IQLoss(),
                valid_loss=IQLoss(),
                max_steps=max_steps,
                scaler_type="standard",
                futr_exog_list=["y_[lag12]"],
                hist_exog_list=None,
                stat_exog_list=["airline1"],
                early_stop_patience_steps=3,
            ),
            NHITS(
                h=12,
                input_size=24,
                loss=IQLoss(),
                valid_loss=IQLoss(),
                max_steps=max_steps,
                scaler_type="standard",
                futr_exog_list=["y_[lag12]"],
                hist_exog_list=None,
                stat_exog_list=["airline1"],
                early_stop_patience_steps=3,
            ),
            TSMixerx(
                h=12,
                input_size=24,
                n_series=2,
                loss=IQLoss(),
                valid_loss=IQLoss(),
                max_steps=max_steps,
                scaler_type="identity",
                futr_exog_list=["y_[lag12]"],
                hist_exog_list=None,
                stat_exog_list=["airline1"],
                early_stop_patience_steps=3,
            ),
            LSTM(
                h=12,
                input_size=24,
                loss=IQLoss(),
                valid_loss=IQLoss(),
                max_steps=max_steps,
                scaler_type="standard",
                futr_exog_list=["y_[lag12]"],
                hist_exog_list=None,
                stat_exog_list=["airline1"],
                early_stop_patience_steps=3,
            ),
            BiTCN(
                h=12,
                input_size=24,
                loss=IQLoss(),
                valid_loss=IQLoss(),
                max_steps=max_steps,
                scaler_type="standard",
                futr_exog_list=["y_[lag12]"],
                hist_exog_list=None,
                stat_exog_list=["airline1"],
                early_stop_patience_steps=3,
            ),
        ],
        freq="M",
    )
    fcst.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
    # %% Test IQLoss prediction with multiple quantiles for different architectures
    # Test IQLoss
    forecasts_q10 = fcst.predict(futr_df=Y_test_df, quantile=0.1)
    forecasts_q50 = fcst.predict(futr_df=Y_test_df, quantile=0.5)
    forecasts_q90 = fcst.predict(futr_df=Y_test_df, quantile=0.9)

    if show_plot:
        model = "NHITS"
        # %% Plot quantile predictions
        forecasts = forecasts_q50.reset_index()
        forecasts = forecasts.merge(forecasts_q10.reset_index())
        forecasts = forecasts.merge(forecasts_q90.reset_index())
        Y_hat_df = forecasts.reset_index(drop=True).drop(
            columns=[TimeSeriesDatasetEnum.UniqueId, TimeSeriesDatasetEnum.Datetime]
        )
        plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
        plot_df = pd.concat([Y_train_df, plot_df])
        plot_df = plot_df[plot_df.unique_id == "Airline1"].drop(TimeSeriesDatasetEnum.UniqueId, axis=1)
        plt.plot(
            plot_df[TimeSeriesDatasetEnum.Datetime],
            plot_df["y"],
            c="black",
            label="True",
        )
        plt.plot(
            plot_df[TimeSeriesDatasetEnum.Datetime],
            plot_df[f"{model}_ql0.5"],
            c="blue",
            label="median",
        )
        plt.fill_between(
            x=plot_df[TimeSeriesDatasetEnum.Datetime][-12:],
            y1=plot_df[f"{model}_ql0.1"][-12:].values,
            y2=plot_df[f"{model}_ql0.9"][-12:].values,
            alpha=0.4,
            label="level 90",
        )
        plt.legend()
        plt.grid()
        plt.show(block=True)


# predict() fixes the quantile; a later fit must sample quantiles again
@pytest.mark.parametrize("loss_cls", [IQLoss, HuberIQLoss])
def test_iqloss_samples_quantiles_when_refit_after_predict(loss_cls):
    Y_train_df, *_ = air_passengers(h=12)
    model = MLP(h=12, input_size=24, loss=loss_cls(), max_steps=2)
    nf = NeuralForecast(models=[model], freq="ME")

    nf.fit(Y_train_df)
    nf.predict(quantiles=[0.9])
    nf.fit(Y_train_df, use_init_models=False)

    loss = nf.models[0].loss
    assert isinstance(loss.q, torch.Tensor) and loss.training


def _calibrated_nf(loss, crossing=False):
    df = AirPassengersPanel[["unique_id", "ds", "y"]]
    model = MLP(
        h=12,
        input_size=24,
        loss=loss(),
        max_steps=2,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        logger=False,
    )
    nf = NeuralForecast(models=[model], freq="ME")
    nf.fit(df)
    n_rows = df.unique_id.nunique() * 12
    decreasing = (np.arange(n_rows) % 2 == 1) if crossing else np.zeros(n_rows, bool)

    # The q-quantile forecast is 100 * q, or 100 * (1 - q) on crossing rows
    def predict(dataset, quantiles, **kwargs):
        q = quantiles[0]
        return np.where(decreasing, 100 * (1 - q), 100 * q)[:, None]

    nf.models[0].predict = predict
    return nf


@pytest.mark.parametrize("loss", [IQLoss, HuberIQLoss])
@pytest.mark.parametrize(
    "quantiles", [[0.1, 0.9], [0.9, 0.1, 0.5], [0.005, 0.999]]
)
def test_iqloss_predict_returns_the_requested_quantiles(loss, quantiles):
    fcsts = _calibrated_nf(loss).predict(quantiles=quantiles)

    for q in quantiles:
        assert np.allclose(fcsts[f"MLP_ql{q}"], 100 * q)


@pytest.mark.parametrize("loss", [IQLoss, HuberIQLoss])
def test_iqloss_predict_uncrosses_quantiles(loss):
    quantiles = [0.9, 0.1, 0.5]
    fcsts = _calibrated_nf(loss, crossing=True).predict(quantiles=quantiles)

    values = fcsts[[f"MLP_ql{q}" for q in sorted(quantiles)]].to_numpy()
    assert (np.diff(values, axis=1) >= 0).all()
