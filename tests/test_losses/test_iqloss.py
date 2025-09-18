# %% Test IQLoss for all types of architectures
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import sys

from neuralforecast import NeuralForecast
from neuralforecast.common.enums import TimeSeriesDatasetEnum
from neuralforecast.models import NBEATSx, NHITS, TSMixerx, LSTM, BiTCN
from neuralforecast.losses.pytorch import IQLoss
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
