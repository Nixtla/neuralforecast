import logging
import os
import shutil
import sys
import tempfile
import warnings
from datetime import date
from pathlib import Path

import git
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pytest
import pytorch_lightning as pl
import s3fs
import torch

# from fastcore.test import test_eq, test_fail
from ray import tune

import neuralforecast
from neuralforecast.auto import (
    AutoDilatedRNN,
    AutoMLP,
    AutoNBEATS,
    AutoNBEATSx,
    AutoRNN,
    AutoTCN,
)
from neuralforecast.core import (
    LSTM,
    DLinear,
    FEDformer,
    NeuralForecast,
    PatchTST,
    PredictionIntervals,
    TimesNet,
    _insample_times,
)
from neuralforecast.losses.pytorch import (
    GMM,
    MAE,
    MSE,
    NBMM,
    PMM,
    DistributionLoss,
    HuberIQLoss,
    HuberMQLoss,
    HuberQLoss,
    IQLoss,
    MQLoss,
    QuantileLoss,
)
from neuralforecast.models.autoformer import Autoformer
from neuralforecast.models.deepar import DeepAR
from neuralforecast.models.dilated_rnn import DilatedRNN
from neuralforecast.models.informer import Informer
from neuralforecast.models.mlp import MLP
from neuralforecast.models.nbeats import NBEATS
from neuralforecast.models.nbeatsx import NBEATSx
from neuralforecast.models.nhits import NHITS
from neuralforecast.models.rnn import RNN
from neuralforecast.models.stemgnn import StemGNN
from neuralforecast.models.tcn import TCN
from neuralforecast.models.tft import TFT
from neuralforecast.models.tsmixer import TSMixer
from neuralforecast.models.tsmixerx import TSMixerx
from neuralforecast.models.vanillatransformer import VanillaTransformer
from neuralforecast.utils import (
    AirPassengersDF,
    AirPassengersPanel,
    AirPassengersStatic,
    generate_series,
)

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


@pytest.fixture
def setup():
    uids = pd.Series(["id_0", "id_1"])
    indptr = np.array([0, 4, 10], dtype=np.int32)
    return uids, indptr


def test_cutoff_deltas(setup):
    uids, indptr = setup
    h = 2
    for step_size, freq, days in zip([1, 2], ["D", "W-THU"], [1, 14]):
        times = np.hstack(
            [
                pd.date_range("2000-01-01", freq=freq, periods=4),
                pd.date_range("2000-10-10", freq=freq, periods=10),
            ]
        )
        times_df = _insample_times(times, uids, indptr, h, freq, step_size=step_size)
        pd.testing.assert_frame_equal(
            times_df.groupby("unique_id")["ds"].min().reset_index(),
            pd.DataFrame(
                {
                    "unique_id": uids,
                    "ds": times[indptr[:-1]],
                }
            ),
        )
        pd.testing.assert_frame_equal(
            times_df.groupby("unique_id")["ds"].max().reset_index(),
            pd.DataFrame(
                {
                    "unique_id": uids,
                    "ds": times[indptr[1:] - 1],
                }
            ),
        )
        cutoff_deltas = (
            times_df.drop_duplicates(["unique_id", "cutoff"])
            .groupby("unique_id")["cutoff"]
            .diff()
            .dropna()
        )
        assert cutoff_deltas.nunique() == 1
        assert cutoff_deltas.unique()[0] == pd.Timedelta(f"{days}D")


@pytest.fixture
def setup_airplane_data():
    AirPassengersPanel_train = AirPassengersPanel[
        AirPassengersPanel["ds"] < AirPassengersPanel["ds"].values[-12]
    ].reset_index(drop=True)
    AirPassengersPanel_test = AirPassengersPanel[
        AirPassengersPanel["ds"] >= AirPassengersPanel["ds"].values[-12]
    ].reset_index(drop=True)
    AirPassengersPanel_test["y"] = np.nan
    AirPassengersPanel_test["y_[lag12]"] = np.nan
    return AirPassengersPanel_train, AirPassengersPanel_test


# Unittest for early stopping without val_size protection
def test_neural_forecast_early_stopping(setup_airplane_data):
    AirPassengersPanel_train, _ = setup_airplane_data
    models = [NHITS(h=12, input_size=12, max_steps=1, early_stop_patience_steps=5)]
    nf = NeuralForecast(models=models, freq="M")
    with pytest.raises(Exception, match="Set val_size>0 if early stopping is enabled."):
        nf.fit(AirPassengersPanel_train)


# test fit+cross_validation behaviour
def test_neural_forecast_fit_cross_validation(setup_airplane_data):
    AirPassengersPanel_train, _ = setup_airplane_data
    models = [NHITS(h=12, input_size=24, max_steps=10)]
    nf = NeuralForecast(models=models, freq="M")
    nf.fit(AirPassengersPanel_train)
    init_fcst = nf.predict()
    init_cv = nf.cross_validation(AirPassengersPanel_train, use_init_models=True)
    after_cv = nf.cross_validation(AirPassengersPanel_train, use_init_models=True)
    nf.fit(AirPassengersPanel_train, use_init_models=True)
    after_fcst = nf.predict()
    assert (init_cv == after_cv).all().all()
    assert (init_fcst == after_fcst).all().all()

# test cross_validation with refit
def test_neural_forecast_refit(setup_airplane_data):
    AirPassengersPanel_train, _ = setup_airplane_data
    models = [
        NHITS(
            h=12,
            input_size=24,
            max_steps=2,
            futr_exog_list=["trend"],
            stat_exog_list=["airline1", "airline2"],
        )
    ]
    nf = NeuralForecast(models=models, freq="M")
    cv_kwargs = dict(
        df=AirPassengersPanel_train,
        static_df=AirPassengersStatic,
        n_windows=4,
        use_init_models=True,
    )
    cv_res_norefit = nf.cross_validation(refit=False, **cv_kwargs)
    cutoffs = cv_res_norefit["cutoff"].unique()
    for refit in [True, 2]:
        cv_res = nf.cross_validation(refit=refit, **cv_kwargs)
        refit = int(refit)
        fltr = lambda df: df["cutoff"].isin(cutoffs[:refit])
        expected = cv_res_norefit[fltr]
        actual = cv_res[fltr]
        # predictions for the no-refit windows should be the same
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True), expected.reset_index(drop=True)
        )
        # predictions after refit should be different
        with pytest.raises(AssertionError, match=r'\(column name="NHITS"\) are different'):
            pd.testing.assert_frame_equal(
                cv_res_norefit.drop(expected.index).reset_index(drop=True),
                cv_res.drop(actual.index).reset_index(drop=True),
            )


def test_neural_forecast_scaling(setup_airplane_data):
    """Test scaling functionality for NeuralForecast models."""
    AirPassengersPanel_train, AirPassengersPanel_test = setup_airplane_data

    # Get initial forecast for comparison
    init_models = [NHITS(h=12, input_size=24, max_steps=10)]
    nf_init = NeuralForecast(models=init_models, freq="M")
    nf_init.fit(AirPassengersPanel_train)
    init_fcst = nf_init.predict()

    models = [NHITS(h=12, input_size=24, max_steps=10)]
    models_exog = [
        NHITS(
            h=12,
            input_size=12,
            max_steps=10,
            hist_exog_list=["trend"],
            futr_exog_list=["trend"],
        )
    ]

    # Test fit+predict with standard scaling
    nf = NeuralForecast(models=models, freq="M", local_scaler_type="standard")
    nf.fit(AirPassengersPanel_train)
    scaled_fcst = nf.predict()
    # Check that the forecasts are similar to the one without scaling
    np.testing.assert_allclose(
        init_fcst["NHITS"].values,
        scaled_fcst["NHITS"].values,
        rtol=0.3,
    )

    # Test with exogenous variables
    nf = NeuralForecast(models=models_exog, freq="M", local_scaler_type="standard")
    nf.fit(AirPassengersPanel_train)
    scaled_exog_fcst = nf.predict(futr_df=AirPassengersPanel_test)
    # Check that the forecasts are similar to the one without exog
    np.testing.assert_allclose(
        scaled_fcst["NHITS"].values,
        scaled_exog_fcst["NHITS"].values,
        rtol=0.3,
    )

    # Test cross-validation with robust scaling
    nf = NeuralForecast(models=models, freq="M", local_scaler_type="robust")
    cv_res = nf.cross_validation(AirPassengersPanel)
    # Check that the forecasts are similar to the original values
    np.testing.assert_allclose(
        cv_res["NHITS"].values,
        cv_res["y"].values,
        rtol=0.3,
    )

    # Test cross-validation with exogenous variables and robust-iqr scaling
    nf = NeuralForecast(models=models_exog, freq="M", local_scaler_type="robust-iqr")
    cv_res_exog = nf.cross_validation(AirPassengersPanel)
    # Check that the forecasts are similar to the original values
    np.testing.assert_allclose(
        cv_res_exog["NHITS"].values,
        cv_res_exog["y"].values,
        rtol=0.2,
    )

    # Test fit+predict_insample with minmax scaling
    nf = NeuralForecast(models=models, freq="M", local_scaler_type="minmax")
    nf.fit(AirPassengersPanel_train)
    insample_res = (
        nf.predict_insample()
        .groupby("unique_id")
        .tail(-12)  # first values aren't reliable
        .merge(
            AirPassengersPanel_train[["unique_id", "ds", "y"]],
            on=["unique_id", "ds"],
            how="left",
            suffixes=("_actual", "_expected"),
        )
    )
    # Check that y is inverted correctly
    np.testing.assert_allclose(
        insample_res["y_actual"].values,
        insample_res["y_expected"].values,
        rtol=1e-5,
    )
    # Check that predictions are in the same scale
    np.testing.assert_allclose(
        insample_res["NHITS"].values,
        insample_res["y_expected"].values,
        rtol=0.7,
    )

    # Test with exogenous variables
    nf = NeuralForecast(models=models_exog, freq="M", local_scaler_type="minmax")
    nf.fit(AirPassengersPanel_train)
    insample_res_exog = (
        nf.predict_insample()
        .groupby("unique_id")
        .tail(-12)  # first values aren't reliable
        .merge(
            AirPassengersPanel_train[["unique_id", "ds", "y"]],
            on=["unique_id", "ds"],
            how="left",
            suffixes=("_actual", "_expected"),
        )
    )
    # Check that y is inverted correctly
    np.testing.assert_allclose(
        insample_res_exog["y_actual"].values,
        insample_res_exog["y_expected"].values,
        rtol=1e-5,
    )
    # Check that predictions are similar to those without exog
    np.testing.assert_allclose(
        insample_res["NHITS"].values,
        insample_res_exog["NHITS"].values,
        rtol=0.2,
    )


def test_neural_forecast_boxcox_scaling(setup_airplane_data):
    """Test BoxCox scaling functionality for NeuralForecast models."""
    AirPassengersPanel_train, _ = setup_airplane_data

    models = [NHITS(h=12, input_size=24, max_steps=10)]

    # Test BoxCox scaling
    nf = NeuralForecast(models=models, freq="M", local_scaler_type="boxcox")
    nf.fit(AirPassengersPanel_train)
    insample_res = (
        nf.predict_insample()
        .groupby("unique_id")
        .tail(-12)  # first values aren't reliable
        .merge(
            AirPassengersPanel_train[["unique_id", "ds", "y"]],
            on=["unique_id", "ds"],
            how="left",
            suffixes=("_actual", "_expected"),
        )
    )
    # Check that y is inverted correctly
    np.testing.assert_allclose(
        insample_res["y_actual"].values,
        insample_res["y_expected"].values,
        rtol=1e-5,
    )
    # Check that predictions are in the same scale
    np.testing.assert_allclose(
        insample_res["NHITS"].values,
        insample_res["y_expected"].values,
        rtol=0.7,
    )
# # test futr_df contents
# models = [
#     NHITS(
#         h=6,
#         input_size=24,
#         max_steps=10,
#         hist_exog_list=["trend"],
#         futr_exog_list=["trend"],
#     )
# ]
# nf = NeuralForecast(models=models, freq="M")
# nf.fit(AirPassengersPanel_train)
# # not enough rows in futr_df raises an error
# test_fail(
#     lambda: nf.predict(futr_df=AirPassengersPanel_test.head()),
#     contains="There are missing combinations",
# )
# # extra rows issues a warning
# with warnings.catch_warnings(record=True) as issued_warnings:
#     warnings.simplefilter("always", UserWarning)
#     nf.predict(futr_df=AirPassengersPanel_test)
# assert any("Dropped 12 unused rows" in str(w.message) for w in issued_warnings)
# # models require futr_df and not provided raises an error
# test_fail(
#     lambda: nf.predict(),
#     contains="Models require the following future exogenous features: {'trend'}",
# )
# # missing feature in futr_df raises an error
# test_fail(
#     lambda: nf.predict(futr_df=AirPassengersPanel_test.drop(columns="trend")),
#     contains="missing from `futr_df`: {'trend'}",
# )
# # null values in futr_df raises an error
# test_fail(
#     lambda: nf.predict(futr_df=AirPassengersPanel_test.assign(trend=np.nan)),
#     contains="Found null values in `futr_df`",
# )
# # Test inplace model fitting
# models = [MLP(h=12, input_size=12, max_steps=1, scaler_type="robust")]
# initial_weights = models[0].mlp[0].weight.detach().clone()
# fcst = NeuralForecast(models=models, freq="M")
# fcst.fit(
#     df=AirPassengersPanel_train, static_df=AirPassengersStatic, use_init_models=True
# )
# after_weights = fcst.models_init[0].mlp[0].weight.detach().clone()
# assert np.allclose(initial_weights, after_weights), "init models should not be modified"
# assert len(fcst.models[0].train_trajectories) > 0, (
#     "models stored trajectories should not be empty"
# )
# # Test predict_insample
# test_size = 12
# n_series = 2
# h = 12


# def get_expected_size(df, h, test_size, step_size):
#     expected_size = 0
#     uids = df["unique_id"].unique()
#     for uid in uids:
#         input_len = len(df[df["unique_id"] == uid])
#         expected_size += ((input_len - test_size - h) / step_size + 1) * h
#     return expected_size


# models = [
#     NHITS(
#         h=h,
#         input_size=24,
#         loss=MQLoss(level=[80]),
#         max_steps=1,
#         alias="NHITS",
#         scaler_type=None,
#     ),
#     RNN(h=h, input_size=-1, loss=MAE(), max_steps=1, alias="RNN", scaler_type=None),
# ]

# nf = NeuralForecast(models=models, freq="M")
# cv = nf.cross_validation(
#     df=AirPassengersPanel_train,
#     static_df=AirPassengersStatic,
#     val_size=0,
#     test_size=test_size,
#     n_windows=None,
# )

# forecasts = nf.predict_insample(step_size=1)

# expected_size = get_expected_size(AirPassengersPanel_train, h, test_size, step_size=1)
# assert len(forecasts) == expected_size, (
#     f"Shape mismatch in predict_insample: {len(forecasts)=}, {expected_size=}"
# )
# # Test predict_insample (different lengths)
# diff_len_df = generate_series(n_series=n_series, max_length=100)

# nf = NeuralForecast(models=models, freq="D")
# cv = nf.cross_validation(
#     df=diff_len_df, val_size=0, test_size=test_size, n_windows=None
# )

# forecasts = nf.predict_insample(step_size=1)
# expected_size = get_expected_size(diff_len_df, h, test_size, step_size=1)
# assert len(forecasts) == expected_size, (
#     f"Shape mismatch in predict_insample: {len(forecasts)=}, {expected_size=}"
# )
# # Test aliases
# config_drnn = {
#     "input_size": tune.choice([-1]),
#     "encoder_hidden_size": tune.choice([5, 10]),
#     "max_steps": 1,
#     "val_check_steps": 1,
#     "step_size": 1,
# }
# models = [
#     # test Auto
#     AutoDilatedRNN(h=12, config=config_drnn, cpus=1, num_samples=2, alias="AutoDIL"),
#     # test BaseWindows
#     NHITS(h=12, input_size=24, loss=MQLoss(level=[80]), max_steps=1, alias="NHITSMQ"),
#     # test BaseRecurrent
#     RNN(
#         h=12,
#         input_size=-1,
#         encoder_hidden_size=10,
#         max_steps=1,
#         stat_exog_list=["airline1"],
#         futr_exog_list=["trend"],
#         hist_exog_list=["y_[lag12]"],
#         alias="MyRNN",
#     ),
#     # test BaseMultivariate
#     StemGNN(
#         h=12,
#         input_size=24,
#         n_series=2,
#         max_steps=1,
#         scaler_type="robust",
#         alias="StemMulti",
#     ),
#     # test model without alias
#     NHITS(h=12, input_size=24, max_steps=1),
# ]
# nf = NeuralForecast(models=models, freq="M")
# nf.fit(df=AirPassengersPanel_train, static_df=AirPassengersStatic)
# forecasts = nf.predict(futr_df=AirPassengersPanel_test)
# test_eq(
#     forecasts.columns.to_list(),
#     [
#         "unique_id",
#         "ds",
#         "AutoDIL",
#         "NHITSMQ-median",
#         "NHITSMQ-lo-80",
#         "NHITSMQ-hi-80",
#         "MyRNN",
#         "StemMulti",
#         "NHITS",
#     ],
# )
# # Unit test for core/model interactions
# config = {
#     "input_size": tune.choice([12, 24]),
#     "hidden_size": 256,
#     "max_steps": 1,
#     "val_check_steps": 1,
#     "step_size": 12,
# }

# config_drnn = {
#     "input_size": tune.choice([-1]),
#     "encoder_hidden_size": tune.choice([5, 10]),
#     "max_steps": 1,
#     "val_check_steps": 1,
#     "step_size": 1,
# }

# fcst = NeuralForecast(
#     models=[
#         AutoDilatedRNN(h=12, config=config_drnn, cpus=1, num_samples=2),
#         DeepAR(
#             h=12,
#             input_size=24,
#             max_steps=1,
#             stat_exog_list=["airline1"],
#             futr_exog_list=["trend"],
#         ),
#         DilatedRNN(
#             h=12,
#             input_size=-1,
#             encoder_hidden_size=10,
#             max_steps=1,
#             stat_exog_list=["airline1"],
#             futr_exog_list=["trend"],
#             hist_exog_list=["y_[lag12]"],
#         ),
#         RNN(
#             h=12,
#             input_size=-1,
#             encoder_hidden_size=10,
#             max_steps=1,
#             inference_input_size=24,
#             stat_exog_list=["airline1"],
#             futr_exog_list=["trend"],
#             hist_exog_list=["y_[lag12]"],
#         ),
#         TCN(
#             h=12,
#             input_size=-1,
#             encoder_hidden_size=10,
#             max_steps=1,
#             stat_exog_list=["airline1"],
#             futr_exog_list=["trend"],
#             hist_exog_list=["y_[lag12]"],
#         ),
#         AutoMLP(h=12, config=config, cpus=1, num_samples=2),
#         NBEATSx(
#             h=12,
#             input_size=12,
#             max_steps=1,
#             stat_exog_list=["airline1"],
#             futr_exog_list=["trend"],
#             hist_exog_list=["y_[lag12]"],
#         ),
#         NHITS(h=12, input_size=24, loss=MQLoss(level=[80]), max_steps=1),
#         NHITS(
#             h=12,
#             input_size=12,
#             max_steps=1,
#             stat_exog_list=["airline1"],
#             futr_exog_list=["trend"],
#             hist_exog_list=["y_[lag12]"],
#         ),
#         DLinear(h=12, input_size=24, max_steps=1),
#         MLP(
#             h=12,
#             input_size=12,
#             max_steps=1,
#             stat_exog_list=["airline1"],
#             futr_exog_list=["trend"],
#             hist_exog_list=["y_[lag12]"],
#         ),
#         TFT(h=12, input_size=24, max_steps=1),
#         VanillaTransformer(h=12, input_size=24, max_steps=1),
#         Informer(h=12, input_size=24, max_steps=1),
#         Autoformer(h=12, input_size=24, max_steps=1),
#         FEDformer(h=12, input_size=24, max_steps=1),
#         PatchTST(h=12, input_size=24, max_steps=1),
#         TimesNet(h=12, input_size=24, max_steps=1),
#         StemGNN(h=12, input_size=24, n_series=2, max_steps=1, scaler_type="robust"),
#         TSMixer(h=12, input_size=24, n_series=2, max_steps=1, scaler_type="robust"),
#         TSMixerx(h=12, input_size=24, n_series=2, max_steps=1, scaler_type="robust"),
#     ],
#     freq="M",
# )
# fcst.fit(df=AirPassengersPanel_train, static_df=AirPassengersStatic)
# forecasts = fcst.predict(futr_df=AirPassengersPanel_test)
# forecasts
# fig, ax = plt.subplots(1, 1, figsize=(20, 7))
# plot_df = pd.concat([AirPassengersPanel_train, forecasts.reset_index()]).set_index("ds")

# plot_df[plot_df["unique_id"] == "Airline1"].drop(
#     ["unique_id", "trend", "y_[lag12]"], axis=1
# ).plot(ax=ax, linewidth=2)

# ax.set_title("AirPassengers Forecast", fontsize=22)
# ax.set_ylabel("Monthly Passengers", fontsize=20)
# ax.set_xlabel("Timestamp [t]", fontsize=20)
# ax.legend(prop={"size": 15})
# ax.grid()
# fig, ax = plt.subplots(1, 1, figsize=(20, 7))
# plot_df = pd.concat([AirPassengersPanel_train, forecasts.reset_index()]).set_index("ds")

# plot_df[plot_df["unique_id"] == "Airline2"].drop(
#     ["unique_id", "trend", "y_[lag12]"], axis=1
# ).plot(ax=ax, linewidth=2)

# ax.set_title("AirPassengers Forecast", fontsize=22)
# ax.set_ylabel("Monthly Passengers", fontsize=20)
# ax.set_xlabel("Timestamp [t]", fontsize=20)
# ax.legend(prop={"size": 15})
# ax.grid()


# def config_optuna(trial):
#     return {
#         "input_size": trial.suggest_categorical("input_size", [12, 24]),
#         "hist_exog_list": trial.suggest_categorical(
#             "hist_exog_list", [["trend"], ["y_[lag12]"], ["trend", "y_[lag12]"]]
#         ),
#         "futr_exog_list": ["trend"],
#         "max_steps": 10,
#         "val_check_steps": 5,
#     }


# config_ray = {
#     "input_size": tune.choice([12, 24]),
#     "hist_exog_list": tune.choice([["trend"], ["y_[lag12]"], ["trend", "y_[lag12]"]]),
#     "futr_exog_list": ["trend"],
#     "max_steps": 10,
#     "val_check_steps": 5,
# }
# # test training with an iterative dataset produces the same results as directly passing in the dataset as a pandas dataframe
# AirPassengersPanel_train["id"] = AirPassengersPanel_train["unique_id"]
# AirPassengersPanel_test["id"] = AirPassengersPanel_test["unique_id"]

# models = [
#     NHITS(h=12, input_size=12, max_steps=10, futr_exog_list=["trend"], random_seed=1),
#     AutoMLP(
#         h=12,
#         config=config_optuna,
#         num_samples=2,
#         backend="optuna",
#         search_alg=optuna.samplers.TPESampler(seed=0),
#     ),  # type: ignore
#     AutoNBEATSx(h=12, config=config_ray, cpus=1, num_samples=2),
# ]
# nf = NeuralForecast(models=models, freq="M")

# # fit+predict with pandas dataframe
# nf.fit(
#     df=AirPassengersPanel_train.drop(columns="unique_id"),
#     use_init_models=True,
#     id_col="id",
# )
# pred_dataframe = nf.predict(
#     futr_df=AirPassengersPanel_test.drop(columns="unique_id")
# ).reset_index()

# # fit+predict with data directory
# with tempfile.TemporaryDirectory() as tmpdir:
#     AirPassengersPanel_train.to_parquet(
#         tmpdir, partition_cols=["unique_id"], index=False
#     )
#     data_directory = sorted([str(path) for path in Path(tmpdir).iterdir()])
#     nf.fit(df=data_directory, use_init_models=True, id_col="id")

# pred_df = AirPassengersPanel_train[
#     AirPassengersPanel_train["unique_id"] == "Airline2"
# ].drop(columns="unique_id")
# futr_df = AirPassengersPanel_test[
#     AirPassengersPanel_test["unique_id"] == "Airline2"
# ].drop(columns="unique_id")

# pred_iterative = nf.predict(df=pred_df, futr_df=futr_df)
# pred_airline2 = pred_dataframe[pred_dataframe["id"] == "Airline2"]
# np.testing.assert_allclose(
#     pred_iterative["NHITS"], pred_airline2["NHITS"], rtol=0, atol=1
# )
# np.testing.assert_allclose(
#     pred_iterative["AutoMLP"], pred_airline2["AutoMLP"], rtol=0, atol=1
# )
# np.testing.assert_allclose(
#     pred_iterative["AutoNBEATSx"], pred_airline2["AutoNBEATSx"], rtol=0, atol=1
# )

# # remove id columns to not impact future tests
# AirPassengersPanel_train = AirPassengersPanel_train.drop(columns="id")
# AirPassengersPanel_test = AirPassengersPanel_test.drop(columns="id")
# config = {
#     "input_size": tune.choice([12, 24]),
#     "hidden_size": 256,
#     "max_steps": 1,
#     "val_check_steps": 1,
#     "step_size": 12,
# }

# config_drnn = {
#     "input_size": tune.choice([-1]),
#     "encoder_hidden_size": tune.choice([5, 10]),
#     "max_steps": 1,
#     "val_check_steps": 1,
#     "step_size": 1,
# }

# fcst = NeuralForecast(
#     models=[
#         DilatedRNN(h=12, input_size=-1, encoder_hidden_size=10, max_steps=1),
#         AutoMLP(h=12, config=config, cpus=1, num_samples=1),
#         NHITS(h=12, input_size=12, max_steps=1),
#     ],
#     freq="M",
# )
# cv_df = fcst.cross_validation(
#     df=AirPassengersPanel, static_df=AirPassengersStatic, n_windows=3, step_size=1
# )


# # test cross validation no leakage
# def test_cross_validation(df, static_df, h, test_size):
#     if (test_size - h) % 1:
#         raise Exception("`test_size - h` should be module `step_size`")

#     n_windows = int((test_size - h) / 1) + 1
#     Y_test_df = df.groupby("unique_id").tail(test_size)
#     Y_train_df = df.drop(Y_test_df.index)
#     config = {
#         "input_size": tune.choice([12, 24]),
#         "step_size": 12,
#         "hidden_size": 256,
#         "max_steps": 1,
#         "val_check_steps": 1,
#     }
#     config_drnn = {
#         "input_size": tune.choice([-1]),
#         "encoder_hidden_size": tune.choice([5, 10]),
#         "max_steps": 1,
#         "val_check_steps": 1,
#     }
#     fcst = NeuralForecast(
#         models=[
#             AutoDilatedRNN(h=12, config=config_drnn, cpus=1, num_samples=1),
#             DilatedRNN(h=12, input_size=-1, encoder_hidden_size=5, max_steps=1),
#             RNN(
#                 h=12,
#                 input_size=-1,
#                 encoder_hidden_size=5,
#                 max_steps=1,
#                 stat_exog_list=["airline1"],
#                 futr_exog_list=["trend"],
#                 hist_exog_list=["y_[lag12]"],
#             ),
#             TCN(
#                 h=12,
#                 input_size=-1,
#                 encoder_hidden_size=5,
#                 max_steps=1,
#                 stat_exog_list=["airline1"],
#                 futr_exog_list=["trend"],
#                 hist_exog_list=["y_[lag12]"],
#             ),
#             AutoMLP(h=12, config=config, cpus=1, num_samples=1),
#             MLP(h=12, input_size=12, max_steps=1, scaler_type="robust"),
#             NBEATSx(
#                 h=12,
#                 input_size=12,
#                 max_steps=1,
#                 stat_exog_list=["airline1"],
#                 futr_exog_list=["trend"],
#                 hist_exog_list=["y_[lag12]"],
#             ),
#             NHITS(h=12, input_size=12, max_steps=1, scaler_type="robust"),
#             NHITS(h=12, input_size=12, loss=MQLoss(level=[80]), max_steps=1),
#             TFT(h=12, input_size=24, max_steps=1, scaler_type="robust"),
#             DLinear(h=12, input_size=24, max_steps=1),
#             VanillaTransformer(h=12, input_size=12, max_steps=1, scaler_type=None),
#             Informer(h=12, input_size=12, max_steps=1, scaler_type=None),
#             Autoformer(h=12, input_size=12, max_steps=1, scaler_type=None),
#             FEDformer(h=12, input_size=12, max_steps=1, scaler_type=None),
#             PatchTST(h=12, input_size=24, max_steps=1, scaler_type=None),
#             TimesNet(h=12, input_size=24, max_steps=1, scaler_type="standard"),
#             StemGNN(h=12, input_size=12, n_series=2, max_steps=1, scaler_type="robust"),
#             TSMixer(h=12, input_size=12, n_series=2, max_steps=1, scaler_type="robust"),
#             TSMixerx(
#                 h=12, input_size=12, n_series=2, max_steps=1, scaler_type="robust"
#             ),
#             DeepAR(
#                 h=12,
#                 input_size=24,
#                 max_steps=1,
#                 stat_exog_list=["airline1"],
#                 futr_exog_list=["trend"],
#             ),
#         ],
#         freq="M",
#     )
#     fcst.fit(df=Y_train_df, static_df=static_df)
#     Y_hat_df = fcst.predict(futr_df=Y_test_df)
#     Y_hat_df = Y_hat_df.merge(Y_test_df, how="left", on=["unique_id", "ds"])
#     last_dates = Y_train_df.groupby("unique_id").tail(1)
#     last_dates = last_dates[["unique_id", "ds"]].rename(columns={"ds": "cutoff"})
#     Y_hat_df = Y_hat_df.merge(last_dates, how="left", on="unique_id")

#     # cross validation
#     fcst = NeuralForecast(
#         models=[
#             AutoDilatedRNN(h=12, config=config_drnn, cpus=1, num_samples=1),
#             DilatedRNN(h=12, input_size=-1, encoder_hidden_size=5, max_steps=1),
#             RNN(
#                 h=12,
#                 input_size=-1,
#                 encoder_hidden_size=5,
#                 max_steps=1,
#                 stat_exog_list=["airline1"],
#                 futr_exog_list=["trend"],
#                 hist_exog_list=["y_[lag12]"],
#             ),
#             TCN(
#                 h=12,
#                 input_size=-1,
#                 encoder_hidden_size=5,
#                 max_steps=1,
#                 stat_exog_list=["airline1"],
#                 futr_exog_list=["trend"],
#                 hist_exog_list=["y_[lag12]"],
#             ),
#             AutoMLP(h=12, config=config, cpus=1, num_samples=1),
#             MLP(h=12, input_size=12, max_steps=1, scaler_type="robust"),
#             NBEATSx(
#                 h=12,
#                 input_size=12,
#                 max_steps=1,
#                 stat_exog_list=["airline1"],
#                 futr_exog_list=["trend"],
#                 hist_exog_list=["y_[lag12]"],
#             ),
#             NHITS(h=12, input_size=12, max_steps=1, scaler_type="robust"),
#             NHITS(h=12, input_size=12, loss=MQLoss(level=[80]), max_steps=1),
#             TFT(h=12, input_size=24, max_steps=1, scaler_type="robust"),
#             DLinear(h=12, input_size=24, max_steps=1),
#             VanillaTransformer(h=12, input_size=12, max_steps=1, scaler_type=None),
#             Informer(h=12, input_size=12, max_steps=1, scaler_type=None),
#             Autoformer(h=12, input_size=12, max_steps=1, scaler_type=None),
#             FEDformer(h=12, input_size=12, max_steps=1, scaler_type=None),
#             PatchTST(h=12, input_size=24, max_steps=1, scaler_type=None),
#             TimesNet(h=12, input_size=24, max_steps=1, scaler_type="standard"),
#             StemGNN(h=12, input_size=12, n_series=2, max_steps=1, scaler_type="robust"),
#             TSMixer(h=12, input_size=12, n_series=2, max_steps=1, scaler_type="robust"),
#             TSMixerx(
#                 h=12, input_size=12, n_series=2, max_steps=1, scaler_type="robust"
#             ),
#             DeepAR(
#                 h=12,
#                 input_size=24,
#                 max_steps=1,
#                 stat_exog_list=["airline1"],
#                 futr_exog_list=["trend"],
#             ),
#         ],
#         freq="M",
#     )
#     Y_hat_df_cv = fcst.cross_validation(
#         df, static_df=static_df, test_size=test_size, n_windows=None
#     )
#     for col in ["ds", "cutoff"]:
#         Y_hat_df_cv[col] = pd.to_datetime(Y_hat_df_cv[col].astype(str))
#         Y_hat_df[col] = pd.to_datetime(Y_hat_df[col].astype(str))
#     pd.testing.assert_frame_equal(
#         Y_hat_df[Y_hat_df_cv.columns],
#         Y_hat_df_cv,
#         check_dtype=False,
#         atol=1e-5,
#     )


# test_cross_validation(AirPassengersPanel, AirPassengersStatic, h=12, test_size=12)
# # test cv with series of different sizes
# series = pd.DataFrame(
#     {
#         "unique_id": np.repeat([0, 1], [10, 15]),
#         "ds": np.arange(25),
#         "y": np.random.rand(25),
#     }
# )
# nf = NeuralForecast(
#     freq=1, models=[MLP(input_size=5, h=5, max_steps=0, enable_progress_bar=False)]
# )
# cv_df = nf.cross_validation(df=series, n_windows=3, step_size=5)
# expected = pd.DataFrame(
#     {
#         "unique_id": np.repeat([0, 1], [5, 10]),
#         "ds": np.hstack([np.arange(5, 10), np.arange(15, 25)]),
#         "cutoff": np.repeat([4, 14, 19], 5),
#     }
# )
# expected = expected.merge(series, on=["unique_id", "ds"])
# pd.testing.assert_frame_equal(expected, cv_df.drop(columns="MLP"))
# # test save and load
# config = {
#     "input_size": tune.choice([12, 24]),
#     "hidden_size": 256,
#     "max_steps": 1,
#     "val_check_steps": 1,
#     "step_size": 12,
# }

# config_drnn = {
#     "input_size": tune.choice([-1]),
#     "encoder_hidden_size": tune.choice([5, 10]),
#     "max_steps": 1,
#     "val_check_steps": 1,
# }

# fcst = NeuralForecast(
#     models=[
#         AutoRNN(h=12, config=config_drnn, cpus=1, num_samples=2, refit_with_val=True),
#         DilatedRNN(h=12, input_size=-1, encoder_hidden_size=5, max_steps=1),
#         AutoMLP(h=12, config=config, cpus=1, num_samples=2),
#         NHITS(
#             h=12,
#             input_size=12,
#             max_steps=1,
#             futr_exog_list=["trend"],
#             hist_exog_list=["y_[lag12]"],
#             alias="Model1",
#         ),
#         StemGNN(h=12, input_size=12, n_series=2, max_steps=1, scaler_type="robust"),
#     ],
#     freq="M",
# )
# prediction_intervals = PredictionIntervals()
# fcst.fit(AirPassengersPanel_train, prediction_intervals=prediction_intervals)
# forecasts1 = fcst.predict(futr_df=AirPassengersPanel_test, level=[50])
# save_paths = ["./examples/debug_run/"]
# try:
#     s3fs.S3FileSystem().ls("s3://nixtla-tmp")
#     pyver = f"{sys.version_info.major}_{sys.version_info.minor}"
#     sha = git.Repo(search_parent_directories=True).head.object.hexsha
#     save_dir = f"{sys.platform}-{pyver}-{sha}"
#     save_paths.append(f"s3://nixtla-tmp/neural/{save_dir}")
# except Exception as e:
#     print(e)

# for path in save_paths:
#     fcst.save(path=path, model_index=None, overwrite=True, save_dataset=True)
#     fcst2 = NeuralForecast.load(path=path)
#     forecasts2 = fcst2.predict(futr_df=AirPassengersPanel_test, level=[50])
#     pd.testing.assert_frame_equal(forecasts1, forecasts2[forecasts1.columns])
# # test save and load without dataset
# shutil.rmtree("examples/debug_run")
# fcst = NeuralForecast(
#     models=[DilatedRNN(h=12, input_size=-1, encoder_hidden_size=5, max_steps=1)],
#     freq="M",
# )
# fcst.fit(AirPassengersPanel_train)
# forecasts1 = fcst.predict(futr_df=AirPassengersPanel_test)
# fcst.save(
#     path="./examples/debug_run/", model_index=None, overwrite=True, save_dataset=False
# )
# fcst2 = NeuralForecast.load(path="./examples/debug_run/")
# forecasts2 = fcst2.predict(df=AirPassengersPanel_train, futr_df=AirPassengersPanel_test)
# np.testing.assert_allclose(forecasts1["DilatedRNN"], forecasts2["DilatedRNN"])
# # test `enable_checkpointing=True` should generate chkpt
# shutil.rmtree("lightning_logs")
# fcst = NeuralForecast(
#     models=[
#         MLP(
#             h=12,
#             input_size=12,
#             max_steps=10,
#             val_check_steps=5,
#             enable_checkpointing=True,
#         ),
#         RNN(
#             h=12,
#             input_size=-1,
#             max_steps=10,
#             val_check_steps=5,
#             enable_checkpointing=True,
#         ),
#     ],
#     freq="M",
# )
# fcst.fit(AirPassengersPanel_train)
# last_log = f"lightning_logs/{os.listdir('lightning_logs')[-1]}"
# no_chkpt_found = ~np.any(
#     [file.endswith("checkpoints") for file in os.listdir(last_log)]
# )
# test_eq(no_chkpt_found, False)
# # test `enable_checkpointing=False` should not generate chkpt
# shutil.rmtree("lightning_logs")
# fcst = NeuralForecast(
#     models=[
#         MLP(h=12, input_size=12, max_steps=10, val_check_steps=5),
#         RNN(h=12, input_size=-1, max_steps=10, val_check_steps=5),
#     ],
#     freq="M",
# )
# fcst.fit(AirPassengersPanel_train)
# last_log = f"lightning_logs/{os.listdir('lightning_logs')[-1]}"
# no_chkpt_found = ~np.any(
#     [file.endswith("checkpoints") for file in os.listdir(last_log)]
# )
# test_eq(no_chkpt_found, True)
# # test short time series
# config = {"input_size": tune.choice([12, 24]), "max_steps": 1, "val_check_steps": 1}

# fcst = NeuralForecast(
#     models=[AutoNBEATS(h=12, config=config, cpus=1, num_samples=2)], freq="M"
# )

# AirPassengersShort = AirPassengersPanel.tail(36 + 144).reset_index(drop=True)
# forecasts = fcst.cross_validation(AirPassengersShort, val_size=48, n_windows=1)
# # test validation scale BaseWindows

# models = [NHITS(h=12, input_size=24, max_steps=50, scaler_type="robust")]
# nf = NeuralForecast(models=models, freq="M")
# nf.fit(AirPassengersPanel_train, val_size=12)
# valid_losses = nf.models[0].valid_trajectories
# assert valid_losses[-1][1] < 40, "Validation loss is too high"
# assert valid_losses[-1][1] > 10, "Validation loss is too low"

# models = [NHITS(h=12, input_size=24, max_steps=50, scaler_type=None)]
# nf = NeuralForecast(models=models, freq="M")
# nf.fit(AirPassengersPanel_train, val_size=12)
# valid_losses = nf.models[0].valid_trajectories
# assert valid_losses[-1][1] < 40, "Validation loss is too high"
# assert valid_losses[-1][1] > 10, "Validation loss is too low"
# # test validation scale BaseRecurrent

# nf = NeuralForecast(
#     models=[
#         LSTM(
#             h=12,
#             input_size=-1,
#             loss=MAE(),
#             scaler_type="robust",
#             encoder_n_layers=2,
#             encoder_hidden_size=128,
#             context_size=10,
#             decoder_hidden_size=128,
#             decoder_layers=2,
#             max_steps=50,
#             val_check_steps=10,
#         )
#     ],
#     freq="M",
# )
# nf.fit(AirPassengersPanel_train, val_size=12)
# valid_losses = nf.models[0].valid_trajectories
# assert valid_losses[-1][1] < 100, "Validation loss is too high"
# assert valid_losses[-1][1] > 30, "Validation loss is too low"
# # Test order of variables does not affect validation loss

# AirPassengersPanel_train["zeros"] = 0
# AirPassengersPanel_train["large_number"] = 100000
# AirPassengersPanel_train["available_mask"] = 1
# AirPassengersPanel_train = AirPassengersPanel_train[
#     ["unique_id", "ds", "zeros", "y", "available_mask", "large_number"]
# ]

# models = [NHITS(h=12, input_size=24, max_steps=50, scaler_type="robust")]
# nf = NeuralForecast(models=models, freq="M")
# nf.fit(AirPassengersPanel_train, val_size=12)
# valid_losses = nf.models[0].valid_trajectories
# assert valid_losses[-1][1] < 40, "Validation loss is too high"
# assert valid_losses[-1][1] > 10, "Validation loss is too low"

# models = [NHITS(h=12, input_size=24, max_steps=50, scaler_type=None)]
# nf = NeuralForecast(models=models, freq="M")
# nf.fit(AirPassengersPanel_train, val_size=12)
# valid_losses = nf.models[0].valid_trajectories
# assert valid_losses[-1][1] < 40, "Validation loss is too high"
# assert valid_losses[-1][1] > 10, "Validation loss is too low"
# # Test fit fails if variable not in dataframe

# # Base Windows
# models = [
#     NHITS(
#         h=12,
#         input_size=24,
#         max_steps=1,
#         hist_exog_list=["not_included"],
#         scaler_type="robust",
#     )
# ]
# nf = NeuralForecast(models=models, freq="M")
# test_fail(
#     nf.fit,
#     contains="historical exogenous variables not found in input dataset",
#     args=(AirPassengersPanel_train,),
# )

# models = [
#     NHITS(
#         h=12,
#         input_size=24,
#         max_steps=1,
#         futr_exog_list=["not_included"],
#         scaler_type="robust",
#     )
# ]
# nf = NeuralForecast(models=models, freq="M")
# test_fail(
#     nf.fit,
#     contains="future exogenous variables not found in input dataset",
#     args=(AirPassengersPanel_train,),
# )

# models = [
#     NHITS(
#         h=12,
#         input_size=24,
#         max_steps=1,
#         stat_exog_list=["not_included"],
#         scaler_type="robust",
#     )
# ]
# nf = NeuralForecast(models=models, freq="M")
# test_fail(
#     nf.fit,
#     contains="static exogenous variables not found in input dataset",
#     args=(AirPassengersPanel_train,),
# )

# # Base Recurrent
# models = [
#     LSTM(
#         h=12,
#         input_size=24,
#         max_steps=1,
#         hist_exog_list=["not_included"],
#         scaler_type="robust",
#     )
# ]
# nf = NeuralForecast(models=models, freq="M")
# test_fail(
#     nf.fit,
#     contains="historical exogenous variables not found in input dataset",
#     args=(AirPassengersPanel_train,),
# )

# models = [
#     LSTM(
#         h=12,
#         input_size=24,
#         max_steps=1,
#         futr_exog_list=["not_included"],
#         scaler_type="robust",
#     )
# ]
# nf = NeuralForecast(models=models, freq="M")
# test_fail(
#     nf.fit,
#     contains="future exogenous variables not found in input dataset",
#     args=(AirPassengersPanel_train,),
# )

# models = [
#     LSTM(
#         h=12,
#         input_size=24,
#         max_steps=1,
#         stat_exog_list=["not_included"],
#         scaler_type="robust",
#     )
# ]
# nf = NeuralForecast(models=models, freq="M")
# test_fail(
#     nf.fit,
#     contains="static exogenous variables not found in input dataset",
#     args=(AirPassengersPanel_train,),
# )
# # Test passing unused variables in dataframe does not affect forecasts

# models = [
#     NHITS(
#         h=12, input_size=24, max_steps=5, hist_exog_list=["zeros"], scaler_type="robust"
#     )
# ]
# nf = NeuralForecast(models=models, freq="M")
# nf.fit(AirPassengersPanel_train)

# Y_hat1 = nf.predict(
#     df=AirPassengersPanel_train[["unique_id", "ds", "y", "zeros", "large_number"]]
# )
# Y_hat2 = nf.predict(df=AirPassengersPanel_train[["unique_id", "ds", "y", "zeros"]])

# pd.testing.assert_frame_equal(
#     Y_hat1,
#     Y_hat2,
#     check_dtype=False,
# )

# models = [
#     LSTM(
#         h=12, input_size=24, max_steps=5, hist_exog_list=["zeros"], scaler_type="robust"
#     )
# ]
# nf = NeuralForecast(models=models, freq="M")
# nf.fit(AirPassengersPanel_train)

# Y_hat1 = nf.predict(
#     df=AirPassengersPanel_train[["unique_id", "ds", "y", "zeros", "large_number"]]
# )
# Y_hat2 = nf.predict(df=AirPassengersPanel_train[["unique_id", "ds", "y", "zeros"]])

# pd.testing.assert_frame_equal(
#     Y_hat1,
#     Y_hat2,
#     check_dtype=False,
# )
# import polars
# from polars.testing import assert_frame_equal

# renamer = {"unique_id": "uid", "ds": "time", "y": "target"}
# inverse_renamer = {v: k for k, v in renamer.items()}
# AirPassengers_pl = polars.from_pandas(AirPassengersPanel_train)
# AirPassengers_pl = AirPassengers_pl.rename(renamer)
# AirPassengersStatic_pl = polars.from_pandas(AirPassengersStatic)
# AirPassengersStatic_pl = AirPassengersStatic_pl.rename({"unique_id": "uid"})
# models = [LSTM(h=12, input_size=24, max_steps=5, scaler_type="robust")]

# # Pandas
# nf = NeuralForecast(models=models, freq="M")
# nf.fit(AirPassengersPanel_train, static_df=AirPassengersStatic)
# insample_preds = nf.predict_insample()
# preds = nf.predict()
# cv_res = nf.cross_validation(df=AirPassengersPanel_train, static_df=AirPassengersStatic)

# # Polars
# nf = NeuralForecast(models=models, freq="1mo")
# nf.fit(
#     AirPassengers_pl,
#     static_df=AirPassengersStatic_pl,
#     id_col="uid",
#     time_col="time",
#     target_col="target",
# )
# insample_preds_pl = nf.predict_insample()
# preds_pl = nf.predict()
# cv_res_pl = nf.cross_validation(
#     df=AirPassengers_pl,
#     static_df=AirPassengersStatic_pl,
#     id_col="uid",
#     time_col="time",
#     target_col="target",
# )


# def assert_equal_dfs(pandas_df, polars_df):
#     mapping = {k: v for k, v in inverse_renamer.items() if k in polars_df}
#     pd.testing.assert_frame_equal(
#         pandas_df,
#         polars_df.rename(mapping).to_pandas(),
#     )


# assert_equal_dfs(preds, preds_pl)
# assert_equal_dfs(insample_preds, insample_preds_pl)
# assert_equal_dfs(cv_res, cv_res_pl)
# # Test predict_insample step_size

# h = 12
# train_end = AirPassengers_pl["time"].max()
# sizes = AirPassengers_pl["uid"].value_counts().to_numpy()

# for step_size, test_size in [(7, 0), (9, 0), (7, 5), (9, 5)]:
#     models = [NHITS(h=h, input_size=12, max_steps=1)]
#     nf = NeuralForecast(models=models, freq="1mo")
#     nf.fit(
#         AirPassengers_pl,
#         id_col="uid",
#         time_col="time",
#         target_col="target",
#     )
#     # Note: only apply set_test_size() upon nf.fit(), otherwise it would have set the test_size = 0
#     nf.models[0].set_test_size(test_size)

#     forecasts = nf.predict_insample(step_size=step_size)
#     n_expected_cutoffs = (sizes[0][1] - test_size - nf.h + step_size) // step_size

#     # compare cutoff values
#     last_cutoff = (
#         train_end - test_size * pd.offsets.MonthEnd() - h * pd.offsets.MonthEnd()
#     )
#     expected_cutoffs = np.flip(
#         np.array(
#             [
#                 last_cutoff - step_size * i * pd.offsets.MonthEnd()
#                 for i in range(n_expected_cutoffs)
#             ]
#         )
#     )
#     pl_cutoffs = (
#         forecasts.filter(polars.col("uid") == nf.uids[1])
#         .select("cutoff")
#         .unique(maintain_order=True)
#     )
#     actual_cutoffs = np.sort(
#         np.array([pd.Timestamp(x["cutoff"]) for x in pl_cutoffs.rows(named=True)])
#     )
#     np.testing.assert_array_equal(
#         expected_cutoffs,
#         actual_cutoffs,
#         err_msg=f"{step_size=},{expected_cutoffs=},{actual_cutoffs=}",
#     )

#     # check forecast-points count per series
#     cutoffs_by_series = forecasts.group_by(["uid", "cutoff"]).count()
#     assert_frame_equal(
#         cutoffs_by_series.filter(polars.col("uid") == "Airline1").select(
#             ["cutoff", "count"]
#         ),
#         cutoffs_by_series.filter(polars.col("uid") == "Airline2").select(
#             ["cutoff", "count"]
#         ),
#         check_row_order=False,
#     )
# # Test if any of the inputs contains NaNs with available_mask = 1, fit shall raise error
# # input type is pandas.DataFrame
# # available_mask is explicitly given

# n_static_features = 2
# n_temporal_features = 4
# temporal_df, static_df = generate_series(
#     n_series=4,
#     min_length=50,
#     max_length=50,
#     n_static_features=n_static_features,
#     n_temporal_features=n_temporal_features,
#     equal_ends=False,
# )
# temporal_df["available_mask"] = 1
# temporal_df.loc[10:20, "available_mask"] = 0
# models = [NHITS(h=12, input_size=24, max_steps=20)]
# nf = NeuralForecast(models=models, freq="D")

# # test case 1: target has NaN values
# test_df1 = temporal_df.copy()
# test_df1.loc[5:7, "y"] = np.nan
# test_fail(lambda: nf.fit(test_df1), contains="Found missing values in ['y']")

# # test case 2: exogenous has NaN values that are correctly flagged with exception
# test_df2 = temporal_df.copy()
# # temporal_0 won't raise ValueError as available_mask = 0
# test_df2.loc[15:18, "temporal_0"] = np.nan
# test_df2.loc[5, "temporal_1"] = np.nan
# test_df2.loc[25, "temporal_2"] = np.nan
# test_fail(
#     lambda: nf.fit(test_df2),
#     contains="Found missing values in ['temporal_1', 'temporal_2']",
# )

# # test case 3: static column has NaN values
# test_df3 = static_df.copy()
# test_df3.loc[3, "static_1"] = np.nan
# test_fail(
#     lambda: nf.fit(temporal_df, static_df=test_df3),
#     contains="Found missing values in ['static_1']",
# )
# # Test if any of the inputs contains NaNs with available_mask = 1, fit shall raise error
# # input type is polars.Dataframe
# # Note that available_mask is not explicitly provided for this test

# pl_df = polars.DataFrame(
#     {
#         "unique_id": [1] * 50,
#         "y": list(range(50)),
#         "temporal_0": list(range(100, 150)),
#         "temporal_1": list(range(200, 250)),
#         "ds": polars.date_range(
#             start=date(2022, 1, 1), end=date(2022, 2, 19), interval="1d", eager=True
#         ),
#     }
# )

# pl_static_df = polars.DataFrame(
#     {
#         "unique_id": [1],
#         "static_0": [1.2],
#         "static_1": [10.9],
#     }
# )

# models = [NHITS(h=12, input_size=24, max_steps=20)]
# nf = NeuralForecast(models=models, freq="1d")

# # test case 1: target has NaN values
# test_pl_df1 = pl_df.clone()
# test_pl_df1[3, "y"] = np.nan
# test_pl_df1[4, "y"] = None
# test_fail(lambda: nf.fit(test_pl_df1), contains="Found missing values in ['y']")

# # test case 2: exogenous has NaN values that are correctly flagged with exception
# test_pl_df2 = pl_df.clone()
# test_pl_df2[15, "temporal_0"] = np.nan
# test_pl_df2[5, "temporal_1"] = np.nan
# test_fail(
#     lambda: nf.fit(test_pl_df2),
#     contains="Found missing values in ['temporal_0', 'temporal_1']",
# )

# # test case 3: static column has NaN values
# test_pl_df3 = pl_static_df.clone()
# test_pl_df3[0, "static_1"] = np.nan
# test_fail(
#     lambda: nf.fit(pl_df, static_df=test_pl_df3),
#     contains="Found missing values in ['static_1']",
# )
# # test customized optimizer behavior such that the user defined optimizer result should differ from default
# # tests consider models implemented using different base classes such as BaseWindows, BaseRecurrent, BaseMultivariate

# for nf_model in [NHITS, RNN, StemGNN]:
#     # default optimizer is based on Adam
#     params = {"h": 12, "input_size": 24, "max_steps": 1}
#     if nf_model.__name__ == "StemGNN":
#         params.update({"n_series": 2})
#     models = [nf_model(**params)]
#     nf = NeuralForecast(models=models, freq="M")
#     nf.fit(AirPassengersPanel_train)
#     default_optimizer_predict = nf.predict()
#     mean = default_optimizer_predict.loc[:, nf_model.__name__].mean()

#     # using a customized optimizer
#     params.update(
#         {
#             "optimizer": torch.optim.Adadelta,
#             "optimizer_kwargs": {"rho": 0.45},
#         }
#     )
#     models2 = [nf_model(**params)]
#     nf2 = NeuralForecast(models=models2, freq="M")
#     nf2.fit(AirPassengersPanel_train)
#     customized_optimizer_predict = nf2.predict()
#     mean2 = customized_optimizer_predict.loc[:, nf_model.__name__].mean()
#     assert mean2 != mean
# # test that if the user-defined optimizer is not a subclass of torch.optim.optimizer, failed with exception
# # tests cover different types of base classes such as BaseWindows, BaseRecurrent, BaseMultivariate
# test_fail(
#     lambda: NHITS(h=12, input_size=24, max_steps=10, optimizer=torch.nn.Module),
#     contains="optimizer is not a valid subclass of torch.optim.Optimizer",
# )
# test_fail(
#     lambda: RNN(h=12, input_size=24, max_steps=10, optimizer=torch.nn.Module),
#     contains="optimizer is not a valid subclass of torch.optim.Optimizer",
# )
# test_fail(
#     lambda: StemGNN(
#         h=12, input_size=24, max_steps=10, n_series=2, optimizer=torch.nn.Module
#     ),
#     contains="optimizer is not a valid subclass of torch.optim.Optimizer",
# )

# # test that if we pass "lr" parameter, we expect warning and it ignores the passed in 'lr' parameter
# # tests consider models implemented using different base classes such as BaseWindows, BaseRecurrent, BaseMultivariate

# for nf_model in [NHITS, RNN, StemGNN]:
#     params = {
#         "h": 12,
#         "input_size": 24,
#         "max_steps": 1,
#         "optimizer": torch.optim.Adadelta,
#         "optimizer_kwargs": {"lr": 0.8, "rho": 0.45},
#     }
#     if nf_model.__name__ == "StemGNN":
#         params.update({"n_series": 2})
#     models = [nf_model(**params)]
#     nf = NeuralForecast(models=models, freq="M")
#     with warnings.catch_warnings(record=True) as issued_warnings:
#         warnings.simplefilter("always", UserWarning)
#         nf.fit(AirPassengersPanel_train)
#         assert any(
#             "ignoring learning rate passed in optimizer_kwargs, using the model's learning rate"
#             in str(w.message)
#             for w in issued_warnings
#         )
# # test that if we pass "optimizer_kwargs" but not "optimizer", we expect a warning
# # tests consider models implemented using different base classes such as BaseWindows, BaseRecurrent, BaseMultivariate

# for nf_model in [NHITS, RNN, StemGNN]:
#     params = {
#         "h": 12,
#         "input_size": 24,
#         "max_steps": 1,
#         "optimizer_kwargs": {"lr": 0.8, "rho": 0.45},
#     }
#     if nf_model.__name__ == "StemGNN":
#         params.update({"n_series": 2})
#     models = [nf_model(**params)]
#     nf = NeuralForecast(models=models, freq="M")
#     with warnings.catch_warnings(record=True) as issued_warnings:
#         warnings.simplefilter("always", UserWarning)
#         nf.fit(AirPassengersPanel_train)
#         assert any(
#             "ignoring optimizer_kwargs as the optimizer is not specified"
#             in str(w.message)
#             for w in issued_warnings
#         )
# # test customized lr_scheduler behavior such that the user defined lr_scheduler result should differ from default
# # tests consider models implemented using different base classes such as BaseWindows, BaseRecurrent, BaseMultivariate

# for nf_model in [NHITS, RNN, StemGNN]:
#     params = {"h": 12, "input_size": 24, "max_steps": 1}
#     if nf_model.__name__ == "StemGNN":
#         params.update({"n_series": 2})
#     models = [nf_model(**params)]
#     nf = NeuralForecast(models=models, freq="M")
#     nf.fit(AirPassengersPanel_train)
#     default_optimizer_predict = nf.predict()
#     mean = default_optimizer_predict.loc[:, nf_model.__name__].mean()

#     # using a customized lr_scheduler, default is StepLR
#     params.update(
#         {
#             "lr_scheduler": torch.optim.lr_scheduler.ConstantLR,
#             "lr_scheduler_kwargs": {"factor": 0.78},
#         }
#     )
#     models2 = [nf_model(**params)]
#     nf2 = NeuralForecast(models=models2, freq="M")
#     nf2.fit(AirPassengersPanel_train)
#     customized_optimizer_predict = nf2.predict()
#     mean2 = customized_optimizer_predict.loc[:, nf_model.__name__].mean()
#     assert mean2 != mean
# # test that if the user-defined lr_scheduler is not a subclass of torch.optim.lr_scheduler, failed with exception
# # tests cover different types of base classes such as BaseWindows, BaseRecurrent, BaseMultivariate
# test_fail(
#     lambda: NHITS(h=12, input_size=24, max_steps=10, lr_scheduler=torch.nn.Module),
#     contains="lr_scheduler is not a valid subclass of torch.optim.lr_scheduler.LRScheduler",
# )
# test_fail(
#     lambda: RNN(h=12, input_size=24, max_steps=10, lr_scheduler=torch.nn.Module),
#     contains="lr_scheduler is not a valid subclass of torch.optim.lr_scheduler.LRScheduler",
# )
# test_fail(
#     lambda: StemGNN(
#         h=12, input_size=24, max_steps=10, n_series=2, lr_scheduler=torch.nn.Module
#     ),
#     contains="lr_scheduler is not a valid subclass of torch.optim.lr_scheduler.LRScheduler",
# )

# # test that if we pass in "optimizer" parameter, we expect warning and it ignores them
# # tests consider models implemented using different base classes such as BaseWindows, BaseRecurrent, BaseMultivariate

# for nf_model in [NHITS, RNN, StemGNN]:
#     params = {
#         "h": 12,
#         "input_size": 24,
#         "max_steps": 1,
#         "lr_scheduler": torch.optim.lr_scheduler.ConstantLR,
#         "lr_scheduler_kwargs": {"optimizer": torch.optim.Adadelta, "factor": 0.22},
#     }
#     if nf_model.__name__ == "StemGNN":
#         params.update({"n_series": 2})
#     models = [nf_model(**params)]
#     nf = NeuralForecast(models=models, freq="M")
#     with warnings.catch_warnings(record=True) as issued_warnings:
#         warnings.simplefilter("always", UserWarning)
#         nf.fit(AirPassengersPanel_train)
#         assert any(
#             "ignoring optimizer passed in lr_scheduler_kwargs, using the model's optimizer"
#             in str(w.message)
#             for w in issued_warnings
#         )
# # test that if we pass in "lr_scheduler_kwargs" but not "lr_scheduler", we expect a warning
# # tests consider models implemented using different base classes such as BaseWindows, BaseRecurrent, BaseMultivariate

# for nf_model in [NHITS, RNN, StemGNN]:
#     params = {
#         "h": 12,
#         "input_size": 24,
#         "max_steps": 1,
#         "lr_scheduler_kwargs": {"optimizer": torch.optim.Adadelta, "factor": 0.22},
#     }
#     if nf_model.__name__ == "StemGNN":
#         params.update({"n_series": 2})
#     models = [nf_model(**params)]
#     nf = NeuralForecast(models=models, freq="M")
#     with warnings.catch_warnings(record=True) as issued_warnings:
#         warnings.simplefilter("always", UserWarning)
#         nf.fit(AirPassengersPanel_train)
#         assert any(
#             "ignoring lr_scheduler_kwargs as the lr_scheduler is not specified"
#             in str(w.message)
#             for w in issued_warnings
#         )

# # test conformal prediction, method=conformal_distribution

# prediction_intervals = PredictionIntervals()

# models = []
# for nf_model in [NHITS, RNN, TSMixer]:
#     params = {"h": 12, "input_size": 24, "max_steps": 1}
#     if nf_model.__name__ == "TSMixer":
#         params.update({"n_series": 2})
#     models.append(nf_model(**params))


# nf = NeuralForecast(models=models, freq="M")
# nf.fit(AirPassengersPanel_train, prediction_intervals=prediction_intervals)
# preds = nf.predict(futr_df=AirPassengersPanel_test, level=[90])
# # test conformal prediction works for polar dataframe

# prediction_intervals = PredictionIntervals()

# models = []
# for nf_model in [NHITS, RNN, TSMixer]:
#     params = {"h": 12, "input_size": 24, "max_steps": 1}
#     if nf_model.__name__ == "TSMixer":
#         params.update({"n_series": 2})
#     models.append(nf_model(**params))


# nf = NeuralForecast(models=models, freq="1mo")
# nf.fit(
#     AirPassengers_pl,
#     prediction_intervals=prediction_intervals,
#     time_col="time",
#     id_col="uid",
#     target_col="target",
# )
# preds = nf.predict(level=[90])
# # test conformal prediction, method=conformal_error

# prediction_intervals = PredictionIntervals(method="conformal_error")

# models = []
# for nf_model in [NHITS, RNN, TSMixer]:
#     params = {"h": 12, "input_size": 24, "max_steps": 1}
#     if nf_model.__name__ == "TSMixer":
#         params.update({"n_series": 2})
#     models.append(nf_model(**params))


# nf = NeuralForecast(models=models, freq="M")
# nf.fit(AirPassengersPanel_train, prediction_intervals=prediction_intervals)
# preds = nf.predict(futr_df=AirPassengersPanel_test, level=[90])
# # test cross validation can support conformal prediction
# prediction_intervals = PredictionIntervals()

# # refit=False, no conformal predictions outputs
# nf = NeuralForecast(models=[NHITS(h=12, input_size=24, max_steps=1)], freq="M")
# test_fail(
#     nf.cross_validation,
#     "Passing prediction_intervals and/or level is only supported with refit=True.",
#     args=(AirPassengersPanel_train, prediction_intervals, [30, 70]),
# )

# # refit=True, we have conformal predictions outputs
# cv2 = nf.cross_validation(
#     AirPassengersPanel_train,
#     prediction_intervals=prediction_intervals,
#     refit=True,
#     level=[30, 70],
# )
# assert all([col in cv2.columns for col in ["NHITS-lo-30", "NHITS-hi-30"]])
# # Test quantile and level argument in predict for different models and errors
# prediction_intervals = PredictionIntervals(method="conformal_error")

# models = []
# for nf_model in [NHITS, LSTM, TSMixer]:
#     params = {"h": 12, "input_size": 24, "max_steps": 1, "loss": MAE()}
#     if nf_model.__name__ == "TSMixer":
#         params.update({"n_series": 2})
#     models.append(nf_model(**params))

#     params = {
#         "h": 12,
#         "input_size": 24,
#         "max_steps": 1,
#         "loss": DistributionLoss(distribution="Normal"),
#     }
#     if nf_model.__name__ == "TSMixer":
#         params.update({"n_series": 2})
#     models.append(nf_model(**params))

#     params = {"h": 12, "input_size": 24, "max_steps": 1, "loss": IQLoss()}
#     if nf_model.__name__ == "TSMixer":
#         params.update({"n_series": 2})
#     models.append(nf_model(**params))

# nf = NeuralForecast(models=models, freq="M")
# nf.fit(AirPassengersPanel_train, prediction_intervals=prediction_intervals)
# # Test default prediction
# preds = nf.predict(futr_df=AirPassengersPanel_test)
# assert list(preds.columns) == [
#     "unique_id",
#     "ds",
#     "NHITS",
#     "NHITS1",
#     "NHITS1-median",
#     "NHITS1-lo-90",
#     "NHITS1-lo-80",
#     "NHITS1-hi-80",
#     "NHITS1-hi-90",
#     "NHITS2_ql0.5",
#     "LSTM",
#     "LSTM1",
#     "LSTM1-median",
#     "LSTM1-lo-90",
#     "LSTM1-lo-80",
#     "LSTM1-hi-80",
#     "LSTM1-hi-90",
#     "LSTM2_ql0.5",
#     "TSMixer",
#     "TSMixer1",
#     "TSMixer1-median",
#     "TSMixer1-lo-90",
#     "TSMixer1-lo-80",
#     "TSMixer1-hi-80",
#     "TSMixer1-hi-90",
#     "TSMixer2_ql0.5",
# ]
# # Test quantile prediction
# preds = nf.predict(futr_df=AirPassengersPanel_test, quantiles=[0.2, 0.3])
# assert list(preds.columns) == [
#     "unique_id",
#     "ds",
#     "NHITS",
#     "NHITS-ql0.2",
#     "NHITS-ql0.3",
#     "NHITS1",
#     "NHITS1_ql0.2",
#     "NHITS1_ql0.3",
#     "NHITS2_ql0.2",
#     "NHITS2_ql0.3",
#     "LSTM",
#     "LSTM-ql0.2",
#     "LSTM-ql0.3",
#     "LSTM1",
#     "LSTM1_ql0.2",
#     "LSTM1_ql0.3",
#     "LSTM2_ql0.2",
#     "LSTM2_ql0.3",
#     "TSMixer",
#     "TSMixer-ql0.2",
#     "TSMixer-ql0.3",
#     "TSMixer1",
#     "TSMixer1_ql0.2",
#     "TSMixer1_ql0.3",
#     "TSMixer2_ql0.2",
#     "TSMixer2_ql0.3",
# ]
# # Test level prediction
# preds = nf.predict(futr_df=AirPassengersPanel_test, level=[80, 90])
# assert list(preds.columns) == [
#     "unique_id",
#     "ds",
#     "NHITS",
#     "NHITS-lo-90",
#     "NHITS-lo-80",
#     "NHITS-hi-80",
#     "NHITS-hi-90",
#     "NHITS1",
#     "NHITS1-lo-90",
#     "NHITS1-lo-80",
#     "NHITS1-hi-80",
#     "NHITS1-hi-90",
#     "NHITS2-lo-90",
#     "NHITS2-lo-80",
#     "NHITS2-hi-80",
#     "NHITS2-hi-90",
#     "LSTM",
#     "LSTM-lo-90",
#     "LSTM-lo-80",
#     "LSTM-hi-80",
#     "LSTM-hi-90",
#     "LSTM1",
#     "LSTM1-lo-90",
#     "LSTM1-lo-80",
#     "LSTM1-hi-80",
#     "LSTM1-hi-90",
#     "LSTM2-lo-90",
#     "LSTM2-lo-80",
#     "LSTM2-hi-80",
#     "LSTM2-hi-90",
#     "TSMixer",
#     "TSMixer-lo-90",
#     "TSMixer-lo-80",
#     "TSMixer-hi-80",
#     "TSMixer-hi-90",
#     "TSMixer1",
#     "TSMixer1-lo-90",
#     "TSMixer1-lo-80",
#     "TSMixer1-hi-80",
#     "TSMixer1-hi-90",
#     "TSMixer2-lo-90",
#     "TSMixer2-lo-80",
#     "TSMixer2-hi-80",
#     "TSMixer2-hi-90",
# ]
# # Re-Test default prediction - note that they are different from the first test (this is expected)
# preds = nf.predict(futr_df=AirPassengersPanel_test)
# assert list(preds.columns) == [
#     "unique_id",
#     "ds",
#     "NHITS",
#     "NHITS1",
#     "NHITS1-median",
#     "NHITS2_ql0.5",
#     "LSTM",
#     "LSTM1",
#     "LSTM1-median",
#     "LSTM2_ql0.5",
#     "TSMixer",
#     "TSMixer1",
#     "TSMixer1-median",
#     "TSMixer2_ql0.5",
# ]
