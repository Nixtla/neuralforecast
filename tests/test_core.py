import os
import platform
import shutil
import sys
import tempfile
import warnings
from datetime import date
from pathlib import Path

import git
import numpy as np
import optuna
import pandas as pd
import polars
import polars.testing
import pytest
import s3fs
import torch
from ray import tune

from neuralforecast.auto import (
    MLP,
    NHITS,
    RNN,
    TCN,
    TFT,
    AutoDilatedRNN,
    Autoformer,
    AutoMLP,
    AutoNBEATSx,
    AutoRNN,
    DeepAR,
    DilatedRNN,
    Informer,
    NBEATSx,
    StemGNN,
    TSMixer,
    TSMixerx,
    VanillaTransformer,
)
from neuralforecast.common.enums import ExplainerEnum
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
    NBMM,
    PMM,
    DistributionLoss,
    MQLoss,
)
from neuralforecast.utils import (
    AirPassengersPanel,
    AirPassengersStatic,
    generate_series,
)

from .test_helpers import assert_equal_dfs, get_expected_size


@pytest.fixture
def setup():
    uids = pd.Series(["id_0", "id_1"])
    indptr = np.array([0, 4, 10], dtype=np.int32)
    return uids, indptr


@pytest.mark.parametrize("step_size, freq, days", [(1, "D", 1), (2, "W-THU", 14)])
def test_cutoff_deltas(setup, step_size, freq, days):
    uids, indptr = setup
    h = 2
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
def setup_airplane_data_polars(setup_airplane_data):
    """Create polars version of airplane data with renamed columns."""
    AirPassengersPanel_train, AirPassengersPanel_test = setup_airplane_data

    # Set up column renaming for Polars
    renamer = {"unique_id": "uid", "ds": "time", "y": "target"}

    # Create Polars dataframes
    AirPassengers_pl = polars.from_pandas(AirPassengersPanel_train)
    AirPassengers_pl = AirPassengers_pl.rename(renamer)

    AirPassengersStatic_pl = polars.from_pandas(AirPassengersStatic)
    AirPassengersStatic_pl = AirPassengersStatic_pl.rename({'unique_id': 'uid'})

    return AirPassengers_pl, AirPassengersStatic_pl


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
    pd.testing.assert_frame_equal(init_cv, after_cv)
    pd.testing.assert_frame_equal(after_fcst, init_fcst)

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
# test futr_df contents
def test_future_df_contents(setup_airplane_data):
    AirPassengersPanel_train, AirPassengersPanel_test = setup_airplane_data

    models = [
        NHITS(
            h=6,
            input_size=24,
            max_steps=10,
            hist_exog_list=["trend"],
            futr_exog_list=["trend"],
        )
    ]
    nf = NeuralForecast(models=models, freq="M")
    nf.fit(AirPassengersPanel_train)

    # Test that not enough rows in futr_df raises an error
    with pytest.raises(Exception) as exc_info:
        nf.predict(futr_df=AirPassengersPanel_test.head())
    assert "There are missing combinations" in str(exc_info.value)

    # Test that extra rows issues a warning
    with warnings.catch_warnings(record=True) as issued_warnings:
        warnings.simplefilter("always", UserWarning)
        nf.predict(futr_df=AirPassengersPanel_test)
    assert any("Dropped 12 unused rows" in str(w.message) for w in issued_warnings)

    # Test that models require futr_df and not providing it raises an error
    with pytest.raises(Exception) as exc_info:
        nf.predict()
    assert "Models require the following future exogenous features: {'trend'}" in str(exc_info.value)

    # Test that missing feature in futr_df raises an error
    with pytest.raises(Exception) as exc_info:
        nf.predict(futr_df=AirPassengersPanel_test.drop(columns="trend"))
    assert "missing from `futr_df`: {'trend'}" in str(exc_info.value)

    # Test that null values in futr_df raises an error
    with pytest.raises(Exception) as exc_info:
        nf.predict(futr_df=AirPassengersPanel_test.assign(trend=np.nan))
    assert "Found null values in `futr_df`" in str(exc_info.value)

# Test inplace model fitting
def test_inplace_model_fitting(setup_airplane_data):
    AirPassengersPanel_train, AirPassengersPanel_test = setup_airplane_data

    models = [MLP(h=12, input_size=12, max_steps=1, scaler_type="robust")]
    initial_weights = models[0].mlp[0].weight.detach().clone()
    fcst = NeuralForecast(models=models, freq="M")
    fcst.fit(
        df=AirPassengersPanel_train, static_df=AirPassengersStatic, use_init_models=True
    )
    after_weights = fcst.models_init[0].mlp[0].weight.detach().clone()
    assert np.allclose(initial_weights, after_weights), "init models should not be modified"
    assert len(fcst.models[0].train_trajectories) > 0, (
        "models stored trajectories should not be empty"
    )

@pytest.fixture
def setup_models_for_insample():
    h = 12

    models = [
        NHITS(
            h=h,
            input_size=24,
            loss=MQLoss(level=[80]),
            max_steps=1,
            alias="NHITS",
            scaler_type=None,
        ),
        RNN(h=h, input_size=-1, loss=MAE(), max_steps=1, alias="RNN", scaler_type=None),
    ]
    return models



# Test predict_insample
def test_predict_insample(setup_airplane_data, setup_models_for_insample):
    AirPassengersPanel_train, AirPassengersPanel_test = setup_airplane_data
    models = setup_models_for_insample
    test_size = 12
    h = 12

    nf = NeuralForecast(models=models, freq="M")
    _ = nf.cross_validation(
        df=AirPassengersPanel_train,
        static_df=AirPassengersStatic,
        val_size=0,
        test_size=test_size,
        n_windows=None,
    )

    forecasts = nf.predict_insample(step_size=1)

    expected_size = get_expected_size(AirPassengersPanel_train, h, test_size, step_size=1)
    assert len(forecasts) == expected_size, (
        f"Shape mismatch in predict_insample: {len(forecasts)=}, {expected_size=}"
    )

# Test predict_insample (different lengths)
def test_predict_insample_diff_lengths(setup_models_for_insample):
    models = setup_models_for_insample
    test_size = 12
    n_series = 2
    h = 12
    diff_len_df = generate_series(n_series=n_series, max_length=100)

    nf = NeuralForecast(models=models, freq="D")
    _ = nf.cross_validation(
        df=diff_len_df, val_size=0, test_size=test_size, n_windows=None
    )

    forecasts = nf.predict_insample(step_size=1)
    expected_size = get_expected_size(diff_len_df, h, test_size, step_size=1)
    assert len(forecasts) == expected_size, (
        f"Shape mismatch in predict_insample: {len(forecasts)=}, {expected_size=}"
    )

@pytest.mark.parametrize("step_size, test_size", [(7, 0), (9, 0), (7, 5), (9, 5)])
def test_predict_insample_step_size(setup_airplane_data, step_size, test_size):
    AirPassengersPanel_train, _ = setup_airplane_data

    h = 12
    train_end = AirPassengersPanel_train['ds'].max()
    sizes = AirPassengersPanel_train['unique_id'].value_counts().to_numpy()

    models = [NHITS(h=h, input_size=12, max_steps=1)]
    nf = NeuralForecast(models=models, freq='M')
    nf.fit(AirPassengersPanel_train)
    # Note: only apply set_test_size() upon nf.fit(), otherwise it would have set the test_size = 0
    nf.models[0].set_test_size(test_size)

    forecasts = nf.predict_insample(step_size=step_size)
    last_cutoff = train_end - test_size * pd.offsets.MonthEnd() - h * pd.offsets.MonthEnd()
    n_expected_cutoffs = (sizes[0] - test_size - nf.h + step_size) // step_size

    # compare cutoff values
    expected_cutoffs = np.flip(np.array([last_cutoff - step_size * i * pd.offsets.MonthEnd() for i in range(n_expected_cutoffs)]))
    actual_cutoffs = np.array([pd.Timestamp(x) for x in forecasts[forecasts['unique_id']==nf.uids[1]]['cutoff'].unique()])
    np.testing.assert_array_equal(expected_cutoffs, actual_cutoffs, err_msg=f"{step_size=},{expected_cutoffs=},{actual_cutoffs=}")

    # check forecast-points count per series
    cutoffs_by_series = forecasts.groupby(['unique_id', 'cutoff']).size().unstack('unique_id')
    pd.testing.assert_series_equal(cutoffs_by_series['Airline1'], cutoffs_by_series['Airline2'], check_names=False)


def test_predict_insample_diff_loss(setup_airplane_data):
    AirPassengersPanel_train, _ = setup_airplane_data


    def get_expected_cols(model, level):
        # index columns
        n_cols = 4
        for model in models:
            if isinstance(loss, (DistributionLoss, PMM, GMM, NBMM)):
                if level is None:
                    # Variations of DistributionLoss return the sample mean as well
                    n_cols += len(loss.quantiles) + 1
                else:
                    # Variations of DistributionLoss return the sample mean as well
                    n_cols += 2 * len(level) + 1
            else:
                if level is None:
                    # Other probabilistic models return the sample mean as well
                    n_cols += 1
                # Other probabilistic models return just the levels
                else:
                    n_cols += len(level) + 1
        return n_cols

    for loss in [
        # IQLoss(),
        DistributionLoss(distribution="Normal", level=[80]),
        PMM(level=[80]),
    ]:
        for level in [None, [80, 90]]:
            # Use CPU accelerator on macOS to avoid MPS LSTM projection limitation
            # MPS (Metal Performance Shaders) doesn't support LSTM with projections
            # which are enabled when recurrent=True
            lstm_kwargs = {"h": 12, "input_size": 12, "loss": loss, "max_steps": 1, "recurrent": True}
            if platform.system() == "Darwin":  # macOS
                lstm_kwargs["accelerator"] = "cpu"

            models = [
                NHITS(h=12, input_size=12, loss=loss, max_steps=1),
                LSTM(**lstm_kwargs),
            ]
            nf = NeuralForecast(models=models, freq='D')

            nf.fit(df=AirPassengersPanel_train)
            df = nf.predict_insample(step_size=1, level=level)
            expected_cols = get_expected_cols(models, level)
            assert df.shape[1] == expected_cols, f'Shape mismatch for {loss} and level={level} in predict_insample: cols={df.shape[1]}, expected_cols={expected_cols}'

# Test aliases
def test_aliases(setup_airplane_data):
    AirPassengersPanel_train, AirPassengersPanel_test = setup_airplane_data

    config_drnn = {
        "input_size": tune.choice([-1]),
        "encoder_hidden_size": tune.choice([5, 10]),
        "max_steps": 1,
        "val_check_steps": 1,
        "step_size": 1,
    }
    models = [
        # test Auto
        AutoDilatedRNN(h=12, config=config_drnn, cpus=1, num_samples=2, alias="AutoDIL"),
        # test BaseWindows
        NHITS(h=12, input_size=24, loss=MQLoss(level=[80]), max_steps=1, alias="NHITSMQ"),
        # test BaseRecurrent
        RNN(
            h=12,
            input_size=-1,
            encoder_hidden_size=10,
            max_steps=1,
            stat_exog_list=["airline1"],
            futr_exog_list=["trend"],
            hist_exog_list=["y_[lag12]"],
            alias="MyRNN",
        ),
        # test BaseMultivariate
        StemGNN(
            h=12,
            input_size=24,
            n_series=2,
            max_steps=1,
            scaler_type="robust",
            alias="StemMulti",
        ),
        # test model without alias
        NHITS(h=12, input_size=24, max_steps=1),
    ]
    nf = NeuralForecast(models=models, freq="M")
    nf.fit(df=AirPassengersPanel_train, static_df=AirPassengersStatic)
    forecasts = nf.predict(futr_df=AirPassengersPanel_test)
    assert forecasts.columns.to_list() == [
            "unique_id",
            "ds",
            "AutoDIL",
            "NHITSMQ-median",
            "NHITSMQ-lo-80",
            "NHITSMQ-hi-80",
            "MyRNN",
            "StemMulti",
            "NHITS",
        ]

def config_optuna(trial):
    return {
        "input_size": trial.suggest_categorical("input_size", [12, 24]),
        "hist_exog_list": trial.suggest_categorical(
            "hist_exog_list", [["trend"], ["y_[lag12]"], ["trend", "y_[lag12]"]]
        ),
        "futr_exog_list": ["trend"],
        "max_steps": 10,
        "val_check_steps": 5,
    }

def test_training_with_an_iterative_dataset(setup_airplane_data):
    AirPassengersPanel_train, AirPassengersPanel_test = setup_airplane_data

    config_ray = {
        "input_size": tune.choice([12, 24]),
        "hist_exog_list": tune.choice([["trend"], ["y_[lag12]"], ["trend", "y_[lag12]"]]),
        "futr_exog_list": ["trend"],
        "max_steps": 10,
        "val_check_steps": 5,
    }
    # test training with an iterative dataset produces the same results as directly passing in the dataset as a pandas dataframe
    AirPassengersPanel_train["id"] = AirPassengersPanel_train["unique_id"]
    AirPassengersPanel_test["id"] = AirPassengersPanel_test["unique_id"]

    models = [
        NHITS(h=12, input_size=12, max_steps=10, futr_exog_list=["trend"], random_seed=1),
        AutoMLP(
            h=12,
            config=config_optuna,
            num_samples=2,
            backend="optuna",
            search_alg=optuna.samplers.TPESampler(seed=0),
        ),  # type: ignore
        AutoNBEATSx(h=12, config=config_ray, cpus=1, num_samples=2),
    ]
    nf = NeuralForecast(models=models, freq="M")

    # fit+predict with pandas dataframe
    nf.fit(
        df=AirPassengersPanel_train.drop(columns="unique_id"),
        use_init_models=True,
        id_col="id",
    )
    pred_dataframe = nf.predict(
        futr_df=AirPassengersPanel_test.drop(columns="unique_id")
    ).reset_index()

    # fit+predict with data directory
    with tempfile.TemporaryDirectory() as tmpdir:
        AirPassengersPanel_train.to_parquet(
            tmpdir, partition_cols=["unique_id"], index=False
        )
        data_directory = sorted([str(path) for path in Path(tmpdir).iterdir()])
        nf.fit(df=data_directory, use_init_models=True, id_col="id")

    pred_df = AirPassengersPanel_train[
        AirPassengersPanel_train["unique_id"] == "Airline2"
    ].drop(columns="unique_id")
    futr_df = AirPassengersPanel_test[
        AirPassengersPanel_test["unique_id"] == "Airline2"
    ].drop(columns="unique_id")

    pred_iterative = nf.predict(df=pred_df, futr_df=futr_df)
    pred_airline2 = pred_dataframe[pred_dataframe["id"] == "Airline2"]
    np.testing.assert_allclose(
        pred_iterative["NHITS"], pred_airline2["NHITS"], rtol=0, atol=1
    )
    np.testing.assert_allclose(
        pred_iterative["AutoMLP"], pred_airline2["AutoMLP"], rtol=0, atol=1
    )
    np.testing.assert_allclose(
        pred_iterative["AutoNBEATSx"], pred_airline2["AutoNBEATSx"], rtol=0, atol=1
    )


# test cross validation no leakage
def test_cross_validation(h=12, test_size=12):
    df = AirPassengersPanel
    static_df = AirPassengersStatic
    if (test_size - h) % 1:
        raise Exception("`test_size - h` should be module `step_size`")

    Y_test_df = df.groupby("unique_id").tail(test_size)
    Y_train_df = df.drop(Y_test_df.index)
    config = {
        "input_size": tune.choice([12, 24]),
        "step_size": 12,
        "hidden_size": 256,
        "max_steps": 1,
        "val_check_steps": 1,
    }
    config_drnn = {
        "input_size": tune.choice([-1]),
        "encoder_hidden_size": tune.choice([5, 10]),
        "max_steps": 1,
        "val_check_steps": 1,
    }
    fcst = NeuralForecast(
        models=[
            AutoDilatedRNN(h=12, config=config_drnn, cpus=1, num_samples=1),
            DilatedRNN(h=12, input_size=-1, encoder_hidden_size=5, max_steps=1),
            RNN(
                h=12,
                input_size=-1,
                encoder_hidden_size=5,
                max_steps=1,
                stat_exog_list=["airline1"],
                futr_exog_list=["trend"],
                hist_exog_list=["y_[lag12]"],
            ),
            TCN(
                h=12,
                input_size=-1,
                encoder_hidden_size=5,
                max_steps=1,
                stat_exog_list=["airline1"],
                futr_exog_list=["trend"],
                hist_exog_list=["y_[lag12]"],
            ),
            AutoMLP(h=12, config=config, cpus=1, num_samples=1),
            MLP(h=12, input_size=12, max_steps=1, scaler_type="robust"),
            NBEATSx(
                h=12,
                input_size=12,
                max_steps=1,
                stat_exog_list=["airline1"],
                futr_exog_list=["trend"],
                hist_exog_list=["y_[lag12]"],
            ),
            NHITS(h=12, input_size=12, max_steps=1, scaler_type="robust"),
            NHITS(h=12, input_size=12, loss=MQLoss(level=[80]), max_steps=1),
            TFT(h=12, input_size=24, max_steps=1, scaler_type="robust"),
            DLinear(h=12, input_size=24, max_steps=1),
            VanillaTransformer(h=12, input_size=12, max_steps=1, scaler_type=None),
            Informer(h=12, input_size=12, max_steps=1, scaler_type=None),
            Autoformer(h=12, input_size=12, max_steps=1, scaler_type=None),
            FEDformer(h=12, input_size=12, max_steps=1, scaler_type=None),
            PatchTST(h=12, input_size=24, max_steps=1, scaler_type=None),
            TimesNet(h=12, input_size=24, max_steps=1, scaler_type="standard"),
            StemGNN(h=12, input_size=12, n_series=2, max_steps=1, scaler_type="robust"),
            TSMixer(h=12, input_size=12, n_series=2, max_steps=1, scaler_type="robust"),
            TSMixerx(
                h=12, input_size=12, n_series=2, max_steps=1, scaler_type="robust"
            ),
            DeepAR(
                h=12,
                input_size=24,
                max_steps=1,
                stat_exog_list=["airline1"],
                futr_exog_list=["trend"],
            ),
        ],
        freq="M",
    )
    fcst.fit(df=Y_train_df, static_df=static_df)
    Y_hat_df = fcst.predict(futr_df=Y_test_df)
    Y_hat_df = Y_hat_df.merge(Y_test_df, how="left", on=["unique_id", "ds"])
    last_dates = Y_train_df.groupby("unique_id").tail(1)
    last_dates = last_dates[["unique_id", "ds"]].rename(columns={"ds": "cutoff"})
    Y_hat_df = Y_hat_df.merge(last_dates, how="left", on="unique_id")

    # cross validation
    fcst = NeuralForecast(
        models=[
            AutoDilatedRNN(h=12, config=config_drnn, cpus=1, num_samples=1),
            DilatedRNN(h=12, input_size=-1, encoder_hidden_size=5, max_steps=1),
            RNN(
                h=12,
                input_size=-1,
                encoder_hidden_size=5,
                max_steps=1,
                stat_exog_list=["airline1"],
                futr_exog_list=["trend"],
                hist_exog_list=["y_[lag12]"],
            ),
            TCN(
                h=12,
                input_size=-1,
                encoder_hidden_size=5,
                max_steps=1,
                stat_exog_list=["airline1"],
                futr_exog_list=["trend"],
                hist_exog_list=["y_[lag12]"],
            ),
            AutoMLP(h=12, config=config, cpus=1, num_samples=1),
            MLP(h=12, input_size=12, max_steps=1, scaler_type="robust"),
            NBEATSx(
                h=12,
                input_size=12,
                max_steps=1,
                stat_exog_list=["airline1"],
                futr_exog_list=["trend"],
                hist_exog_list=["y_[lag12]"],
            ),
            NHITS(h=12, input_size=12, max_steps=1, scaler_type="robust"),
            NHITS(h=12, input_size=12, loss=MQLoss(level=[80]), max_steps=1),
            TFT(h=12, input_size=24, max_steps=1, scaler_type="robust"),
            DLinear(h=12, input_size=24, max_steps=1),
            VanillaTransformer(h=12, input_size=12, max_steps=1, scaler_type=None),
            Informer(h=12, input_size=12, max_steps=1, scaler_type=None),
            Autoformer(h=12, input_size=12, max_steps=1, scaler_type=None),
            FEDformer(h=12, input_size=12, max_steps=1, scaler_type=None),
            PatchTST(h=12, input_size=24, max_steps=1, scaler_type=None),
            TimesNet(h=12, input_size=24, max_steps=1, scaler_type="standard"),
            StemGNN(h=12, input_size=12, n_series=2, max_steps=1, scaler_type="robust"),
            TSMixer(h=12, input_size=12, n_series=2, max_steps=1, scaler_type="robust"),
            TSMixerx(
                h=12, input_size=12, n_series=2, max_steps=1, scaler_type="robust"
            ),
            DeepAR(
                h=12,
                input_size=24,
                max_steps=1,
                stat_exog_list=["airline1"],
                futr_exog_list=["trend"],
            ),
        ],
        freq="M",
    )
    Y_hat_df_cv = fcst.cross_validation(
        df, static_df=static_df, test_size=test_size, n_windows=None
    )
    for col in ["ds", "cutoff"]:
        Y_hat_df_cv[col] = pd.to_datetime(Y_hat_df_cv[col].astype(str))
        Y_hat_df[col] = pd.to_datetime(Y_hat_df[col].astype(str))
    pd.testing.assert_frame_equal(
        Y_hat_df[Y_hat_df_cv.columns],
        Y_hat_df_cv,
        check_dtype=False,
        atol=1e-5,
    )


# test cv with series of different sizes
def test_cv_with_series_of_different_sizes():
    series = pd.DataFrame(
        {
            "unique_id": np.repeat([0, 1], [10, 15]),
            "ds": np.arange(25),
            "y": np.random.rand(25),
        }
    )
    nf = NeuralForecast(
        freq=1, models=[MLP(input_size=5, h=5, max_steps=0, enable_progress_bar=False)]
    )
    cv_df = nf.cross_validation(df=series, n_windows=3, step_size=5)
    expected = pd.DataFrame(
        {
            "unique_id": np.repeat([0, 1], [5, 10]),
            "ds": np.hstack([np.arange(5, 10), np.arange(15, 25)]),
            "cutoff": np.repeat([4, 14, 19], 5),
        }
    )
    expected = expected.merge(series, on=["unique_id", "ds"])
    pd.testing.assert_frame_equal(expected, cv_df.drop(columns="MLP"))

# test save and load
def test_save_load(setup_airplane_data):
    AirPassengersPanel_train, AirPassengersPanel_test = setup_airplane_data

    config = {
        "input_size": tune.choice([12, 24]),
        "hidden_size": 256,
        "max_steps": 1,
        "val_check_steps": 1,
        "step_size": 12,
    }

    config_drnn = {
        "input_size": tune.choice([-1]),
        "encoder_hidden_size": tune.choice([5, 10]),
        "max_steps": 1,
        "val_check_steps": 1,
    }

    fcst = NeuralForecast(
        models=[
            AutoRNN(h=12, config=config_drnn, cpus=1, num_samples=2, refit_with_val=True),
            DilatedRNN(h=12, input_size=-1, encoder_hidden_size=5, max_steps=1),
            AutoMLP(h=12, config=config, cpus=1, num_samples=2),
            NHITS(
                h=12,
                input_size=12,
                max_steps=1,
                futr_exog_list=["trend"],
                hist_exog_list=["y_[lag12]"],
                alias="Model1",
            ),
            StemGNN(h=12, input_size=12, n_series=2, max_steps=1, scaler_type="robust"),
        ],
        freq="M",
    )
    prediction_intervals = PredictionIntervals()
    fcst.fit(AirPassengersPanel_train, prediction_intervals=prediction_intervals)
    forecasts1 = fcst.predict(futr_df=AirPassengersPanel_test, level=[50])
    save_paths = ["./examples/debug_run/"]
    try:
        s3fs.S3FileSystem().ls("s3://nixtla-tmp")
        pyver = f"{sys.version_info.major}_{sys.version_info.minor}"
        sha = git.Repo(search_parent_directories=True).head.object.hexsha
        save_dir = f"{sys.platform}-{pyver}-{sha}"
        save_paths.append(f"s3://nixtla-tmp/neural/{save_dir}")
    except Exception as e:
        print(e)

    for path in save_paths:
        fcst.save(path=path, model_index=None, overwrite=True, save_dataset=True)
        fcst2 = NeuralForecast.load(path=path)
        forecasts2 = fcst2.predict(futr_df=AirPassengersPanel_test, level=[50])
        pd.testing.assert_frame_equal(forecasts1, forecasts2[forecasts1.columns])


# test save and load without dataset
def test_save_load_no_dataset(setup_airplane_data):
    AirPassengersPanel_train, AirPassengersPanel_test = setup_airplane_data

    shutil.rmtree("examples/debug_run")
    fcst = NeuralForecast(
        models=[DilatedRNN(h=12, input_size=-1, encoder_hidden_size=5, max_steps=1)],
        freq="M",
    )
    fcst.fit(AirPassengersPanel_train)
    forecasts1 = fcst.predict(futr_df=AirPassengersPanel_test)
    fcst.save(
        path="./examples/debug_run/", model_index=None, overwrite=True, save_dataset=False
    )
    fcst2 = NeuralForecast.load(path="./examples/debug_run/")
    forecasts2 = fcst2.predict(df=AirPassengersPanel_train, futr_df=AirPassengersPanel_test)
    np.testing.assert_allclose(forecasts1["DilatedRNN"], forecasts2["DilatedRNN"])

# test `enable_checkpointing=True` should generate chkpt
def test_enable_checkpointing(setup_airplane_data):
    AirPassengersPanel_train, _ = setup_airplane_data

    try:
        shutil.rmtree("lightning_logs")
    except:
        print("Directory does not exist")

    fcst = NeuralForecast(
        models=[
            MLP(
                h=12,
                input_size=12,
                max_steps=10,
                val_check_steps=5,
                enable_checkpointing=True,
            ),
            RNN(
                h=12,
                input_size=-1,
                max_steps=10,
                val_check_steps=5,
                enable_checkpointing=True,
            ),
        ],
        freq="M",
    )
    fcst.fit(AirPassengersPanel_train)
    last_log = f"lightning_logs/{os.listdir('lightning_logs')[-1]}"
    no_chkpt_found = ~np.any(
        [file.endswith("checkpoints") for file in os.listdir(last_log)]
    )
    assert no_chkpt_found ==  False

# test `enable_checkpointing=False` should not generate chkpt
def test_no_checkpointing(setup_airplane_data):
    AirPassengersPanel_train, _ = setup_airplane_data

    try:
        shutil.rmtree("lightning_logs")
    except:
        print("Directory does not exist")

    fcst = NeuralForecast(
        models=[
            MLP(h=12, input_size=12, max_steps=10, val_check_steps=5),
            RNN(h=12, input_size=-1, max_steps=10, val_check_steps=5),
        ],
        freq="M",
    )
    fcst.fit(AirPassengersPanel_train)
    last_log = f"lightning_logs/{os.listdir('lightning_logs')[-1]}"
    no_chkpt_found = ~np.any(
        [file.endswith("checkpoints") for file in os.listdir(last_log)]
    )
    assert no_chkpt_found == True


# test validation scale BaseWindows
@pytest.mark.parametrize("scaler_type", ["robust", None])
def test_validation_scale_basewindows(setup_airplane_data, scaler_type):
    AirPassengersPanel_train, _ = setup_airplane_data

    models = [NHITS(h=12, input_size=24, max_steps=50, scaler_type=scaler_type)]
    nf = NeuralForecast(models=models, freq="M")
    nf.fit(AirPassengersPanel_train, val_size=12)
    valid_losses = nf.models[0].valid_trajectories
    assert valid_losses[-1][1] < 40, "Validation loss is too high"
    assert valid_losses[-1][1] > 10, "Validation loss is too low"


# test validation scale BaseRecurrent
def test_validation_scale_baserecurrent(setup_airplane_data):
    AirPassengersPanel_train, _ = setup_airplane_data
    nf = NeuralForecast(
        models=[
            LSTM(
                h=12,
                input_size=-1,
                loss=MAE(),
                scaler_type="robust",
                encoder_n_layers=2,
                encoder_hidden_size=128,
                context_size=10,
                decoder_hidden_size=128,
                decoder_layers=2,
                max_steps=50,
                val_check_steps=10,
            )
        ],
        freq="M",
    )
    nf.fit(AirPassengersPanel_train, val_size=12)
    valid_losses = nf.models[0].valid_trajectories
    assert valid_losses[-1][1] < 100, "Validation loss is too high"
    assert valid_losses[-1][1] > 30, "Validation loss is too low"


# Test order of variables does not affect validation loss
@pytest.mark.parametrize("scaler_type", ["robust", None])
def test_order_of_variables_no_effect_on_val_loss(setup_airplane_data, scaler_type):
    AirPassengersPanel_train, _ = setup_airplane_data
    AirPassengersPanel_train["zeros"] = 0
    AirPassengersPanel_train["large_number"] = 100000
    AirPassengersPanel_train["available_mask"] = 1
    AirPassengersPanel_train = AirPassengersPanel_train[
        ["unique_id", "ds", "zeros", "y", "available_mask", "large_number"]
    ]

    models = [NHITS(h=12, input_size=24, max_steps=50, scaler_type=scaler_type)]
    nf = NeuralForecast(models=models, freq="M")
    nf.fit(AirPassengersPanel_train, val_size=12)
    valid_losses = nf.models[0].valid_trajectories
    assert valid_losses[-1][1] < 40, "Validation loss is too high"
    assert valid_losses[-1][1] > 10, "Validation loss is too low"



@pytest.mark.parametrize("model,expected_error", [
    (NHITS(h=12, input_size=24, max_steps=50, hist_exog_list=["not_included"], scaler_type="robust"),
     "historical exogenous variables not found in input dataset"),
    (NHITS(h=12, input_size=24, max_steps=50, futr_exog_list=["not_included"], scaler_type="robust"),
     "future exogenous variables not found in input dataset"),
    (NHITS(h=12, input_size=24, max_steps=50, stat_exog_list=["not_included"], scaler_type="robust"),
     "static exogenous variables not found in input dataset"),
    (LSTM(h=12, input_size=24, max_steps=50, hist_exog_list=["not_included"], scaler_type="robust"),
     "historical exogenous variables not found in input dataset"),
    (LSTM(h=12, input_size=24, max_steps=50, futr_exog_list=["not_included"], scaler_type="robust"),
     "future exogenous variables not found in input dataset"),
    (LSTM(h=12, input_size=24, max_steps=50, stat_exog_list=["not_included"], scaler_type="robust"),
     "static exogenous variables not found in input dataset"),
])
def test_neural_forecast_missing_variables(setup_airplane_data, model, expected_error):
    """Test that fit fails appropriately when variables are not in dataframe."""
    AirPassengersPanel_train, _ = setup_airplane_data

    nf = NeuralForecast(models=[model], freq="M")
    with pytest.raises(Exception) as exc_info:
        nf.fit(AirPassengersPanel_train)
    assert expected_error in str(exc_info.value)


def test_neural_forecast_unused_variables(setup_airplane_data):
    """Test that passing unused variables in dataframe does not affect forecasts."""
    AirPassengersPanel_train, _ = setup_airplane_data
    AirPassengersPanel_train['zeros'] = 0
    AirPassengersPanel_train['large_number'] = 100000
    AirPassengersPanel_train['available_mask'] = 1
    AirPassengersPanel_train = AirPassengersPanel_train[['unique_id','ds','zeros','y','available_mask','large_number']]

    models = [
        NHITS(
            h=12, input_size=24, max_steps=5, hist_exog_list=["zeros"], scaler_type="robust"
        )
    ]
    nf = NeuralForecast(models=models, freq="M")
    nf.fit(AirPassengersPanel_train)

    # Test that including unused variables doesn't change predictions
    Y_hat1 = nf.predict(
        df=AirPassengersPanel_train[["unique_id", "ds", "y", "zeros", "large_number"]]
    )
    Y_hat2 = nf.predict(df=AirPassengersPanel_train[["unique_id", "ds", "y", "zeros"]])

    pd.testing.assert_frame_equal(
        Y_hat1,
        Y_hat2,
        check_dtype=False,
    )

    models = [
        LSTM(
            h=12, input_size=24, max_steps=5, hist_exog_list=["zeros"], scaler_type="robust"
        )
    ]
    nf = NeuralForecast(models=models, freq="M")
    nf.fit(AirPassengersPanel_train)

    Y_hat1 = nf.predict(
        df=AirPassengersPanel_train[["unique_id", "ds", "y", "zeros", "large_number"]]
    )
    Y_hat2 = nf.predict(df=AirPassengersPanel_train[["unique_id", "ds", "y", "zeros"]])

    pd.testing.assert_frame_equal(
        Y_hat1,
        Y_hat2,
        check_dtype=False,
    )


def test_neural_forecast_pandas_polars_compatibility(setup_airplane_data, setup_airplane_data_polars):
    """Test that NeuralForecast produces identical results with Pandas and Polars dataframes."""
    AirPassengersPanel_train, _ = setup_airplane_data
    AirPassengers_pl, AirPassengersStatic_pl = setup_airplane_data_polars

    AirPassengersStatic = pd.DataFrame(
        {
            "unique_id": ["H196", "H256"],
            "airline1": [0, 1],
            "airline2": [1, 0],
        }
    )

    models = [LSTM(h=12, input_size=24, max_steps=5, scaler_type="robust")]

    # Test with Pandas
    nf_pandas = NeuralForecast(models=models, freq="M")
    nf_pandas.fit(AirPassengersPanel_train, static_df=AirPassengersStatic)
    insample_preds = nf_pandas.predict_insample()
    preds = nf_pandas.predict()
    cv_res = nf_pandas.cross_validation(df=AirPassengersPanel_train, static_df=AirPassengersStatic)

    # Test with Polars
    nf_polars = NeuralForecast(models=models, freq="1mo")
    nf_polars.fit(
        AirPassengers_pl,
        static_df=AirPassengersStatic_pl,
        id_col="uid",
        time_col="time",
        target_col="target",
    )
    insample_preds_pl = nf_polars.predict_insample()
    preds_pl = nf_polars.predict()
    cv_res_pl = nf_polars.cross_validation(
        df=AirPassengers_pl,
        static_df=AirPassengersStatic_pl,
        id_col="uid",
        time_col="time",
        target_col="target",
    )

    # Assert that results are identical between Pandas and Polars
    assert_equal_dfs(preds, preds_pl)
    assert_equal_dfs(insample_preds, insample_preds_pl)
    assert_equal_dfs(cv_res, cv_res_pl)


def test_predict_insample_step_size_polars(setup_airplane_data_polars):
    """Test predict_insample with different step_size values using Polars dataframes."""
    AirPassengers_pl, _ = setup_airplane_data_polars

    h = 12
    train_end = AirPassengers_pl["time"].max()
    sizes = AirPassengers_pl["uid"].value_counts().to_numpy()

    for step_size, test_size in [(7, 0), (9, 0), (7, 5), (9, 5)]:
        models = [NHITS(h=h, input_size=12, max_steps=1)]
        nf = NeuralForecast(models=models, freq="1mo")
        nf.fit(
            AirPassengers_pl,
            id_col="uid",
            time_col="time",
            target_col="target",
        )
        # Note: only apply set_test_size() upon nf.fit(), otherwise it would have set the test_size = 0
        nf.models[0].set_test_size(test_size)

        forecasts = nf.predict_insample(step_size=step_size)
        n_expected_cutoffs = (sizes[0][1] - test_size - nf.h + step_size) // step_size

        # Compare cutoff values
        last_cutoff = (
            train_end - test_size * pd.offsets.MonthEnd() - h * pd.offsets.MonthEnd()
        )
        expected_cutoffs = np.flip(
            np.array(
                [
                    last_cutoff - step_size * i * pd.offsets.MonthEnd()
                    for i in range(n_expected_cutoffs)
                ]
            )
        )
        pl_cutoffs = (
            forecasts.filter(polars.col("uid") == nf.uids[1])
            .select("cutoff")
            .unique(maintain_order=True)
        )
        actual_cutoffs = np.sort(
            np.array([pd.Timestamp(x["cutoff"]) for x in pl_cutoffs.rows(named=True)])
        )
        np.testing.assert_array_equal(
            expected_cutoffs,
            actual_cutoffs,
            err_msg=f"{step_size=},{expected_cutoffs=},{actual_cutoffs=}",
        )

        # Check forecast-points count per series
        cutoffs_by_series = forecasts.group_by(["uid", "cutoff"]).len()
        polars.testing.assert_frame_equal(
            cutoffs_by_series.filter(polars.col("uid") == "Airline1").select(
                ["cutoff", "len"]
            ),
            cutoffs_by_series.filter(polars.col("uid") == "Airline2").select(
                ["cutoff", "len"]
            ),
            check_row_order=False,
        )


# Test if any of the inputs contains NaNs with available_mask = 1, fit shall raise error
# input type is pandas.DataFrame
# available_mask is explicitly given
def test_masks_pandas():
    n_static_features = 2
    n_temporal_features = 4
    temporal_df, static_df = generate_series(
        n_series=4,
        min_length=50,
        max_length=50,
        n_static_features=n_static_features,
        n_temporal_features=n_temporal_features,
        equal_ends=False,
    )
    temporal_df["available_mask"] = 1
    temporal_df.loc[10:20, "available_mask"] = 0
    models = [NHITS(h=12, input_size=24, max_steps=20)]
    nf = NeuralForecast(models=models, freq="D")

    # test case 1: target has NaN values
    test_df1 = temporal_df.copy()
    test_df1.loc[5:7, "y"] = np.nan

    with pytest.raises(ValueError) as exc_info:
        nf.fit(test_df1)
    assert "Found missing values in ['y']" in str(exc_info.value)

    # test case 2: exogenous has NaN values that are correctly flagged with exception
    test_df2 = temporal_df.copy()
    # temporal_0 won't raise ValueError as available_mask = 0
    test_df2.loc[15:18, "temporal_0"] = np.nan
    test_df2.loc[5, "temporal_1"] = np.nan
    test_df2.loc[25, "temporal_2"] = np.nan

    with pytest.raises(ValueError) as exc_info:
        nf.fit(test_df2),
    assert "Found missing values in ['temporal_1', 'temporal_2']" in str(exc_info.value)

    # test case 3: static column has NaN values
    test_df3 = static_df.copy()
    test_df3.loc[3, "static_1"] = np.nan
    with pytest.raises(ValueError) as exc_info:
        nf.fit(temporal_df, static_df=test_df3),
    assert "Found missing values in ['static_1']" in str(exc_info.value)

# Test if any of the inputs contains NaNs with available_mask = 1, fit shall raise error
# input type is polars.Dataframe
# Note that available_mask is not explicitly provided for this test
def test_polars_nans_with_masks():
    pl_df = polars.DataFrame(
        {
            "unique_id": [1] * 50,
            "y": list(range(50)),
            "temporal_0": list(range(100, 150)),
            "temporal_1": list(range(200, 250)),
            "ds": polars.date_range(
                start=date(2022, 1, 1), end=date(2022, 2, 19), interval="1d", eager=True
            ),
        }
    )

    pl_static_df = polars.DataFrame(
        {
            "unique_id": [1],
            "static_0": [1.2],
            "static_1": [10.9],
        }
    )

    models = [NHITS(h=12, input_size=24, max_steps=20)]
    nf = NeuralForecast(models=models, freq="1d")

    # test case 1: target has NaN values
    test_pl_df1 = pl_df.clone()
    test_pl_df1[3, "y"] = np.nan
    test_pl_df1[4, "y"] = None
    with pytest.raises(ValueError) as exc_info:
        nf.fit(test_pl_df1)
    assert "Found missing values in ['y']" in str(exc_info.value)

    # test case 2: exogenous has NaN values that are correctly flagged with exception
    test_pl_df2 = pl_df.clone()
    test_pl_df2[15, "temporal_0"] = np.nan
    test_pl_df2[5, "temporal_1"] = np.nan
    with pytest.raises(ValueError) as exc_info:
        nf.fit(test_pl_df2),

    assert "Found missing values in ['temporal_0', 'temporal_1']" in str(exc_info.value)


    # test case 3: static column has NaN values
    test_pl_df3 = pl_static_df.clone()
    test_pl_df3[0, "static_1"] = np.nan
    with pytest.raises(ValueError) as exc_info:
        nf.fit(pl_df, static_df=test_pl_df3),
    assert "Found missing values in ['static_1']" in str(exc_info.value)

# test customized optimizer behavior such that the user defined optimizer result should differ from default
# tests consider models implemented using different base classes such as BaseWindows, BaseRecurrent, BaseMultivariate
@pytest.mark.parametrize("model", [NHITS])
def test_customized_behavior(setup_airplane_data, model):
    AirPassengersPanel_train, _ = setup_airplane_data
    # default optimizer is based on Adam
    params = {"h": 12, "input_size": 24, "max_steps": 1}

    models = [model(**params)]
    nf = NeuralForecast(models=models, freq="M")
    nf.fit(AirPassengersPanel_train)
    default_optimizer_predict = nf.predict()
    mean = default_optimizer_predict.loc[:, model.__name__].mean()

    # using a customized optimizer
    params.update(
        {
            "optimizer": torch.optim.Adadelta,
            "optimizer_kwargs": {"rho": 0.45},
        }
    )
    models2 = [model(**params)]
    nf2 = NeuralForecast(models=models2, freq="M")
    nf2.fit(AirPassengersPanel_train)
    customized_optimizer_predict = nf2.predict()
    mean2 = customized_optimizer_predict.loc[:, model.__name__].mean()
    assert mean2 != mean


@pytest.mark.parametrize("model", [NHITS])
def test_neural_forecast_invalid_optimizer(model):
    """Test that invalid optimizers raise appropriate exceptions for different model types."""

    # Test that if the user-defined optimizer is not a subclass of torch.optim.Optimizer,
    # it fails with exception. Tests cover different types of base classes such as
    # BaseWindows, BaseRecurrent, BaseMultivariate

    # Test BaseWindows model (NHITS)
    with pytest.raises(Exception) as exc_info:
        model(h=12, input_size=24, max_steps=10, optimizer=torch.nn.Module)
    assert "optimizer is not a valid subclass of torch.optim.Optimizer" in str(exc_info.value)


@pytest.mark.parametrize("model", [NHITS])
def test_neural_forecast_optimizer_lr_warning(setup_airplane_data, model):
    """Test that passing 'lr' parameter in optimizer_kwargs produces expected warnings."""
    AirPassengersPanel_train, _ = setup_airplane_data

    # Test that if we pass "lr" parameter, we expect warning and it ignores the passed in 'lr' parameter
    # Tests consider models implemented using different base classes such as BaseWindows, BaseRecurrent, BaseMultivariate

    params = {
    "h": 12,
        "input_size": 24,
        "max_steps": 1,
        "optimizer": torch.optim.Adadelta,
        "optimizer_kwargs": {"lr": 0.8, "rho": 0.45},
    }

    models = [model(**params)]
    nf = NeuralForecast(models=models, freq="M")

    with warnings.catch_warnings(record=True) as issued_warnings:
        warnings.simplefilter("always", UserWarning)
        nf.fit(AirPassengersPanel_train)
        assert any(
            "ignoring learning rate passed in optimizer_kwargs, using the model's learning rate"
            in str(w.message)
            for w in issued_warnings
        ), f"Expected learning rate warning not found for {model.__name__}"


# test that if we pass "optimizer_kwargs" but not "optimizer", we expect a warning
# tests consider models implemented using different base classes such as BaseWindows, BaseRecurrent, BaseMultivariate
@pytest.mark.parametrize("model", [NHITS])
def test_neuralforecast_optimizer_kwargs(setup_airplane_data, model):
    AirPassengersPanel_train, _ = setup_airplane_data
    params = {
        "h": 12,
        "input_size": 24,
            "max_steps": 1,
            "optimizer_kwargs": {"lr": 0.8, "rho": 0.45},
        }

    models = [model(**params)]
    nf = NeuralForecast(models=models, freq="M")
    with warnings.catch_warnings(record=True) as issued_warnings:
        warnings.simplefilter("always", UserWarning)
        nf.fit(AirPassengersPanel_train)
        assert any(
            "ignoring optimizer_kwargs as the optimizer is not specified"
            in str(w.message)
            for w in issued_warnings
        )


@pytest.mark.parametrize("model", [NHITS])
def test_neuralforecast_customized_lr_scheduler(setup_airplane_data, model):
    """Test customized lr_scheduler behavior to ensure user-defined lr_scheduler results differ from default."""
    AirPassengersPanel_train, _ = setup_airplane_data

    # Test customized lr_scheduler behavior such that the user defined lr_scheduler result should differ from default
    # Tests consider models implemented using different base classes such as BaseWindows, BaseRecurrent, BaseMultivariate

    params = {"h": 12, "input_size": 24, "max_steps": 1}

    # Test with default lr_scheduler
    models = [model(**params)]
    nf = NeuralForecast(models=models, freq="M")
    nf.fit(AirPassengersPanel_train)
    default_optimizer_predict = nf.predict()
    mean = default_optimizer_predict.loc[:, model.__name__].mean()

    # Test with customized lr_scheduler (default is StepLR, using ConstantLR instead)
    params.update(
        {
            "lr_scheduler": torch.optim.lr_scheduler.ConstantLR,
            "lr_scheduler_kwargs": {"factor": 0.78},
        }
    )
    models2 = [model(**params)]
    nf2 = NeuralForecast(models=models2, freq="M")
    nf2.fit(AirPassengersPanel_train)
    customized_optimizer_predict = nf2.predict()
    mean2 = customized_optimizer_predict.loc[:, model.__name__].mean()

    # Assert that customized lr_scheduler produces different results
    assert mean2 != mean, f"Customized lr_scheduler should produce different results for {model.__name__}"

@pytest.mark.parametrize("model", [NHITS])
def test_neuralforecast_invalid_lr_scheduler(model):
    """Test that invalid lr_schedulers raise appropriate exceptions for different model types."""

    # Test that if the user-defined lr_scheduler is not a subclass of torch.optim.lr_scheduler,
    # it fails with exception. Tests cover different types of base classes such as
    # BaseWindows, BaseRecurrent, BaseMultivariate

    # Test BaseWindows model (NHITS)
    with pytest.raises(Exception) as exc_info:
        model(h=12, input_size=24, max_steps=10, lr_scheduler=torch.nn.Module)
    assert "lr_scheduler is not a valid subclass of torch.optim.lr_scheduler.LRScheduler" in str(exc_info.value)

@pytest.mark.parametrize("model", [NHITS])
def test_neuralforecast_lr_scheduler_optimizer_warning(setup_airplane_data, model):
    """Test that passing 'optimizer' parameter in lr_scheduler_kwargs produces expected warnings."""
    AirPassengersPanel_train, _ = setup_airplane_data

    # Test that if we pass in "optimizer" parameter in lr_scheduler_kwargs, we expect warning and it ignores them
    # Tests consider models implemented using different base classes such as BaseWindows, BaseRecurrent, BaseMultivariate

    params = {
        "h": 12,
        "input_size": 24,
        "max_steps": 1,
        "lr_scheduler": torch.optim.lr_scheduler.ConstantLR,
        "lr_scheduler_kwargs": {"optimizer": torch.optim.Adadelta, "factor": 0.22},
    }

    models = [model(**params)]
    nf = NeuralForecast(models=models, freq="M")

    with warnings.catch_warnings(record=True) as issued_warnings:
        warnings.simplefilter("always", UserWarning)
        nf.fit(AirPassengersPanel_train)
        assert any(
            "ignoring optimizer passed in lr_scheduler_kwargs, using the model's optimizer"
            in str(w.message)
            for w in issued_warnings
        ), f"Expected optimizer warning not found for {model.__name__}"


@pytest.mark.parametrize("model", [NHITS])
def test_neuralforecast_lr_scheduler_kwargs_warning(setup_airplane_data, model):
    """Test that passing lr_scheduler_kwargs without lr_scheduler produces expected warnings."""
    AirPassengersPanel_train, _ = setup_airplane_data

    # Test that if we pass in "lr_scheduler_kwargs" but not "lr_scheduler", we expect a warning
    # Tests consider models implemented using different base classes such as BaseWindows, BaseRecurrent, BaseMultivariate
    params = {
        "h": 12,
        "input_size": 24,
        "max_steps": 1,
        "lr_scheduler_kwargs": {"optimizer": torch.optim.Adadelta, "factor": 0.22},
    }

    models = [model(**params)]
    nf = NeuralForecast(models=models, freq="M")

    with warnings.catch_warnings(record=True) as issued_warnings:
        warnings.simplefilter("always", UserWarning)
        nf.fit(AirPassengersPanel_train)
        assert any(
            "ignoring lr_scheduler_kwargs as the lr_scheduler is not specified"
            in str(w.message)
            for w in issued_warnings
        ), f"Expected lr_scheduler_kwargs warning not found for {model.__name__}"


@pytest.mark.parametrize("model", [NHITS])
def test_neuralforecast_conformal_prediction(setup_airplane_data, setup_airplane_data_polars, model):
    """Test conformal prediction, method=conformal_distribution."""
    AirPassengersPanel_train, AirPassengersPanel_test = setup_airplane_data
    AirPassengers_pl, AirPassengersStatic_pl = setup_airplane_data_polars

    prediction_intervals = PredictionIntervals()


    params = {"h": 12, "input_size": 24, "max_steps": 1}
    models = [model(**params)]


    nf = NeuralForecast(models=models, freq="M")
    nf.fit(AirPassengersPanel_train, prediction_intervals=prediction_intervals)
    preds = nf.predict(futr_df=AirPassengersPanel_test, level=[90])

    assert "unique_id" in preds.columns
    assert "ds" in preds.columns
    assert any(col.startswith(model.__name__) for col in preds.columns)

    # test conformal prediction works for polar dataframe
    nf = NeuralForecast(models=models, freq="1mo")
    nf.fit(
        AirPassengers_pl,
        prediction_intervals=prediction_intervals,
        time_col="time",
        id_col="uid",
        target_col="target",
    )
    preds = nf.predict(level=[90])
    assert "uid" in preds.columns
    assert any(col.startswith(model.__name__) for col in preds.columns)

def test_neuralforecast_cross_validation_conformal_prediction(setup_airplane_data):
    """Test cross validation can support conformal prediction with proper refit parameter."""
    AirPassengersPanel_train, _ = setup_airplane_data

    # Test cross validation can support conformal prediction
    prediction_intervals = PredictionIntervals()

    # Test that refit=False with conformal predictions raises an error
    nf = NeuralForecast(models=[NHITS(h=12, input_size=24, max_steps=1)], freq="M")
    with pytest.raises(Exception) as exc_info:
        nf.cross_validation(
            AirPassengersPanel_train,
            prediction_intervals=prediction_intervals,
            level=[30, 70]
        )
    assert "Passing prediction_intervals is only supported with refit=True." in str(exc_info.value)

    # Test that refit=True produces conformal predictions outputs
    cv2 = nf.cross_validation(
        AirPassengersPanel_train,
        prediction_intervals=prediction_intervals,
        refit=True,
        level=[30, 70],
    )
    assert all([col in cv2.columns for col in ["NHITS-lo-30", "NHITS-hi-30"]]), \
        "Expected conformal prediction columns not found in cross validation results"

@pytest.mark.parametrize("model", [NHITS])
def test_neuralforecast_quantile_level_prediction(setup_airplane_data, model):
    """Test quantile and level argument in predict for different models and errors."""
    AirPassengersPanel_train, AirPassengersPanel_test = setup_airplane_data

    prediction_intervals = PredictionIntervals(method="conformal_error")

    # Create a simple model with MAE loss and no scaler to avoid MPS compatibility issues
    params = {"h": 12, "input_size": 24, "max_steps": 1, "loss": MAE(), "scaler_type": None}

    model = model(**params)
    nf = NeuralForecast(models=[model], freq="M")
    nf.fit(AirPassengersPanel_train, prediction_intervals=prediction_intervals)

    # Test default prediction
    preds = nf.predict(futr_df=AirPassengersPanel_test)
    assert "unique_id" in preds.columns
    assert "ds" in preds.columns
    assert any(col.startswith(str(model)) for col in preds.columns)

    # Test quantile prediction (with conformal prediction)
    preds_quantile = nf.predict(futr_df=AirPassengersPanel_test, quantiles=[0.2, 0.3])
    assert "unique_id" in preds_quantile.columns
    assert "ds" in preds_quantile.columns
    # Should have quantile columns for conformal predictions
    quantile_cols = [col for col in preds_quantile.columns if "-ql0.2" in col or "-ql0.3" in col]
    assert len(quantile_cols) > 0

    # Test level prediction (with conformal prediction)
    preds_level = nf.predict(futr_df=AirPassengersPanel_test, level=[80, 90])
    assert "unique_id" in preds_level.columns
    assert "ds" in preds_level.columns
    # Should have level columns for conformal predictions
    level_cols = [col for col in preds_level.columns if "-lo-" in col or "-hi-" in col]
    assert len(level_cols) > 0

@pytest.mark.parametrize("explainer", [ExplainerEnum.IntegratedGradients, ExplainerEnum.InputXGradient])
@pytest.mark.parametrize("use_polars", [True, False])
@pytest.mark.parametrize("horizons", [list(range(12)), [0, 5]])
@pytest.mark.parametrize("recursive_horizon", [True, False])
def test_explainability(explainer, use_polars, horizons, recursive_horizon):
    "Test that explanations are returned or skipped depending on model and configuration"
    Y_train_df = AirPassengersPanel[AirPassengersPanel['ds'] < AirPassengersPanel['ds'].values[-12]].reset_index(drop=True)
    Y_test_df = AirPassengersPanel[AirPassengersPanel['ds'] >= AirPassengersPanel['ds'].values[-12]].reset_index(drop=True)
    futr_df = Y_test_df.drop(columns=["y", "y_[lag12]"])
    static_df = AirPassengersStatic.drop(columns=["airline2"])

    h = 12
    h_train = h
    input_size = 2*h
    n_series = Y_train_df["unique_id"].nunique()
    n_stat_exog = len(static_df.drop(columns="unique_id").columns)

    if recursive_horizon:
        # For recursive test: train with h=6, predict with h=12
        h_train = 6
        input_size = 6

    base_config = {
        "h": h_train,
        "input_size": input_size,
        "scaler_type": "robust",
        "max_steps": 2,
        "accelerator": "cpu",
    }

    models = [
        NHITS(**base_config),
        NHITS(
            **base_config,
            hist_exog_list=["y_[lag12]"],
            futr_exog_list=["trend"],
            stat_exog_list=['airline1'],
            alias="NHITS-exog",
        ),
        NHITS(
            **base_config,
            loss=MQLoss(level=[80]),
            alias="NHITS-MQLoss"
        ),
        NHITS( # Gets skipped because of DistributionLoss
            **base_config,
            loss=DistributionLoss(distribution="Normal", level=[80]),
            alias="NHITS-DistributionLoss"
        ),
        LSTM(
            **base_config,
            recurrent=False,
        ),
        LSTM( # Gets skiped when explainer is IntegratedGradients
            **base_config,
            recurrent=True,
            alias="LSTM-recurrent"
        ),
        TSMixer( # Gets skipped because it's multivariate
            **base_config,
            n_series=2,
        )
    ]
    if recursive_horizon:
        recursive_config = {
            "h": h_train,
            "input_size": input_size,
            "scaler_type": "robust",
            "max_steps": 2,
            "accelerator": "cpu",
        }
        models = [
            NHITS(
                **recursive_config,
                futr_exog_list=["trend"],
                stat_exog_list=['airline1'],
            ),
            LSTM(
                **recursive_config,
                recurrent=False,
            ),
        ]

    freq="ME"
    if use_polars:
        Y_train_df = polars.from_pandas(Y_train_df)
        static_df = polars.from_pandas(static_df)
        futr_df = polars.from_pandas(futr_df)
        freq="1mo"
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=Y_train_df, static_df=static_df)

    outputs = [0]
    preds_df, explanations = nf.explain(
        outputs=outputs, # Get only 1 ouput
        horizons=horizons, # Get all horizons
        static_df=static_df,
        futr_df=futr_df,
        h=h, 
        explainer=explainer
    )

    # Determine which models should have explanations
    expected_explanations = set()
    skipped_models = set()
    
    for model in models:
        model_name = model.alias or model.__class__.__name__
        
        # Check skip conditions
        if model.MULTIVARIATE:
            skipped_models.add(model_name)
        elif hasattr(model.loss, 'is_distribution_output') and model.loss.is_distribution_output:
            skipped_models.add(model_name)
        elif model.RECURRENT and explainer == ExplainerEnum.IntegratedGradients:
            skipped_models.add(model_name)
        else:
            expected_explanations.add(model_name)
    
    assert set(explanations.keys()) == expected_explanations, f"Expected {expected_explanations}, got {set(explanations.keys())}"

    # Verify skipped models have no explanations
    for model_name in skipped_models:
        assert model_name not in explanations

    # Verify explained models have predictions
    for model_name in expected_explanations:
        assert any(model_name in col for col in preds_df.columns), f"Model {model_name} should have predictions but doesn't"

    # Test explained model
    for model_name in expected_explanations:
        expl = explanations[model_name]

        # Basic structure tests
        assert expl["insample"] is not None
        expected_input_size = input_size
        if model_name == "LSTM-recurrent":
            expected_input_size = input_size + h_train
        
        batch_size = n_series
        n_series_ = 1
        expected_insample_shape = (
            batch_size,           # batch_size
            len(horizons),        # horizons
            n_series_,            # n_series (1 for univariate)
            len(outputs),         # n_outputs
            expected_input_size,  # n_input_steps
            2                     # (y_attr, mask_attr)
        )
        assert expl["insample"].shape == expected_insample_shape
        
        # Test additivity for additive explainers
        if explainer == "IntegratedGradients":
            expected_baseline_shape = (
                batch_size,           # batch_size
                len(horizons),        # horizons
                n_series_,            # n_series (1 for univariate)
                len(outputs)          # n_outputs
            )
            assert expl["baseline_predictions"] is not None
            assert expl["baseline_predictions"].shape == expected_baseline_shape
            _test_model_additivity(preds_df, expl, model_name, use_polars, n_series, h, horizons)
        else:
            assert expl["baseline_predictions"] is None
            
        
        # Check exogenous if model has them
        model = next(m for m in models if (m.alias or m.__class__.__name__) == model_name)
        if model.futr_exog_list:
            if recursive_horizon:
                futr_temporal_size = model.input_size + model.h
            else:
                futr_temporal_size = model.input_size + h
            expected_futr_shape = (
                batch_size,                 # batch size
                len(horizons),              # horizons
                n_series_,                  # n_series (1 for univariate)
                len(outputs),               # n_outputs
                futr_temporal_size,         # n_input_steps (past + future)
                len(model.futr_exog_list),  # number of features
            )
            assert expl["futr_exog"] is not None
            assert expl["futr_exog"].shape == expected_futr_shape
        if model.hist_exog_list:
            expected_hist_shape = (
                batch_size,                # batch size
                len(horizons),             # horizons
                n_series_,                 # n_series (1 for univariate)
                len(outputs),              # n_outputs
                model.input_size,          # n_input_steps (past)
                len(model.hist_exog_list), # number of features
            )
            assert expl["hist_exog"] is not None
            assert expl["hist_exog"].shape == expected_hist_shape
        if model.stat_exog_list:
            expected_stat_shape = (
                batch_size,    # batch size
                len(horizons), # horizons
                n_series_,     # n_series (1 for univariate)
                len(outputs),  # n_outputs
                n_stat_exog,   # number of features
            )
            assert expl["stat_exog"] is not None
            assert expl["stat_exog"].shape == expected_stat_shape

def _test_model_additivity(preds_df, expl, model_name, use_polars, n_series, h, horizons):
    """Test if sum of attributions and baseline predictions equal forecasts"""
    pred_col = [col for col in preds_df.columns if col.startswith(model_name)][0]
    if use_polars:
        preds = preds_df[pred_col].to_numpy()
    else:
        preds = preds_df[pred_col].values

    # Sum over n_outputs (-1) and n_series (-2), shape (batch_size, h, n_series, n_outputs) -> (batch_size, h)
    baseline = expl["baseline_predictions"].sum(dim=(-1, -2))  
    
    # Sum over channels (-1), input_sequence (-2), n_outputs (-3), n_series (-4)
    sum_dims = (-1, -2, -3, -4) 
    insample_attr = expl["insample"].sum(dim=sum_dims)
    futr_attr = expl["futr_exog"].sum(dim=sum_dims) if not isinstance(expl["futr_exog"], list) else 0
    hist_attr = expl["hist_exog"].sum(dim=sum_dims) if not isinstance(expl["hist_exog"], list) else 0
    
    # Static doesn't have the input_sequence dimension, as it is static across that dimension
    sum_dims = (-1, -2, -3)  # Sum over channels (-1), n_outputs (-2), n_series (-3)
    stat_attr = expl["stat_exog"].sum(dim=sum_dims) if not isinstance(expl["stat_exog"], list) else 0
    
    total_attr = insample_attr + futr_attr + hist_attr + stat_attr
    pred_from_attr = baseline + total_attr  # Shape: (n_series, h)

    preds = preds.reshape(n_series, h)
    preds = preds[:, horizons]

    np.testing.assert_allclose(
        pred_from_attr.cpu().numpy(),
        preds,
        rtol=1e-3,
        err_msg="Attribution predictions do not match model predictions"
    )
