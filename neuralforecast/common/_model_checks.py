


__all__ = ['seed', 'test_size', 'FREQ', 'N_SERIES_1', 'df', 'max_ds', 'Y_TRAIN_DF_1', 'Y_TEST_DF_1', 'N_SERIES_2', 'Y_TRAIN_DF_2',
           'Y_TEST_DF_2', 'N_SERIES_3', 'STATIC_3', 'Y_TRAIN_DF_3', 'Y_TEST_DF_3', 'N_SERIES_4', 'STATIC_4',
           'Y_TRAIN_DF_4', 'Y_TEST_DF_4', 'check_loss_functions', 'check_airpassengers', 'check_model']


import pandas as pd

import neuralforecast.losses.pytorch as losses
from neuralforecast.utils import (
    AirPassengersPanel,
    AirPassengersStatic,
    generate_series,
)

from .. import NeuralForecast

seed = 0
test_size = 14
FREQ = "D"

# 1 series, no exogenous
N_SERIES_1 = 1
df = generate_series(
    n_series=N_SERIES_1, seed=seed, freq=FREQ, equal_ends=True, max_length=75
)
max_ds = df.ds.max() - pd.Timedelta(test_size, FREQ)
Y_TRAIN_DF_1 = df[df.ds < max_ds]
Y_TEST_DF_1 = df[df.ds >= max_ds]

# 5 series, no exogenous
N_SERIES_2 = 5
df = generate_series(
    n_series=N_SERIES_2, seed=seed, freq=FREQ, equal_ends=True, max_length=75
)
max_ds = df.ds.max() - pd.Timedelta(test_size, FREQ)
Y_TRAIN_DF_2 = df[df.ds < max_ds]
Y_TEST_DF_2 = df[df.ds >= max_ds]

# 1 series, with static and temporal exogenous
N_SERIES_3 = 1
df, STATIC_3 = generate_series(
    n_series=N_SERIES_3,
    n_static_features=2,
    n_temporal_features=2,
    seed=seed,
    freq=FREQ,
    equal_ends=True,
    max_length=75,
)
max_ds = df.ds.max() - pd.Timedelta(test_size, FREQ)
Y_TRAIN_DF_3 = df[df.ds < max_ds]
Y_TEST_DF_3 = df[df.ds >= max_ds]

# 5 series, with static and temporal exogenous
N_SERIES_4 = 5
df, STATIC_4 = generate_series(
    n_series=N_SERIES_4,
    n_static_features=2,
    n_temporal_features=2,
    seed=seed,
    freq=FREQ,
    equal_ends=True,
)
max_ds = df.ds.max() - pd.Timedelta(test_size, FREQ)
Y_TRAIN_DF_4 = df[df.ds < max_ds]
Y_TEST_DF_4 = df[df.ds >= max_ds]


# Generic test for a given config for a model
def _run_model_tests(model_class, config):
    if model_class.RECURRENT:
        config["inference_input_size"] = config["input_size"]

    # DF_1
    if model_class.MULTIVARIATE:
        config["n_series"] = N_SERIES_1
    if isinstance(config["loss"], losses.relMSE):
        config["loss"].y_train = Y_TRAIN_DF_1["y"].values
    if isinstance(config["valid_loss"], losses.relMSE):
        config["valid_loss"].y_train = Y_TRAIN_DF_1["y"].values

    model = model_class(**config)
    fcst = NeuralForecast(models=[model], freq=FREQ)
    fcst.fit(df=Y_TRAIN_DF_1, val_size=24)
    _ = fcst.predict(futr_df=Y_TEST_DF_1)
    # DF_2
    if model_class.MULTIVARIATE:
        config["n_series"] = N_SERIES_2
    if isinstance(config["loss"], losses.relMSE):
        config["loss"].y_train = Y_TRAIN_DF_2["y"].values
    if isinstance(config["valid_loss"], losses.relMSE):
        config["valid_loss"].y_train = Y_TRAIN_DF_2["y"].values
    model = model_class(**config)
    fcst = NeuralForecast(models=[model], freq=FREQ)
    fcst.fit(df=Y_TRAIN_DF_2, val_size=24)
    _ = fcst.predict(futr_df=Y_TEST_DF_2)

    if model.EXOGENOUS_STAT and model.EXOGENOUS_FUTR:
        # DF_3
        if model_class.MULTIVARIATE:
            config["n_series"] = N_SERIES_3
        if isinstance(config["loss"], losses.relMSE):
            config["loss"].y_train = Y_TRAIN_DF_3["y"].values
        if isinstance(config["valid_loss"], losses.relMSE):
            config["valid_loss"].y_train = Y_TRAIN_DF_3["y"].values
        model = model_class(**config)
        fcst = NeuralForecast(models=[model], freq=FREQ)
        fcst.fit(df=Y_TRAIN_DF_3, static_df=STATIC_3, val_size=24)
        _ = fcst.predict(futr_df=Y_TEST_DF_3)

        # DF_4
        if model_class.MULTIVARIATE:
            config["n_series"] = N_SERIES_4
        if isinstance(config["loss"], losses.relMSE):
            config["loss"].y_train = Y_TRAIN_DF_4["y"].values
        if isinstance(config["valid_loss"], losses.relMSE):
            config["valid_loss"].y_train = Y_TRAIN_DF_4["y"].values
        model = model_class(**config)
        fcst = NeuralForecast(models=[model], freq=FREQ)
        fcst.fit(df=Y_TRAIN_DF_4, static_df=STATIC_4, val_size=24)
        _ = fcst.predict(futr_df=Y_TEST_DF_4)


# Tests a model against every loss function
def check_loss_functions(model_class):
    loss_list = [
        losses.MAE(),
        losses.MSE(),
        losses.RMSE(),
        losses.MAPE(),
        losses.SMAPE(),
        losses.MASE(seasonality=7),
        losses.QuantileLoss(q=0.5),
        losses.MQLoss(),
        losses.IQLoss(),
        losses.HuberIQLoss(),
        losses.DistributionLoss("Normal"),
        losses.DistributionLoss("StudentT"),
        losses.DistributionLoss("Poisson"),
        losses.DistributionLoss("NegativeBinomial"),
        losses.DistributionLoss("Tweedie", rho=1.5),
        losses.DistributionLoss("ISQF"),
        losses.PMM(),
        losses.PMM(weighted=True),
        losses.GMM(),
        losses.GMM(weighted=True),
        losses.NBMM(),
        losses.NBMM(weighted=True),
        losses.HuberLoss(),
        losses.TukeyLoss(),
        losses.HuberQLoss(q=0.5),
        losses.HuberMQLoss(),
    ]
    for loss in loss_list:
        test_name = f"{model_class.__name__}: checking {loss._get_name()}"
        print(f"{test_name}")
        config = {
            "max_steps": 2,
            "h": 7,
            "input_size": 28,
            "loss": loss,
            "valid_loss": None,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "val_check_steps": 2,
            "batch_size": 8,
            "windows_batch_size": 8,
            "valid_batch_size": 8,
            "inference_windows_batch_size": 8,
        }
        try:
            _run_model_tests(model_class, config)
        except RuntimeError:
            raise Exception(f"{test_name} failed.")
        except Exception:
            print(f"{test_name} skipped on raised Exception.")
            pass


# Tests a model against the AirPassengers dataset
def check_airpassengers(model_class):
    h = 12
    Y_train_df = AirPassengersPanel[
        AirPassengersPanel.ds < AirPassengersPanel["ds"].values[-h]
    ]  # 132 train
    Y_test_df = AirPassengersPanel[
        AirPassengersPanel.ds >= AirPassengersPanel["ds"].values[-h]
    ].reset_index(
        drop=True
    )  # 12 test
    config = {
        "max_steps": 2,
        "h": h,
        "input_size": 24,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "val_check_steps": 2,
        "batch_size": 8,
        "windows_batch_size": 8,
        "valid_batch_size": 8,
        "inference_windows_batch_size": 8,
    }

    if model_class.MULTIVARIATE:
        config["n_series"] = Y_train_df["unique_id"].nunique()
    # Normal forecast
    fcst = NeuralForecast(models=[model_class(**config)], freq="M")
    fcst.fit(df=Y_train_df, static_df=AirPassengersStatic)
    fcst.predict(futr_df=Y_test_df)

    # Longer horizon forecast
    fcst.predict(h=(h + 3))

    # Cross-validation
    fcst = NeuralForecast(models=[model_class(**config)], freq="M")
    fcst.cross_validation(
        df=AirPassengersPanel, static_df=AirPassengersStatic, n_windows=2, step_size=12
    )


# Add unit test functions to this function
def check_model(model_class, checks=["losses", "airpassengers"]):
    """
    Check model with various tests. Options for checks are:
        "losses": test the model against all loss functions
        "airpassengers": test the model against the airpassengers dataset for forecasting and cross-validation

    Args:
        model_class (nn.Module): Model class to check.
        checks (list): List of checks to run.

    Returns:
        None
    """
    if "losses" in checks:
        check_loss_functions(model_class)
    if "airpassengers" in checks:
        try:
            check_airpassengers(model_class)
        except RuntimeError:
            raise Exception(
                f"{model_class.__name__}: AirPassengers forecast test failed."
            )
