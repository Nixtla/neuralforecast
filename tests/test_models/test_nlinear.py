import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNLinear, NLinear
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model
from neuralforecast.common.enums import TimeSeriesDatasetEnum

from .test_helpers import check_args
from ray import tune


def test_nlinear_model(suppress_warnings):
    check_model(NLinear, ["airpassengers"])


def test_autonlinear_longer_horizon(longer_horizon_test):
    config = {
        "input_size": tune.choice(
            [longer_horizon_test.input_size, longer_horizon_test.input_size * 2]
        ),
        "h": None,
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "max_steps": tune.quniform(lower=2, upper=6, q=2),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(lower=1, upper=20),
    }

    fcst = NeuralForecast(
        models=[
            AutoNLinear(
                h=longer_horizon_test.h,
                config=config,
                num_samples=3,
            )
        ],
        freq="ME",
    )
    fcst.fit(df=longer_horizon_test.train_df)
    longer_h_fcst_df = fcst.predict(
        futr_df=longer_horizon_test.test_df, h=longer_horizon_test.longer_h
    )
    group_cnt = longer_h_fcst_df.groupby(TimeSeriesDatasetEnum.UniqueId)[
        "AutoNLinear"
    ].count()
    expected = pd.Series(
        data=[longer_horizon_test.longer_h] * 2,
        index=["Airline1", "Airline2"],
        name="AutoNLinear",
    )
    expected.index.name = TimeSeriesDatasetEnum.UniqueId
    pd.testing.assert_series_equal(group_cnt, expected)


def test_autonlinear(setup_dataset):
    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoNLinear, exclude_args=["cls_model"])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoNLinear.get_default_config(h=12, backend="optuna")

    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update({"max_steps": 2, "val_check_steps": 1, "input_size": 12})
        return config

    model = AutoNLinear(
        h=12, config=my_config_new, backend="optuna", num_samples=1, cpus=1
    )
    assert model.config(MockTrial())["h"] == 12
    model.fit(dataset=setup_dataset)

    # Unit test for situation: Ray with updated default config
    my_config = AutoNLinear.get_default_config(h=12, backend="ray")
    my_config["max_steps"] = 2
    my_config["val_check_steps"] = 1
    my_config["input_size"] = 12
    model = AutoNLinear(h=12, config=my_config, backend="ray", num_samples=1, cpus=1)
    model.fit(dataset=setup_dataset)
