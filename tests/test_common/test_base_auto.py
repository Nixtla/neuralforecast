import logging
import warnings

import optuna
import pandas as pd
import pytest
import pytorch_lightning as pl
from ray import tune

from neuralforecast.common._base_auto import BaseAuto
from neuralforecast.losses.numpy import mae
from neuralforecast.losses.pytorch import MAE, MSE
from neuralforecast.models.mlp import MLP
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengersDF as Y_df

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

@pytest.fixture
def setup_module():
    """Setup for the test module."""
    Y_train_df = Y_df[Y_df.ds <= "1959-12-31"]  # 132 train
    Y_test_df = Y_df[Y_df.ds > "1959-12-31"]  # 12 test

    dataset, *_ = TimeSeriesDataset.from_df(Y_train_df)
    return dataset, Y_train_df,Y_test_df


class RayLogLossesCallback(tune.Callback):
    def on_trial_complete(self, iteration, trials, trial, **info):
        result = trial.last_result
        print(40 * "-" + "Trial finished" + 40 * "-")
        print(
            f"Train loss: {result['train_loss']:.2f}. Valid loss: {result['loss']:.2f}"
        )
        print(80 * "-")

def test_ray_tune(setup_module):
    dataset, _, Y_test_df = setup_module
    config = {
        "hidden_size": tune.choice([512]),
        "num_layers": tune.choice([3, 4]),
        "input_size": 12,
        "max_steps": 10,
    "val_check_steps": 5,
    }
    auto = BaseAuto(
        h=12,
        loss=MAE(),
        valid_loss=MSE(),
        cls_model=MLP,
        config=config,
        num_samples=2,
        cpus=1,
        gpus=0,
        callbacks=[RayLogLossesCallback()],
    )
    auto.fit(dataset=dataset)
    y_hat = auto.predict(dataset=dataset)
    assert mae(Y_test_df["y"].values, y_hat[:, 0]) < 200


def config_f(trial):
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [512]),
        "num_layers": trial.suggest_categorical("num_layers", [3, 4]),
        "input_size": 12,
        "max_steps": 10,
        "val_check_steps": 5,
    }


class OptunaLogLossesCallback:
    def __call__(self, study, trial):
        metrics = trial.user_attrs["METRICS"]
        print(40 * "-" + "Trial finished" + 40 * "-")
        print(
            f"Train loss: {metrics['train_loss']:.2f}. Valid loss: {metrics['loss']:.2f}"
        )
        print(80 * "-")


def test_optuna_tune(setup_module):
    dataset, Y_train_df, Y_test_df = setup_module
    auto2 = BaseAuto(
        h=12,
        loss=MAE(),
        valid_loss=MSE(),
        cls_model=MLP,
    config=config_f,
    search_alg=optuna.samplers.RandomSampler(),
    num_samples=2,
    backend="optuna",
    callbacks=[OptunaLogLossesCallback()],
)
    auto2.fit(dataset=dataset)
    assert isinstance(auto2.results, optuna.Study)
    y_hat2 = auto2.predict(dataset=dataset)
    assert mae(Y_test_df["y"].values, y_hat2[:, 0]) < 200
    Y_test_df["AutoMLP"] = y_hat2

    pd.concat([Y_train_df, Y_test_df]).drop("unique_id", axis=1).set_index("ds").plot()

@pytest.fixture
def setup_config():
    return {
        "hidden_size": tune.choice([512]),
        "num_layers": tune.choice([3, 4]),
        "input_size": 12,
        "max_steps": 1,
        "val_check_steps": 1,
    }


def test_instantiation(setup_config):
    config = setup_config

    # Test instantiation
    auto = BaseAuto(
        h=12,
        loss=MAE(),
        valid_loss=MSE(),
        cls_model=MLP,
        config=config,
        num_samples=2,
        cpus=1,
        gpus=0,
    )
    assert str(type(auto.loss)) == "<class 'neuralforecast.losses.pytorch.MAE'>"
    assert str(type(auto.valid_loss)) == "<class 'neuralforecast.losses.pytorch.MSE'>"

def test_validation_default(setup_config):
    auto = BaseAuto(
        h=12,
        loss=MSE(),
        valid_loss=None,
        cls_model=MLP,
        config=setup_config,
        num_samples=2,
        cpus=1,
        gpus=0,
    )
    assert str(type(auto.loss)) == "<class 'neuralforecast.losses.pytorch.MSE'>"
    assert str(type(auto.valid_loss)) == "<class 'neuralforecast.losses.pytorch.MSE'>"
