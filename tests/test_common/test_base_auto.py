import logging
import warnings
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd
import pytest
from ray import tune

from neuralforecast.auto import AutoMLP
from neuralforecast.common._base_auto import BaseAuto, OptunaOptions, RayOptions
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
        ray_options=RayOptions(cpus=1, gpus=0),
        callbacks=[RayLogLossesCallback()],
    )
    auto.fit(dataset=dataset)
    y_hat = auto.predict(dataset=dataset)
    assert np.mean(np.abs(Y_test_df["y"].values - y_hat[:, 0])) < 200


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
    assert np.mean(np.abs(Y_test_df["y"].values - y_hat2[:, 0])) < 200
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
        ray_options=RayOptions(cpus=1, gpus=0),
    )
    assert str(type(auto.loss)) == "<class 'neuralforecast.losses.pytorch.MAE'>"
    assert str(type(auto.valid_loss)) == "<class 'neuralforecast.losses.pytorch.MSE'>"


def test_ray_gpus_default_is_single_gpu_per_trial(setup_config):
    # #1291: reserving every GPU per trial makes the model launch DDP inside the
    # Ray actor and crash. Default to one GPU per trial; respect explicit overrides.
    with patch(
        "neuralforecast.common._base_auto.torch.cuda.device_count", return_value=4
    ):
        auto = BaseAuto(
            h=12,
            loss=MAE(),
            valid_loss=MSE(),
            cls_model=MLP,
            config=setup_config,
            num_samples=1,
            backend="ray",
        )
        assert auto.gpus == 1  # not 4

        override = BaseAuto(
            h=12,
            loss=MAE(),
            valid_loss=MSE(),
            cls_model=MLP,
            config=setup_config,
            num_samples=1,
            backend="ray",
            ray_options=RayOptions(gpus=2),
        )
        assert override.gpus == 2  # explicit value untouched

def test_validation_default(setup_config):
    auto = BaseAuto(
        h=12,
        loss=MSE(),
        valid_loss=None,
        cls_model=MLP,
        config=setup_config,
        num_samples=2,
        ray_options=RayOptions(cpus=1, gpus=0),
    )
    assert str(type(auto.loss)) == "<class 'neuralforecast.losses.pytorch.MSE'>"
    assert str(type(auto.valid_loss)) == "<class 'neuralforecast.losses.pytorch.MSE'>"


def test_config_missing_required_param_raises():
    # A config that omits a required no-default arg of the underlying model used to fail deep
    # inside ray with an opaque "No best trial found" error. Now it must fail fast
    # at construction time with a message naming the missing key.
    with pytest.raises(ValueError, match="input_size"):
        AutoMLP(
            h=4,
            config={
                "max_steps": tune.choice([5]),
                "random_seed": tune.choice([0]),
            },
            backend="ray",
        )

    def config_fn(trial):
        return {
            "max_steps": trial.suggest_categorical("max_steps", [5]),
            "random_seed": trial.suggest_categorical("random_seed", [0]),
        }

    with pytest.raises(ValueError, match="input_size"):
        AutoMLP(h=4, config=config_fn, backend="optuna")


def test_default_config_copy_is_valid_user_config(setup_module):
    # https://github.com/Nixtla/neuralforecast/issues/571
    # The default configs express the input size as `input_size_multiplier`,
    # which used to be translated into `input_size` only when `config=None`,
    # making configs built on top of `default_config` invalid.
    dataset, _, _ = setup_module
    config = AutoMLP.default_config.copy()
    # tweak the search space, keeping it small so the test runs fast
    config["input_size_multiplier"] = [1, 2]
    config["hidden_size"] = tune.choice([8])
    config["num_layers"] = 2
    config["max_steps"] = 1
    config["val_check_steps"] = 1
    auto = AutoMLP(
        h=12,
        config=config,
        num_samples=1,
        ray_options=RayOptions(cpus=1, gpus=0),
    )
    assert "input_size_multiplier" not in auto.config
    assert "input_size" in auto.config
    auto.fit(dataset=dataset)
    y_hat = auto.predict(dataset=dataset)
    assert y_hat.shape[0] > 0


def test_optuna_config_translates_input_size_multiplier(setup_module):
    # https://github.com/Nixtla/neuralforecast/issues/571 (optuna variant)
    dataset, _, _ = setup_module

    def config_multiplier(trial):
        return {
            "input_size_multiplier": trial.suggest_categorical(
                "input_size_multiplier", [1, 2]
            ),
            "hidden_size": trial.suggest_categorical("hidden_size", [8]),
            "num_layers": 2,
            "max_steps": 1,
            "val_check_steps": 1,
        }

    auto = AutoMLP(
        h=12,
        config=config_multiplier,
        backend="optuna",
        search_alg=optuna.samplers.RandomSampler(seed=0),
        num_samples=1,
    )
    auto.fit(dataset=dataset)
    y_hat = auto.predict(dataset=dataset)
    assert y_hat.shape[0] > 0


def test_ray_time_budget(setup_module):
    dataset, _, _ = setup_module
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
        num_samples=1,
        time_budget=60,
        ray_options=RayOptions(cpus=1, gpus=0),
    )
    auto.fit(dataset=dataset)
    y_hat = auto.predict(dataset=dataset)
    assert y_hat.shape[0] > 0


def test_ray_run_config_storage_path(setup_module, tmp_path):
    dataset, _, _ = setup_module
    from ray import air

    storage_path = tmp_path / "ray_results"
    config = {
        "hidden_size": tune.choice([512]),
        "num_layers": tune.choice([3, 4]),
        "input_size": 12,
        "max_steps": 1,
        "val_check_steps": 1,
    }
    auto = BaseAuto(
        h=12,
        loss=MAE(),
        valid_loss=MSE(),
        cls_model=MLP,
        config=config,
        num_samples=1,
        ray_options=RayOptions(
            run_config=air.RunConfig(storage_path=str(storage_path), name="nf_test"),
            cpus=1,
            gpus=0,
        ),
    )
    auto.fit(dataset=dataset)
    assert (storage_path / "nf_test").is_dir()


def test_optuna_create_study_kwargs_persistence(setup_module, tmp_path):
    dataset, _, _ = setup_module
    db_path = tmp_path / "study.db"
    create_study_kwargs = {
        "study_name": "nf_persist",
        "storage": f"sqlite:///{db_path}",
        "load_if_exists": True,
    }
    auto1 = BaseAuto(
        h=12,
        loss=MAE(),
        valid_loss=MSE(),
        cls_model=MLP,
        config=config_f,
        search_alg=optuna.samplers.RandomSampler(seed=0),
        num_samples=1,
        backend="optuna",
        optuna_options=OptunaOptions(create_study_kwargs=create_study_kwargs),
    )
    auto1.fit(dataset=dataset)
    assert db_path.exists()
    assert len(auto1.results.trials) == 1

    auto2 = BaseAuto(
        h=12,
        loss=MAE(),
        valid_loss=MSE(),
        cls_model=MLP,
        config=config_f,
        search_alg=optuna.samplers.RandomSampler(seed=0),
        num_samples=1,
        backend="optuna",
        optuna_options=OptunaOptions(create_study_kwargs=create_study_kwargs),
    )
    auto2.fit(dataset=dataset)
    # The reloaded study keeps the first run's trial and appends the new one.
    assert len(auto2.results.trials) == 2


def config_with_failure(trial):
    cfg = {
        "hidden_size": trial.suggest_categorical("hidden_size", [512]),
        "num_layers": trial.suggest_categorical("num_layers", [3]),
        "input_size": 12,
        "max_steps": 1,
        "val_check_steps": 1,
    }
    if getattr(trial, "number", -1) == 0:
        raise RuntimeError("simulated failing trial")
    return cfg


def test_optuna_study_kwargs_catch(setup_module):
    dataset, _, _ = setup_module
    auto = BaseAuto(
        h=12,
        loss=MAE(),
        valid_loss=MSE(),
        cls_model=MLP,
        config=config_with_failure,
        search_alg=optuna.samplers.RandomSampler(seed=0),
        num_samples=2,
        backend="optuna",
        optuna_options=OptunaOptions(study_kwargs={"catch": (RuntimeError,)}),
    )
    auto.fit(dataset=dataset)
    assert isinstance(auto.results, optuna.Study)
    states = [t.state for t in auto.results.trials]
    assert optuna.trial.TrialState.FAIL in states
    assert optuna.trial.TrialState.COMPLETE in states
    assert auto.results.best_trial.state == optuna.trial.TrialState.COMPLETE


def test_optuna_options_wrong_backend_warns(setup_config):
    with pytest.warns(UserWarning, match=r"optuna_options.*backend='ray'"):
        BaseAuto(
            h=12,
            loss=MAE(),
            valid_loss=MSE(),
            cls_model=MLP,
            config=setup_config,
            num_samples=1,
            ray_options=RayOptions(cpus=1, gpus=0),
            optuna_options=OptunaOptions(study_kwargs={"catch": (RuntimeError,)}),
        )


def test_ray_options_wrong_backend_warns():
    from ray import air

    with pytest.warns(UserWarning, match=r"ray_options.*backend='optuna'"):
        BaseAuto(
            h=12,
            loss=MAE(),
            valid_loss=MSE(),
            cls_model=MLP,
            config=config_f,
            search_alg=optuna.samplers.RandomSampler(seed=0),
            num_samples=1,
            backend="optuna",
            ray_options=RayOptions(run_config=air.RunConfig(name="nf_test")),
        )


def test_ray_scheduler_passthrough(setup_module):
    dataset, _, _ = setup_module
    from ray.tune.schedulers import ASHAScheduler

    class RecordingASHA(ASHAScheduler):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.added = 0

        def on_trial_add(self, tune_controller, trial):
            self.added += 1
            return super().on_trial_add(tune_controller, trial)

    scheduler = RecordingASHA(grace_period=1)
    config = {
        "hidden_size": tune.choice([512]),
        "num_layers": tune.choice([3, 4]),
        "input_size": 12,
        "max_steps": 1,
        "val_check_steps": 1,
    }
    auto = BaseAuto(
        h=12,
        loss=MAE(),
        valid_loss=MSE(),
        cls_model=MLP,
        config=config,
        num_samples=2,
        ray_options=RayOptions(scheduler=scheduler, cpus=1, gpus=0),
    )
    auto.fit(dataset=dataset)
    assert scheduler.added == 2


def test_create_study_kwargs_wrong_backend_warns(setup_config):
    with pytest.warns(UserWarning, match=r"optuna_options.*backend='ray'"):
        BaseAuto(
            h=12,
            loss=MAE(),
            valid_loss=MSE(),
            cls_model=MLP,
            config=setup_config,
            num_samples=1,
            ray_options=RayOptions(cpus=1, gpus=0),
            optuna_options=OptunaOptions(create_study_kwargs={"study_name": "ignored"}),
        )


def test_optuna_create_study_kwargs_override_warns(setup_module):
    dataset, _, _ = setup_module
    auto = BaseAuto(
        h=12,
        loss=MAE(),
        valid_loss=MSE(),
        cls_model=MLP,
        config=config_f,
        search_alg=optuna.samplers.RandomSampler(seed=0),
        num_samples=1,
        backend="optuna",
        optuna_options=OptunaOptions(create_study_kwargs={"direction": "minimize"}),
    )
    with pytest.warns(UserWarning, match="overrides default values for \\['direction'\\]"):
        auto.fit(dataset=dataset)


def test_optuna_study_kwargs_override_warns(setup_module):
    dataset, _, _ = setup_module
    auto = BaseAuto(
        h=12,
        loss=MAE(),
        valid_loss=MSE(),
        cls_model=MLP,
        config=config_f,
        search_alg=optuna.samplers.RandomSampler(seed=0),
        num_samples=1,
        backend="optuna",
        optuna_options=OptunaOptions(study_kwargs={"n_trials": 1}),
    )
    with pytest.warns(UserWarning, match="overrides default values for \\['n_trials'\\]"):
        auto.fit(dataset=dataset)


def test_optuna_time_budget(setup_module):
    dataset, _, _ = setup_module
    auto = BaseAuto(
        h=12,
        loss=MAE(),
        valid_loss=MSE(),
        cls_model=MLP,
        config=config_f,
        search_alg=optuna.samplers.RandomSampler(),
        num_samples=1,
        time_budget=60,
        backend="optuna",
    )
    auto.fit(dataset=dataset)
    assert isinstance(auto.results, optuna.Study)
    y_hat = auto.predict(dataset=dataset)
    assert y_hat.shape[0] > 0
