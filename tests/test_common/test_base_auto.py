import logging
import warnings

import numpy as np
import optuna
import pandas as pd
import pytest
from ray import tune

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


@pytest.mark.parametrize("legacy_kwarg", ["cpus", "gpus"])
def test_legacy_cpus_gpus_emit_deprecation_warning(setup_config, legacy_kwarg):
    with pytest.warns(DeprecationWarning, match=rf"`{legacy_kwarg}` is deprecated"):
        auto = BaseAuto(
            h=12,
            loss=MAE(),
            valid_loss=MSE(),
            cls_model=MLP,
            config=setup_config,
            num_samples=1,
            **{legacy_kwarg: 2},
        )
    assert getattr(auto.ray_options, legacy_kwarg) == 2


@pytest.mark.parametrize("legacy_kwarg", ["cpus", "gpus"])
def test_legacy_cpus_gpus_conflict_raises_type_error(setup_config, legacy_kwarg):
    with pytest.raises(TypeError, match=rf"`{legacy_kwarg}` and `ray_options"):
        BaseAuto(
            h=12,
            loss=MAE(),
            valid_loss=MSE(),
            cls_model=MLP,
            config=setup_config,
            num_samples=1,
            ray_options=RayOptions(**{legacy_kwarg: 2}),
            **{legacy_kwarg: 2},
        )


@pytest.mark.parametrize("legacy_kwarg", ["cpus", "gpus"])
def test_legacy_cpus_gpus_on_optuna_warns_once(legacy_kwarg, recwarn):
    BaseAuto(
        h=12,
        loss=MAE(),
        valid_loss=MSE(),
        cls_model=MLP,
        config=config_f,
        search_alg=optuna.samplers.RandomSampler(seed=0),
        num_samples=1,
        backend="optuna",
        **{legacy_kwarg: 2},
    )
    # Only the "ignored on backend='optuna'" warning, not the deprecation.
    matching = [
        w for w in recwarn.list if f"`{legacy_kwarg}`" in str(w.message)
    ]
    assert len(matching) == 1
    assert "backend='optuna'" in str(matching[0].message)
    assert not issubclass(matching[0].category, DeprecationWarning)
