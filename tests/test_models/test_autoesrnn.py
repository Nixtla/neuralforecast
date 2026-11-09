"""AutoESRNN configuration and real Optuna integration checks."""

import numpy as np
import optuna
import pandas as pd
import pytest

from neuralforecast import NeuralForecast
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.losses.pytorch import MAE, QuantileLoss
from neuralforecast.tsdataset import TimeSeriesDataset


def _panel():
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": pd.date_range("2020-01-01", periods=18),
                    "y": level + np.arange(18) % 3 + 0.1 * np.arange(18),
                }
            )
            for uid, level in [("a", 10.0), ("b", 30.0)]
        ],
        ignore_index=True,
    )


def _config(trial):
    return dict(
        input_size=3,
        season_length=3,
        encoder_hidden_size=trial.suggest_categorical("encoder_hidden_size", [4, 8]),
        dilations=((1,),),
        max_steps=2,
        val_check_steps=1,
        batch_size=1,
        windows_batch_size=4,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )


def test_autoesrnn_default_spaces_and_baseauto_signature():
    from neuralforecast.auto import AutoESRNN
    from .test_helpers import check_args

    check_args(AutoESRNN, exclude_args=["cls_model"])
    ray_config = AutoESRNN.get_default_config(h=2, backend="ray")
    assert ray_config["scaler_type"] == "identity"
    assert ray_config["devices"] == 1
    assert ray_config["season_length"] == ()
    optuna_config = AutoESRNN.get_default_config(h=2, backend="optuna")
    assert callable(optuna_config)
    model = AutoESRNN(h=2, config=_config, backend="optuna", num_samples=2)
    assert isinstance(model.valid_loss, MAE)
    assert isinstance(model.loss, QuantileLoss) and model.loss.q == 0.5
    assert model.config(MockTrial())["h"] == 2
    # Exercise actual search-space conversion, beyond the MockTrial schema.
    trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=3)).ask()
    sampled = optuna_config(trial)
    from neuralforecast.models import ESRNN

    candidate = ESRNN(**{**sampled, "h": 2, "loss": model.loss, "logger": False})
    assert candidate.input_size in (2, 4, 6)
    assert candidate.season_lengths == ()


def test_autoesrnn_optuna_fit_predict_and_checkpoint(tmp_path):
    from neuralforecast.auto import AutoESRNN

    auto = AutoESRNN(
        h=2,
        config=_config,
        backend="optuna",
        num_samples=2,
        search_alg=optuna.samplers.RandomSampler(seed=3),
    )
    nf = NeuralForecast(models=[auto], freq="D")
    nf.fit(df=_panel(), val_size=2)
    fitted = nf.models[0]
    assert len(fitted.results.trials) == 2
    assert np.isfinite(fitted.results.best_value)
    forecast = nf.predict()
    assert len(forecast) == 4
    assert np.isfinite(forecast.filter(like="AutoESRNN").to_numpy()).all()
    path = str(tmp_path / "auto")
    nf.save(path=path, save_dataset=True)
    restored = NeuralForecast.load(path)
    pd.testing.assert_frame_equal(restored.predict(), forecast)
    subset = _panel().query("unique_id == 'b'")
    actual = restored.predict(df=subset)
    np.testing.assert_allclose(
        actual.filter(like="AutoESRNN"),
        forecast.query("unique_id == 'b'").filter(like="AutoESRNN"),
    )


def test_wrapped_esrnn_rejects_core_scaling_before_fit():
    from neuralforecast.auto import AutoESRNN

    with pytest.raises(ValueError, match="local_scaler"):
        NeuralForecast(
            models=[AutoESRNN(h=2, config=_config, backend="optuna")],
            freq="D",
            local_scaler_type="standard",
        )


def test_wrapped_esrnn_rejects_insample_without_state_mutation():
    from neuralforecast.auto import AutoESRNN

    nf = NeuralForecast(
        models=[AutoESRNN(h=2, config=_config, backend="optuna", num_samples=1)],
        freq="D",
    )
    nf.fit(df=_panel(), val_size=2)
    before = nf.predict()
    with pytest.raises(NotImplementedError, match="ESRNN"):
        nf.predict_insample()
    assert nf.h == 2
    pd.testing.assert_frame_equal(nf.predict(), before)


def test_autoesrnn_refuses_distributed_training_before_trials():
    from neuralforecast.auto import AutoESRNN

    dataset, *_ = TimeSeriesDataset.from_df(_panel())
    model = AutoESRNN(h=2, config=_config, backend="optuna", num_samples=1)
    with pytest.raises(ValueError, match="distributed"):
        model.fit(dataset, val_size=2, distributed_config=object())


def test_autoesrnn_forwards_current_optuna_options():
    from neuralforecast.auto import AutoESRNN
    from neuralforecast.common._base_auto import OptunaOptions

    options = OptunaOptions(
        study_kwargs={"n_trials": 1},
        create_study_kwargs={"study_name": "esrnn-options"},
    )
    model = AutoESRNN(
        h=2,
        config=_config,
        backend="optuna",
        num_samples=2,
        time_budget=60,
        optuna_options=options,
    )
    dataset, *_ = TimeSeriesDataset.from_df(_panel())
    model.fit(dataset, val_size=2)
    assert model.results.study_name == "esrnn-options"
    assert len(model.results.trials) == 1


def test_autoesrnn_enforces_single_gpu_ray_reservations():
    from neuralforecast.auto import AutoESRNN
    from neuralforecast.common._base_auto import RayOptions

    with pytest.raises(ValueError, match="one GPU"):
        AutoESRNN(h=2, ray_options=RayOptions(gpus=2))
