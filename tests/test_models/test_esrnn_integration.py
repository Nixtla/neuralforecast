"""Public API and rolling evaluation checks for ESRNN."""

import numpy as np
import pandas as pd
import pytest


def _panel():
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": name,
                    "ds": pd.date_range("2020-01-01", periods=n),
                    "y": level + 0.2 * np.arange(n) + np.tile([0.0, 2.0, 1.0], 8)[:n],
                }
            )
            for name, n, level in [("a", 20, 10.0), ("b", 18, 50.0)]
        ],
        ignore_index=True,
    )


def _small_model(**kwargs):
    from neuralforecast.models import ESRNN

    config = dict(
        h=2,
        input_size=3,
        season_length=3,
        encoder_hidden_size=4,
        dilations=((1, 2), (2,)),
        max_steps=2,
        val_check_steps=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    return ESRNN(**{**config, **kwargs})


def test_fit_predict_validation_and_reload_preserve_series_mapping(tmp_path):
    from neuralforecast import NeuralForecast

    df = _panel()
    nf = NeuralForecast(models=[_small_model(batch_size=1)], freq="D")
    nf.fit(df=df.sample(frac=1, random_state=3), val_size=2)
    forecast = nf.predict()
    column = [c for c in forecast if c.startswith("ESRNN")][0]
    assert forecast.shape == (4, 3)
    assert np.isfinite(forecast[column]).all()
    assert (forecast[column] > 0).all()
    assert nf.models[0].valid_trajectories
    subset = df[df.unique_id == "b"].iloc[::-1]
    subset_prediction = nf.predict(df=subset)
    np.testing.assert_allclose(
        subset_prediction[column], forecast.loc[forecast.unique_id == "b", column]
    )
    path = str(tmp_path / "model")
    nf.save(path=path, save_dataset=False)
    loaded = NeuralForecast.load(path)
    actual = loaded.predict(df=df.iloc[::-1])
    np.testing.assert_allclose(actual[column], forecast[column], rtol=1e-6)
    np.testing.assert_allclose(
        loaded.predict(df=subset)[column], subset_prediction[column], rtol=1e-6
    )


@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.inf, -np.inf])
def test_reject_nonpositive_or_nonfinite_targets(bad_value):
    from neuralforecast import NeuralForecast

    df = _panel()
    df.loc[4, "y"] = bad_value
    with pytest.raises(ValueError, match="positive|finite"):
        NeuralForecast(models=[_small_model()], freq="D").fit(df=df)


@pytest.mark.parametrize("kind", ["gap", "duplicate", "mask", "short"])
def test_reject_invalid_chronological_histories(kind):
    from neuralforecast import NeuralForecast

    df = _panel()
    if kind == "gap":
        df = df.drop(index=4)
    elif kind == "duplicate":
        df = pd.concat([df, df.iloc[[4]]])
    elif kind == "mask":
        df["available_mask"] = 1.0
        df.loc[4, "available_mask"] = 0.0
    else:
        df = df.groupby("unique_id").head(6)
    with pytest.raises(
        ValueError, match="contiguous|gap|duplicate|available|history|observations"
    ):
        NeuralForecast(models=[_small_model()], freq="D").fit(df=df)


def test_reject_unknown_and_truncated_prediction_histories():
    from neuralforecast import NeuralForecast

    df = _panel()
    nf = NeuralForecast(models=[_small_model()], freq="D")
    nf.fit(df=df)
    with pytest.raises(ValueError, match="unseen|Unknown|unknown"):
        nf.predict(df=df.replace({"unique_id": {"b": "c"}}))
    with pytest.raises(ValueError, match="start|truncated|history"):
        nf.predict(df=df.groupby("unique_id").tail(10))


def test_reject_scaling_and_distributional_losses():
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import DistributionLoss, MQLoss

    with pytest.raises(ValueError, match="identity"):
        _small_model(scaler_type="standard")
    with pytest.raises(ValueError, match="local_scaler"):
        NeuralForecast(models=[_small_model()], freq="D", local_scaler_type="standard")
    for loss in (DistributionLoss(distribution="Normal"), MQLoss()):
        with pytest.raises(ValueError, match="point|loss"):
            _small_model(loss=loss)
    with pytest.raises(ValueError, match="exogenous"):
        _small_model(hist_exog_list=["x"])


@pytest.mark.parametrize("refit", [False, True])
def test_rolling_cross_validation(refit):
    from neuralforecast import NeuralForecast

    nf = NeuralForecast(models=[_small_model()], freq="D")
    actual = nf.cross_validation(
        df=_panel(), n_windows=2, step_size=1, val_size=2, refit=refit
    )
    column = [c for c in actual if c.startswith("ESRNN")][0]
    assert len(actual) == 8
    assert np.isfinite(actual[column]).all()
    assert (actual[column] > 0).all()
    if not refit:
        # The same fitted parameters and same observed prefix must give the same
        # forecast when requested as an ordinary prediction at a CV cutoff.
        first = actual.sort_values("cutoff").groupby("unique_id").head(2)
        history = _panel().merge(first[["unique_id", "cutoff"]].drop_duplicates())
        history = history[history.ds <= history.cutoff].drop(columns="cutoff")
        forecast = nf.predict(df=history)
        np.testing.assert_allclose(
            forecast[column], first.sort_values(["unique_id", "ds"])[column], rtol=1e-6
        )


@pytest.mark.parametrize("season_length", [(), (2, 3)])
def test_unequal_starts_integer_times_and_stored_dataset_reload(
    tmp_path, season_length
):
    from neuralforecast import NeuralForecast

    df = _panel()
    df["ds"] = df.groupby("unique_id").cumcount()
    df.loc[df.unique_id == "b", "ds"] += 5
    nf = NeuralForecast(models=[_small_model(season_length=season_length)], freq=1)
    nf.fit(df=df, val_size=2)
    predicted = nf.predict()
    path = str(tmp_path / "saved")
    nf.save(path=path, save_dataset=True)
    actual = NeuralForecast.load(path).predict()
    pd.testing.assert_frame_equal(actual, predicted)


@pytest.mark.parametrize(
    "options",
    [
        {"start_padding_enabled": True},
        {"exclude_insample_y": True},
        {"inference_input_size": 20},
        {"devices": 2},
        {"strategy": "ddp"},
        {"windows_batch_size": 0},
    ],
)
def test_reject_unsupported_training_modes(options):
    with pytest.raises(ValueError, match="ESRNN|windows_batch_size"):
        _small_model(**options)


def test_reject_distributed_and_extended_horizon():
    from neuralforecast import NeuralForecast
    from neuralforecast.tsdataset import TimeSeriesDataset

    dataset, *_ = TimeSeriesDataset.from_df(_panel())
    with pytest.raises(ValueError, match="distributed"):
        _small_model().fit(dataset, distributed_config=object())
    nf = NeuralForecast(models=[_small_model()], freq="D")
    nf.fit(df=_panel())
    with pytest.raises(ValueError, match="horizon"):
        nf.predict(h=4)


def test_fit_local_parameters_exclude_held_out_targets():
    from neuralforecast import NeuralForecast
    import torch

    data = _panel()
    altered = data.copy()
    held_out = altered.groupby("unique_id").tail(4).index
    altered.loc[held_out, "y"] *= 100
    fitted = []
    for df in (data, altered):
        nf = NeuralForecast(models=[_small_model()], freq="D")
        nf.cross_validation(df=df, n_windows=1, val_size=2, refit=False)
        fitted.append(nf.models[0])
    for left, right in zip(fitted[0].parameters(), fitted[1].parameters()):
        torch.testing.assert_close(left, right)


def test_insample_rejection_preserves_ordinary_prediction_state():
    from neuralforecast import NeuralForecast

    nf = NeuralForecast(models=[_small_model()], freq="D")
    nf.fit(df=_panel())
    before = nf.predict()
    horizon = nf.h
    test_size = nf.models[0].get_test_size()
    with pytest.raises(NotImplementedError, match="ESRNN|insample"):
        nf.predict_insample()
    assert nf.h == horizon
    assert nf.models[0].get_test_size() == test_size
    pd.testing.assert_frame_equal(nf.predict(), before)


def test_conformal_intervals_use_esrnn_point_forecasts():
    from neuralforecast import NeuralForecast
    from neuralforecast.utils import PredictionIntervals

    nf = NeuralForecast(models=[_small_model()], freq="D")
    nf.fit(df=_panel(), prediction_intervals=PredictionIntervals(n_windows=2))
    intervals = nf.predict(level=[80])
    assert len(intervals) == 4
    assert np.isfinite(intervals.filter(like="ESRNN").to_numpy()).all()
    assert (intervals["ESRNN-lo-80"] <= intervals["ESRNN"]).all()
    assert (intervals["ESRNN"] <= intervals["ESRNN-hi-80"]).all()


@pytest.mark.parametrize("options", [{}, {"devices": "auto"}, {"devices": None}])
def test_effective_trainer_configuration_stays_on_one_device(monkeypatch, options):
    from neuralforecast.models import ESRNN
    import torch

    # Simulate CUDA discovery without needing a physical GPU. The behavior
    # under test is the effective Trainer configuration after BaseModel setup.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = ESRNN(h=2, input_size=3, **options)
    assert model.trainer_kwargs["devices"] == 1


def test_multiple_nodes_are_rejected_before_trainer_creation():
    with pytest.raises(ValueError, match="single|node|distributed"):
        _small_model(num_nodes=2)


def test_sample_weights_are_rejected_before_training():
    from neuralforecast import NeuralForecast

    nf = NeuralForecast(models=[_small_model()], freq="D")
    with pytest.raises(ValueError, match="sample weights"):
        nf.fit(df=_panel().assign(sample_weight=1.0))


def test_direct_fit_with_minimal_two_timestamp_history():
    from neuralforecast.tsdataset import TimeSeriesDataset

    df = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": pd.date_range("2020-01-01", periods=2),
            "y": [2.0, 3.0],
        }
    )
    dataset, *_ = TimeSeriesDataset.from_df(df)
    model = _small_model(
        h=1,
        input_size=1,
        season_length=(),
        max_steps=1,
        encoder_hidden_size=1,
        dilations=((1,),),
        es_backend="torch",
    )
    model.fit(dataset)
    future, *_ = TimeSeriesDataset.from_df(
        pd.DataFrame(
            {"unique_id": ["a"], "ds": [pd.Timestamp("2020-01-03")], "y": [np.nan]}
        )
    )
    predicted = model.predict(dataset.append(future), test_size=1)
    assert predicted.shape == (1, 1)
    assert np.isfinite(predicted).all() and (predicted > 0).all()


@pytest.mark.parametrize("offset", [0, 30])
def test_esrnn_rejects_noncontiguous_explicit_validation(offset):
    from neuralforecast import NeuralForecast

    train = _panel().query("unique_id == 'a'").copy()
    validation = train.head(2).copy()
    validation["ds"] += pd.Timedelta(days=offset)
    forecaster = NeuralForecast(models=[_small_model()], freq="D")
    with pytest.raises(ValueError, match="ESRNN.*contiguous"):
        forecaster.fit(df=train, val_df=validation)
