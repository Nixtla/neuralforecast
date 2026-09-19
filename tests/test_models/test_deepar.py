import numpy as np
import pandas as pd
import pytest
import torch

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoDeepAR, DeepAR, RayOptions
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model

from .test_helpers import check_args


def test_deepar(suppress_warnings):
    check_model(DeepAR, ["airpassengers"])


def test_autodeepar(setup_dataset):
    dataset = setup_dataset

    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoDeepAR, exclude_args=['cls_model'])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoDeepAR.get_default_config(h=12, backend='optuna')
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update({'max_steps': 1, 'val_check_steps': 1, 'input_size': 12, 'lstm_hidden_size': 8})
        return config

    model = AutoDeepAR(h=12, config=my_config_new, backend='optuna', num_samples=1)
    assert model.config(MockTrial())['h'] == 12
    model.fit(dataset=dataset)

    # Unit test for situation: Ray with updated default config
    my_config = AutoDeepAR.get_default_config(h=12, backend='ray')
    my_config['max_steps'] = 1
    my_config['val_check_steps'] = 1
    my_config['input_size'] = 12
    my_config['lstm_hidden_size'] = 8
    model = AutoDeepAR(h=12, config=my_config, backend='ray', num_samples=1, ray_options=RayOptions(cpus=1))
    model.fit(dataset=dataset)


def _small_deepar(**kwargs):
    config = dict(
        h=3,
        input_size=5,
        lstm_n_layers=1,
        lstm_hidden_size=8,
        lstm_dropout=0.0,
        trajectory_samples=8,
        max_steps=2,
        val_check_steps=1,
        num_lr_decays=0,
        batch_size=2,
        windows_batch_size=4,
        inference_windows_batch_size=4,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    config.update(kwargs)
    return DeepAR(**config)


@pytest.mark.parametrize("mixed", [False, True])
def test_deepar_hist_exog_forward(mixed):
    model = _small_deepar(
        hist_exog_list=["hist_a", "hist_b"],
        futr_exog_list=["calendar"] if mixed else None,
        stat_exog_list=["group"] if mixed else None,
    ).eval()
    y = torch.randn(2, 7, 1)
    hist = torch.randn(2, 7, 2, requires_grad=True)
    futr = torch.randn(2, 7, 1) if mixed else None
    stat = torch.randn(2, 1) if mixed else None
    batch = dict(insample_y=y, hist_exog=hist, futr_exog=futr, stat_exog=stat)

    output = model(batch)
    assert model.hist_encoder.input_size == 3 + 2 * mixed
    assert output.shape == (2, model.h, model.loss.outputsize_multiplier)
    output.square().sum().backward()
    assert torch.isfinite(hist.grad).all()
    assert hist.grad.abs().sum() > 0
    with torch.no_grad():
        changed = model({**batch, "hist_exog": hist + 1.0})
    assert not torch.allclose(output, changed)


@pytest.mark.parametrize("mixed", [False, True])
def test_deepar_hist_exog_disabled_preserves_forward(mixed):
    model = _small_deepar(
        futr_exog_list=["calendar"] if mixed else None,
        stat_exog_list=["group"] if mixed else None,
    ).eval()
    y = torch.randn(2, 7, 1)
    futr = torch.randn(2, 7, 1) if mixed else None
    stat = torch.randn(2, 1) if mixed else None
    # Also preserve callers that omit the unused historical input entirely.
    output = model(dict(insample_y=y, futr_exog=futr, stat_exog=stat))
    original_input = y
    if mixed:
        original_input = torch.cat((y, futr, stat[:, None].expand(-1, 7, -1)), 2)
    hidden, _ = model.hist_encoder(original_input)
    expected = model.decoder(hidden)[:, -model.h :]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_deepar_hist_exog_recurrent_state():
    model = _small_deepar(h=1, hist_exog_list=["hist"]).eval()
    y, hist = torch.randn(2, 7, 1), torch.randn(2, 7, 1)
    batch = dict(insample_y=y, hist_exog=hist, futr_exog=None, stat_exog=None)
    expected = model(batch)
    model.maintain_state = True
    model({**batch, "insample_y": y[:, :-1], "hist_exog": hist[:, :-1]})
    actual = model({**batch, "insample_y": y[:, -1:], "hist_exog": hist[:, -1:]})
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("scaler_type", ["identity", "standard"])
@pytest.mark.parametrize("future_value", [1e6, float("nan")])
def test_deepar_hist_exog_masks_forecast_values(scaler_type, future_value):
    model = _small_deepar(
        hist_exog_list=["hist"], futr_exog_list=["calendar"], scaler_type=scaler_type
    ).eval()
    length = model.input_size + model.h
    temporal = torch.arange(2 * length * 4, dtype=torch.float32)
    temporal = temporal.reshape(2, length, 4, 1) / 10
    temporal[:, :, 3] = 1
    cols = pd.Index(["y", "hist", "calendar", "available_mask"])
    parsed = []
    for value in (0.0, future_value):
        values = temporal.clone()
        values[:, model.input_size :, 1] = value
        windows = dict(
            temporal=values, temporal_cols=cols, static=None, static_cols=None
        )
        windows = model._normalization(windows, y_idx=0)
        parsed.append(model._parse_windows({"y_idx": 0}, windows))

    reference, changed = parsed
    torch.testing.assert_close(reference[4], changed[4], rtol=0, atol=0)
    assert torch.count_nonzero(changed[4][:, model.input_size - 1 :]) == 0
    outputs = []
    for insample_y, mask, _, _, hist, futr, stat in parsed:
        outputs.append(
            model(
                dict(
                    insample_y=insample_y,
                    insample_mask=mask,
                    hist_exog=hist,
                    futr_exog=futr,
                    stat_exog=stat,
                )
            )
        )
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


def test_deepar_hist_exog_categorical_embedding():
    model = _small_deepar(
        hist_exog_list=["hist", "category"],
        cat_exog_list=["category"],
        categorical_cardinalities={"category": 3},
        cat_emb_dim=2,
    )
    length = model.input_size + model.h
    temporal = torch.ones(2, length, 4, 1)
    temporal[:, :, 1, 0] = torch.arange(length, dtype=torch.float32)
    temporal[:, :, 2, 0] = torch.arange(length) % 3
    # Unavailable category IDs must be masked before the embedding lookup.
    temporal[:, model.input_size :, 2] = float("nan")
    windows = dict(
        temporal=temporal,
        temporal_cols=pd.Index(["y", "hist", "category", "available_mask"]),
        static=None,
        static_cols=None,
    )
    y, mask, _, _, hist, futr, stat = model._parse_windows({"y_idx": 0}, windows)
    assert model.hist_exog_size == 3
    assert model.hist_encoder.input_size == 4
    output = model(
        dict(
            insample_y=y,
            insample_mask=mask,
            hist_exog=hist,
            futr_exog=futr,
            stat_exog=stat,
        )
    )
    assert torch.isfinite(output).all()
    output.square().sum().backward()


@pytest.mark.parametrize("h_train", [1, 3])
@pytest.mark.parametrize("mixed", [False, True])
def test_deepar_hist_exog_fit_predict_without_future_history(h_train, mixed):
    dates = pd.date_range("2020-01-01", periods=24, freq="D")
    frames = []
    for uid in range(2):
        hist = np.sin(np.arange(24) / 3.0) + uid
        frames.append(
            pd.DataFrame({
                "unique_id": uid,
                "ds": dates,
                "y": 10 + hist,
                "hist": hist,
                "calendar": dates.dayofweek,
            })
        )
    df = pd.concat(frames, ignore_index=True)
    model = _small_deepar(
        h_train=h_train,
        hist_exog_list=["hist"],
        futr_exog_list=["calendar"] if mixed else None,
        stat_exog_list=["group"] if mixed else None,
        scaler_type="standard",
    )
    nf = NeuralForecast(models=[model], freq="D")
    static_df = pd.DataFrame({"unique_id": [0, 1], "group": [0, 1]})
    nf.fit(df=df, static_df=static_df if mixed else None, val_size=3)
    futr_df = None
    if mixed:
        futr_df = nf.make_future_dataframe()
        futr_df["calendar"] = futr_df["ds"].dt.dayofweek
    predictions = nf.predict(futr_df=futr_df)
    assert len(predictions) == 6
    forecasts = predictions.filter(like="DeepAR")
    assert not forecasts.empty
    assert np.isfinite(forecasts.to_numpy()).all()
