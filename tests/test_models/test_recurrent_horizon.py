import numpy as np
import pandas as pd
import pytest

from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.utils import AirPassengersDF


def _fit_recurrent_model(batch_size, scaler):
    parts = []
    for series in range(5):
        part = AirPassengersDF.copy()
        part["unique_id"] = series
        part["y"] *= series + 1
        parts.append(part)
    nf = NeuralForecast(
        models=[
            LSTM(
                h=2,
                input_size=4,
                inference_input_size=6,
                recurrent=True,
                encoder_hidden_size=8,
                encoder_n_layers=1,
                valid_batch_size=batch_size,
                scaler_type=scaler,
                max_steps=2,
                accelerator="cpu",
                logger=False,
                enable_progress_bar=False,
            )
        ],
        freq="ME",
    )
    nf.fit(df=pd.concat(parts, ignore_index=True))
    return nf


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("scaler", ["identity", "standard"])
def test_recurrent_long_horizon_across_batches(batch_size, scaler):
    nf = _fit_recurrent_model(batch_size, scaler)
    model = nf.models[0]
    default = nf.predict()
    batched = nf.predict(h=6)
    assert len(batched) == 30
    assert not batched.isna().any().any()
    assert model.h == model.predict_horizon == 2

    model.valid_batch_size = 8
    single_batch = nf.predict(h=6)
    np.testing.assert_allclose(batched["LSTM"], single_batch["LSTM"], rtol=1e-5)
    np.testing.assert_allclose(nf.predict()["LSTM"], default["LSTM"], rtol=1e-5)


@pytest.mark.parametrize("explain", [False, True])
def test_recurrent_horizon_restored_after_prediction_error(monkeypatch, explain):
    nf = _fit_recurrent_model(2, "standard")
    model = nf.models[0]
    before = nf.predict()
    forward = model.forward
    calls = 0

    def interrupted(windows_batch):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted prediction")
        return forward(windows_batch)

    with monkeypatch.context() as patch:
        patch.setattr(model, "forward", interrupted)
        with pytest.raises(RuntimeError, match="interrupted prediction"):
            if explain:
                model.predict(dataset=nf.dataset, h=6, explainer_config={})
            else:
                nf.predict(h=6)

    assert model.h == model.predict_horizon == 2
    assert model.input_size == model.input_size_backup
    assert model.n_predicts == 1
    assert not model.maintain_state
    assert model.rnn_state is None
    np.testing.assert_allclose(nf.predict()["LSTM"], before["LSTM"], rtol=1e-5)
    assert len(nf.predict(h=6)) == 30
