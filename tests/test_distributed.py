from types import SimpleNamespace

import fugue.api as fa
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS


def _make_distributed_worker():
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 14,
            "ds": pd.date_range("2026-01-01", periods=14),
            "y": 1.0,
        }
    )
    model = NHITS(
        h=7,
        input_size=7,
        max_steps=1,
        accelerator="cpu",
        enable_progress_bar=False,
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.id_col = "unique_id"
    nf.time_col = "ds"
    nf.target_col = "y"
    nf.dataset = SimpleNamespace(files=["unused"])
    engine = SimpleNamespace(read=SimpleNamespace(parquet=lambda *args: df))
    return nf, engine


def _run_transform(*, df, using, params, **kwargs):
    return using(df, **params)


def test_distributed_predict_initializes_static_scalers(monkeypatch):
    nf, engine = _make_distributed_worker()

    def predict(worker, **kwargs):
        assert worker.scalers_ == {}
        assert worker.static_scalers_ == {}
        return kwargs["df"][["unique_id", "ds"]].assign(NHITS=0.0)

    monkeypatch.setattr(fa, "transform", _run_transform)
    monkeypatch.setattr(NeuralForecast, "predict", predict)

    result = nf._predict_distributed(None, None, None, engine)

    assert result.shape == (14, 3)


def test_distributed_simulate_initializes_static_scalers(monkeypatch):
    nf, engine = _make_distributed_worker()

    def simulate(worker, **kwargs):
        assert worker.scalers_ == {}
        assert worker.static_scalers_ == {}
        return kwargs["df"][["unique_id", "ds"]].assign(sample_id=0, NHITS=0.0)

    monkeypatch.setattr(fa, "transform", _run_transform)
    monkeypatch.setattr(NeuralForecast, "simulate", simulate)

    result = nf._simulate_distributed(None, None, None, engine, n_paths=1)

    assert result.shape == (14, 4)
