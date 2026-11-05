import numpy as np

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import IQLoss
from neuralforecast.models import MLP
from neuralforecast.utils import AirPassengersPanel, uncross_quantiles


def _fitted():
    df = AirPassengersPanel[["unique_id", "ds", "y"]]
    model = MLP(
        h=12,
        input_size=24,
        loss=IQLoss(),
        max_steps=2,
        accelerator="cpu",
        enable_progress_bar=False,
        logger=False,
    )
    nf = NeuralForecast(models=[model], freq="ME")
    nf.fit(df)
    n_rows = df.unique_id.nunique() * 12
    # A calibrated model: the q-quantile forecast is always 100 * q
    nf.models[0].predict = lambda dataset, quantiles, **kw: np.full(
        (n_rows, 1), 100 * quantiles[0]
    )
    return nf


def test_predict_returns_the_requested_quantile():
    fcsts = _fitted().predict(quantiles=[0.1, 0.9])

    assert np.allclose(fcsts["MLP_ql0.1"], 10)
    assert np.allclose(fcsts["MLP_ql0.9"], 90)


def test_simulate_follows_the_requested_quantiles():
    sims = _fitted().simulate(n_paths=5000, seed=0)

    assert np.allclose(np.percentile(sims["MLP"], [10, 50, 90]), [10, 50, 90], atol=2)


def test_uncross_quantiles_keeps_the_column_order():
    fcsts = np.array([[5.0, 1.0, 9.0]])

    assert uncross_quantiles(fcsts, [0.5, 0.1, 0.9]).tolist() == [[5.0, 1.0, 9.0]]
    assert uncross_quantiles(fcsts, [0.1, 0.5, 0.9]).tolist() == [[1.0, 5.0, 9.0]]
