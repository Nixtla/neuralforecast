import numpy as np

from neuralforecast import NeuralForecast
from neuralforecast.models import MLP
from neuralforecast.utils import AirPassengersPanel


def calibrated_iqloss_nf(loss, crossing=False):
    """IQLoss model whose q-quantile forecast is 100 * q.

    With ``crossing=True``, every other row swaps the 0.1 and 0.9 forecasts, so
    its levels cross while their sorted values are still 100 * q.
    """
    df = AirPassengersPanel[["unique_id", "ds", "y"]]
    model = MLP(
        h=12,
        input_size=24,
        loss=loss(),
        max_steps=2,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        logger=False,
    )
    nf = NeuralForecast(models=[model], freq="ME")
    nf.fit(df)
    n_rows = df.unique_id.nunique() * 12
    crossed = (np.arange(n_rows) % 2 == 1) if crossing else np.zeros(n_rows, bool)

    def predict(dataset, quantiles, **kwargs):
        q = quantiles[0]
        swapped = np.isclose(q, 0.1) or np.isclose(q, 0.9)
        return np.where(crossed & swapped, 100 * (1 - q), 100 * q)[:, None]

    nf.models[0].predict = predict
    return nf
