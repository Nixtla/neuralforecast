import numpy as np
import pandas as pd
import polars as pl
import pytest

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.utils import AirPassengersPanel


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("ragged", [False, True])
def test_val_df_predict_insample(backend, ragged):
    df = AirPassengersPanel[["unique_id", "ds", "y"]]
    train = df.groupby("unique_id", observed=True).head(120)
    if ragged:
        train = train.drop(train[train.unique_id == "Airline2"].head(12).index)
    val = df.groupby("unique_id", observed=True).tail(24)
    expected = pd.concat([train, val]).sort_values(["unique_id", "ds"])
    # Validation rows need not arrive in series/time order.
    val = val.sample(frac=1, random_state=42)
    if backend == "polars":
        train, val = pl.from_pandas(train), pl.from_pandas(val)

    nf = NeuralForecast(
        models=[
            NHITS(
                h=12,
                input_size=24,
                max_steps=2,
                accelerator="cpu",
                enable_progress_bar=False,
                logger=False,
            )
        ],
        freq="ME" if backend == "pandas" else "1mo",
    )
    nf.fit(train, val_df=val)
    predictions = nf.predict_insample()
    np.testing.assert_array_equal(nf.ds, expected.ds.to_numpy())
    assert len(nf.ds) == nf.dataset.temporal.shape[0]
    if backend == "polars":
        predictions = predictions.to_pandas()
    actuals = predictions[["unique_id", "ds", "y"]].drop_duplicates()
    pd.testing.assert_frame_equal(
        actuals.sort_values(["unique_id", "ds"]).reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )
