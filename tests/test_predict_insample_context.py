import numpy as np
import pandas as pd
import pytest

from neuralforecast import NeuralForecast
from neuralforecast.models import NLinear


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("horizon", [1, 3])
@pytest.mark.parametrize("step_size", [1, 2, 7, 1000])
@pytest.mark.parametrize("test_size", [0, 5])
def test_sparse_insample_forecasts_keep_available_history(
    engine, horizon, step_size, test_size
):
    frames = []
    for uid, size in [("short", 23), ("long", 34)]:
        time = np.arange(size)
        frames.append(
            pd.DataFrame(
                {"unique_id": uid, "ds": time, "y": time + np.sin(time / 3) + 10}
            )
        )
    data = pd.concat(frames, ignore_index=True)
    if engine == "polars":
        data = pytest.importorskip("polars").from_pandas(data)
    nf = NeuralForecast(
        models=[
            NLinear(
                h=horizon,
                input_size=6,
                max_steps=1,
                val_check_steps=1,
                inference_windows_batch_size=1,
                accelerator="cpu",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
        ],
        freq=1,
    )
    nf.fit(data)
    nf.models[0].set_test_size(test_size)
    dense = nf.predict_insample(step_size=1)
    sparse = nf.predict_insample(step_size=step_size)
    if engine == "polars":
        dense, sparse = dense.to_pandas(), sparse.to_pandas()
    expected = []
    for _, group in dense.groupby("unique_id", sort=False):
        cutoffs = group["cutoff"].drop_duplicates().iloc[::-step_size]
        expected.append(group[group["cutoff"].isin(cutoffs)])
    keys = ["unique_id", "cutoff", "ds"]
    pd.testing.assert_frame_equal(
        sparse.sort_values(keys).reset_index(drop=True),
        pd.concat(expected).sort_values(keys).reset_index(drop=True),
        check_exact=True,
    )
