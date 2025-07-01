import numpy as np
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import GMM
from neuralforecast.models import HINT, NHITS

# Unit test to check hierarchical coherence
# Probabilistic coherent => Sample coherent => Mean coherence


def sort_df_hier(Y_df, S_df):
    # NeuralForecast core, sorts unique_id lexicographically
    # by default, this class matches S_df and Y_hat_df order.
    Y_df.unique_id = Y_df.unique_id.astype("category")
    Y_df.unique_id = Y_df.unique_id.cat.set_categories(S_df.index)
    Y_df = Y_df.sort_values(by=["unique_id", "ds"])
    return Y_df


def setup_synthetic_data():
    # -----Create synthetic dataset-----
    np.random.seed(123)
    train_steps = 20
    num_levels = 7
    level = np.arange(0, 100, 0.1)
    qs = [[50 - lv / 2, 50 + lv / 2] for lv in level]
    quantiles = np.sort(np.concatenate(qs) / 100)

    levels = ["Top", "Mid1", "Mid2", "Bottom1", "Bottom2", "Bottom3", "Bottom4"]
    unique_ids = np.repeat(levels, train_steps)

    S = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    S_dict = {col: S[:, i] for i, col in enumerate(levels[3:])}
    S_df = pd.DataFrame(S_dict, index=levels)

    ds = (
        pd.date_range(start="2018-03-31", periods=train_steps, freq="Q").tolist()
        * num_levels
    )
    # Create Y_df
    y_lists = [S @ np.random.uniform(low=100, high=500, size=4) for i in range(train_steps)]
    y = [elem for tup in zip(*y_lists) for elem in tup]
    Y_df = pd.DataFrame({"unique_id": unique_ids, "ds": ds, "y": y})
    Y_df = sort_df_hier(Y_df, S_df)

    return Y_df, S, quantiles


# ------Fit/Predict HINT Model------
# Model + Distribution + Reconciliation
def test_hint_model():
    Y_df, S, quantiles = setup_synthetic_data()
    nhits = NHITS(
        h=4,
        input_size=4,
        loss=GMM(n_components=2, quantiles=quantiles, num_samples=len(quantiles)),
        max_steps=5,
        early_stop_patience_steps=2,
        val_check_steps=1,
        scaler_type="robust",
        learning_rate=1e-3,
    )
    model = HINT(h=4, model=nhits, S=S, reconciliation="BottomUp")

    # Fit and Predict
    nf = NeuralForecast(models=[model], freq="Q")
    forecasts = nf.cross_validation(df=Y_df, val_size=4, n_windows=1)

    # ---Check Hierarchical Coherence---
    parent_children_dict = {0: [1, 2], 1: [3, 4], 2: [5, 6]}
    # check coherence for each horizon time step
    for _, df in forecasts.groupby("ds"):
        hint_mean = df["HINT"].values
        for parent_idx, children_list in parent_children_dict.items():
            parent_value = hint_mean[parent_idx]
            children_sum = hint_mean[children_list].sum()
            np.testing.assert_allclose(children_sum, parent_value, rtol=1e-6)
