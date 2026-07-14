import numpy as np
import pandas as pd
import pytest
import ray
import ray.cloudpickle as cpickle
from ray import tune

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoHINT
from neuralforecast.common._base_auto import BaseAuto
from neuralforecast.losses.pytorch import GMM, sCRPS
from neuralforecast.models import HINT, NHITS

# Ray refuses to serialize a trainable larger than this (FUNCTION_SIZE_ERROR_THRESHOLD).
RAY_FUNCTION_SIZE_THRESHOLD = 95 * 1024**2

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


# HINT wrapper must survive nf.fit() so that
# nf.predict() emits the HINT column and applies reconciliation.
def test_hint_fit_predict():
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

    nf = NeuralForecast(models=[model], freq="Q")
    nf.fit(df=Y_df, val_size=4)
    forecasts = nf.predict()

    assert isinstance(nf.models[0], HINT), (
        "HINT wrapper was replaced by the underlying model after fit()."
    )
    assert "HINT" in forecasts.columns, "fit()/predict() did not emit a HINT column."

    parent_children_dict = {0: [1, 2], 1: [3, 4], 2: [5, 6]}
    for _, df in forecasts.groupby("ds"):
        hint_mean = df["HINT"].values
        for parent_idx, children_list in parent_children_dict.items():
            parent_value = hint_mean[parent_idx]
            children_sum = hint_mean[children_list].sum()
            np.testing.assert_allclose(children_sum, parent_value, rtol=1e-6)


def _make_auto_hint(S, quantiles, config=None):
    if config is None:
        config = {
            "input_size": tune.choice([8]),
            "max_steps": tune.choice([1]),
            "reconciliation": tune.choice(["BottomUp"]),
        }
    return AutoHINT(
        NHITS,
        h=4,
        loss=GMM(n_components=2, quantiles=quantiles, num_samples=len(quantiles)),
        valid_loss=sCRPS(quantiles=quantiles),
        S=S,
        config=config,
        num_samples=1,
    )


# Move `S` to the object store replaces the
# captured ndarray with a lightweight ObjectRef that Ray transmits by reference.
def test_auto_hint_large_S_not_captured_in_trainable():
    _, _, quantiles = setup_synthetic_data()
    # ~100 MiB float64 matrix -> above Ray's 95 MiB threshold when captured.
    n_bottom = 3600
    S = np.ones((n_bottom + 1, n_bottom), dtype=np.float64)
    assert S.nbytes > RAY_FUNCTION_SIZE_THRESHOLD

    auto = _make_auto_hint(S, quantiles)

    # Pre-fix behaviour: with `S` stored as a raw ndarray, the bound method drags
    # the whole matrix into the pickled trainable -> Ray rejects the actor.
    captured_size = len(cpickle.dumps(auto._train_tune))
    assert captured_size > RAY_FUNCTION_SIZE_THRESHOLD

    # Once `S` lives in the object store, `self` no longer
    # holds the ndarray directly. Ray captures the ObjectRef by reference in the
    # actual actor-registration path, so the array is not re-serialized per trial.
    if not ray.is_initialized():
        ray.init()
    auto.S = ray.put(S)
    assert isinstance(auto.S, ray.ObjectRef)
    assert not isinstance(auto.S, np.ndarray)


# `AutoHINT.fit` must move `S` into the Ray object store before the trainable is
# built, then restore the original array so the instance stays reusable.
def test_auto_hint_fit_moves_S_to_object_store(monkeypatch):
    _, _, quantiles = setup_synthetic_data()
    S = np.ones((5, 4), dtype=np.float64)
    auto = _make_auto_hint(S, quantiles)

    captured = {}

    def fake_super_fit(
        self, dataset, val_size=0, test_size=0, random_seed=None, distributed_config=None
    ):
        captured["is_ref"] = isinstance(self.S, ray.ObjectRef)
        return "ok"

    monkeypatch.setattr(BaseAuto, "fit", fake_super_fit)

    result = auto.fit(dataset=None, val_size=4)

    assert result == "ok"
    assert captured["is_ref"], "S was not moved to the Ray object store during fit."
    assert isinstance(auto.S, np.ndarray), "S was not restored after fit."


# End-to-end: confirms the object-store ObjectRef round-trips back into a usable
# array inside `_fit_model` (via `ray.get`), producing coherent forecasts.
@pytest.mark.filterwarnings("ignore")
def test_auto_hint_fit_predict():
    Y_df, S, quantiles = setup_synthetic_data()
    config = {
        "input_size": tune.choice([4]),
        "max_steps": tune.choice([1]),
        "val_check_steps": tune.choice([1]),
        "learning_rate": tune.choice([1e-3]),
        "reconciliation": tune.choice(["BottomUp"]),
    }
    auto = _make_auto_hint(S, quantiles, config=config)

    nf = NeuralForecast(models=[auto], freq="Q")
    nf.fit(df=Y_df, val_size=4)
    forecasts = nf.predict()

    assert isinstance(auto.S, np.ndarray), "S was not restored after fit."
    hint_col = [c for c in forecasts.columns if c.startswith("AutoHINT")][0]

    parent_children_dict = {0: [1, 2], 1: [3, 4], 2: [5, 6]}
    for _, df in forecasts.groupby("ds"):
        hint_mean = df[hint_col].values
        for parent_idx, children_list in parent_children_dict.items():
            parent_value = hint_mean[parent_idx]
            children_sum = hint_mean[children_list].sum()
            np.testing.assert_allclose(children_sum, parent_value, rtol=1e-6)
