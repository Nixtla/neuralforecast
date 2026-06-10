import numpy as np
import pandas as pd
import pytest

from neuralforecast import NeuralForecast
from neuralforecast.models import MLP, NHITS, VanillaTransformer, LSTM, MLPMultivariate

CITIES = ["paris", "london", "tokyo", "berlin"]
DOWS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _panel(n_series=4, n=48, seed=0, cols=("city", "dow")):
    rng = np.random.default_rng(seed)
    rows = []
    for uid in range(n_series):
        for t in range(n):
            row = {
                "unique_id": f"s{uid}",
                "ds": t,
                "y": np.sin(t / 5) + uid + rng.standard_normal() * 0.1,
            }
            if "city" in cols:
                row["city"] = CITIES[uid % len(CITIES)]  # hist categorical
            if "dow" in cols:
                row["dow"] = DOWS[t % 7]  # futr (calendar) categorical
            rows.append(row)
    return pd.DataFrame(rows)


def _futr_df(n_series=4, n=48, h=6, unseen=False):
    rows = []
    for uid in range(n_series):
        for t in range(n, n + h):
            dow = "UNSEEN_DAY" if (unseen and t == n) else DOWS[t % 7]
            rows.append({"unique_id": f"s{uid}", "ds": t, "dow": dow})
    return pd.DataFrame(rows)


def _model(cls, **kwargs):
    base = dict(h=6, input_size=12, max_steps=2, random_seed=1, scaler_type="standard")
    base.update(kwargs)
    return cls(**base)


@pytest.mark.parametrize(
    "cls,hist_cat,futr_cat",
    [
        (MLP, ["city"], ["dow"]),  # both streams
        (NHITS, ["city"], []),  # historical only (different architecture)
        (VanillaTransformer, [], ["dow"]),  # future only (futr-exog-only transformer)
    ],
)
def test_categoricals_run(cls, hist_cat, futr_cat):
    cards = {}
    if hist_cat:
        cards["city"] = 4
    if futr_cat:
        cards["dow"] = 7
    df = _panel(cols=tuple(hist_cat + futr_cat))
    model = _model(
        cls,
        hist_exog_list=hist_cat,
        futr_exog_list=futr_cat,
        hist_cat_exog_list=hist_cat,
        futr_cat_exog_list=futr_cat,
        categorical_cardinalities=cards,
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df)
    futr_df = _futr_df() if futr_cat else None
    preds = nf.predict(futr_df=futr_df)
    assert preds.shape[0] == 4 * 6
    assert np.isfinite(preds[cls.__name__].to_numpy()).all()


def test_effective_exog_sizes():
    df = _panel()
    model = _model(
        MLP,
        hist_exog_list=["city"],
        futr_exog_list=["dow"],
        hist_cat_exog_list=["city"],
        futr_cat_exog_list=["dow"],
        categorical_cardinalities={"city": 4, "dow": 7},
        cat_emb_dim=5,
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df)
    # No continuous temporal features remain, so size == embedding dim.
    assert nf.models[0].hist_exog_size == 5
    assert nf.models[0].futr_exog_size == 5


def test_embeddings_differ_from_numeric_baseline():
    df = _panel(cols=("city",))
    emb = _model(
        MLP,
        hist_exog_list=["city"],
        hist_cat_exog_list=["city"],
        categorical_cardinalities={"city": 4},
    )
    nf_emb = NeuralForecast(models=[emb], freq=1)
    nf_emb.fit(df)
    p_emb = nf_emb.predict()["MLP"].to_numpy()

    # Baseline: same column passed as a plain numeric (label-encoded) feature.
    df_num = df.copy()
    df_num["city"] = df_num["city"].map({c: i for i, c in enumerate(CITIES)})
    num = _model(MLP, hist_exog_list=["city"])
    nf_num = NeuralForecast(models=[num], freq=1)
    nf_num.fit(df_num)
    p_num = nf_num.predict()["MLP"].to_numpy()

    assert not np.allclose(p_emb, p_num)


def test_unseen_category_resolves_to_oov():
    df = _panel(cols=("dow",))
    model = _model(
        MLP,
        futr_exog_list=["dow"],
        futr_cat_exog_list=["dow"],
        categorical_cardinalities={"dow": 7},
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df)
    # 'UNSEEN_DAY' is not in the training vocabulary -> OOV index 0, no crash.
    preds = nf.predict(futr_df=_futr_df(unseen=True))
    assert np.isfinite(preds["MLP"].to_numpy()).all()


def test_save_load_reproduces_predictions(tmp_path):
    df = _panel()
    model = _model(
        MLP,
        hist_exog_list=["city"],
        futr_exog_list=["dow"],
        hist_cat_exog_list=["city"],
        futr_cat_exog_list=["dow"],
        categorical_cardinalities={"city": 4, "dow": 7},
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df)
    p1 = nf.predict(futr_df=_futr_df())["MLP"].to_numpy()

    nf.save(path=str(tmp_path), overwrite=True, save_dataset=True)
    nf2 = NeuralForecast.load(path=str(tmp_path))
    assert nf2.categorical_vocab_ == nf.categorical_vocab_
    p2 = nf2.predict(futr_df=_futr_df())["MLP"].to_numpy()
    np.testing.assert_allclose(p1, p2, rtol=1e-5, atol=1e-5)


def test_no_categoricals_is_inert():
    # When no categorical lists are declared, sizes equal the raw counts and the
    # vocabulary stays empty -> the carve-out is a no-op.
    df = _panel(cols=("city",))
    model = _model(MLP, hist_exog_list=["city"])  # numeric path
    df_num = df.copy()
    df_num["city"] = df_num["city"].map({c: i for i, c in enumerate(CITIES)})
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df_num)
    assert nf.categorical_vocab_ == {}
    assert nf.models[0].hist_exog_size == 1


# ---------------------------------------------------------------------------
# Guards: unsupported configurations must raise.
# ---------------------------------------------------------------------------


def test_recurrent_model_rejects_categoricals():
    with pytest.raises(Exception, match="categorical"):
        LSTM(
            h=6,
            input_size=12,
            max_steps=2,
            hist_exog_list=["city"],
            hist_cat_exog_list=["city"],
            categorical_cardinalities={"city": 4},
        )


def test_multivariate_model_rejects_categoricals():
    with pytest.raises(Exception, match="categorical"):
        MLPMultivariate(
            h=6,
            input_size=12,
            n_series=4,
            max_steps=2,
            hist_exog_list=["city"],
            hist_cat_exog_list=["city"],
            categorical_cardinalities={"city": 4},
        )


def test_categorical_must_be_subset_of_exog_list():
    with pytest.raises(Exception, match="must also be listed"):
        _model(
            MLP,
            hist_exog_list=[],
            hist_cat_exog_list=["city"],
            categorical_cardinalities={"city": 4},
        )


def test_missing_cardinality_raises():
    with pytest.raises(Exception, match="cardinalities"):
        _model(
            MLP,
            hist_exog_list=["city"],
            hist_cat_exog_list=["city"],
        )


def test_too_many_categories_raises():
    df = _panel(cols=("city",))  # 4 cities
    model = _model(
        MLP,
        hist_exog_list=["city"],
        hist_cat_exog_list=["city"],
        categorical_cardinalities={"city": 2},  # under-declared
    )
    nf = NeuralForecast(models=[model], freq=1)
    with pytest.raises(ValueError, match="distinct values"):
        nf.fit(df)


def test_explain_guard():
    df = _panel(cols=("city",))
    model = _model(
        MLP,
        hist_exog_list=["city"],
        hist_cat_exog_list=["city"],
        categorical_cardinalities={"city": 4},
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df)
    with pytest.raises(NotImplementedError, match="categorical"):
        nf.explain()


def test_simulate_guard():
    df = _panel(cols=("city",))
    model = _model(
        MLP,
        hist_exog_list=["city"],
        hist_cat_exog_list=["city"],
        categorical_cardinalities={"city": 4},
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df)
    with pytest.raises(NotImplementedError, match="categorical"):
        nf.simulate()
