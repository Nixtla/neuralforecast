import numpy as np
import pandas as pd
import pytest
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import (
    MLP,
    NHITS,
    VanillaTransformer,
    LSTM,
    GRU,
    RNN,
    DeepAR,
    MLPMultivariate,
    TSMixerx,
    XLinear,
    TimeXer,
)
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.auto import AutoMLP

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
    # Force CPU: the MacOS MPS backend exhausts its small shared pool when many
    # models are trained in one session.
    base = dict(
        h=6,
        input_size=12,
        max_steps=2,
        random_seed=1,
        scaler_type="standard",
        accelerator="cpu",
    )
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
        cat_exog_list=hist_cat + futr_cat,
        categorical_cardinalities=cards,
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df)
    futr_df = _futr_df() if futr_cat else None
    preds = nf.predict(futr_df=futr_df)
    assert preds.shape[0] == 4 * 6
    assert np.isfinite(preds[cls.__name__].to_numpy()).all()


def test_mixed_continuous_and_categorical_hist_stream():
    df = _panel(cols=("city",))
    df["temp"] = (np.arange(len(df), dtype=float) % 10)  # continuous
    model = _model(
        MLP,
        hist_exog_list=["temp", "city"],
        cat_exog_list=["city"],
        categorical_cardinalities={"city": 4},
        cat_emb_dim=5,
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df)
    assert nf.models[0].hist_exog_size == 1 + 5  # 1 continuous + 5-dim embedding
    assert "temp" not in nf.categorical_vocab_  # continuous column not embedded
    preds = nf.predict()
    assert preds.shape[0] == 4 * 6
    assert np.isfinite(preds["MLP"].to_numpy()).all()


def test_mixed_continuous_and_categorical_futr_stream():
    df = _panel(cols=("dow",))
    df["price"] = (np.arange(len(df), dtype=float) % 5)  # continuous
    model = _model(
        MLP,
        futr_exog_list=["price", "dow"],
        cat_exog_list=["dow"],
        categorical_cardinalities={"dow": 7},
        cat_emb_dim=5,
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df)
    assert nf.models[0].futr_exog_size == 1 + 5
    futr = _futr_df()  # has `dow`; add the continuous futr column too
    futr["price"] = (np.arange(len(futr), dtype=float) % 5)
    preds = nf.predict(futr_df=futr)
    assert preds.shape[0] == 4 * 6
    assert np.isfinite(preds["MLP"].to_numpy()).all()


def test_static_categorical_runs():
    # Static categorical feature passed via static_df and embedded.
    df = _panel(cols=())  # only y, no temporal exog
    static_df = pd.DataFrame(
        {"unique_id": [f"s{u}" for u in range(4)], "cluster": ["A", "B", "A", "C"]}
    )
    model = _model(
        MLP,
        stat_exog_list=["cluster"],
        cat_exog_list=["cluster"],
        categorical_cardinalities={"cluster": 3},
        cat_emb_dim=5,
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df, static_df=static_df)
    # No continuous static features remain, so size == embedding dim.
    assert nf.models[0].stat_exog_size == 5
    preds = nf.predict()
    assert preds.shape[0] == 4 * 6
    assert np.isfinite(preds["MLP"].to_numpy()).all()


def test_effective_exog_sizes():
    df = _panel()
    model = _model(
        MLP,
        hist_exog_list=["city"],
        futr_exog_list=["dow"],
        cat_exog_list=["city", "dow"],
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
        cat_exog_list=["city"],
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


def test_oov_uses_mean_of_category_embeddings():
    # Index 0 (OOV/unseen) maps to the mean of the learned category rows.
    model = _model(
        MLP,
        hist_exog_list=["city"],
        cat_exog_list=["city"],
        categorical_cardinalities={"city": 4},
    )
    emb = model.hist_cat_embeddings[0]
    out = model._lookup_embedding(emb, torch.tensor([0, 1]))
    assert torch.allclose(out[0], emb.weight[1:].mean(dim=0))  # OOV -> mean
    assert torch.allclose(out[1], emb.weight[1])  # real category -> its row


def test_unseen_category_resolves_to_oov():
    df = _panel(cols=("dow",))
    model = _model(
        MLP,
        futr_exog_list=["dow"],
        cat_exog_list=["dow"],
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
        cat_exog_list=["city", "dow"],
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


def test_val_df_is_encoded():
    # val_df goes through align/from_df, which bypasses _prepare_fit's encoding;
    # without encoding it, the string categorical column would break or mismatch.
    df = _panel(cols=("city",))
    val_df = _panel(cols=("city",), n=12, seed=1)  # same series, string `city`
    model = _model(
        MLP,
        hist_exog_list=["city"],
        cat_exog_list=["city"],
        categorical_cardinalities={"city": 4},
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df, val_df=val_df)
    preds = nf.predict()
    assert preds.shape[0] == 4 * 6
    assert np.isfinite(preds["MLP"].to_numpy()).all()


def test_cross_validation_no_refit_with_categoricals():
    # No-refit CV builds the vocab from the training portion (test_size held out)
    # and resets stale vocab; here we just assert it runs end-to-end.
    df = _panel(cols=("dow",))
    model = _model(
        MLP,
        futr_exog_list=["dow"],
        cat_exog_list=["dow"],
        categorical_cardinalities={"dow": 7},
    )
    nf = NeuralForecast(models=[model], freq=1)
    cv = nf.cross_validation(df, n_windows=2, h=6, refit=False)
    assert len(cv) > 0
    assert np.isfinite(cv["MLP"].to_numpy()).all()


def test_refit_stored_dataset_preserves_vocab():
    # fit(df=None) retrains on the stored (already-encoded) dataset; it must not
    # wipe the vocabulary, so predict(futr_df=...) still encodes categoricals.
    df = _panel(cols=("dow",))
    model = _model(
        MLP,
        futr_exog_list=["dow"],
        cat_exog_list=["dow"],
        categorical_cardinalities={"dow": 7},
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df)
    vocab = dict(nf.categorical_vocab_)
    nf.fit()  # df=None -> reuse stored dataset
    assert nf.categorical_vocab_ == vocab
    preds = nf.predict(futr_df=_futr_df())
    assert np.isfinite(preds["MLP"].to_numpy()).all()


def test_use_fitted_cross_validation_preserves_vocab():
    # use_fitted CV on a new df snapshots/restores state; the fitted vocabulary
    # must survive the (reset+rebuild) that runs on the holdout df.
    df = _panel(cols=("dow",))
    model = _model(
        MLP,
        futr_exog_list=["dow"],
        cat_exog_list=["dow"],
        categorical_cardinalities={"dow": 7},
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df)
    vocab_before = dict(nf.categorical_vocab_)
    holdout = _panel(cols=("dow",), n=30, seed=2)
    nf.cross_validation(df=holdout, n_windows=2, h=6, refit=False, use_fitted=True)
    assert nf.categorical_vocab_ == vocab_before


def test_prediction_intervals_preserve_full_vocab():
    # 'flag' == "B" only in the tail. The conformal CV trains on df-minus-tail,
    # but the final vocab must still include "B" (as in a plain fit) rather than
    # making it permanent OOV.
    from neuralforecast.utils import PredictionIntervals

    rng = np.random.default_rng(0)
    rows = []
    for uid in range(4):
        for t in range(60):
            flag = "B" if t >= 54 else "A"
            rows.append(
                {
                    "unique_id": f"s{uid}",
                    "ds": t,
                    "y": np.sin(t / 5) + uid + rng.standard_normal() * 0.1,
                    "flag": flag,
                }
            )
    df = pd.DataFrame(rows)
    model = _model(
        MLP,
        hist_exog_list=["flag"],
        cat_exog_list=["flag"],
        categorical_cardinalities={"flag": 2},
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df, prediction_intervals=PredictionIntervals(n_windows=2))
    assert nf.categorical_vocab_["flag"] == {"A": 1, "B": 2}


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


@pytest.mark.parametrize("cls", [LSTM, GRU, RNN])
def test_recurrent_categoricals_run(cls):
    # Recursive path (recurrent=True): hist + futr categoricals embedded per step.
    df = _panel(cols=("city", "dow"))
    model = _model(
        cls,
        recurrent=True,
        hist_exog_list=["city"],
        futr_exog_list=["dow"],
        cat_exog_list=["city", "dow"],
        categorical_cardinalities={"city": 4, "dow": 7},
        cat_emb_dim=5,
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df)
    assert nf.models[0].hist_exog_size == 5
    assert nf.models[0].futr_exog_size == 5
    preds = nf.predict(futr_df=_futr_df())
    assert preds.shape[0] == 4 * 6
    assert np.isfinite(preds[cls.__name__].to_numpy()).all()


def test_deepar_categoricals_run():
    # DeepAR is always recurrent and uses a distribution loss; futr + static cats.
    df = _panel(cols=("dow",))
    static_df = pd.DataFrame(
        {"unique_id": [f"s{u}" for u in range(4)], "cluster": ["A", "B", "A", "C"]}
    )
    model = _model(
        DeepAR,
        futr_exog_list=["dow"],
        stat_exog_list=["cluster"],
        cat_exog_list=["dow", "cluster"],
        categorical_cardinalities={"dow": 7, "cluster": 3},
        cat_emb_dim=5,
        loss=DistributionLoss(distribution="Normal", level=[80]),
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df, static_df=static_df)
    assert nf.models[0].futr_exog_size == 5
    assert nf.models[0].stat_exog_size == 5
    preds = nf.predict(futr_df=_futr_df())
    assert preds.shape[0] == 4 * 6
    assert np.isfinite(preds["DeepAR"].to_numpy()).all()


def test_recurrent_embeddings_differ_from_numeric_baseline():
    # Confirms the embedding is wired into the recursive forward.
    df = _panel(cols=("city",))
    emb = _model(
        LSTM,
        recurrent=True,
        hist_exog_list=["city"],
        cat_exog_list=["city"],
        categorical_cardinalities={"city": 4},
    )
    nf_emb = NeuralForecast(models=[emb], freq=1)
    nf_emb.fit(df)
    p_emb = nf_emb.predict()["LSTM"].to_numpy()

    df_num = df.copy()
    df_num["city"] = df_num["city"].map({c: i for i, c in enumerate(CITIES)})
    num = _model(LSTM, recurrent=True, hist_exog_list=["city"])
    nf_num = NeuralForecast(models=[num], freq=1)
    nf_num.fit(df_num)
    p_num = nf_num.predict()["LSTM"].to_numpy()

    assert not np.allclose(p_emb, p_num)


@pytest.mark.parametrize("cls", [MLPMultivariate, TSMixerx, XLinear])
def test_multivariate_categoricals_run(cls):
    # hist + futr + static categoricals embedded on the feature axis.
    df = _panel(cols=("city", "dow"))
    static_df = pd.DataFrame(
        {"unique_id": [f"s{u}" for u in range(4)], "cluster": ["A", "B", "A", "C"]}
    )
    model = _model(
        cls,
        n_series=4,
        hist_exog_list=["city"],
        futr_exog_list=["dow"],
        stat_exog_list=["cluster"],
        cat_exog_list=["city", "dow", "cluster"],
        categorical_cardinalities={"city": 4, "dow": 7, "cluster": 3},
        cat_emb_dim=5,
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df, static_df=static_df)
    # No continuous features remain, so each size == embedding dim.
    assert nf.models[0].hist_exog_size == 5
    assert nf.models[0].futr_exog_size == 5
    assert nf.models[0].stat_exog_size == 5
    preds = nf.predict(futr_df=_futr_df())
    assert preds.shape[0] == 4 * 6
    assert np.isfinite(preds[cls.__name__].to_numpy()).all()


def test_timexer_categoricals_run():
    # TimeXer has EXOGENOUS_FUTR = False, so only hist + static categoricals.
    # It reshapes hist_exog into per-variate tokens ([B, L, X * N]), so each
    # embedding dimension becomes its own variate: the least trivial consumer
    # of the expanded feature axis.
    df = _panel(cols=("city",))
    static_df = pd.DataFrame(
        {"unique_id": [f"s{u}" for u in range(4)], "cluster": ["A", "B", "A", "C"]}
    )
    model = _model(
        TimeXer,
        n_series=4,
        patch_len=6,  # must divide input_size (12); default 16 is too large here
        hist_exog_list=["city"],
        stat_exog_list=["cluster"],
        cat_exog_list=["city", "cluster"],
        categorical_cardinalities={"city": 4, "cluster": 3},
        cat_emb_dim=5,
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df, static_df=static_df)
    assert nf.models[0].hist_exog_size == 5
    assert nf.models[0].stat_exog_size == 5
    preds = nf.predict()
    assert preds.shape[0] == 4 * 6
    assert np.isfinite(preds["TimeXer"].to_numpy()).all()


def test_multivariate_mixed_continuous_and_categorical():
    # Continuous features are kept and concatenated ahead of the embeddings.
    df = _panel(cols=("city",))
    df["temp"] = np.arange(len(df), dtype=float) % 10  # continuous hist
    model = _model(
        MLPMultivariate,
        n_series=4,
        hist_exog_list=["temp", "city"],
        cat_exog_list=["city"],
        categorical_cardinalities={"city": 4},
        cat_emb_dim=5,
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df)
    assert nf.models[0].hist_exog_size == 1 + 5  # 1 continuous + 5-dim embedding
    preds = nf.predict()
    assert np.isfinite(preds["MLPMultivariate"].to_numpy()).all()


def test_multivariate_embeddings_differ_from_numeric_baseline():
    # Confirms the embedding is actually wired into the multivariate forward.
    df = _panel(cols=("city",))
    emb = _model(
        MLPMultivariate,
        n_series=4,
        hist_exog_list=["city"],
        cat_exog_list=["city"],
        categorical_cardinalities={"city": 4},
    )
    nf_emb = NeuralForecast(models=[emb], freq=1)
    nf_emb.fit(df)
    p_emb = nf_emb.predict()["MLPMultivariate"].to_numpy()

    # Baseline: same column passed as a plain numeric (label-encoded) feature.
    df_num = df.copy()
    df_num["city"] = df_num["city"].map({c: i for i, c in enumerate(CITIES)})
    num = _model(MLPMultivariate, n_series=4, hist_exog_list=["city"])
    nf_num = NeuralForecast(models=[num], freq=1)
    nf_num.fit(df_num)
    p_num = nf_num.predict()["MLPMultivariate"].to_numpy()

    assert not np.allclose(p_emb, p_num)


def test_categorical_must_be_subset_of_exog_list():
    with pytest.raises(Exception, match="must also be listed"):
        _model(
            MLP,
            hist_exog_list=[],
            cat_exog_list=["city"],
            categorical_cardinalities={"city": 4},
        )


def test_missing_cardinality_raises():
    with pytest.raises(Exception, match="cardinalities"):
        _model(
            MLP,
            hist_exog_list=["city"],
            cat_exog_list=["city"],
        )


def test_too_many_categories_raises():
    df = _panel(cols=("city",))  # 4 cities
    model = _model(
        MLP,
        hist_exog_list=["city"],
        cat_exog_list=["city"],
        categorical_cardinalities={"city": 2},  # under-declared
    )
    nf = NeuralForecast(models=[model], freq=1)
    with pytest.raises(ValueError, match="distinct values"):
        nf.fit(df)


def test_conflicting_cross_model_cardinalities_raise():
    df = _panel(cols=("city",))
    m1 = _model(
        MLP,
        hist_exog_list=["city"],
        cat_exog_list=["city"],
        categorical_cardinalities={"city": 4},
    )
    m2 = _model(
        NHITS,
        hist_exog_list=["city"],
        cat_exog_list=["city"],
        categorical_cardinalities={"city": 6},  # disagrees with m1
    )
    nf = NeuralForecast(models=[m1, m2], freq=1)
    with pytest.raises(ValueError, match="conflicting"):
        nf.fit(df)


def test_auto_model_with_categoricals_runs():
    # Auto* models read cat_exog_list / categorical_cardinalities from their
    # config; the vocab is built and the search trains on the encoded data.
    df = _panel(cols=("city",))

    def config(trial):
        return {
            "input_size": 12,
            "max_steps": 2,
            "accelerator": "cpu",
            "hist_exog_list": ["city"],
            "cat_exog_list": ["city"],
            "categorical_cardinalities": {"city": 4},
        }

    model = AutoMLP(h=6, config=config, num_samples=1, backend="optuna")
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df)
    assert set(nf.categorical_vocab_["city"]) == set(CITIES)
    preds = nf.predict()
    assert preds.shape[0] == 4 * 6
    assert np.isfinite(preds["AutoMLP"].to_numpy()).all()


@pytest.mark.parametrize(
    "explainer",
    ["IntegratedGradients", "InputXGradient", "ShapleyValueSampling"],
)
def test_explain_categoricals_aggregate_to_feature(explainer):
    # Attributions over the embedded axis are summed back to one value per
    # original feature, so the feature axis matches `hist_exog_list` /
    # `futr_exog_list` (1 each here) rather than the embedding dim (5).
    pytest.importorskip("captum")
    df = _panel(cols=("city", "dow"))
    model = _model(
        MLP,
        hist_exog_list=["city"],
        futr_exog_list=["dow"],
        cat_exog_list=["city", "dow"],
        categorical_cardinalities={"city": 4, "dow": 7},
        cat_emb_dim=5,
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df)
    _, explanations = nf.explain(futr_df=_futr_df(), explainer=explainer)
    hist = explanations["MLP"]["hist_exog"]  # [b, h, series, out, n_features, temporal]
    futr = explanations["MLP"]["futr_exog"]
    assert hist.shape[-2] == 1  # 'city' collapsed from emb_dim 5 -> 1 feature
    assert futr.shape[-2] == 1  # 'dow'  collapsed from emb_dim 5 -> 1 feature
    assert np.isfinite(hist.numpy()).all()
    assert np.isfinite(futr.numpy()).all()


def test_explain_mixed_continuous_and_categorical():
    # Continuous feature keeps its own attribution; the categorical one collapses,
    # so a hist stream of [temp, city] yields 2 feature attributions (not 1 + 5).
    pytest.importorskip("captum")
    df = _panel(cols=("city",))
    df["temp"] = np.arange(len(df), dtype=float) % 10
    model = _model(
        MLP,
        hist_exog_list=["temp", "city"],
        cat_exog_list=["city"],
        categorical_cardinalities={"city": 4},
        cat_emb_dim=5,
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df)
    assert nf.models[0].hist_exog_size == 1 + 5  # embedded size
    _, explanations = nf.explain()
    hist = explanations["MLP"]["hist_exog"]
    assert hist.shape[-2] == 2  # temp + city
    assert np.isfinite(hist.numpy()).all()


def test_explain_static_categorical():
    pytest.importorskip("captum")
    df = _panel(cols=())
    static_df = pd.DataFrame(
        {"unique_id": [f"s{u}" for u in range(4)], "cluster": ["A", "B", "A", "C"]}
    )
    model = _model(
        MLP,
        stat_exog_list=["cluster"],
        cat_exog_list=["cluster"],
        categorical_cardinalities={"cluster": 3},
        cat_emb_dim=5,
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df, static_df=static_df)
    _, explanations = nf.explain()
    stat = explanations["MLP"]["stat_exog"]  # [..., n_static_features]
    assert stat.shape[-1] == 1  # 'cluster' collapsed from emb_dim 5 -> 1 feature
    assert np.isfinite(stat.numpy()).all()


def test_simulate_distribution_loss_with_futr_categorical():
    # futr categorical: futr_df must be encoded with the fitted vocab before
    # alignment, otherwise the raw string values crash the copula sampling.
    df = _panel(cols=("dow",))
    model = _model(
        NHITS,
        futr_exog_list=["dow"],
        cat_exog_list=["dow"],
        categorical_cardinalities={"dow": 7},
        loss=DistributionLoss(distribution="Normal"),
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df)
    sims = nf.simulate(n_paths=5, futr_df=_futr_df())
    assert sims.shape[0] == 4 * 6 * 5  # n_series * h * n_paths
    assert np.isfinite(sims["NHITS"].to_numpy()).all()

    # simulate() validates futr_df like predict(): a required (categorical)
    # column missing from futr_df raises before alignment.
    bad_futr = _futr_df().drop(columns=["dow"])
    with pytest.raises(ValueError, match="missing from `futr_df`"):
        nf.simulate(n_paths=5, futr_df=bad_futr)


def test_simulate_point_loss_conformal_with_hist_categorical():
    # Point-loss models simulate through the conformal path (_simulate_conformal),
    # which reuses the conformity scores + vocabulary built during fit().
    from neuralforecast.utils import PredictionIntervals

    df = _panel(cols=("city",))
    model = _model(
        NHITS,
        hist_exog_list=["city"],
        cat_exog_list=["city"],
        categorical_cardinalities={"city": 4},
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df, prediction_intervals=PredictionIntervals(n_windows=2))
    sims = nf.simulate(n_paths=5)
    assert sims.shape[0] == 4 * 6 * 5
    assert np.isfinite(sims["NHITS"].to_numpy()).all()


def test_simulate_static_categorical():
    # Static categoricals are embedded and the stored (encoded) dataset is reused.
    df = _panel(cols=())
    static_df = pd.DataFrame(
        {"unique_id": [f"s{u}" for u in range(4)], "cluster": ["A", "B", "A", "C"]}
    )
    model = _model(
        NHITS,
        stat_exog_list=["cluster"],
        cat_exog_list=["cluster"],
        categorical_cardinalities={"cluster": 3},
        loss=DistributionLoss(distribution="Normal"),
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df, static_df=static_df)
    sims = nf.simulate(n_paths=5)
    assert sims.shape[0] == 4 * 6 * 5
    assert np.isfinite(sims["NHITS"].to_numpy()).all()


def test_simulate_with_provided_df():
    df = _panel(cols=("city", "dow"))
    static_df = pd.DataFrame(
        {"unique_id": [f"s{u}" for u in range(4)], "cluster": ["A", "B", "A", "C"]}
    )
    model = _model(
        NHITS,
        hist_exog_list=["city"],
        futr_exog_list=["dow"],
        stat_exog_list=["cluster"],
        cat_exog_list=["city", "dow", "cluster"],
        categorical_cardinalities={"city": 4, "dow": 7, "cluster": 3},
        loss=DistributionLoss(distribution="Normal"),
    )
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df, static_df=static_df)

    futr = _futr_df()
    sims_df = nf.simulate(
        df=df, static_df=static_df, futr_df=futr, n_paths=5, seed=0
    )
    sims_stored = nf.simulate(futr_df=futr, n_paths=5, seed=0)

    assert sims_df.shape[0] == 4 * 6 * 5
    assert np.isfinite(sims_df["NHITS"].to_numpy()).all()
    np.testing.assert_allclose(
        sims_df["NHITS"].to_numpy(),
        sims_stored["NHITS"].to_numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.fixture(scope="module")
def spark_session():
    pytest.importorskip("pyspark")
    pytest.importorskip("fugue")
    from pyspark.sql import SparkSession

    try:
        spark = (
            SparkSession.builder.master("local[1]")
            .config("spark.sql.shuffle.partitions", "1")
            .getOrCreate()
        )
    except Exception as e:  # e.g. no Java runtime available
        pytest.skip(f"Could not start a local SparkSession: {e}")
    yield spark
    spark.stop()


def test_distributed_fit_predict_with_categoricals(spark_session, tmp_path):
    # Distributed (Spark) fit builds the vocabulary from the Spark frame and
    # encodes it driver-side before writing parquet; predict encodes the futr /
    # static frames driver-side before the union / join. Workers never encode.
    from neuralforecast import DistributedConfig

    spark = spark_session
    pdf = _panel(cols=("city", "dow"))  # 'city' hist-cat, 'dow' futr-cat
    static_pdf = pd.DataFrame(
        {"unique_id": [f"s{u}" for u in range(4)], "cluster": ["A", "B", "A", "C"]}
    )
    spark_df = spark.createDataFrame(pdf)
    spark_static = spark.createDataFrame(static_pdf)

    model = _model(
        MLP,
        hist_exog_list=["city"],
        futr_exog_list=["dow"],
        stat_exog_list=["cluster"],
        cat_exog_list=["city", "dow", "cluster"],
        categorical_cardinalities={"city": 4, "dow": 7, "cluster": 3},
    )
    nf = NeuralForecast(models=[model], freq=1)
    dist_cfg = DistributedConfig(
        partitions_path=str(tmp_path / "partitions"), num_nodes=1, devices=1
    )
    nf.fit(spark_df, static_df=spark_static, distributed_config=dist_cfg)

    # Vocabulary was built distributedly from the Spark frames.
    assert set(nf.categorical_vocab_) == {"city", "dow", "cluster"}
    assert nf.categorical_vocab_["cluster"] == {"A": 1, "B": 2, "C": 3}

    spark_futr = spark.createDataFrame(_futr_df())
    preds = nf.predict(futr_df=spark_futr, engine=spark).toPandas()
    assert preds.shape[0] == 4 * 6
    assert np.isfinite(preds["MLP"].to_numpy()).all()


def test_distributed_simulate_with_categoricals(spark_session, tmp_path):
    # Distributed simulation works with categoricals too (a distribution-loss
    # model, since point-loss conformal simulation needs prediction intervals,
    # which distributed training does not support for any model).
    from neuralforecast import DistributedConfig

    spark = spark_session
    pdf = _panel(cols=("dow",))  # 'dow' futr-cat
    spark_df = spark.createDataFrame(pdf)

    model = _model(
        NHITS,
        futr_exog_list=["dow"],
        cat_exog_list=["dow"],
        categorical_cardinalities={"dow": 7},
        loss=DistributionLoss(distribution="Normal"),
    )
    nf = NeuralForecast(models=[model], freq=1)
    dist_cfg = DistributedConfig(
        partitions_path=str(tmp_path / "sim_partitions"), num_nodes=1, devices=1
    )
    nf.fit(spark_df, distributed_config=dist_cfg)

    spark_futr = spark.createDataFrame(_futr_df())
    sims = nf.simulate(futr_df=spark_futr, engine=spark, n_paths=5).toPandas()
    assert np.isfinite(sims["NHITS"].to_numpy()).all()
