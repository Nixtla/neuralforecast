"""Phase 0: does a nonlinear coordinate transform help at all?

`invariant` (arcsinh, see neuralforecast/common/_scalers.py) is already a
nonlinear coordinate scaler shipped with the library. If it never beats
`robust`, the Yeo-Johnson hypothesis is falsified before we write any code.

Emits per-(series, seed) MASE in long form so deltas can be paired on
unique_id, and per-series training-set skewness so the win can be stratified.

Pre-registered hypothesis
-------------------------
`invariant` beats `robust` in the TOP training-skewness tercile and is
neutral in the bottom tercile. A wash across all series is the expected
average and is NOT the quantity of interest.

Usage (Colab) — paste this whole file into one cell, run it, then in the
next cell call the functions directly:

    run_dataset('M4-Weekly')    # smallest, start here
    run_all()                   # all four datasets, then summarize
    analyze()                   # summarize whatever is already on disk

Usage (CLI):

    python phase0_run.py --dataset M4-Weekly
    python phase0_run.py --analyze

Resumable: finished (dataset, model, scaler, seed) cells are skipped, so a
disconnected Colab runtime can just re-run the same call.
"""

import os
import gc
import time
import logging
import argparse
import warnings

import numpy as np
import pandas as pd


# neuralforecast / datasetsforecast are imported lazily inside run_dataset so
# that `--analyze` works in a bare pandas environment.

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
os.environ["NIXTLA_ID_AS_COL"] = "1"

DATA_DIR = "./data"
RESULTS_DIR = "./results"
SCORES_CSV = f"{RESULTS_DIR}/phase0_scores.csv"
SERIES_CSV = f"{RESULTS_DIR}/phase0_series_meta.csv"
TIMING_CSV = f"{RESULTS_DIR}/phase0_timing.csv"

SCALERS = ["robust", "invariant", "identity"]
MODELS = ["NHITS", "PatchTST"]
SEEDS = [1, 2, 3, 4, 5]
BASELINE = "robust"

MAX_STEPS = 1000
PATIENCE = 3


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# (loader group, horizon, freq, seasonality)
DATASETS = {
    "M3-Monthly": ("M3", "Monthly", 18, "M", 12),
    "M4-Daily": ("M4", "Daily", 14, 1, 7),
    "M4-Weekly": ("M4", "Weekly", 13, 1, 1),
    "M4-Monthly": ("M4", "Monthly", 18, 1, 12),
}


def get_dataset(name):
    from datasetsforecast.m3 import M3
    from datasetsforecast.m4 import M4

    source, group, horizon, freq, seasonality = DATASETS[name]
    if source == "M3":
        Y_df, *_ = M3.load(DATA_DIR, group)
    else:
        Y_df, *_ = M4.load(DATA_DIR, group)
        Y_df["ds"] = Y_df["ds"].astype(int)
    Y_df = Y_df[["unique_id", "ds", "y"]].reset_index(drop=True)
    return Y_df, horizon, freq, seasonality


def split(Y_df, horizon):
    test_df = Y_df.groupby("unique_id").tail(horizon)
    train_df = Y_df.drop(test_df.index).reset_index(drop=True)
    return train_df, test_df.reset_index(drop=True)


def series_metadata(train_df, dataset):
    """Training-set skewness and CV per series. Computed on train only so the
    stratification is available at decision time, not just post hoc."""
    g = train_df.groupby("unique_id")["y"]
    meta = pd.DataFrame(
        {
            "skew": g.skew(),
            "cv": g.std() / g.mean().abs().replace(0, np.nan),
            "n_obs": g.size(),
            "min_y": g.min(),
        }
    ).reset_index()
    meta["dataset"] = dataset
    meta["all_positive"] = meta["min_y"] > 0
    return meta


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def build_model(model_name, horizon, scaler_type, seed):
    from neuralforecast.models import NHITS, PatchTST
    from neuralforecast.losses.pytorch import MAE

    common = dict(
        h=horizon,
        input_size=2 * horizon,
        scaler_type=scaler_type,
        loss=MAE(),
        max_steps=MAX_STEPS,
        early_stop_patience_steps=PATIENCE,
        val_check_steps=50,
        random_seed=seed,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    if model_name == "NHITS":
        return NHITS(**common)
    if model_name == "PatchTST":
        return PatchTST(**common)
    raise ValueError(model_name)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def completed_cells():
    if not os.path.exists(SCORES_CSV):
        return set()
    done = pd.read_csv(SCORES_CSV, usecols=["dataset", "model", "scaler", "seed"])
    return set(map(tuple, done.drop_duplicates().values))


def append_csv(df, path):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def run_dataset(dataset):
    from functools import partial
    from utilsforecast.losses import mase
    from utilsforecast.evaluation import evaluate
    from neuralforecast import NeuralForecast

    Y_df, horizon, freq, seasonality = get_dataset(dataset)
    train_df, test_df = split(Y_df, horizon)

    if not os.path.exists(SERIES_CSV):
        append_csv(series_metadata(train_df, dataset), SERIES_CSV)
    else:
        seen = pd.read_csv(SERIES_CSV)
        if dataset not in set(seen["dataset"]):
            append_csv(series_metadata(train_df, dataset), SERIES_CSV)

    done = completed_cells()
    metric = partial(mase, seasonality=seasonality)

    for model_name in MODELS:
        for scaler in SCALERS:
            for seed in SEEDS:
                cell = (dataset, model_name, scaler, seed)
                if cell in done:
                    print(f"skip  {cell}", flush=True)
                    continue

                print(f"run   {cell}", flush=True)
                start = time.time()
                try:
                    model = build_model(model_name, horizon, scaler, seed)
                    nf = NeuralForecast(models=[model], freq=freq)
                    nf.fit(train_df, val_size=horizon)
                    preds = nf.predict()
                except Exception as e:  # identity can diverge on raw M4 scales
                    print(f"FAIL  {cell}: {type(e).__name__}: {e}", flush=True)
                    append_csv(
                        pd.DataFrame([{**dict(zip(
                            ["dataset", "model", "scaler", "seed"], cell)),
                            "elapsed_s": round(time.time() - start),
                            "status": f"fail:{type(e).__name__}"}]),
                        TIMING_CSV,
                    )
                    continue
                elapsed = time.time() - start

                col = preds.columns[-1]
                eval_df = test_df.merge(preds, on=["unique_id", "ds"], how="left")
                eval_df = eval_df.rename(columns={col: "pred"})

                scores = evaluate(
                    eval_df,
                    metrics=[metric],
                    models=["pred"],
                    train_df=train_df,
                    target_col="y",
                )
                scores = scores[["unique_id", "pred"]].rename(columns={"pred": "mase"})
                scores["dataset"] = dataset
                scores["model"] = model_name
                scores["scaler"] = scaler
                scores["seed"] = seed

                append_csv(scores, SCORES_CSV)
                append_csv(
                    pd.DataFrame([{
                        "dataset": dataset, "model": model_name, "scaler": scaler,
                        "seed": seed, "elapsed_s": round(elapsed),
                        "status": "ok",
                        "n_nan": int(scores["mase"].isna().sum()),
                        "median_mase": float(scores["mase"].median()),
                    }]),
                    TIMING_CSV,
                )
                print(
                    f"done  {cell}  median MASE={scores['mase'].median():.4f}"
                    f"  ({elapsed:.0f}s)",
                    flush=True,
                )

                del nf, model, preds
                gc.collect()


# ---------------------------------------------------------------------------
# Analysis: paired deltas, stratified by training skewness
# ---------------------------------------------------------------------------


def analyze():
    if not os.path.exists(SCORES_CSV):
        print(f"no results yet at {SCORES_CSV} — run run_dataset(...) first")
        return
    scores = pd.read_csv(SCORES_CSV)
    meta = pd.read_csv(SERIES_CSV).drop_duplicates(["dataset", "unique_id"])

    wide = scores.pivot_table(
        index=["dataset", "model", "seed", "unique_id"],
        columns="scaler",
        values="mase",
    ).reset_index()

    available = [s for s in SCALERS if s in wide.columns and s != BASELINE]
    if BASELINE not in wide.columns:
        print(f"no {BASELINE} baseline in results yet")
        return

    wide = wide.merge(meta[["dataset", "unique_id", "skew"]],
                      on=["dataset", "unique_id"], how="left")

    # Terciles of training skewness, computed within each dataset.
    wide["skew_bin"] = (
        wide.groupby("dataset")["skew"]
        .transform(lambda s: pd.qcut(s, 3, labels=["low", "mid", "high"],
                                     duplicates="drop"))
    )

    rows = []
    for scaler in available:
        d = wide.dropna(subset=[BASELINE, scaler]).copy()
        # Paired relative delta: negative means `scaler` beat `robust`.
        d["delta"] = (d[scaler] - d[BASELINE]) / d[BASELINE]
        d["win"] = d[scaler] < d[BASELINE]
        for keys, grp in d.groupby(["dataset", "model", "skew_bin"],
                                   observed=True):
            rows.append({
                "dataset": keys[0], "model": keys[1], "skew_bin": keys[2],
                "scaler": scaler,
                "n": len(grp),
                "median_delta_pct": round(100 * grp["delta"].median(), 2),
                "mean_delta_pct": round(100 * grp["delta"].mean(), 2),
                "win_rate": round(grp["win"].mean(), 3),
            })

    summary = pd.DataFrame(rows).sort_values(
        ["scaler", "dataset", "model", "skew_bin"]
    )
    out = f"{RESULTS_DIR}/phase0_summary.csv"
    summary.to_csv(out, index=False)

    pd.set_option("display.width", 160)
    print("\nPaired vs `robust` — negative median_delta_pct = nonlinear wins")
    print("Pre-registered: expect gains in skew_bin=high, neutral in low.\n")
    print(summary.to_string(index=False))
    print(f"\nwrote {out}")

    # Headline: is the effect monotone in skewness?
    print("\nPooled across datasets/models (series-weighted):\n")
    summary["_d"] = summary["median_delta_pct"] * summary["n"]
    summary["_w"] = summary["win_rate"] * summary["n"]
    pooled = summary.groupby(["scaler", "skew_bin"], observed=True)[
        ["_d", "_w", "n"]
    ].sum().reset_index()
    pooled["median_delta_pct"] = (pooled["_d"] / pooled["n"]).round(2)
    pooled["win_rate"] = (pooled["_w"] / pooled["n"]).round(3)
    print(pooled[["scaler", "skew_bin", "n", "median_delta_pct",
                  "win_rate"]].to_string(index=False))


def run_all(datasets=None):
    """Notebook entry point: run every dataset, then summarize."""
    for name in datasets or list(DATASETS):
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}", flush=True)
        run_dataset(name)
    analyze()


def _in_notebook():
    """True under Jupyter/Colab, where sys.argv is the kernel launcher's."""
    try:
        return get_ipython().__class__.__name__ in (  # noqa: F821
            "ZMQInteractiveShell",
            "Shell",
        )
    except NameError:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=list(DATASETS))
    parser.add_argument("--analyze", action="store_true")
    args, _ = parser.parse_known_args(argv)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.analyze:
        analyze()
    elif args.dataset:
        run_dataset(args.dataset)
    else:
        parser.error("pass --dataset <name> or --analyze")


# Pasting this file into a notebook cell should define the functions and stop,
# not try to parse the kernel's argv.
if __name__ == "__main__" and not _in_notebook():
    main()
elif _in_notebook():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(
        "phase0 loaded. Call:\n"
        "    run_dataset('M4-Weekly')   # one dataset\n"
        "    run_all()                  # all four, then summarize\n"
        "    analyze()                  # summarize whatever is on disk"
    )
