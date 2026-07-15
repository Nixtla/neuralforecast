"""Regression tests for multi-device training of multivariate models.

Multivariate models need all series in every batch, so we shard the
time-windows across devices instead of the series.
"""
import types

import numpy as np
import pandas as pd
import pytest
import torch

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import NHITS, SOFTS, MLPMultivariate, TSMixer

MULTIVARIATE_MODELS = [TSMixer, MLPMultivariate, SOFTS]


def _model_kwargs(model_cls, **overrides):
    kwargs = {"h": 2, "input_size": 4, "max_steps": 1}
    if model_cls in MULTIVARIATE_MODELS:
        kwargs["n_series"] = 3
    kwargs.update(overrides)
    return kwargs


def _fake_trainer(world_size, global_rank):
    return types.SimpleNamespace(world_size=world_size, global_rank=global_rank)


class TestShardMultivariateWindows:
    def test_windows_are_partitioned_disjointly_across_ranks(self):
        fc = torch.arange(10)
        shards = []
        for rank in range(3):
            model = TSMixer(**_model_kwargs(TSMixer))
            model._trainer = _fake_trainer(world_size=3, global_rank=rank)
            shards.append(model._shard_multivariate_windows(fc))
        # every window assigned exactly once, no overlap
        assert torch.equal(torch.cat(shards).sort().values, fc)
        assert sum(len(s) for s in shards) == len(fc)

    def test_fallback_when_fewer_windows_than_devices(self):
        # No rank may end up with an empty batch (would deadlock DDP).
        model = TSMixer(**_model_kwargs(TSMixer))
        model._trainer = _fake_trainer(world_size=4, global_rank=2)
        fc = torch.arange(3)
        assert torch.equal(model._shard_multivariate_windows(fc), fc)

    @pytest.mark.parametrize(
        "model_cls, trainer",
        [
            (TSMixer, None),  # no trainer attached
            (TSMixer, _fake_trainer(world_size=1, global_rank=0)),  # single device
            (NHITS, _fake_trainer(world_size=2, global_rank=1)),  # univariate
        ],
    )
    def test_returns_all_windows_when_sharding_not_applicable(self, model_cls, trainer):
        model = model_cls(**_model_kwargs(model_cls))
        if trainer is not None:
            model._trainer = trainer
        fc = torch.arange(10)
        assert torch.equal(model._shard_multivariate_windows(fc), fc)


def _make_df(n_series=4, length=60):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=length, freq="15min")
    return pd.concat(
        [
            pd.DataFrame(
                {"unique_id": f"s{i}", "ds": dates, "y": rng.standard_normal(length).cumsum()}
            )
            for i in range(n_series)
        ],
        ignore_index=True,
    )


@pytest.mark.parametrize("model_cls", MULTIVARIATE_MODELS)
def test_multivariate_trains_on_multi_process_ddp(model_cls):
    """Multivariate models train on >1 device without crashing.

    CPU multi-process DDP (gloo) reproduces the same batch-sharding as
    multi-GPU, so no GPU is required.
    """
    if not torch.distributed.is_gloo_available():
        pytest.skip("gloo backend required for CPU DDP")
    df = _make_df()
    n_series = df["unique_id"].nunique()
    model = model_cls(
        **_model_kwargs(model_cls, n_series=n_series, max_steps=2),
        batch_size=n_series,
        loss=MAE(),
        accelerator="cpu",
        devices=2,
        strategy="ddp_spawn",
    )
    nf = NeuralForecast(models=[model], freq="15min")
    nf.fit(df=df)


def test_validation_loss_is_window_count_weighted():
    """on_validation_epoch_end combines batches/devices weighted by window
    count, not as an unweighted mean of per-batch means.

    Two batches with means 2.0 (3 windows) and 4.0 (1 window) must yield the
    weighted mean 2.5, not the unweighted mean-of-means 3.0.
    """
    model = NHITS(**_model_kwargs(NHITS))
    model.val_size = 2
    model.log = lambda *args, **kwargs: None
    model.all_gather = lambda t: t.unsqueeze(0)  # single-device passthrough
    model.validation_step_outputs = [
        torch.tensor([6.0, 3.0]),  # loss_sum=6 over 3 windows -> mean 2.0
        torch.tensor([4.0, 1.0]),  # loss_sum=4 over 1 window  -> mean 4.0
    ]

    model.on_validation_epoch_end()

    _, avg_loss = model.valid_trajectories[-1]
    assert avg_loss == pytest.approx(2.5)


@pytest.mark.parametrize("model_cls", [NHITS, SOFTS])
def test_validation_runs_on_multi_process_ddp(model_cls):
    """Validation completes on >1 device without hanging or crashing.

    Exercises the all_gather in on_validation_epoch_end and the window
    sharding added to validation_step for both univariate and multivariate
    models.
    """
    if not torch.distributed.is_gloo_available():
        pytest.skip("gloo backend required for CPU DDP")
    df = _make_df()
    n_series = df["unique_id"].nunique()
    model = model_cls(
        **_model_kwargs(model_cls, n_series=n_series, max_steps=2),
        batch_size=n_series,
        loss=MAE(),
        accelerator="cpu",
        devices=2,
        strategy="ddp_spawn",
    )
    nf = NeuralForecast(models=[model], freq="15min")
    nf.fit(df=df, val_size=4)
