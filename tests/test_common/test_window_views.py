import numpy as np
import pandas as pd
import pytest
import torch

from neuralforecast.models import MLP, MLPMultivariate


@pytest.mark.parametrize("step", ["train", "val", "predict"])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("multivariate", [False, True])
def test_deferred_windows_keep_samples_and_gradients(
    step, stride, weighted, multivariate
):
    n, length = 3, 24
    columns = ["y", "available_mask", "x"]
    if weighted:
        columns.insert(1, "sample_weight")
    temporal = torch.arange(n * len(columns) * length, dtype=torch.float64)
    temporal = temporal.reshape(n, len(columns), length) / 16
    temporal[:, columns.index("available_mask")] = 1
    temporal[0, columns.index("available_mask"), 4:7] = 0
    temporal[1, columns.index("available_mask"), 10:13] = 0
    if weighted:
        temporal[:, columns.index("sample_weight")] = torch.arange(length) % 4 + 1
    temporal.requires_grad_()
    batch = dict(
        temporal=temporal,
        temporal_cols=pd.Index(columns),
        static=torch.arange(n * 2, dtype=torch.float64).reshape(n, 2),
        static_cols=pd.Index(["a", "b"]),
    )
    kwargs = dict(
        h=3,
        input_size=4,
        step_size=stride,
        hist_exog_list=["x"],
        training_data_availability_threshold=[0.5, 0.5],
    )
    model = MLPMultivariate(n_series=n, **kwargs) if multivariate else MLP(**kwargs)
    model.val_size, model.test_size = 6, 3
    model.predict_step_size = stride
    # Build the materialized reference directly, independently of the model's
    # window creation and sampling helpers.
    if step == "train":
        values = torch.nn.functional.pad(temporal[:, :, :-9], (0, model.h))
    elif step == "val":
        values = temporal[:, :, -13:-3]
    else:
        values = temporal[:, :, -7:]
    reference = values.unfold(-1, model.input_size + model.h, stride)
    starts = reference.shape[2]
    if multivariate:
        reference = reference.permute(2, 3, 1, 0)
        reference_static = batch["static"]
    else:
        reference = reference.permute(0, 2, 3, 1).flatten(0, 1).unsqueeze(-1)
        reference_static = batch["static"].repeat_interleave(starts, dim=0)
    reference_cols = batch["temporal_cols"]
    reference_weights = None
    if step == "train" and weighted:
        weight_idx = reference_cols.get_loc("sample_weight")
        reference_weights = reference[:, model.input_size :, weight_idx : weight_idx + 1].mean(1)
        reference = reference[:, :, [i for i in range(len(columns)) if i != weight_idx]]
        reference_cols = reference_cols.delete(weight_idx)
    if step == "train":
        mask = reference[:, :, reference_cols.get_loc("available_mask")]
        valid = (mask[:, : model.input_size].sum((1, 2)) >= max(1, int(model.input_size * 0.5 * model.n_series))) & (
            mask[:, model.input_size :].sum((1, 2)) >= max(1, int(model.h * 0.5 * model.n_series))
        )
        valid = valid.nonzero().squeeze(-1)
    else:
        valid = torch.arange(len(reference))
    deferred = model._create_windows(batch, step)
    assert reference_cols.equals(deferred[-1])
    torch.testing.assert_close(valid, deferred[3], rtol=0, atol=0)
    if not multivariate:
        # The deferred view owns only the underlying series, never all windows.
        assert deferred[0].ndim == 5
        assert (
            deferred[0].untyped_storage().nbytes()
            <= n * len(columns) * (length + model.h) * 8
        )
    idxs = torch.tensor([len(valid) - 1, 0, 0, len(valid) // 2])
    selected = valid[idxs]
    samples = [dict(
        temporal=reference[selected],
        static=reference_static if multivariate else reference_static[selected],
        sample_weight=None if reference_weights is None else reference_weights[selected],
    )]
    windows, static, static_cols, valid, weights, cols = deferred
    samples.append(model._sample_windows(
        windows, static, static_cols, cols, idxs, valid, weights
    ))
    for key in ["temporal", "static", "sample_weight"]:
        if samples[0][key] is None:
            assert samples[1][key] is None
        else:
            torch.testing.assert_close(samples[0][key], samples[1][key], rtol=0, atol=0)
    objectives = [s["temporal"].square().sum() for s in samples]
    if samples[0]["sample_weight"] is not None:
        objectives = [
            value + sample["sample_weight"].square().sum()
            for value, sample in zip(objectives, samples)
        ]
    grads = [
        torch.autograd.grad(value, temporal, retain_graph=True)[0]
        for value in objectives
    ]
    torch.testing.assert_close(grads[0], grads[1], rtol=0, atol=0)


def test_training_does_not_materialize_all_windows(monkeypatch):
    n, length, lookback, h = 8, 512, 48, 12
    rng = np.random.default_rng(12)
    df = pd.DataFrame(
        dict(
            unique_id=np.repeat(np.arange(n), length),
            ds=np.tile(np.arange(length), n),
            y=rng.normal(size=n * length),
        )
    )
    from neuralforecast.tsdataset import TimeSeriesDataset, TimeSeriesDataModule

    dataset, *_ = TimeSeriesDataset.from_df(df)
    batch = next(iter(TimeSeriesDataModule(dataset, batch_size=n).train_dataloader()))
    model = MLP(h=h, input_size=lookback, windows_batch_size=16)
    model.val_size = model.test_size = 0
    sample = model._sample_windows
    storage = []

    def record(windows_temporal, **kwargs):
        storage.append(windows_temporal.untyped_storage().nbytes())
        return sample(windows_temporal=windows_temporal, **kwargs)

    monkeypatch.setattr(model, "_sample_windows", record)
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert storage[0] <= n * 2 * (length + h) * 4
