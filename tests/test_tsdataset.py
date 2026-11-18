import numpy as np
import pandas as pd
import polars
import pytest
import torch

from neuralforecast.tsdataset import TimeSeriesDataModule, TimeSeriesDataset
from neuralforecast.utils import generate_series


@pytest.fixture
def setup_data():
    temporal_df = generate_series(
        n_series=1000, n_temporal_features=0, equal_ends=False
    )
    sorted_temporal_df = temporal_df.sort_values(["unique_id", "ds"])
    unsorted_temporal_df = sorted_temporal_df.sample(frac=1.0)

    return temporal_df, sorted_temporal_df, unsorted_temporal_df


# Testing sort_df=True functionality
def test_sort_df(setup_data):
    temporal_df, sorted_temporal_df, unsorted_temporal_df = setup_data
    dataset, indices, dates, ds = TimeSeriesDataset.from_df(df=unsorted_temporal_df)

    np.testing.assert_allclose(
        dataset.temporal[:, :-1],
        sorted_temporal_df.drop(columns=["unique_id", "ds"]).values,
    )
    np.testing.assert_array_equal(indices, pd.Series(sorted_temporal_df["unique_id"].unique()))
    np.testing.assert_array_equal(dates, temporal_df.groupby("unique_id", observed=True)["ds"].max().values)


def test_data_module(setup_data):
    temporal_df, sorted_temporal_df, unsorted_temporal_df = setup_data
    dataset, indices, dates, ds = TimeSeriesDataset.from_df(df=unsorted_temporal_df)

    batch_size = 128

    data = TimeSeriesDataModule(dataset=dataset, batch_size=batch_size, drop_last=True)
    for batch in data.train_dataloader():
        assert batch["temporal"].shape == (batch_size, 2, 500)
        np.testing.assert_array_equal(batch["temporal_cols"], ["y", "available_mask"])


@pytest.mark.parametrize(
    "n_series, train_batches, valid_batches",
    [(5, [2, 2], [4, 1]), (3, [2], [3])],
)
def test_data_module_drop_last_keeps_every_validation_series(
    n_series, train_batches, valid_batches
):
    # `drop_last` is for training batches: validation must still see every series.
    temporal_df = generate_series(n_series=n_series, n_temporal_features=0)
    dataset, *_ = TimeSeriesDataset.from_df(df=temporal_df)

    data = TimeSeriesDataModule(
        dataset=dataset, batch_size=2, valid_batch_size=4, drop_last=True
    )

    assert [len(b["temporal"]) for b in data.train_dataloader()] == train_batches
    assert [len(b["temporal"]) for b in data.val_dataloader()] == valid_batches


def test_static_features(setup_data):
    temporal_df, sorted_temporal_df, unsorted_temporal_df = setup_data
    batch_size = 128
    n_static_features = 2
    n_temporal_features = 4
    temporal_df, static_df = generate_series(
        n_series=1000,
        n_static_features=n_static_features,
        n_temporal_features=n_temporal_features,
        equal_ends=False,
    )

    dataset, indices, dates, ds = TimeSeriesDataset.from_df(
        df=temporal_df, static_df=static_df
    )
    data = TimeSeriesDataModule(dataset=dataset, batch_size=batch_size, drop_last=True)

    for batch in data.train_dataloader():
        assert batch["temporal"].shape == (batch_size, n_temporal_features + 2, 500)
        np.testing.assert_array_equal(
            batch["temporal_cols"],
            ["y"]
            + [f"temporal_{i}" for i in range(n_temporal_features)]
            + ["available_mask"],
        )

        assert batch["static"].shape == (batch_size, n_static_features)
        np.testing.assert_array_equal(batch["static_cols"], [f"static_{i}" for i in range(n_static_features)])


# Testing sort_df=True functionality
def test_sort_df_mask():
    temporal_df = generate_series(n_series=2, n_temporal_features=2, equal_ends=True)
    temporal_df = temporal_df.groupby("unique_id").tail(10)
    temporal_df = temporal_df.reset_index()
    temporal_full_df = temporal_df.sort_values(["unique_id", "ds"]).reset_index(
        drop=True
    )
    temporal_full_df.loc[temporal_full_df.ds > "2001-05-11", ["y", "temporal_0"]] = None

    split1_df = temporal_full_df.loc[temporal_full_df.ds <= "2001-05-11"]
    split2_df = temporal_full_df.loc[temporal_full_df.ds > "2001-05-11"]
    # Testing available mask
    temporal_df_w_mask = temporal_df.copy()
    temporal_df_w_mask["available_mask"] = 1

    # Mask with all 1's
    dataset, indices, dates, ds = TimeSeriesDataset.from_df(df=temporal_df_w_mask)
    mask_average = dataset.temporal[:, -1].mean()
    np.testing.assert_almost_equal(mask_average, 1.0000)

    # Add 0's to available mask
    temporal_df_w_mask.loc[temporal_df_w_mask.ds > "2001-05-11", "available_mask"] = 0
    dataset, indices, dates, ds = TimeSeriesDataset.from_df(df=temporal_df_w_mask)
    mask_average = dataset.temporal[:, -1].mean()
    np.testing.assert_almost_equal(mask_average, 0.7000)

    # Available mask not in last column
    temporal_df_w_mask = temporal_df_w_mask[
        ["unique_id", "ds", "y", "available_mask", "temporal_0", "temporal_1"]
    ]
    dataset, indices, dates, ds = TimeSeriesDataset.from_df(df=temporal_df_w_mask)
    mask_average = dataset.temporal[:, 1].mean()
    np.testing.assert_almost_equal(mask_average, 0.7000)

    # To test correct future_df wrangling of the `update_df` method
    # We are checking that we are able to recover the AirPassengers dataset
    # using the dataframe or splitting it into parts and initializing.

    # FULL DATASET
    dataset_full, indices_full, dates_full, ds_full = TimeSeriesDataset.from_df(
        df=temporal_full_df
    )
    # SPLIT_1 DATASET
    dataset_1, indices_1, dates_1, ds_1 = TimeSeriesDataset.from_df(df=split1_df)
    dataset_1 = dataset_1.update_dataset(dataset_1, split2_df)
    np.testing.assert_almost_equal(
        dataset_full.temporal.numpy(), dataset_1.temporal.numpy()
    )
    np.testing.assert_almost_equal(dataset_full.max_size, dataset_1.max_size)
    np.testing.assert_almost_equal(dataset_full.indptr, dataset_1.indptr)

@pytest.fixture
def temporal_df():
    n_static_features = 0
    n_temporal_features = 2

    return generate_series(n_series=100,
                                  min_length=50,
                                  max_length=100,
                                  n_static_features=n_static_features,
                                  n_temporal_features=n_temporal_features,
                                  equal_ends=False)


# Testing trim_dataset functionality
def test_trim_dataset(temporal_df):

    n_static_features = 2
    n_temporal_features = 4
    _, static_df = generate_series(n_series=1000,
                                            n_static_features=n_static_features,
                                            n_temporal_features=n_temporal_features,
                                            equal_ends=False)
    dataset, indices, dates, ds = TimeSeriesDataset.from_df(
        df=temporal_df, static_df=static_df
    )
    left_trim = 10
    right_trim = 20
    dataset_trimmed = dataset.trim_dataset(
        dataset, left_trim=left_trim, right_trim=right_trim
    )

    np.testing.assert_almost_equal(
        dataset.temporal[
            dataset.indptr[50] + left_trim : dataset.indptr[51] - right_trim
        ].numpy(),
        dataset_trimmed.temporal[
            dataset_trimmed.indptr[50] : dataset_trimmed.indptr[51]
        ].numpy(),
    )


def test_polars_integration(temporal_df):
    _, static_df = generate_series(n_series=1000,
                                   n_static_features=2,
                                   n_temporal_features=4,
                                   equal_ends=False)
    dataset, indices, dates, ds = TimeSeriesDataset.from_df(df=temporal_df, static_df=static_df)



    temporal_df2 = temporal_df.copy()
    for col in ('unique_id', 'temporal_0', 'temporal_1'):
        temporal_df2[col] = temporal_df2[col].cat.codes
    temporal_pl = polars.from_pandas(temporal_df2).sample(fraction=1.0)
    static_pl = polars.from_pandas(static_df.assign(unique_id=lambda df: df['unique_id'].astype('int64')))
    dataset_pl, indices_pl, dates_pl, ds_pl = TimeSeriesDataset.from_df(df=temporal_pl, static_df=static_df)

    # Compare array-like attributes with .all()
    for attr in ('static_cols', 'temporal_cols'):
        assert getattr(dataset, attr).equals(getattr(dataset_pl, attr))

    # Compare scalar attributes directly
    for attr in ('min_size', 'max_size', 'n_groups'):
        assert getattr(dataset, attr) == getattr(dataset_pl, attr)

    torch.testing.assert_close(dataset.temporal, dataset_pl.temporal)
    torch.testing.assert_close(dataset.static, dataset_pl.static)
    pd.testing.assert_series_equal(indices.astype('int64'), indices_pl.to_pandas().astype('int64'))
    pd.testing.assert_index_equal(dates, pd.Index(dates_pl, name='ds'))
    np.testing.assert_array_equal(ds, ds_pl)
    np.testing.assert_array_equal(dataset.indptr, dataset_pl.indptr)


@pytest.mark.parametrize("frame", [pd.DataFrame, polars.DataFrame])
def test_static_features_follow_the_series_in_df(frame):
    df = frame(
        {"unique_id": ["b", "b", "c", "c"], "ds": [1, 2, 1, 2], "y": [1.0, 2.0, 3.0, 4.0]}
    )
    # Holds an extra series that sorts first, and is itself out of order.
    static_df = frame({"unique_id": ["c", "a", "b"], "group": [30.0, 10.0, 20.0]})

    dataset, uids, *_ = TimeSeriesDataset.from_df(df=df, static_df=static_df)

    assert uids.to_list() == ["b", "c"]
    np.testing.assert_array_equal(dataset.static.numpy(), [[20.0], [30.0]])


@pytest.mark.parametrize(
    "static_ids",
    [["b"], ["a", "b"], ["b", "c", "c"], ["b", "b"]],
    ids=["short", "wrong", "dup", "dup_for_missing"],
)
def test_static_df_must_cover_each_series_once(static_ids):
    df = pd.DataFrame(
        {"unique_id": ["b", "b", "c", "c"], "ds": [1, 2, 1, 2], "y": [1.0, 2.0, 3.0, 4.0]}
    )
    static_df = pd.DataFrame(
        {"unique_id": static_ids, "group": np.arange(len(static_ids), dtype=float)}
    )

    with pytest.raises(ValueError, match="exactly one row for each series"):
        TimeSeriesDataset.from_df(df=df, static_df=static_df)


def test_eq_same_values(setup_data):
    temporal_df, sorted_temporal_df, unsorted_temporal_df = setup_data
    dataset, *_ = TimeSeriesDataset.from_df(df=unsorted_temporal_df)
    replica, *_ = TimeSeriesDataset.from_df(df=sorted_temporal_df)

    assert dataset == dataset
    assert dataset == replica


def test_eq_differing_values(setup_data):
    temporal_df, *_ = setup_data
    dataset, *_ = TimeSeriesDataset.from_df(df=temporal_df)

    modified_df = temporal_df.copy()
    modified_df.loc[modified_df.index[0], "y"] += 1.0
    modified, *_ = TimeSeriesDataset.from_df(df=modified_df)

    assert dataset != modified


def test_eq_static_features(setup_data):
    temporal_df, *_ = setup_data
    _, static_df = generate_series(
        n_series=1000, n_static_features=2, n_temporal_features=0, equal_ends=False
    )
    with_static, *_ = TimeSeriesDataset.from_df(df=temporal_df, static_df=static_df)
    replica, *_ = TimeSeriesDataset.from_df(df=temporal_df, static_df=static_df)
    without_static, *_ = TimeSeriesDataset.from_df(df=temporal_df)

    assert with_static == replica
    assert with_static != without_static


def test_eq_non_dataset_returns_false(setup_data):
    temporal_df, *_ = setup_data
    dataset, *_ = TimeSeriesDataset.from_df(df=temporal_df)

    assert (dataset == "not-a-dataset") is False


def _dataset(temporal, indptr, cols=("y",), y_idx=0, static=None, static_cols=None):
    return TimeSeriesDataset(
        temporal=temporal,
        temporal_cols=list(cols),
        indptr=np.array(indptr),
        y_idx=y_idx,
        static=static,
        static_cols=static_cols,
    )


def test_eq_requires_exact_values():
    """A near-equal value is a different dataset: equality carries no tolerance."""
    base = torch.tensor([[112.0], [113.0]])
    dataset = _dataset(base, [0, 2])
    nudged = _dataset(torch.tensor([[112.001], [113.0]]), [0, 2])

    assert dataset != nudged


def test_eq_treats_a_matching_nan_as_equal():
    """equal_nan=True is a stated design choice: the same NaN compares equal."""
    nan = float("nan")
    left = _dataset(torch.tensor([[1.0], [nan]]), [0, 2])
    right = _dataset(torch.tensor([[1.0], [nan]]), [0, 2])

    assert left == right


def test_eq_compares_the_series_boundaries():
    """Same flat values split into different series are different datasets."""
    flat = torch.tensor([[1.0], [2.0], [3.0]])
    one_series = _dataset(flat, [0, 3])
    two_series = _dataset(flat.clone(), [0, 1, 3])

    assert one_series != two_series


def test_eq_compares_the_column_names():
    """Same values under a different column name are a different dataset."""
    values = torch.tensor([[1.0], [2.0]])
    target = _dataset(values, [0, 2], cols=("y",))
    renamed = _dataset(values.clone(), [0, 2], cols=("target",))

    assert target != renamed


def test_eq_compares_static_values():
    """The same static columns with different values are a different dataset."""
    temporal = torch.tensor([[1.0], [2.0]])
    left = _dataset(temporal, [0, 2], static=torch.tensor([[10.0]]), static_cols=["s"])
    right = _dataset(
        temporal.clone(), [0, 2], static=torch.tensor([[11.0]]), static_cols=["s"]
    )

    assert left != right


def test_eq_compares_the_static_column_names():
    """The same static values under different names are a different dataset."""
    temporal, static = torch.tensor([[1.0], [2.0]]), torch.tensor([[10.0]])
    left = _dataset(temporal, [0, 2], static=static, static_cols=["a"])
    right = _dataset(temporal.clone(), [0, 2], static=static.clone(), static_cols=["b"])

    assert left != right


def test_eq_compares_y_idx():
    """Datasets that differ only in which column is the target are different."""
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert _dataset(values, [0, 2], cols=("y", "x")) != _dataset(
        values.clone(), [0, 2], cols=("y", "x"), y_idx=1
    )


def test_eq_allows_static_without_column_names():
    """static set with static_cols=None compares values and does not raise."""
    temporal = torch.tensor([[1.0], [2.0]])
    left = _dataset(temporal, [0, 2], static=torch.tensor([[10.0]]))
    same = _dataset(temporal.clone(), [0, 2], static=torch.tensor([[10.0]]))
    other = _dataset(temporal.clone(), [0, 2], static=torch.tensor([[11.0]]))

    assert left == same
    assert left != other
