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