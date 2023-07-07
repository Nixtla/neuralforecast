---
title: PyTorch Dataset/Loader
---

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
%load_ext autoreload
%autoreload 2
```

</details>

:::

> Torch Dataset for Time Series

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
from fastcore.test import test_eq
from nbdev.showdoc import show_doc
from neuralforecast.utils import generate_series
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class TimeSeriesLoader(DataLoader):
    """TimeSeriesLoader DataLoader.
    [Source code](https://github.com/Nixtla/neuralforecast1/blob/main/neuralforecast/tsdataset.py).

    Small change to PyTorch's Data loader. 
    Combines a dataset and a sampler, and provides an iterable over the given dataset.

    The class `~torch.utils.data.DataLoader` supports both map-style and
    iterable-style datasets with single- or multi-process loading, customizing
    loading order and optional automatic batching (collation) and memory pinning.    
    
    **Parameters:**<br>
    `batch_size`: (int, optional): how many samples per batch to load (default: 1).<br>
    `shuffle`: (bool, optional): set to `True` to have the data reshuffled at every epoch (default: `False`).<br>
    `sampler`: (Sampler or Iterable, optional): defines the strategy to draw samples from the dataset.<br>
                Can be any `Iterable` with `__len__` implemented. If specified, `shuffle` must not be specified.<br>
    """
    def __init__(self, dataset, **kwargs):
        if 'collate_fn' in kwargs:
            kwargs.pop('collate_fn')
        kwargs_ = {**kwargs, **dict(collate_fn=self._collate_fn)}
        DataLoader.__init__(self, dataset=dataset, **kwargs_)
    
    def _collate_fn(self, batch):
        elem = batch[0]
        elem_type = type(elem)

        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel, device=elem.device)
                out = elem.new(storage).resize_(len(batch), *list(elem.size()))
            return torch.stack(batch, 0, out=out)

        elif isinstance(elem, Mapping):
            if elem['static'] is None:
                return dict(temporal=self.collate_fn([d['temporal'] for d in batch]),
                            temporal_cols = elem['temporal_cols'])
            
            return dict(static=self.collate_fn([d['static'] for d in batch]),
                        static_cols = elem['static_cols'],
                        temporal=self.collate_fn([d['temporal'] for d in batch]),
                        temporal_cols = elem['temporal_cols'])

        raise TypeError(f'Unknown {elem_type}')
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(TimeSeriesLoader)
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class TimeSeriesDataset(Dataset):

    def __init__(self,
                 temporal,
                 temporal_cols,
                 indptr,
                 max_size: int,
                 min_size: int,
                 static=None,
                 static_cols=None,
                 sorted=False):
        super().__init__()
        
        self.temporal = torch.tensor(temporal, dtype=torch.float)
        self.temporal_cols = pd.Index(list(temporal_cols))

        if static is not None:
            self.static = torch.tensor(static, dtype=torch.float)
            self.static_cols = static_cols
        else:
            self.static = static
            self.static_cols = static_cols

        self.indptr = indptr
        self.n_groups = self.indptr.size - 1
        self.max_size = max_size
        self.min_size = min_size

        # Upadated flag. To protect consistency, dataset can only be updated once
        self.updated = False
        self.sorted = sorted

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # Parse temporal data and pad its left
            temporal = torch.zeros(size=(len(self.temporal_cols), self.max_size),
                                   dtype=torch.float32)
            ts = self.temporal[self.indptr[idx] : self.indptr[idx + 1], :]
            temporal[:len(self.temporal_cols), -len(ts):] = ts.permute(1, 0)

            # Add static data if available
            static = None if self.static is None else self.static[idx,:]

            item = dict(temporal=temporal, temporal_cols=self.temporal_cols,
                        static=static, static_cols=self.static_cols)

            return item
        raise ValueError(f'idx must be int, got {type(idx)}')

    def __len__(self):
        return self.n_groups

    def __repr__(self):
        return f'TimeSeriesDataset(n_data={self.data.size:,}, n_groups={self.n_groups:,})'

    def __eq__(self, other):
        if not hasattr(other, 'data') or not hasattr(other, 'indptr'):
            return False
        return np.allclose(self.data, other.data) and np.array_equal(self.indptr, other.indptr)

    @staticmethod
    def update_dataset(dataset, future_df):
        """Add future observations to the dataset.
        """        
        
        # Protect consistency
        future_df = future_df.copy()

        # Add Nones to missing columns (without available_mask)
        temporal_cols = dataset.temporal_cols.copy()
        for col in temporal_cols:
            if col not in future_df.columns:
                future_df[col] = None
            if col == 'available_mask':
                future_df[col] = 1
        
        # Sort columns to match self.temporal_cols (without available_mask)
        future_df = future_df[ ['unique_id','ds'] + temporal_cols.tolist() ]

        # Process future_df
        futr_dataset, *_ = dataset.from_df(df=future_df, sort_df=dataset.sorted)

        # Define and fill new temporal with updated information
        len_temporal, col_temporal = dataset.temporal.shape
        new_temporal = torch.zeros(size=(len_temporal+len(future_df), col_temporal))
        new_indptr = [0]
        new_max_size = 0

        acum = 0
        for i in range(dataset.n_groups):
            series_length = dataset.indptr[i + 1] - dataset.indptr[i]
            new_length = series_length + futr_dataset.indptr[i + 1] - futr_dataset.indptr[i]
            new_temporal[acum:(acum+series_length), :] = dataset.temporal[dataset.indptr[i] : dataset.indptr[i + 1], :]
            new_temporal[(acum+series_length):(acum+new_length), :] = \
                                 futr_dataset.temporal[futr_dataset.indptr[i] : futr_dataset.indptr[i + 1], :]
            
            acum += new_length
            new_indptr.append(acum)
            if new_length > new_max_size:
                new_max_size = new_length
        
        # Define new dataset
        updated_dataset = TimeSeriesDataset(temporal=new_temporal,
                                            temporal_cols=dataset.temporal_cols.copy(),
                                            indptr=np.array(new_indptr).astype(np.int32),
                                            max_size=new_max_size,
                                            min_size=dataset.min_size,
                                            static=dataset.static,
                                            static_cols=dataset.static_cols,
                                            sorted=dataset.sorted)

        return updated_dataset
    
    @staticmethod
    def trim_dataset(dataset, left_trim: int = 0, right_trim: int = 0):
        """
        Trim temporal information from a dataset.
        Returns temporal indexes [t+left:t-right] for all series.
        """
        if dataset.min_size <= left_trim + right_trim:
            raise Exception(f'left_trim + right_trim ({left_trim} + {right_trim}) \
                                must be lower than the shorter time series ({dataset.min_size})')

        # Define and fill new temporal with trimmed information        
        len_temporal, col_temporal = dataset.temporal.shape
        total_trim = (left_trim + right_trim) * dataset.n_groups
        new_temporal = torch.zeros(size=(len_temporal-total_trim, col_temporal))
        new_indptr = [0]

        acum = 0
        for i in range(dataset.n_groups):
            series_length = dataset.indptr[i + 1] - dataset.indptr[i]
            new_length = series_length - left_trim - right_trim
            new_temporal[acum:(acum+new_length), :] = dataset.temporal[dataset.indptr[i]+left_trim : \
                                                                       dataset.indptr[i + 1]-right_trim, :]
            acum += new_length
            new_indptr.append(acum)

        new_max_size = dataset.max_size-left_trim-right_trim
        new_min_size = dataset.min_size-left_trim-right_trim
        
        # Define new dataset
        updated_dataset = TimeSeriesDataset(temporal=new_temporal,
                                            temporal_cols= dataset.temporal_cols.copy(),
                                            indptr=np.array(new_indptr).astype(np.int32),
                                            max_size=new_max_size,
                                            min_size=new_min_size,
                                            static=dataset.static,
                                            static_cols=dataset.static_cols,
                                            sorted=dataset.sorted)

        return updated_dataset

    @staticmethod
    def from_df(df, static_df=None, sort_df=False):
        # TODO: protect on equality of static_df + df indexes

        # Define indexes if not given
        if df.index.name != 'unique_id':
            df = df.set_index('unique_id')
            if static_df is not None:
                static_df = static_df.set_index('unique_id')

        df = df.set_index('ds', append=True)
        
        # Sort data by index
        if not df.index.is_monotonic_increasing and sort_df:
            df = df.sort_index()

            if static_df is not None:
                static_df = static_df.sort_index()

        # Create auxiliary temporal indices 'indptr'
        temporal = df.values.astype(np.float32)
        temporal_cols = df.columns
        indices_sizes = df.index.get_level_values('unique_id').value_counts(sort=False)
        indices = indices_sizes.index
        sizes = indices_sizes.values
        max_size = max(sizes)
        min_size = min(sizes)
        cum_sizes = sizes.cumsum()
        dates = df.index.get_level_values('ds')[cum_sizes - 1]
        indptr = np.append(0, cum_sizes).astype(np.int32)

        # Add Available mask efficiently (without adding column to df)
        if 'available_mask' not in df.columns:
            available_mask = np.ones((len(temporal),1), dtype=np.float32)
            temporal = np.append(temporal, available_mask, axis=1)
            temporal_cols = temporal_cols.append(pd.Index(['available_mask']))

        # Static features
        if static_df is not None:
            static = static_df.values
            static_cols = static_df.columns
        else:
            static = None
            static_cols = None

        dataset = TimeSeriesDataset(
                    temporal=temporal, temporal_cols=temporal_cols,
                    static=static, static_cols=static_cols,
                    indptr=indptr, max_size=max_size, min_size=min_size, sorted=sort_df)
        return dataset, indices, dates, df.index
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(TimeSeriesDataset)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# Testing sort_df=True functionality
temporal_df = generate_series(n_series=1000, 
                         n_temporal_features=0, equal_ends=False)
sorted_temporal_df = temporal_df.sort_values(['unique_id', 'ds'])
unsorted_temporal_df = sorted_temporal_df.sample(frac=1.0)
dataset, indices, dates, ds = TimeSeriesDataset.from_df(df=unsorted_temporal_df,
                                                        sort_df=True)

np.testing.assert_allclose(dataset.temporal[:,:-1], 
                           sorted_temporal_df.drop(columns='ds').values)
test_eq(indices, sorted_temporal_df.index.unique(level='unique_id'))
test_eq(dates, temporal_df.groupby('unique_id')['ds'].max().values)
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class TimeSeriesDataModule(pl.LightningDataModule):
    
    def __init__(
            self, 
            dataset: TimeSeriesDataset,
            batch_size=32, 
            valid_batch_size=1024,
            num_workers=0,
            drop_last=False
        ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
    
    def train_dataloader(self):
        loader = TimeSeriesLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=self.drop_last
        )
        return loader
    
    def val_dataloader(self):
        loader = TimeSeriesLoader(
            self.dataset, 
            batch_size=self.valid_batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=self.drop_last
        )
        return loader
    
    def predict_dataloader(self):
        loader = TimeSeriesLoader(
            self.dataset,
            batch_size=self.valid_batch_size, 
            num_workers=self.num_workers,
            shuffle=False
        )
        return loader
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(TimeSeriesDataModule)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
batch_size = 128
data = TimeSeriesDataModule(dataset=dataset, 
                            batch_size=batch_size, drop_last=True)
for batch in data.train_dataloader():
    test_eq(batch['temporal'].shape, (batch_size, 2, 500))
    test_eq(batch['temporal_cols'], ['y', 'available_mask'])
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
batch_size = 128
n_static_features = 2
n_temporal_features = 4
temporal_df, static_df = generate_series(n_series=1000,
                                         n_static_features=n_static_features,
                                         n_temporal_features=n_temporal_features, 
                                         equal_ends=False)

dataset, indices, dates, ds = TimeSeriesDataset.from_df(df=temporal_df,
                                                        static_df=static_df,
                                                        sort_df=True)
data = TimeSeriesDataModule(dataset=dataset,
                            batch_size=batch_size, drop_last=True)

for batch in data.train_dataloader():
    test_eq(batch['temporal'].shape, (batch_size, n_temporal_features + 2, 500))
    test_eq(batch['temporal_cols'],
            ['y'] + [f'temporal_{i}' for i in range(n_temporal_features)] + ['available_mask'])
    
    test_eq(batch['static'].shape, (batch_size, n_static_features))
    test_eq(batch['static_cols'], [f'static_{i}' for i in range(n_static_features)])
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# Testing sort_df=True functionality
temporal_df = generate_series(n_series=2,
                              n_temporal_features=2, equal_ends=True)
temporal_df = temporal_df.groupby('unique_id').tail(10)
temporal_df = temporal_df.reset_index()
temporal_full_df = temporal_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
temporal_full_df.loc[temporal_full_df.ds > '2001-05-11', ['y', 'temporal_0']] = None

split1_df = temporal_full_df.loc[temporal_full_df.ds <= '2001-05-11']
split2_df = temporal_full_df.loc[temporal_full_df.ds > '2001-05-11']
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# Testing available mask
temporal_df_w_mask = temporal_df.copy()
temporal_df_w_mask['available_mask'] = 1

# Mask with all 1's
dataset, indices, dates, ds = TimeSeriesDataset.from_df(df=temporal_df_w_mask,
                                                        sort_df=True)
mask_average = dataset.temporal[:, -1].mean()
np.testing.assert_almost_equal(mask_average, 1.0000)

# Add 0's to available mask
temporal_df_w_mask.loc[temporal_df_w_mask.ds > '2001-05-11', 'available_mask'] = 0
dataset, indices, dates, ds = TimeSeriesDataset.from_df(df=temporal_df_w_mask,
                                                        sort_df=True)
mask_average = dataset.temporal[:, -1].mean()
np.testing.assert_almost_equal(mask_average, 0.7000)

# Available mask not in last column
temporal_df_w_mask = temporal_df_w_mask[['unique_id','ds','y','available_mask', 'temporal_0','temporal_1']]
dataset, indices, dates, ds = TimeSeriesDataset.from_df(df=temporal_df_w_mask,
                                                        sort_df=True)
mask_average = dataset.temporal[:, 1].mean()
np.testing.assert_almost_equal(mask_average, 0.7000)
```

</details>

:::

<details>
<summary>Code</summary>

``` python
# To test correct future_df wrangling of the `update_df` method
# We are checking that we are able to recover the AirPassengers dataset
# using the dataframe or splitting it into parts and initializing.
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# FULL DATASET
dataset_full, indices_full, dates_full, ds_full = TimeSeriesDataset.from_df(df=temporal_full_df,
                                                                            sort_df=False)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# SPLIT_1 DATASET
dataset_1, indices_1, dates_1, ds_1 = TimeSeriesDataset.from_df(df=split1_df,
                                                                sort_df=False)
dataset_1 = dataset_1.update_dataset(dataset_1, split2_df)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
np.testing.assert_almost_equal(dataset_full.temporal.numpy(), dataset_1.temporal.numpy())
test_eq(dataset_full.max_size, dataset_1.max_size)
test_eq(dataset_full.indptr, dataset_1.indptr)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# Testing trim_dataset functionality
n_static_features = 0
n_temporal_features = 2
temporal_df = generate_series(n_series=100,
                              min_length=50,
                              max_length=100,
                              n_static_features=n_static_features,
                              n_temporal_features=n_temporal_features, 
                              equal_ends=False)
dataset, indices, dates, ds = TimeSeriesDataset.from_df(df=temporal_df,
                                                        static_df=static_df,
                                                        sort_df=True)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
left_trim = 10
right_trim = 20
dataset_trimmed = dataset.trim_dataset(dataset, left_trim=left_trim, right_trim=right_trim)

np.testing.assert_almost_equal(dataset.temporal[dataset.indptr[50]+left_trim:dataset.indptr[51]-right_trim].numpy(),
                               dataset_trimmed.temporal[dataset_trimmed.indptr[50]:dataset_trimmed.indptr[51]].numpy())
```

</details>

:::

