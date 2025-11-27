---
description: >-
  PyTorch Dataset and DataLoader classes for time series. TimeSeriesDataset and TimeSeriesDataModule for efficient batch processing with Lightning integration.
output-file: tsdataset.html
title: PyTorch Dataset/Loader
---

## Torch Time Series Dataset

::: neuralforecast.tsdataset.TimeSeriesLoader

::: neuralforecast.tsdataset.BaseTimeSeriesDataset

::: neuralforecast.tsdataset.LocalFilesTimeSeriesDataset

::: neuralforecast.tsdataset.TimeSeriesDataset

::: neuralforecast.tsdataset.TimeSeriesDataModule

### Example

```python
import lightning.pytorch as L
import torch.utils.data as data
from pytorch_lightning.demos.boring_classes import RandomDataset

class MyDataModule(L.LightningDataModule):
    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        ...

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        dataset = RandomDataset(1, 100)
        self.train, self.val, self.test = data.random_split(
            dataset, [80, 10, 10], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return data.DataLoader(self.train)

    def val_dataloader(self):
        return data.DataLoader(self.val)

    def test_dataloader(self):
        return data.DataLoader(self.test)

    def on_exception(self, exception):
        # clean up state after the trainer faced an exception
        ...

    def teardown(self):
        # clean up state after the trainer stops, delete files...
        # called on every process in DDP
        ...*
```


```python
# To test correct future_df wrangling of the `update_df` method
# We are checking that we are able to recover the AirPassengers dataset
# using the dataframe or splitting it into parts and initializing.
```

