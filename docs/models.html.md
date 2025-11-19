---
description: >-
  NeuralForecast contains user-friendly implementations of neural forecasting
  models that allow for easy transition of computing capabilities (GPU/CPU),
  computation parallelization, and hyperparameter tuning.
output-file: models.html
title: AutoModels
---

All the NeuralForecast models are "global" because we train them with
all the series from the input pd.DataFrame data `Y_df`, yet the
optimization objective is, momentarily, "univariate" as it does not
consider the interaction between the output predictions across time
series. Like the StatsForecast library, `core.NeuralForecast` allows you
to explore collections of models efficiently and contains functions for
convenient wrangling of input and output pd.DataFrames predictions.

First we load the AirPassengers dataset such that you can run all the
examples.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengersDF as Y_df
```

```python
# Split train/test and declare time series dataset
Y_train_df = Y_df[Y_df.ds<='1959-12-31'] # 132 train
Y_test_df = Y_df[Y_df.ds>'1959-12-31']   # 12 test
dataset, *_ = TimeSeriesDataset.from_df(Y_train_df)
```

# Automatic Forecasting

## RNN-Based

::: neuralforecast.auto.AutoRNN
    options:
      members: []
      heading_level: 3

::: neuralforecast.auto.AutoLSTM
    options:
      members: []
      heading_level: 3

::: neuralforecast.auto.AutoGRU
    options:
      members: []
      heading_level: 3

::: neuralforecast.auto.AutoDilatedRNN
    options:
      members: []
      heading_level: 3

::: neuralforecast.auto.AutoTCN
    options:
      members: []
      heading_level: 3

## MLP-Based

::: neuralforecast.auto.AutoMLP
    options:
      members: []
      heading_level: 3

::: neuralforecast.auto.AutoNHITS
    options:
      members: []
      heading_level: 3

::: neuralforecast.auto.AutoNBEATS
    options:
      members: []
      heading_level: 3

## Transformer-Based

::: neuralforecast.auto.AutoAutoformer
    options:
      members: []
      heading_level: 3

::: neuralforecast.auto.AutoInformer
    options:
      members: []
      heading_level: 3

::: neuralforecast.auto.AutoFEDformer
    options:
      members: []
      heading_level: 3

::: neuralforecast.auto.AutoPatchTST
    options:
      members: []
      heading_level: 3

::: neuralforecast.auto.AutoTFT
    options:
      members: []
      heading_level: 3

::: neuralforecast.auto.AutoVanillaTransformer
    options:
      members: []
      heading_level: 3

::: neuralforecast.auto.AutoiTransformer
    options:
      members: []
      heading_level: 3
