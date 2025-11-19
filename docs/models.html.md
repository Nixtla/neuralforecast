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

# 1. Automatic Forecasting

## A. RNN-Based

::: neuralforecast.auto.AutoRNN
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoRNN.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=-1, encoder_hidden_size=8)
model = AutoRNN(h=12, config=config, num_samples=1, cpus=1)

model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoRNN(h=12, config=None, num_samples=1, cpus=1, backend='optuna')
```

::: neuralforecast.auto.AutoLSTM
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoLSTM.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=-1, encoder_hidden_size=8)
model = AutoLSTM(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoLSTM(h=12, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoGRU
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoGRU.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=-1, encoder_hidden_size=8)
model = AutoGRU(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoGRU(h=12, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoTCN
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoTCN.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=-1, encoder_hidden_size=8)
model = AutoTCN(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoTCN(h=12, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoDeepAR
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoDeepAR.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, lstm_hidden_size=8)
model = AutoDeepAR(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoDeepAR(h=12, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoDilatedRNN
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoDilatedRNN.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=-1, encoder_hidden_size=8)
model = AutoDilatedRNN(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoDilatedRNN(h=12, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoBiTCN
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoBiTCN.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, hidden_size=8)
model = AutoBiTCN(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoBiTCN(h=12, config=None, backend='optuna')
```

## B. MLP-Based

::: neuralforecast.auto.AutoMLP
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoMLP.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, hidden_size=8)
model = AutoMLP(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoMLP(h=12, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoNBEATS
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoNBEATS.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12,
              mlp_units=3*[[8, 8]])
model = AutoNBEATS(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoNBEATS(h=12, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoNBEATSx
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoNBEATSx.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12,
              mlp_units=3*[[8, 8]])
model = AutoNBEATSx(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoNBEATSx(h=12, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoNHITS
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoNHITS.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, 
              mlp_units=3 * [[8, 8]])
model = AutoNHITS(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoNHITS(h=12, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoDLinear
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoDLinear.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12)
model = AutoDLinear(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoDLinear(h=12, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoNLinear
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoNLinear.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12)
model = AutoNLinear(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoNLinear(h=12, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoTiDE
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoTiDE.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12)
model = AutoTiDE(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoTiDE(h=12, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoDeepNPTS
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoDeepNPTS.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12)
model = AutoDeepNPTS(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoDeepNPTS(h=12, config=None, backend='optuna')
```

## C. KAN-Based

::: neuralforecast.auto.AutoKAN
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoKAN.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12)
model = AutoKAN(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoKAN(h=12, config=None, backend='optuna')
```

## D. Transformer-Based

::: neuralforecast.auto.AutoTFT
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoTFT.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, hidden_size=8)
model = AutoTFT(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoTFT(h=12, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoVanillaTransformer
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoVanillaTransformer.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, hidden_size=8)
model = AutoVanillaTransformer(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoVanillaTransformer(h=12, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoInformer
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoInformer.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, hidden_size=8)
model = AutoInformer(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoInformer(h=12, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoAutoformer
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoAutoformer.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, hidden_size=8)
model = AutoAutoformer(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoAutoformer(h=12, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoFEDformer
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoFEDFormer.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, hidden_size=64)
model = AutoFEDformer(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoFEDformer(h=12, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoPatchTST
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoPatchTST.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, hidden_size=16)
model = AutoPatchTST(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoPatchTST(h=12, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoiTransformer
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoiTransformer.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, hidden_size=16)
model = AutoiTransformer(h=12, n_series=1, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoiTransformer(h=12, n_series=1, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoTimeXer
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoTimeXer.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, patch_len=12)
model = AutoTimeXer(h=12, n_series=1, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoTimeXer(h=12, n_series=1, config=None, backend='optuna')
```

## E. CNN Based

::: neuralforecast.auto.AutoTimesNet
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoTimesNet.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, hidden_size=32)
model = AutoTimesNet(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoTimesNet(h=12, config=None, backend='optuna')
```

## F. Multivariate

::: neuralforecast.auto.AutoStemGNN
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoStemGNN.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12)
model = AutoStemGNN(h=12, n_series=1, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoStemGNN(h=12, n_series=1, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoHINT
    options:
      members: [__init__]
      heading_level: 3

```python
# Perform a simple hyperparameter optimization with 
# NHITS and then reconcile with HINT
from neuralforecast.losses.pytorch import GMM, sCRPS

base_config = dict(max_steps=1, val_check_steps=1, input_size=8)
base_model = AutoNHITS(h=4, loss=GMM(n_components=2, quantiles=quantiles), 
                       config=base_config, num_samples=1, cpus=1)
model = HINT(h=4, S=S_df.values,
             model=base_model,  reconciliation='MinTraceOLS')

model.fit(dataset=dataset)
y_hat = model.predict(dataset=hint_dataset)

# Perform a conjunct hyperparameter optimization with 
# NHITS + HINT reconciliation configurations
nhits_config = {
       "learning_rate": tune.choice([1e-3]),                                     # Initial Learning rate
       "max_steps": tune.choice([1]),                                            # Number of SGD steps
       "val_check_steps": tune.choice([1]),                                      # Number of steps between validation
       "input_size": tune.choice([5 * 12]),                                      # input_size = multiplier * horizon
       "batch_size": tune.choice([7]),                                           # Number of series in windows
       "windows_batch_size": tune.choice([256]),                                 # Number of windows in batch
       "n_pool_kernel_size": tune.choice([[2, 2, 2], [16, 8, 1]]),               # MaxPool's Kernelsize
       "n_freq_downsample": tune.choice([[168, 24, 1], [24, 12, 1], [1, 1, 1]]), # Interpolation expressivity ratios
       "activation": tune.choice(['ReLU']),                                      # Type of non-linear activation
       "n_blocks":  tune.choice([[1, 1, 1]]),                                    # Blocks per each 3 stacks
       "mlp_units":  tune.choice([[[512, 512], [512, 512], [512, 512]]]),        # 2 512-Layers per block for each stack
       "interpolation_mode": tune.choice(['linear']),                            # Type of multi-step interpolation
       "random_seed": tune.randint(1, 10),
       "reconciliation": tune.choice(['BottomUp', 'MinTraceOLS', 'MinTraceWLS'])
    }
model = AutoHINT(h=4, S=S_df.values,
                 cls_model=NHITS,
                 config=nhits_config,
                 loss=GMM(n_components=2, level=[80, 90]),
                 valid_loss=sCRPS(level=[80, 90]),
                 num_samples=1, cpus=1)
model.fit(dataset=dataset)
y_hat = model.predict(dataset=hint_dataset)
```

::: neuralforecast.auto.AutoTSMixer
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoTSMixer.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12)
model = AutoTSMixer(h=12, n_series=1, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoTSMixer(h=12, n_series=1, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoTSMixerx
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoTSMixerx.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12)
model = AutoTSMixerx(h=12, n_series=1, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoTSMixerx(h=12, n_series=1, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoMLPMultivariate
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoMLPMultivariate.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12)
model = AutoMLPMultivariate(h=12, n_series=1, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoMLPMultivariate(h=12, n_series=1, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoSOFTS
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoSOFTS.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, hidden_size=16)
model = AutoSOFTS(h=12, n_series=1, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoSOFTS(h=12, n_series=1, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoTimeMixer
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoTimeMixer.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, d_model=16)
model = AutoTimeMixer(h=12, n_series=1, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoTimeMixer(h=12, n_series=1, config=None, backend='optuna')
```

::: neuralforecast.auto.AutoRMoK
    options:
      members: [__init__]
      heading_level: 3

```python
# Use your own config or AutoRMoK.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, learning_rate=1e-2)
model = AutoRMoK(h=12, n_series=1, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Optuna
model = AutoRMoK(h=12, n_series=1, config=None, backend='optuna')
```