---
description: >-
  AutoModel classes for NeuralForecast hyperparameter optimization. Automated grid search, Bayesian optimization with Ray Tune for 34 forecasting architectures.
output-file: models.html
title: Automatic Forecasting
---

## 1. Introduction

All `NeuralForecast` models work out of the box with sensible default parameters. However, to achieve optimal forecasting performance on your specific dataset, hyperparameter optimization is highly recommended.

**Hyperparameter optimization** is the process of automatically finding the best configuration for a model by systematically exploring different combinations of parameters such as learning rate, hidden layer sizes, number of layers, and other architectural choices. Unlike model parameters that are learned during training, hyperparameters must be set before training begins.

NeuralForecast provides `AutoModel` classes that automate this optimization process. Each `AutoModel` wraps a corresponding forecasting model and uses techniques like grid search, random search, or Bayesian optimization to explore the hyperparameter space and identify the configuration that minimizes validation loss.

## BaseAuto Class

All `AutoModel` classes inherit from `BaseAuto`, which provides a unified interface for hyperparameter optimization. `BaseAuto` handles the complete optimization workflow:

1. **Search Space Definition**: Defines which hyperparameters to explore and their ranges
2. **Temporal Cross-Validation**: Splits data temporally to avoid look-ahead bias
3. **Training & Evaluation**: Runs multiple trials with different hyperparameter configurations
4. **Model Selection**: Selects the configuration with the best validation performance
5. **Refitting**: Trains the final model with optimal hyperparameters

The optimization process uses temporal cross-validation where the validation set sequentially precedes the test set. This ensures that hyperparameter selection is based on realistic forecasting scenarios. The validation loss guides the selection process, so it's important that the validation period is representative of future forecasting conditions.

::: neuralforecast.common._base_auto.BaseAuto
    options:
      members: []

## 2. Available AutoModels

NeuralForecast provides 34 `AutoModel` variants, each wrapping a specific forecasting model with automatic hyperparameter optimization. Each `AutoModel` has a `default_config` attribute that defines sensible search spaces for its corresponding model.

### RNN-Based Models
Recurrent neural networks for sequential forecasting:

- `AutoRNN`: [Basic recurrent neural network](./models.rnn.html)
- `AutoLSTM`: [Long Short-Term Memory network](./models.lstm.html)
- `AutoGRU`: [Gated Recurrent Unit network](./models.gru.html)
- `AutoDilatedRNN`: [RNN with dilated recurrent connections for capturing long-range dependencies](./models.dilated_rnn.html)
- `AutoxLSTM`: Extended LSTM with enhanced memory capabilities

### Transformer-Based Models
Attention-based architectures for capturing complex temporal patterns:

- `AutoTFT`: [Temporal Fusion Transformer with multi-horizon forecasting](./models.tft.html)
- `AutoVanillaTransformer`: [Standard transformer architecture](./models.vanillatransformer.hml)
- `AutoInformer`: [Efficient transformer for long sequence forecasting](./models.informer.html)
- `AutoAutoformer`: [Auto-correlation based transformer](./models.autoformer.html)
- `AutoFEDformer`: [Frequency enhanced decomposition transformer](./models.fedformer.html)
- `AutoPatchTST`: [Patched time series transformer](./models.patchtst.html)
- `AutoiTransformer`: [Inverted transformer for multivariate forecasting](./models.itransformer.html)
- `AutoTimeXer`: [Cross-series attention transformer](./models.timemixer.html)

### CNN-Based Models
Convolutional architectures for local pattern recognition:

- `AutoTCN`: [Temporal Convolutional Network with causal convolutions](./models.tcn.html)
- `AutoBiTCN`: [Bidirectional TCN](./models.bitcn.html)
- `AutoTimesNet`: [Multi-periodic convolution network](./models.timesnet.html)

### Linear and MLP Models
Simple yet effective linear and feed-forward architectures:

- `AutoMLP`: [Multi-layer Perceptron](./models.mlp.html)
- `AutoDLinear`: [Decomposition linear model](./models.dlinear.html)
- `AutoNLinear`: [Normalized linear model](./models.nlinear.html)
- `AutoTSMixer`: [Time Series Mixer architecture](./models.tsmixer.html)
- `AutoTSMixerx`: [TSMixer with exogenous variable support](./models.tsmixerx.html)
- `AutoMLPMultivariate`: [MLP for multivariate time series](./models.mlpmultivariate.html)

### Specialized Models
Models designed for specific forecasting scenarios:

- `AutoNBEATS`: [Neural Basis Expansion Analysis for interpretable forecasting](./models.nbeats.html)
- `AutoNBEATSx`: [NBEATS with exogenous variables](./models.nbeatsx.html)
- `AutoNHITS`: [Neural Hierarchical Interpolation for multi-horizon forecasting](./models.nhits.html)
- `AutoDeepAR`: [Probabilistic forecasting with autoregressive RNN](./models.deepar.html)
- `AutoDeepNPTS`: [Deep Non-Parametric Time Series model](./models.deepnpts.html)
- `AutoTiDE`: [Time-series Dense Encoder](./models.tide.html)
- `AutoKAN`: [Kolmogorov-Arnold Network for time series](./models.kan.html)
- `AutoStemGNN`: [Graph neural network for multivariate forecasting](./models.stemgnn.html)
- `AutoSOFTS`: [Spectral Optimal Fourier Transform model](./models.softs.html)
- `AutoTimeMixer`: [Temporal mixing architecture](./models.timemixer.html)
- `AutoRMoK`: [Random Mixture of Kernels](./models.rmok.html)
- `AutoHINT`: [Hierarchical forecasting with automatic reconciliation](./models.hint.html)

## 3. Usage Examples

### Data Preparation

First, prepare your time series data and create a `TimeSeriesDataset`:

```python
import numpy as np
import pandas as pd
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengersDF as Y_df

# Split data temporally: train and test
Y_train_df = Y_df[Y_df.ds <= '1959-12-31']  # 132 train observations
Y_test_df = Y_df[Y_df.ds > '1959-12-31']    # 12 test observations

# Create TimeSeriesDataset
dataset, *_ = TimeSeriesDataset.from_df(Y_train_df)
```

### Basic Usage

The simplest way to use an `AutoModel` is with its default search space:

```python
from neuralforecast.auto import AutoRNN

# Use your own config or AutoRNN.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=-1, encoder_hidden_size=8)
model = AutoRNN(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset, val_size=12)
y_hat = model.predict(dataset=dataset)
```

### Hierarchical Forecasting with AutoHINT

`AutoHINT` combines hyperparameter optimization with hierarchical reconciliation. This is useful when forecasting hierarchical time series (e.g., product hierarchies, geographic hierarchies).

#### Optimize Model, Then Apply Fixed Reconciliation

```python
from neuralforecast.auto import AutoNHITS
from neuralforecast.models.hint import HINT
from neuralforecast.losses.pytorch import GMM, sCRPS

base_model = AutoNHITS(
    h=4,
    loss=GMM(n_components=2, level=[80, 90]),  # Probabilistic loss
    num_samples=10
)

# Apply hierarchical reconciliation with the optimized model
# S: summing matrix defining the hierarchical structure
model = HINT(
    h=4,
    S=S_df.values,
    model=base_model,
    reconciliation='MinTraceOLS'
)

model.fit(dataset=dataset, val_size=4)
y_hat = model.predict(dataset=dataset)
```

#### Joint Optimization of Model and Reconciliation Method

```python
from neuralforecast.auto import AutoHINT
from neuralforecast.models.nhits import NHITS
from ray import tune

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

model = AutoHINT(
    h=4,
    S=S_df.values,
    cls_model=NHITS,
    config=nhits_config,
    loss=GMM(n_components=2, level=[80, 90]),
    valid_loss=sCRPS(level=[80, 90]),
    num_samples=20
)

model.fit(dataset=dataset, val_size=4)
y_hat = model.predict(dataset=dataset)
```