---
description: >-
  NeuralForecast contains a collection NumPy loss functions aimed to be used
  during the models' evaluation.
output-file: losses.numpy.html
title: NumPy Evaluation
---


The most important train signal is the forecast error, which is the
difference between the observed value $y_{\tau}$ and the prediction
$\hat{y}_{\tau}$, at time $y_{\tau}$:

$$e_{\tau} = y_{\tau}-\hat{y}_{\tau} \qquad \qquad \tau \in \{t+1,\dots,t+H \}$$

The train loss summarizes the forecast errors in different evaluation
metrics.

# 1. Scale-dependent Errors

These metrics are on the same scale as the data.

## Mean Absolute Error

::: neuralforecast.losses.numpy.mae
    options:
      members: []
      heading_level: 3

![](imgs_losses/mae_loss.png)

## Mean Squared Error

::: neuralforecast.losses.numpy.mse
    options:
      members: []
      heading_level: 3

![](imgs_losses/mse_loss.png)

## Root Mean Squared Error

::: neuralforecast.losses.numpy.rmse
    options:
      members: []
      heading_level: 3

![](imgs_losses/rmse_loss.png)

# 2. Percentage errors

These metrics are unit-free, suitable for comparisons across series.

## Mean Absolute Percentage Error

::: neuralforecast.losses.numpy.mape
    options:
      members: []
      heading_level: 3

![](imgs_losses/mape_loss.png)

## SMAPE

::: neuralforecast.losses.numpy.smape
    options:
      members: []
      heading_level: 3

# 3. Scale-independent Errors

These metrics measure the relative improvements versus baselines.

## Mean Absolute Scaled Error

::: neuralforecast.losses.numpy.mase
    options:
      members: []
      heading_level: 3

![](imgs_losses/mase_loss.png)

## Relative Mean Absolute Error

::: neuralforecast.losses.numpy.rmae
    options:
      members: []
      heading_level: 3

![](imgs_losses/rmae_loss.png)

# 4. Probabilistic Errors

These measure absolute deviation non-symmetrically, that produce
under/over estimation.

## Quantile Loss

::: neuralforecast.losses.numpy.quantile_loss
    options:
      members: []
      heading_level: 3

![](imgs_losses/q_loss.png)

## Multi-Quantile Loss

::: neuralforecast.losses.numpy.mqloss
    options:
      members: []
      heading_level: 3

![](imgs_losses/mq_loss.png)

# Examples and Validation


```python
import unittest
import torch as t 
import numpy as np

from neuralforecast.losses.pytorch import (
    MAE, MSE, RMSE,      # unscaled errors
    MAPE, SMAPE,         # percentage errors
    MASE,                # scaled error
    QuantileLoss, MQLoss # probabilistic errors
)

from neuralforecast.losses.numpy import (
    mae, mse, rmse,              # unscaled errors
    mape, smape,                 # percentage errors
    mase,                        # scaled error
    quantile_loss, mqloss        # probabilistic errors
)
```

