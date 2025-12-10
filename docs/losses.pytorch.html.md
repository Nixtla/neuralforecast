---
description: >-
  PyTorch loss functions for neural forecast training: MAE, MSE, MAPE, quantile losses, distribution losses, and robust losses for model optimization.
output-file: losses.pytorch.html
title: PyTorch Losses
---


The most important train signal is the forecast error, which is the
difference between the observed value $y_{\tau}$ and the prediction
$\hat{y}_{\tau}$, at time $y_{\tau}$:

$$e_{\tau} = y_{\tau}-\hat{y}_{\tau} \qquad \qquad \tau \in \{t+1,\dots,t+H \}$$

The train loss summarizes the forecast errors in different train
optimization objectives.

All the losses are `torch.nn.modules` which helps to automatically moved
them across CPU/GPU/TPU devices with Pytorch Lightning.

::: neuralforecast.losses.pytorch.BasePointLoss
    options:
      members: [__init__, __call__]
      heading_level: 3

# 1. Scale-dependent Errors

These metrics are on the same scale as the data.

## Mean Absolute Error (MAE)

::: neuralforecast.losses.pytorch.MAE
    options:
      members: [__init__, __call__]
      heading_level: 3

![](imgs_losses/mae_loss.png)

## Mean Squared Error (MSE)

::: neuralforecast.losses.pytorch.MSE
    options:
      members: [__init__, __call__]
      heading_level: 3

![](imgs_losses/mse_loss.png)

## Root Mean Squared Error (RMSE)

::: neuralforecast.losses.pytorch.RMSE
    options:
      members: [__init__, __call__]
      heading_level: 3

![](imgs_losses/rmse_loss.png)

# 2. Percentage errors

These metrics are unit-free, suitable for comparisons across series.

## Mean Absolute Percentage Error (MAPE)

::: neuralforecast.losses.pytorch.MAPE
    options:
      members: [__init__, __call__]
      heading_level: 3

![](imgs_losses/mape_loss.png)

## Symmetric MAPE (sMAPE)

::: neuralforecast.losses.pytorch.SMAPE
    options:
      members: [__init__, __call__]
      heading_level: 3

# 3. Scale-independent Errors

These metrics measure the relative improvements versus baselines.

## Mean Absolute Scaled Error (MASE)

::: neuralforecast.losses.pytorch.MASE
    options:
      members: [__init__, __call__]
      heading_level: 3

![](imgs_losses/mase_loss.png)

## Relative Mean Squared Error (relMSE)

::: neuralforecast.losses.pytorch.relMSE
    options:
      members: [__init__, __call__]
      heading_level: 3

# 4. Probabilistic Errors

These methods use statistical approaches for estimating unknown
probability distributions using observed data.

Maximum likelihood estimation involves finding the parameter values that
maximize the likelihood function, which measures the probability of
obtaining the observed data given the parameter values. MLE has good
theoretical properties and efficiency under certain satisfied
assumptions.

On the non-parametric approach, quantile regression measures
non-symmetrically deviation, producing under/over estimation.

## Quantile Loss

::: neuralforecast.losses.pytorch.QuantileLoss
    options:
      members: [__init__, __call__]
      heading_level: 3

![](imgs_losses/q_loss.png)

## Multi Quantile Loss (MQLoss)

::: neuralforecast.losses.pytorch.MQLoss
    options:
      members: [__init__, __call__]
      heading_level: 3

![](imgs_losses/mq_loss.png)

## Implicit Quantile Loss (IQLoss)

::: neuralforecast.losses.pytorch.QuantileLayer
    options:
      members: [__init__, __call__]
      heading_level: 3

::: neuralforecast.losses.pytorch.IQLoss
    options:
      members: [__init__, __call__]
      heading_level: 3

## DistributionLoss

::: neuralforecast.losses.pytorch.DistributionLoss
    options:
      members: [__init__, __call__]
      heading_level: 3

## Poisson Mixture Mesh (PMM)

::: neuralforecast.losses.pytorch.PMM
    options:
      members: [__init__, __call__]
      heading_level: 3

![](imgs_losses/pmm.png)

## Gaussian Mixture Mesh (GMM)

::: neuralforecast.losses.pytorch.GMM
    options:
      members: [__init__, __call__]
      heading_level: 3

![](imgs_losses/gmm.png)

## Negative Binomial Mixture Mesh (NBMM)

::: neuralforecast.losses.pytorch.NBMM
    options:
      members: [__init__, __call__]
      heading_level: 3

# 5. Robustified Errors

## Huber Loss

::: neuralforecast.losses.pytorch.HuberLoss
    options:
      members: [__init__, __call__]
      heading_level: 3

![](imgs_losses/huber_loss.png)

## Tukey Loss

::: neuralforecast.losses.pytorch.TukeyLoss
    options:
      members: [__init__, __call__]
      heading_level: 3

![](imgs_losses/tukey_loss.png)

## Huberized Quantile Loss

::: neuralforecast.losses.pytorch.HuberQLoss
    options:
      members: [__init__, __call__]
      heading_level: 3

![](imgs_losses/huber_qloss.png)

## Huberized MQLoss

::: neuralforecast.losses.pytorch.HuberMQLoss
    options:
      members: [__init__, __call__]
      heading_level: 3

![](imgs_losses/hmq_loss.png)

## Huberized IQLoss

::: neuralforecast.losses.pytorch.HuberIQLoss
    options:
      members: [__init__, __call__]
      heading_level: 3

# 6. Others

## Accuracy

::: neuralforecast.losses.pytorch.Accuracy
    options:
      members: [__init__, __call__]
      heading_level: 3

## Scaled Continuous Ranked Probability Score (sCRPS)

::: neuralforecast.losses.pytorch.sCRPS
    options:
      members: [__init__, __call__]
      heading_level: 3
