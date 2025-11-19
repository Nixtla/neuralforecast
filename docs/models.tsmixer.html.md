---
description: >-
  Time-Series Mixer (`TSMixer`) is a MLP-based multivariate time-series
  forecasting model. `TSMixer` jointly learns temporal and cross-sectional
  representations of the time-series by repeatedly combining time- and feature
  information using stacked mixing layers. A mixing layer consists of a
  sequential time- and feature Multi Layer Perceptron (`MLP`). Note: this model
  cannot handle exogenous inputs. If you want to use additional exogenous
  inputs, use `TSMixerx`.
output-file: models.tsmixer.html
title: TSMixer
---

![Figure 1. TSMixer for multivariate time series forecasting.](imgs_models/tsmixer.png)
*Figure 1. TSMixer for multivariate time
series forecasting.*

## TSMixer

::: neuralforecast.models.tsmixer.TSMixer
    options:
      members:
        - fit
        - predict
      heading_level: 3
