---
description: >-
  Time-Series Mixer exogenous (`TSMixerx`) is a MLP-based multivariate
  time-series forecasting model, with capability for additional exogenous
  inputs. `TSMixerx` jointly learns temporal and cross-sectional representations
  of the time-series by repeatedly combining time- and feature information using
  stacked mixing layers. A mixing layer consists of a sequential time- and
  feature Multi Layer Perceptron (`MLP`).
output-file: models.tsmixerx.html
title: TSMixerx
---

![Figure 2. TSMixerX for multivariate time series forecasting.](imgs_models/tsmixerx.png)
*Figure 2. TSMixerX for multivariate time
series forecasting.*

## TSMixerx

::: neuralforecast.models.tsmixerx.TSMixerx
    options:
      members:
        - fit
        - predict
      heading_level: 3
