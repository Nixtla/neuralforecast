---
description: >-
  Time-series Dense Encoder (`TiDE`) is a MLP-based univariate time-series
  forecasting model. `TiDE` uses Multi-layer Perceptrons (MLPs) in an
  encoder-decoder model for long-term time-series forecasting. In addition, this
  model can handle exogenous inputs.
output-file: models.tide.html
title: TiDE
---

![Figure 1. TiDE architecture.](imgs_models/tide.png)
*Figure 1. TiDE architecture.*

## TiDE

::: neuralforecast.models.tide.TiDE
    options:
      members:
        - fit
        - predict
      heading_level: 3
