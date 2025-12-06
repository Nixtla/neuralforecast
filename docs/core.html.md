---
description: >-
  NeuralForecast core class for high-level time series forecasting. Fits multiple PyTorch models on pandas DataFrames with parallelization and distributed computation.
output-file: core.html
title: Core
---

NeuralForecast contains two main components, PyTorch implementations deep
learning predictive models, as well as parallelization and distributed
computation utilities. The first component comprises low-level PyTorch model
estimator classes like `models.NBEATS` and `models.RNN`. The second component is a high-level `core.NeuralForecast` wrapper class that operates with sets of time series data stored in pandas DataFrames.

##

::: neuralforecast.core.NeuralForecast
    handler: python
    options:
      docstring_style: google
      members:
        - fit
        - predict
        - cross_validation
        - predict_insample
        - save
        - load
      heading_level: 3
      show_root_heading: true
      show_source: true