---
description: >-
  XLinear: A MLP-based model for multivariate forecasting with exogenous features.
output-file: models.xlinear.html
title: XLinear
---

XLinear is a MLP-based model for multivariate time series forecasting
that uses gating mechanisms for temporal and cross-channel interactions.
The architecture consists of temporal gating with a global token to capture
global temporal patterns, followed by cross-channel gating to model
dependencies between different time series.

**References**

- [Xinyang, C., et al. "XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs"](https://arxiv.org/abs/2601.09237)

![Figure 1. Architecture of XLinear](imgs_models/xlinear.png)
*Figure 1. Architecture of XLinear*

## XLinear

::: neuralforecast.models.xlinear.XLinear
    options:
      members:
        - fit
        - predict
      heading_level: 3

### Usage Example


```python
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import XLinear
from neuralforecast.losses.pytorch import MAE
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = XLinear(h=12, 
            input_size=24,
            n_series=2,
            stat_exog_list=['airline1'],
            hist_exog_list=["y_[lag12]"],
            futr_exog_list=['trend'],            
            loss = MAE(),
            scaler_type='robust',
            learning_rate=1e-3,
            max_steps=200,
            val_check_steps=10,
            early_stop_patience_steps=2)

fcst = NeuralForecast(
    models=[model],
    freq='ME'
)
fcst.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
forecasts = fcst.predict(futr_df=Y_test_df)

# Plot predictions
Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['XLinear'], c='blue', label='median')
plt.grid()
plt.legend()
plt.plot()
```