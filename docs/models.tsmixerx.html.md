---
description: >-
  TSMixerx: TSMixer with exogenous variables. MLP-based multivariate forecasting combines temporal-feature mixing with static and future covariate support.
output-file: models.tsmixerx.html
title: TSMixerx
---

Time-Series Mixer exogenous (`TSMixerx`) is a MLP-based multivariate
time-series forecasting model, with capability for additional exogenous
inputs. `TSMixerx` jointly learns temporal and cross-sectional representations
of the time-series by repeatedly combining time and feature information using
stacked mixing layers. A mixing layer consists of a sequential time- and
feature Multi Layer Perceptron (`MLP`).

![Figure 2. TSMixerX for multivariate time series forecasting.](imgs_models/tsmixerx.png)
*Figure 2. TSMixerX for multivariate time series forecasting.*

## 1. TSMixerx

::: neuralforecast.models.tsmixerx.TSMixerx
    options:
      members:
        - fit
        - predict
      heading_level: 3

### Usage Examples

Train model and forecast future values with `predict` method.

```python
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import TSMixerx
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
from neuralforecast.losses.pytorch import GMM

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = TSMixerx(h=12,
                input_size=24,
                n_series=2,
                stat_exog_list=['airline1'],
                futr_exog_list=['trend'],
                n_block=4,
                ff_dim=4,
                revin=True,
                scaler_type='robust',
                max_steps=500,
                early_stop_patience_steps=-1,
                val_check_steps=5,
                learning_rate=1e-3,
                loss = GMM(n_components=10, weighted=True),
                batch_size=32
                )

fcst = NeuralForecast(models=[model], freq='ME')
fcst.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
forecasts = fcst.predict(futr_df=Y_test_df)

# Plot predictions
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['TSMixerx-median'], c='blue', label='median')
plt.fill_between(x=plot_df['ds'][-12:], 
                 y1=plot_df['TSMixerx-lo-90'][-12:].values,
                 y2=plot_df['TSMixerx-hi-90'][-12:].values,
                 alpha=0.4, label='level 90')
ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Year', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
```

Using `cross_validation` to forecast multiple historic values.

```python
fcst = NeuralForecast(models=[model], freq='M')
forecasts = fcst.cross_validation(df=AirPassengersPanel, static_df=AirPassengersStatic, n_windows=2, step_size=12)

# Plot predictions
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
Y_hat_df = forecasts.loc['Airline1']
Y_df = AirPassengersPanel[AirPassengersPanel['unique_id']=='Airline1']

plt.plot(Y_df['ds'], Y_df['y'], c='black', label='True')
plt.plot(Y_hat_df['ds'], Y_hat_df['TSMixerx-median'], c='blue', label='Forecast')
ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Year', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
```

## 2. Auxiliary Functions

### 2.1 Mixing layers

A mixing layer consists of a sequential time- and feature Multi Layer
Perceptron ([`MLP`](./models.mlp.html#mlp)).

::: neuralforecast.models.tsmixerx.MixingLayerWithStaticExogenous
    options:
      members: []

::: neuralforecast.models.tsmixerx.MixingLayer
    options:
      members: []

::: neuralforecast.models.tsmixerx.FeatureMixing
    options:
      members: []

::: neuralforecast.models.tsmixerx.TemporalMixing
    options:
      members: []

### 2.2 Reversible InstanceNormalization

An Instance Normalization Layer that is reversible, based on [this reference implementation](https://github.com/google-research/google-research/blob/master/tsmixer/tsmixer_basic/models/rev_in.py).