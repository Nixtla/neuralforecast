---
output-file: models.tsmixer.html
title: TSMixer
---

Time-Series Mixer (`TSMixer`) is a MLP-based multivariate time-series
forecasting model. `TSMixer` jointly learns temporal and cross-sectional
representations of the time-series by repeatedly combining time- and feature
information using stacked mixing layers. A mixing layer consists of a
sequential time- and feature Multi Layer Perceptron (`MLP`). Note: this model
cannot handle exogenous inputs. If you want to use additional exogenous
inputs, use `TSMixerx`.

![Figure 1. TSMixer for multivariate time series forecasting.](imgs_models/tsmixer.png)
*Figure 1. TSMixer for multivariate time series forecasting.*

## 1. TSMixer

::: neuralforecast.models.tsmixer.TSMixer
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
from neuralforecast.models import TSMixer
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
from neuralforecast.losses.pytorch import MAE, MQLoss

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = TSMixer(h=12,
                input_size=24,
                n_series=2, 
                n_block=4,
                ff_dim=4,
                dropout=0,
                revin=True,
                scaler_type='standard',
                max_steps=500,
                early_stop_patience_steps=-1,
                val_check_steps=5,
                learning_rate=1e-3,
                loss=MQLoss(),
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

plot_df = plot_df[plot_df.unique_id=='Airline2'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['TSMixer-median'], c='blue', label='median')
plt.fill_between(x=plot_df['ds'][-12:], 
                 y1=plot_df['TSMixer-lo-90'][-12:].values,
                 y2=plot_df['TSMixer-hi-90'][-12:].values,
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
plt.plot(Y_hat_df['ds'], Y_hat_df['TSMixer-median'], c='blue', label='Forecast')
ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Year', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
```

## 2. Auxiliary Functions

### 2.1 Mixing layers

A mixing layer consists of a sequential time- and feature Multi Layer
Perceptron
([`MLP`](./models.mlp.html#mlp)).

::: neuralforecast.models.tsmixer.MixingLayer
    options:
      members: []

::: neuralforecast.models.tsmixer.FeatureMixing
    options:
      members: []

::: neuralforecast.models.tsmixer.TemporalMixing
    options:
      members: []