---
description: >-
  TimeMixer: Temporal mixing architecture for multivariate time series forecasting with multi-scale decomposition and frequency-domain feature extraction.
output-file: models.timemixer.html
title: TimeMixer
---

![Figure 1. Architecture of SOFTS.](imgs_models/timemixer.png)
*Figure 1. Architecture of SOFTS.*

## 1. TimeMixer

::: neuralforecast.models.timemixer.TimeMixer
    options:
      members:
        - fit
        - predict
      heading_level: 3


### Usage example

```python 
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import TimeMixer
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
from neuralforecast.losses.pytorch import MAE

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = TimeMixer(h=12,
                input_size=24,
                n_series=2,
                scaler_type='standard',
                max_steps=500,
                early_stop_patience_steps=-1,
                val_check_steps=5,
                learning_rate=1e-3,
                loss = MAE(),
                valid_loss=MAE(),
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
plt.plot(plot_df['ds'], plot_df['TimeMixer'], c='blue', label='median')
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
plt.plot(Y_hat_df['ds'], Y_hat_df['TimeMixer'], c='blue', label='Forecast')
ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Year', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
```

## 2. Auxiliary Functions

### 2.1 Embedding

::: neuralforecast.models.timemixer.DataEmbedding_wo_pos
    options:
      members: []

::: neuralforecast.models.timemixer.DFT_series_decomp
    options:
      members: []

### 2.2 Mixing

::: neuralforecast.models.timemixer.PastDecomposableMixing
    options:
      members: []

::: neuralforecast.models.timemixer.MultiScaleTrendMixing
    options:
      members: []

::: neuralforecast.models.timemixer.MultiScaleSeasonMixing
    options:
      members: []