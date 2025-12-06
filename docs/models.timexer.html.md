---
description: >-
  TimeXer: Cross-series attention transformer for multivariate forecasting with patch-based processing and exogenous variable support for complex temporal patterns.
output-file: models.timexer.html
title: TimeXer
---

![Figure 1. Architecture of TimeXer.](imgs_models/timexer.png)
*Figure 1. Architecture of TimeXer.*

## 1. TimeXer

::: neuralforecast.models.timexer.TimeXer
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
from neuralforecast.models import TimeXer
from neuralforecast.losses.pytorch import MSE
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic, augment_calendar_df

AirPassengersPanel, calendar_cols = augment_calendar_df(df=AirPassengersPanel, freq='M')

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = TimeXer(h=12,
                input_size=24,
                n_series=2,
                futr_exog_list=["trend", "month"],
                patch_len=12,
                hidden_size=128,
                n_heads=16,
                e_layers=2,
                d_ff=256,
                factor=1,
                dropout=0.1,
                use_norm=True,
                loss=MSE(),
                valid_loss=MAE(),
                early_stop_patience_steps=3,
                batch_size=32)

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
plt.plot(plot_df['ds'], plot_df['TimeXer'], c='blue', label='Forecast')
ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Year', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
```

## 2. Auxiliary Functions


::: neuralforecast.models.timexer.FlattenHead
    options:
      members: []

::: neuralforecast.models.timexer.Encoder
    options:
      members: []

::: neuralforecast.models.timexer.EncoderLayer
    options:
      members: []

::: neuralforecast.models.timexer.EnEmbedding
    options:
      members: []