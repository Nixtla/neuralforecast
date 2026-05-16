---
description: >-
  SOFTSSharp: SOFTS extension with stochastic variable-position encoding for multivariate time series forecasting.
output-file: models.softssharp.html
title: SOFTSSharp
---

SOFTSSharp extends SOFTS by stochastically adding variable-position embeddings
and multiple dropout layers inside the STAD aggregation-redistribution component
while preserving linear complexity.

## 1. SOFTSSharp

::: neuralforecast.models.softssharp.SOFTSSharp
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
from neuralforecast.models import SOFTSSharp
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
from neuralforecast.losses.pytorch import MASE

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]].reset_index(drop=True)
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True)

model = SOFTSSharp(h=12,
                   input_size=24,
                   n_series=2,
                   hidden_size=256,
                   d_core=256,
                   e_layers=2,
                   d_ff=64,
                   dropout=0.1,
                   pe_keep_prob=0.5,
                   use_norm=True,
                   loss=MASE(seasonality=4),
                   early_stop_patience_steps=3,
                   batch_size=32)

fcst = NeuralForecast(models=[model], freq='ME')
fcst.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
forecasts = fcst.predict(futr_df=Y_test_df)

fig, ax = plt.subplots(1, 1, figsize = (20, 7))
Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['SOFTSSharp'], c='blue', label='Forecast')
ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Year', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
```

## 2. Auxiliary functions

::: neuralforecast.models.softssharp.PositionalEmbedding
    options:
      members: []

::: neuralforecast.models.softssharp.STADSharp
    options:
      members: []
