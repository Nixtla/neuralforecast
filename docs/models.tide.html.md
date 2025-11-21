---
output-file: models.tide.html
title: TiDE
---

Time-series Dense Encoder (`TiDE`) is a MLP-based univariate time-series
forecasting model. `TiDE` uses Multi-layer Perceptrons (MLPs) in an
encoder-decoder model for long-term time-series forecasting. In addition, this
model can handle exogenous inputs.

![Figure 1. TiDE architecture.](imgs_models/tide.png)
*Figure 1. TiDE architecture.*

## 1. TiDE

::: neuralforecast.models.tide.TiDE
    options:
      members:
        - fit
        - predict
      heading_level: 3

### Usage Examples

```python
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import TiDE
from neuralforecast.losses.pytorch import GMM
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

fcst = NeuralForecast(
    models=[
            TiDE(h=12,
                input_size=24,
                loss=GMM(n_components=7, return_params=True, level=[80,90], weighted=True),
                max_steps=100,
                scaler_type='standard',
                futr_exog_list=['y_[lag12]'],
                hist_exog_list=None,
                stat_exog_list=['airline1'],
                ),     
    ],
    freq='ME'
)
fcst.fit(df=Y_train_df, static_df=AirPassengersStatic)
forecasts = fcst.predict(futr_df=Y_test_df)

# Plot quantile predictions
Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['TiDE-median'], c='blue', label='median')
plt.fill_between(x=plot_df['ds'][-12:], 
                 y1=plot_df['TiDE-lo-90'][-12:].values,
                 y2=plot_df['TiDE-hi-90'][-12:].values,
                 alpha=0.4, label='level 90')
plt.legend()
plt.grid()
```

## 2. Auxiliary Functions

::: neuralforecast.models.tide.MLPResidual
    options:
      members: []