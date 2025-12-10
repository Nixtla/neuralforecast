---
description: >-
  KAN: Kolmogorov-Arnold Networks for time series forecasting. MLP alternative using learnable activation functions for improved non-linear pattern modeling.
output-file: models.kan.html
title: KAN
---

Kolmogorov-Arnold Networks (KANs) are an alternative to Multi-Layer
Perceptrons (MLPs). This model uses KANs similarly as our MLP model.

**References**

- [Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle,
James Halverson, Marin Soljačić, Thomas Y. Hou, Max Tegmark. “KAN:
Kolmogorov–Arnold Networks”](https://arxiv.org/html/2404.19756v1)

![Figure 1. KAN compared to MLP.](imgs_models/kan.png)
*Figure 1. KAN compared to
MLP.*

## 1. KAN

::: neuralforecast.models.kan.KAN
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
from neuralforecast.models import KAN
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

fcst = NeuralForecast(
    models=[
            KAN(h=12,
                input_size=24,
                loss = DistributionLoss(distribution="Normal"),
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
plt.plot(plot_df['ds'], plot_df['KAN-median'], c='blue', label='median')
plt.fill_between(x=plot_df['ds'][-12:], 
                 y1=plot_df['KAN-lo-90'][-12:].values,
                 y2=plot_df['KAN-hi-90'][-12:].values,
                 alpha=0.4, label='level 90')
plt.legend()
plt.grid()
```

## 2. Auxiliary functions

::: neuralforecast.models.kan.KANLinear
    options:
      members: []