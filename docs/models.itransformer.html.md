---
description: >-
  iTransformer: Inverted Transformer architecture for multivariate time series forecasting with attention on time points and feed-forward on series dimensions.
output-file: models.itransformer.html
title: iTransformer
---

The iTransformer model simply takes the Transformer architecture but it
applies the attention and feed-forward network on the inverted
dimensions. This means that time points of each individual series are
embedded into tokens. That way, the attention mechanisms learn
multivariate correlation and the feed-forward network learns non-linear
relationships.

**References**

- [Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu
Wang, Lintao Ma, Mingsheng Long. “iTransformer: Inverted Transformers
Are Effective for Time Series
Forecasting”](https://arxiv.org/abs/2310.06625)

![Figure 1. Architecture of iTransformer.](imgs_models/itransformer.png)
*Figure 1. Architecture of
iTransformer.*

## 1. iTransformer

::: neuralforecast.models.itransformer.iTransformer
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
from neuralforecast.models import iTransformer
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
from neuralforecast.losses.pytorch import MSE

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = iTransformer(h=12,
                     input_size=24,
                     n_series=2,
                     hidden_size=128,
                     n_heads=2,
                     e_layers=2,
                     d_layers=1,
                     d_ff=4,
                     factor=1,
                     dropout=0.1,
                     use_norm=True,
                     loss=MSE(),
                     valid_loss=MAE(),
                     early_stop_patience_steps=3,
                     batch_size=32,
                     max_steps=100)

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
plt.plot(plot_df['ds'], plot_df['iTransformer'], c='blue', label='Forecast')
ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Year', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
```