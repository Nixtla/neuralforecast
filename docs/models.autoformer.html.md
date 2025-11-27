---
description: >-
  Autoformer: Transformer with auto-correlation mechanism and progressive decomposition for reliable long-horizon time series forecasting with trend-seasonality.
output-file: models.autoformer.html
title: Autoformer
---

The Autoformer model tackles the challenge of finding reliable
dependencies on intricate temporal patterns of long-horizon forecasting.

The architecture has the following distinctive features: - In-built
progressive decomposition in trend and seasonal compontents based on a
moving average filter. - Auto-Correlation mechanism that discovers the
period-based dependencies by calculating the autocorrelation and
aggregating similar sub-series based on the periodicity. - Classic
encoder-decoder proposed by Vaswani et al. (2017) with a multi-head
attention mechanism.

The Autoformer model utilizes a three-component approach to define its
embedding: - It employs encoded autoregressive features obtained from a
convolution network. - Absolute positional embeddings obtained from
calendar features are utilized.

**References**

- [Wu, Haixu, Jiehui Xu, Jianmin Wang, and Mingsheng
Long. “Autoformer: Decomposition transformers with auto-correlation for
long-term series
forecasting”](https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html)


![Figure 1. Autoformer Architecture.](imgs_models/autoformer.png)
*Figure 1. Autoformer
Architecture.*

## 1. Autoformer

::: neuralforecast.models.autoformer.Autoformer
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
from neuralforecast.models import Autoformer
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic, augment_calendar_df

AirPassengersPanel, calendar_cols = augment_calendar_df(df=AirPassengersPanel, freq='M')

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = Autoformer(h=12,
                 input_size=24,
                 hidden_size = 16,
                 conv_hidden_size = 32,
                 n_head=2,
                 loss=MAE(),
                 futr_exog_list=calendar_cols,
                 scaler_type='robust',
                 learning_rate=1e-3,
                 max_steps=300,
                 val_check_steps=50,
                 early_stop_patience_steps=2)

nf = NeuralForecast(
    models=[model],
    freq='ME'
)
nf.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
forecasts = nf.predict(futr_df=Y_test_df)

Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

if model.loss.is_distribution_output:
    plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
    plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
    plt.plot(plot_df['ds'], plot_df['Autoformer-median'], c='blue', label='median')
    plt.fill_between(x=plot_df['ds'][-12:], 
                    y1=plot_df['Autoformer-lo-90'][-12:].values, 
                    y2=plot_df['Autoformer-hi-90'][-12:].values,
                    alpha=0.4, label='level 90')
    plt.grid()
    plt.legend()
    plt.plot()
else:
    plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
    plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
    plt.plot(plot_df['ds'], plot_df['Autoformer'], c='blue', label='Forecast')
    plt.legend()
    plt.grid()
```

## 2. Auxiliary functions

::: neuralforecast.models.autoformer.Decoder
    options:
      members: []

::: neuralforecast.models.autoformer.DecoderLayer
    options:
      members: []

::: neuralforecast.models.autoformer.Encoder
    options:
      members: []

::: neuralforecast.models.autoformer.EncoderLayer
    options:
      members: []

::: neuralforecast.models.autoformer.LayerNorm
    options:
      members: []

::: neuralforecast.models.autoformer.AutoCorrelationLayer
    options:
      members: []

::: neuralforecast.models.autoformer.AutoCorrelation
    options:
      members: []
