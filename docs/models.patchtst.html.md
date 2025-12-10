---
description: >-
  PatchTST: Efficient Transformer model for multivariate forecasting using patched time series and channel-independence for scalable long-term predictions.
output-file: models.patchtst.html
title: PatchTST
---

The PatchTST model is an efficient Transformer-based model for
multivariate time series forecasting.

It is based on two key components: - segmentation of time series into
windows (patches) which are served as input tokens to Transformer -
channel-independence. where each channel contains a single univariate
time series.

**References**

- [Nie, Y., Nguyen, N. H., Sinthong, P., &
Kalagnanam, J. (2022). “A Time Series is Worth 64 Words: Long-term
Forecasting with
Transformers”](https://arxiv.org/pdf/2211.14730.pdf)


![Figure 1. PatchTST.](imgs_models/patchtst.png)
*Figure 1. PatchTST.*

## 1. PatchTST

::: neuralforecast.models.patchtst.PatchTST
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
from neuralforecast.models import PatchTST
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic, augment_calendar_df

AirPassengersPanel, calendar_cols = augment_calendar_df(df=AirPassengersPanel, freq='M')

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = PatchTST(h=12,
                 input_size=104,
                 patch_len=24,
                 stride=24,
                 revin=False,
                 hidden_size=16,
                 n_heads=4,
                 scaler_type='robust',
                 loss=DistributionLoss(distribution='StudentT', level=[80, 90]),
                 learning_rate=1e-3,
                 max_steps=500,
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
    plt.plot(plot_df['ds'], plot_df['PatchTST-median'], c='blue', label='median')
    plt.fill_between(x=plot_df['ds'][-12:], 
                    y1=plot_df['PatchTST-lo-90'][-12:].values, 
                    y2=plot_df['PatchTST-hi-90'][-12:].values,
                    alpha=0.4, label='level 90')
    plt.grid()
    plt.legend()
    plt.plot()
else:
    plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
    plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
    plt.plot(plot_df['ds'], plot_df['PatchTST'], c='blue', label='Forecast')
    plt.legend()
    plt.grid()
```

## 2. Backbone

### Auxiliary Functions

::: neuralforecast.models.patchtst.get_activation_fn
    options:
      members: []

::: neuralforecast.models.patchtst.Transpose
    options:
      members: []

### Positional Encoding

::: neuralforecast.models.patchtst.positional_encoding
    options:
      members: []

::: neuralforecast.models.patchtst.Coord1dPosEncoding
    options:
      members: []

::: neuralforecast.models.patchtst.Coord2dPosEncoding
    options:
      members: []

::: neuralforecast.models.patchtst.PositionalEncoding
    options:
      members: []

### Encoder

::: neuralforecast.models.patchtst.TSTEncoderLayer
    options:
      members: []

::: neuralforecast.models.patchtst.TSTEncoder
    options:
      members: []

::: neuralforecast.models.patchtst.TSTiEncoder
    options:
      members: []

::: neuralforecast.models.patchtst.Flatten_Head
    options:
      members: []

::: neuralforecast.models.patchtst.PatchTST_backbone
    options:
      members: []