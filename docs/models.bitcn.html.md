---
description: >-
  BiTCN: Bidirectional Temporal Convolutional Network for forecasting. Parameter-efficient architecture with forward-backward encoding for probabilistic predictions.
output-file: models.bitcn.html
title: BiTCN
---

Bidirectional Temporal Convolutional Network (BiTCN) is a forecasting
architecture based on two temporal convolutional networks (TCNs). The
first network (‘forward’) encodes future covariates of the time series,
whereas the second network (‘backward’) encodes past observations and
covariates. This method allows to preserve the temporal information of
sequence data, and is computationally more efficient than common RNN
methods (LSTM, GRU, …). As compared to Transformer-based methods, BiTCN
has a lower space complexity, i.e. it requires orders of magnitude less
parameters.

This model may be a good choice if you seek a small model (small amount
of trainable parameters) with few hyperparameters to tune (only 2).

**References**

- [Olivier Sprangers, Sebastian Schelter, Maarten de
Rijke (2023). Parameter-Efficient Deep Probabilistic Forecasting.
International Journal of Forecasting 39, no. 1 (1 January 2023): 332–45.
URL:
https://doi.org/10.1016/j.ijforecast.2021.11.011.](https://doi.org/10.1016/j.ijforecast.2021.11.011)
- [Shaojie Bai, Zico Kolter, Vladlen Koltun. (2018). An Empirical
Evaluation of Generic Convolutional and Recurrent Networks for Sequence
Modeling. Computing Research Repository, abs/1803.01271. URL:
https://arxiv.org/abs/1803.01271.](https://arxiv.org/abs/1803.01271)
- [van den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O.,
Graves, A., Kalchbrenner, N., Senior, A. W., & Kavukcuoglu, K. (2016).
Wavenet: A generative model for raw audio. Computing Research
Repository, abs/1609.03499. URL: http://arxiv.org/abs/1609.03499.
arXiv:1609.03499.](https://arxiv.org/abs/1609.03499)


![Figure 1. Visualization of a stack of dilated causal convolutional layers.](imgs_models/bitcn.png)
*Figure 1. Visualization of a stack of dilated causal convolutional layers.*

## 1. BiTCN

::: neuralforecast.models.bitcn.BiTCN
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
from neuralforecast.losses.pytorch import GMM
from neuralforecast.models import BiTCN
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

fcst = NeuralForecast(
    models=[
            BiTCN(h=12,
                input_size=24,
                loss=GMM(n_components=7, level=[80,90]),
                max_steps=100,
                scaler_type='standard',
                futr_exog_list=['y_[lag12]'],
                hist_exog_list=None,
                stat_exog_list=['airline1'],
                windows_batch_size=2048,
                val_check_steps=10,
                early_stop_patience_steps=-1,
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
plt.plot(plot_df['ds'], plot_df['BiTCN-median'], c='blue', label='median')
plt.fill_between(x=plot_df['ds'][-12:], 
                 y1=plot_df['BiTCN-lo-90'][-12:].values,
                 y2=plot_df['BiTCN-hi-90'][-12:].values,
                 alpha=0.4, label='level 90')
plt.legend()
plt.grid()
```

## 2. Auxilary functions

::: neuralforecast.models.bitcn.TCNCell
    options:
      members: []

::: neuralforecast.models.bitcn.CustomConv1d
    options:
      members: []