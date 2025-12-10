---
description: >-
  GRU: Gated Recurrent Unit model for sequential forecasting. Improves upon LSTM with simplified gating mechanism and MLP decoder for time series predictions.
output-file: models.gru.html
title: GRU
---

Cho et. al proposed the Gated Recurrent Unit
([`GRU`](./models.gru.html#gru))
to improve on LSTM and Elman cells. The predictions at each time are
given by a MLP decoder. This architecture follows closely the original
Multi Layer Elman
[`RNN`](./models.rnn.html#rnn)
with the main difference being its use of the GRU cells. The predictions
are obtained by transforming the hidden states into contexts
$\mathbf{c}_{[t+1:t+H]}$, that are decoded and adapted into
$\mathbf{\hat{y}}_{[t+1:t+H],[q]}$ through MLPs.

where $\mathbf{h}_{t}$, is the hidden state for time $t$,
$\mathbf{y}_{t}$ is the input at time $t$ and $\mathbf{h}_{t-1}$ is the
hidden state of the previous layer at $t-1$, $\mathbf{x}^{(s)}$ are
static exogenous inputs, $\mathbf{x}^{(h)}_{t}$ historic exogenous,
$\mathbf{x}^{(f)}_{[:t+H]}$ are future exogenous available at the time
of the prediction.

**References**

- [Junyoung Chung, Caglar Gulcehre, KyungHyun Cho,
Yoshua Bengio (2014). “Empirical Evaluation of Gated Recurrent Neural
Networks on Sequence Modeling”.](https:arxivorg/abs/1412.3555)
- [Kyunghyun Cho, Bart van Merrienboer, Dzmitry Bahdanau, Yoshua Bengio
(2014). “On the Properties of Neural Machine Translation:
Encoder-Decoder Approaches”.](https://arxiv.org/abs/1409.1259)


![Figure 1. Gated Recurrent Unit Cell.](imgs_models/gru.png)
*Figure 1. Gated Recurrent Unit
Cell.*

## GRU

::: neuralforecast.models.gru.GRU
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
# from neuralforecast.models import GRU
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

fcst = NeuralForecast(
    models=[GRU(h=12, input_size=24,
                loss=DistributionLoss(distribution='Normal', level=[80, 90]),
                scaler_type='robust',
                encoder_n_layers=2,
                encoder_hidden_size=128,
                decoder_hidden_size=128,
                decoder_layers=2,
                max_steps=200,
                futr_exog_list=None,
                hist_exog_list=['y_[lag12]'],
                stat_exog_list=['airline1'],
                )
    ],
    freq='ME'
)
fcst.fit(df=Y_train_df, static_df=AirPassengersStatic)
forecasts = fcst.predict(futr_df=Y_test_df)

Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['GRU-median'], c='blue', label='median')
plt.fill_between(x=plot_df['ds'][-12:], 
                 y1=plot_df['GRU-lo-90'][-12:].values, 
                 y2=plot_df['GRU-hi-90'][-12:].values,
                 alpha=0.4, label='level 90')
plt.legend()
plt.grid()
plt.plot()
```
