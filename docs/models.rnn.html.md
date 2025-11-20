---
output-file: models.rnn.html
title: RNN
---

Elman proposed this classic recurrent neural network
([`RNN`](./models.rnn.html#rnn))
in 1990, where each layer uses the following recurrent transformation:
$$\mathbf{h}^{l}_{t} = \mathrm{Activation}([\mathbf{y}_{t},\mathbf{x}^{(h)}_{t},\mathbf{x}^{(s)}] W^{\intercal}_{ih} + b_{ih}  +  \mathbf{h}^{l}_{t-1} W^{\intercal}_{hh} + b_{hh})$$

where $\mathbf{h}^{l}_{t}$, is the hidden state of RNN layer $l$ for
time $t$, $\mathbf{y}_{t}$ is the input at time $t$ and
$\mathbf{h}_{t-1}$ is the hidden state of the previous layer at $t-1$,
$\mathbf{x}^{(s)}$ are static exogenous inputs, $\mathbf{x}^{(h)}_{t}$
historic exogenous, $\mathbf{x}^{(f)}_{[:t+H]}$ are future exogenous
available at the time of the prediction. The available activations are
`tanh`, and `relu`. The predictions are obtained by transforming the
hidden states into contexts $\mathbf{c}_{[t+1:t+H]}$, that are decoded
and adapted into $\mathbf{\hat{y}}_{[t+1:t+H],[q]}$ through MLPs.

**References**

- [Jeffrey L. Elman (1990). “Finding Structure in
Time”.](https://onlinelibrary.wiley.com/doiabs/10.1207/s15516709cog1402_1)
- [Cho, K., van Merrienboer, B., Gülcehre, C., Bougares, F., Schwenk, H.,
& Bengio, Y. (2014). Learning phrase representations using RNN
encoder-decoder for statistical machine
translation.](http://arxiv.org/abs/1406.1078)


![Figure 1. Single Layer Elman RNN with MLP decoder.](imgs_models/rnn.png)
*Figure 1. Single Layer Elman RNN with MLP
decoder.*

## RNN

::: neuralforecast.models.rnn.RNN
    options:
      members:
        - fit
        - predict
      heading_level: 3

## Usage Example


```python
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import RNN
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

fcst = NeuralForecast(
    models=[RNN(h=12,
                input_size=24,
                inference_input_size=24,
                loss=MQLoss(level=[80, 90]),
                valid_loss=MQLoss(level=[80, 90]),
                scaler_type='standard',
                encoder_n_layers=2,
                encoder_hidden_size=128,
                decoder_hidden_size=128,
                decoder_layers=2,
                max_steps=200,
                futr_exog_list=['y_[lag12]'],
                stat_exog_list=['airline1'],
                )
    ],
    freq='ME'
)
fcst.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
forecasts = fcst.predict(futr_df=Y_test_df)

Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['RNN-median'], c='blue', label='median')
plt.fill_between(x=plot_df['ds'][-12:], 
                 y1=plot_df['RNN-lo-90'][-12:].values, 
                 y2=plot_df['RNN-hi-90'][-12:].values,
                 alpha=0.4, label='level 90')
plt.legend()
plt.grid()
plt.plot()
```
