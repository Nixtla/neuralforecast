---
output-file: models.lstm.html
title: LSTM
---

The Long Short-Term Memory Recurrent Neural Network
([`LSTM`](./models.lstm.html#lstm)),
uses a multilayer
[`LSTM`](./models.lstm.html#lstm)
encoder and an
[`MLP`](./models.mlp.html#mlp)
decoder. It builds upon the LSTM-cell that improves the exploding and
vanishing gradients of classic
[`RNN`](./models.rnn.html#rnn)’s.
This network has been extensively used in sequential prediction tasks
like language modeling, phonetic labeling, and forecasting. The
predictions are obtained by transforming the hidden states into contexts
$\mathbf{c}_{[t+1:t+H]}$, that are decoded and adapted into
$\mathbf{\hat{y}}_{[t+1:t+H],[q]}$ through MLPs.

where $\mathbf{h}_{t}$, is the hidden state for time $t$,
$\mathbf{y}_{t}$ is the input at time $t$ and $\mathbf{h}_{t-1}$ is the
hidden state of the previous layer at $t-1$, $\mathbf{x}^{(s)}$ are
static exogenous inputs, $\mathbf{x}^{(h)}_{t}$ historic exogenous,
$\mathbf{x}^{(f)}_{[:t+H]}$ are future exogenous available at the time
of the prediction.

**References**

- [Jeffrey L. Elman (1990). “Finding Structure in
Time”.](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1)
- [Haşim
Sak, Andrew Senior, Françoise Beaufays (2014). “Long Short-Term Memory
Based Recurrent Neural Network Architectures for Large Vocabulary Speech
Recognition.”](https://arxiv.org/abs/1402.1128)


![Figure 1. Long Short-Term Memory Cell.](imgs_models/lstm.png)
*Figure 1. Long Short-Term Memory
Cell.*

## 1. LSTM

::: neuralforecast.models.lstm.LSTM
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
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

nf = NeuralForecast(
    models=[LSTM(h=12, 
                 input_size=8,
                 loss=DistributionLoss(distribution="Normal", level=[80, 90]),
                 scaler_type='robust',
                 encoder_n_layers=2,
                 encoder_hidden_size=128,
                 decoder_hidden_size=128,
                 decoder_layers=2,
                 max_steps=200,
                 futr_exog_list=['y_[lag12]'],
                 stat_exog_list=['airline1'],
                 recurrent=True,
                 h_train=1,
                 )
    ],
    freq='ME'
)
nf.fit(df=Y_train_df, static_df=AirPassengersStatic)
Y_hat_df = nf.predict(futr_df=Y_test_df)

# Plots
Y_hat_df = Y_hat_df.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['LSTM-median'], c='blue', label='median')
plt.fill_between(x=plot_df['ds'][-12:], 
                 y1=plot_df['LSTM-lo-90'][-12:].values,
                 y2=plot_df['LSTM-hi-90'][-12:].values,
                 alpha=0.4, label='level 90')
plt.grid()
plt.plot()
```
