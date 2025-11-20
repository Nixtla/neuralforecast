---
output-file: models.tcn.html
title: TCN
---

For long time in deep learning, sequence modelling was synonymous with
recurrent networks, yet several papers have shown that simple
convolutional architectures can outperform canonical recurrent networks
like LSTMs by demonstrating longer effective memory. By skipping
temporal connections the causal convolution filters can be applied to
larger time spans while remaining computationally efficient.

The predictions are obtained by transforming the hidden states into
contexts $\mathbf{c}_{[t+1:t+H]}$, that are decoded and adapted into
$\mathbf{\hat{y}}_{[t+1:t+H],[q]}$ through MLPs.

where $\mathbf{h}_{t}$, is the hidden state for time $t$,
$\mathbf{y}_{t}$ is the input at time $t$ and $\mathbf{h}_{t-1}$ is the
hidden state of the previous layer at $t-1$, $\mathbf{x}^{(s)}$ are
static exogenous inputs, $\mathbf{x}^{(h)}_{t}$ historic exogenous,
$\mathbf{x}^{(f)}_{[:t+H]}$ are future exogenous available at the time
of the prediction.

**References**

- [van den Oord, A., Dieleman, S., Zen, H., Simonyan,
K., Vinyals, O., Graves, A., Kalchbrenner, N., Senior, A. W., &
Kavukcuoglu, K. (2016). Wavenet: A generative model for raw audio.
Computing Research Repository, abs/1609.03499. URL:
http://arxiv.org/abs/1609.03499.
arXiv:1609.03499.](https://arxiv.org/abs/1609.03499)
- [Shaojie Bai,
Zico Kolter, Vladlen Koltun. (2018). An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling. Computing
Research Repository, abs/1803.01271. URL:
https://arxiv.org/abs/1803.01271.](https://arxiv.org/abs/1803.01271)


![Figure 1. Visualization of a stack of dilated causal convolutional layers.](imgs_models/tcn.png)
*Figure 1. Visualization of a stack of
dilated causal convolutional layers.*

## TCN

::: neuralforecast.models.tcn.TCN
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
from neuralforecast.models import TCN
from neuralforecast.losses.pytorch import  DistributionLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

fcst = NeuralForecast(
    models=[TCN(h=12,
                input_size=-1,
                loss=DistributionLoss(distribution='Normal', level=[80, 90]),
                learning_rate=5e-4,
                kernel_size=2,
                dilations=[1,2,4,8,16],
                encoder_hidden_size=128,
                context_size=10,
                decoder_hidden_size=128,
                decoder_layers=2,
                max_steps=500,
                scaler_type='robust',
                futr_exog_list=['y_[lag12]'],
                hist_exog_list=None,
                stat_exog_list=['airline1'],
                )
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
plt.plot(plot_df['ds'], plot_df['TCN-median'], c='blue', label='median')
plt.fill_between(x=plot_df['ds'][-12:], 
                 y1=plot_df['TCN-lo-90'][-12:].values,
                 y2=plot_df['TCN-hi-90'][-12:].values,
                 alpha=0.4, label='level 90')
plt.legend()
plt.grid()
plt.plot()
```
