---
description: >-
  NHITS: Neural Hierarchical Interpolation for Time Series. MLP architecture with multi-rate processing for long-horizon forecasting, 50x faster than Informer.
output-file: models.nhits.html
title: NHITS
---

Long-horizon forecasting is challenging because of the *volatility* of
the predictions and the *computational complexity*. To solve this
problem we created the Neural Hierarchical Interpolation for Time Series
(NHITS).
[`NHITS`](./models.nhits.html#nhits)
builds upon
[`NBEATS`](./models.nbeats.html#nbeats)
and specializes its partial outputs in the different frequencies of the
time series through hierarchical interpolation and multi-rate input
processing. On the long-horizon forecasting task
[`NHITS`](./models.nhits.html#nhits)
improved accuracy by 25% on AAAI’s best paper award the
[`Informer`](./models.informer.html#informer),
while being 50x faster.

The model is composed of several MLPs with ReLU non-linearities. Blocks
are connected via doubly residual stacking principle with the backcast
$\mathbf{\tilde{y}}_{t-L:t,l}$ and forecast
$\mathbf{\hat{y}}_{t+1:t+H,l}$ outputs of the l-th block. Multi-rate
input pooling, hierarchical interpolation and backcast residual
connections together induce the specialization of the additive
predictions in different signal bands, reducing memory footprint and
computational time, thus improving the architecture parsimony and
accuracy.

**References**

- [Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados,
Yoshua Bengio (2019). “N-BEATS: Neural basis expansion analysis for
interpretable time series
forecasting”.](https://arxiv.org/abs/1905.10437)
- [Cristian Challu,
Kin G. Olivares, Boris N. Oreshkin, Federico Garza, Max
Mergenthaler-Canseco, Artur Dubrawski (2023). “NHITS: Neural
Hierarchical Interpolation for Time Series Forecasting”. Accepted at the
Thirty-Seventh AAAI Conference on Artificial
Intelligence.](https://arxiv.org/abs/2201.12886)
- [Zhou, H.; Zhang,
S.; Peng, J.; Zhang, S.; Li, J.; Xiong, H.; and Zhang, W. (2020).
“Informer: Beyond Efficient Transformer for Long Sequence Time-Series
Forecasting”. Association for the Advancement of Artificial Intelligence
Conference 2021 (AAAI 2021).](https://arxiv.org/abs/2012.07436)

![Figure 1. Neural Hierarchical Interpolation for Time Series (NHITS).](imgs_models/nhits.png)
*Figure 1. Neural Hierarchical
Interpolation for Time Series (NHITS).*

## NHITS

::: neuralforecast.models.nhits.NHITS
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
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = NHITS(h=12,
              input_size=24,
              loss=DistributionLoss(distribution='StudentT', level=[80, 90], return_params=True),
              stat_exog_list=['airline1'],
              futr_exog_list=['trend'],
              n_freq_downsample=[2, 1, 1],
              scaler_type='robust',
              max_steps=200,
              early_stop_patience_steps=2,
              inference_windows_batch_size=1,
              val_check_steps=10,
              learning_rate=1e-3)

fcst = NeuralForecast(models=[model], freq='ME')
fcst.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
forecasts = fcst.predict(futr_df=Y_test_df)

# Plot quantile predictions
Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['NHITS-median'], c='blue', label='median')
plt.fill_between(x=plot_df['ds'][-12:], 
                 y1=plot_df['NHITS-lo-90'][-12:].values, 
                 y2=plot_df['NHITS-hi-90'][-12:].values,
                 alpha=0.4, label='level 90')
plt.legend()
plt.grid()
plt.plot()
```
