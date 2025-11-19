---
output-file: models.nbeats.html
title: NBEATS
---

The Neural Basis Expansion Analysis
([`NBEATS`](https://nixtlaverse.nixtla.io/neuralforecast/models.nbeats.html#nbeats))
is an
[`MLP`](https://nixtlaverse.nixtla.io/neuralforecast/models.mlp.html#mlp)-based
deep neural architecture with backward and forward residual links. The
network has two variants: (1) in its interpretable configuration,
[`NBEATS`](https://nixtlaverse.nixtla.io/neuralforecast/models.nbeats.html#nbeats)
sequentially projects the signal into polynomials and harmonic basis to
learn trend and seasonality components; (2) in its generic
configuration, it substitutes the polynomial and harmonic basis for
identity basis and larger network’s depth. The Neural Basis Expansion
Analysis with Exogenous
([`NBEATSx`](https://nixtlaverse.nixtla.io/neuralforecast/models.nbeatsx.html#nbeatsx)),
incorporates projections to exogenous temporal variables available at
the time of the prediction.

This method proved state-of-the-art performance on the M3, M4, and
Tourism Competition datasets, improving accuracy by 3% over the `ESRNN`
M4 competition winner.

**References**
 -[Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados,
Yoshua Bengio (2019). “N-BEATS: Neural basis expansion analysis for
interpretable time series
forecasting”.](https://arxiv.org/abs/1905.10437)

![Figure 1. Neural Basis Expansion Analysis.](imgs_models/nbeats.png)
*Figure 1. Neural Basis Expansion
Analysis.*

## NBEATS

::: neuralforecast.models.nbeats.NBEATS
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
from neuralforecast.models import NBEATS
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = NBEATS(h=12, input_size=24,
               basis='changepoint',
               n_basis=2,
               loss=DistributionLoss(distribution='Poisson', level=[80, 90]),
               stack_types = ['identity', 'trend', 'seasonality'],
               max_steps=100,
               val_check_steps=10,
               early_stop_patience_steps=2)

fcst = NeuralForecast(
    models=[model],
    freq='ME'
)
fcst.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
forecasts = fcst.predict(futr_df=Y_test_df)

# Plot quantile predictions
Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['NBEATS-median'], c='blue', label='median')
plt.fill_between(x=plot_df['ds'][-12:], 
                 y1=plot_df['NBEATS-lo-90'][-12:].values, 
                 y2=plot_df['NBEATS-hi-90'][-12:].values,
                 alpha=0.4, label='level 90')
plt.grid()
plt.legend()
plt.plot()
```
