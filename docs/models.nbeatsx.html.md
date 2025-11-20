---
output-file: models.nbeatsx.html
title: NBEATSx
---

The Neural Basis Expansion Analysis
([`NBEATS`](./models.nbeats.html#nbeats))
is an
[`MLP`](./models.mlp.html#mlp)-based
deep neural architecture with backward and forward residual links. The
network has two variants: (1) in its interpretable configuration,
[`NBEATS`](./neuralforecast/models.nbeats.html#nbeats)
sequentially projects the signal into polynomials and harmonic basis to
learn trend and seasonality components; (2) in its generic
configuration, it substitutes the polynomial and harmonic basis for
identity basis and larger network’s depth. The Neural Basis Expansion
Analysis with Exogenous
([`NBEATSx`](./models.nbeatsx.html#nbeatsx)),
incorporates projections to exogenous temporal variables available at
the time of the prediction.

This method proved state-of-the-art
performance on the M3, M4, and Tourism Competition datasets, improving
accuracy by 3% over the `ESRNN` M4 competition winner. For Electricity
Price Forecasting tasks
[`NBEATSx`](./models.nbeatsx.html#nbeatsx)
model improved accuracy by 20% and 5% over `ESRNN` and
[`NBEATS`](./models.nbeats.html#nbeats),
and 5% on task-specialized
architectures.

**References**

- [Boris N. Oreshkin, Dmitri
Carpov, Nicolas Chapados, Yoshua Bengio (2019). “N-BEATS: Neural basis
expansion analysis for interpretable time series
forecasting”.](https://arxiv.org/abs/1905.10437)

- [Kin G. Olivares,
Cristian Challu, Grzegorz Marcjasz, Rafał Weron, Artur Dubrawski (2021).
“Neural basis expansion analysis with exogenous variables: Forecasting
electricity prices with NBEATSx”.](https://arxiv.org/abs/2104.05522)


![Figure 1. Neural Basis Expansion Analysis with Exogenous Variables.](imgs_models/nbeatsx.png)
*Figure 1. Neural Basis Expansion Analysis
with Exogenous Variables.*

## NBEATSx

::: neuralforecast.models.nbeatsx.NBEATSx
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
from neuralforecast.models import NBEATSx
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = NBEATSx(h=12, input_size=24,
                loss=MQLoss(level=[80, 90]),
                scaler_type='robust',
                dropout_prob_theta=0.5,
                stat_exog_list=['airline1'],
                futr_exog_list=['trend'],
                stack_types = ["identity", "trend", "seasonality", "exogenous"],
                n_blocks = [1,1,1,1],
                max_steps=200,
                val_check_steps=10,
                early_stop_patience_steps=2)

nf = NeuralForecast(
    models=[model],
    freq='ME'
)
nf.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
Y_hat_df = nf.predict(futr_df=Y_test_df)

# Plot quantile predictions
Y_hat_df = Y_hat_df.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['NBEATSx-median'], c='blue', label='median')
plt.fill_between(x=plot_df['ds'][-12:], 
                 y1=plot_df['NBEATSx-lo-90'][-12:].values, 
                 y2=plot_df['NBEATSx-hi-90'][-12:].values,
                 alpha=0.4, label='level 90')
plt.legend()
plt.grid()
plt.plot()
```
