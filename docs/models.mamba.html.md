---
description: >-
  Mamba: linear-time sequence model with selective state spaces. Input-dependent state-space recurrence with O(L) scaling in sequence length, replacing transformer self-attention for time series forecasting.
output-file: models.mamba.html
title: Mamba
---

`Mamba` is a sequence model based on selective state-space models (SSMs) that
achieves linear-time scaling in the sequence length while remaining competitive
with attention-based architectures on long-range tasks. Unlike standard SSMs
whose parameters are time-invariant, Mamba makes the SSM parameters
input-dependent (the *selective* mechanism), which allows the model to
selectively propagate or forget information based on the current token.

The neuralforecast adaptation builds an encoder input by concatenating the
target series with historic, static, and the past portion of future
exogenous features. The window is linearly projected to `hidden_size`,
processed by a stack of `e_layers` Mamba residual blocks, then projected
from the input length to the forecast horizon. The future portion of the
future-exogenous features is concatenated to each horizon step before the
final output projection, so known-future covariates inform every forecast
step. The selective scan uses a chunked parallel scan — a closed-form
cumsum formulation within fixed-size chunks composed sequentially across
chunks — keeping the cumulative exponent bounded for numerical stability in
fp32 during training, while still avoiding the per-step Python loop. It
runs efficiently on CPU/GPU/MPS without depending on the `mamba_ssm` CUDA
kernels.

**References**

-[Albert Gu, Tri Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"](https://arxiv.org/abs/2312.00752)

## Mamba

::: neuralforecast.models.mamba.Mamba
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
from neuralforecast.models import Mamba
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = Mamba(h=12, input_size=24,
              loss=DistributionLoss(distribution='Normal', level=[80, 90]),
              scaler_type='robust',
              learning_rate=1e-3,
              max_steps=200,
              val_check_steps=10,
              early_stop_patience_steps=2)

fcst = NeuralForecast(
    models=[model],
    freq='ME'
)
fcst.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
forecasts = fcst.predict(futr_df=Y_test_df)

# Plot predictions
Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['Mamba-median'], c='blue', label='median')
plt.fill_between(x=plot_df['ds'][-12:], 
                 y1=plot_df['Mamba-lo-90'][-12:].values, 
                 y2=plot_df['Mamba-hi-90'][-12:].values,
                 alpha=0.4, label='level 90')
plt.grid()
plt.legend()
plt.plot()
```
