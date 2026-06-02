---
description: >-
  xLSTM: An extension of the LSTM model.
output-file: models.xlstm.html
title: xLSTM
---

xLSTM is an RNN-based model for sequence modeling that extends the classical Long Short-Term Memory architecture with exponential gating and novel memory structures. The architecture introduces two new memory cell variants: sLSTM, which uses a scalar memory with a scalar update rule and a new cross-cell memory mixing mechanism, and mLSTM, which replaces the scalar cell state with a matrix memory updated via a covariance (outer product) rule, enabling full parallelizability. These cell variants are integrated into residual blocks to form xLSTM blocks, which are then stacked into full xLSTM architectures.

**References**

- [Maximilian, B., et al. "xLSTM: Extended Long Short-Term Memory"](https://arxiv.org/abs/2405.04517)

![Figure 1. Architecture of xLSTM](./imgs_models/xlstm.png)
*Figure 1. Architecture of xLSTM*

## xLSTM

::: neuralforecast.models.xlstm.xLSTM
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
from neuralforecast.models import xLSTM
from neuralforecast.losses.pytorch import MAE
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = xLSTM(h=12, 
            input_size=24,
            stat_exog_list=['airline1'],
            hist_exog_list=["y_[lag12]"],
            futr_exog_list=['trend'],            
            loss = MAE(),
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
plt.plot(plot_df['ds'], plot_df['xLSTM'], c='blue', label='median')
plt.grid()
plt.legend()
plt.plot()
```