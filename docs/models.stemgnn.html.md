---
output-file: models.stemgnn.html
title: StemGNN
---

The Spectral Temporal Graph Neural Network
([`StemGNN`](https://nixtlaverse.nixtla.io/neuralforecast/models.stemgnn.html#stemgnn))
is a Graph-based multivariate time-series forecasting model.
[`StemGNN`](https://nixtlaverse.nixtla.io/neuralforecast/models.stemgnn.html#stemgnn)
jointly learns temporal dependencies and inter-series correlations in
the spectral domain, by combining Graph Fourier Transform (GFT) and
Discrete Fourier Transform (DFT).

This method proved state-of-the-art performance on geo-temporal datasets
such as `Solar`, `METR-LA`, and `PEMS-BAY`, and

**References**
 -[Defu Cao, Yujing Wang, Juanyong Duan, Ce Zhang, Xia
Zhu, Congrui Huang, Yunhai Tong, Bixiong Xu, Jing Bai, Jie Tong, Qi
Zhang (2020). “Spectral Temporal Graph Neural Network for Multivariate
Time-series
Forecasting”.](https://proceedings.neurips.cc/paper/2020/hash/cdf6581cb7aca4b7e19ef136c6e601a5-Abstract.html)

![Figure 1. StemGNN.](imgs_models/stemgnn.png)
*Figure 1. StemGNN.*

## StemGNN

::: neuralforecast.models.stemgnn.StemGNN
    options:
      members:
        - fit
        - predict
      heading_level: 3

## Usage Examples

Train model and forecast future values with `predict` method.


```python
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import StemGNN
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
from neuralforecast.losses.pytorch import MAE

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = StemGNN(h=12,
                input_size=24,
                n_series=2,
                scaler_type='standard',
                max_steps=500,
                early_stop_patience_steps=-1,
                val_check_steps=10,
                learning_rate=1e-3,
                loss=MAE(),
                valid_loss=MAE(),
                batch_size=32
                )

fcst = NeuralForecast(models=[model], freq='ME')
fcst.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
forecasts = fcst.predict(futr_df=Y_test_df)

# Plot predictions
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['StemGNN'], c='blue', label='Forecast')
ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Year', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
```

Using `cross_validation` to forecast multiple historic values.


```python
fcst = NeuralForecast(models=[model], freq='M')
forecasts = fcst.cross_validation(df=AirPassengersPanel, static_df=AirPassengersStatic, n_windows=2, step_size=12)

# Plot predictions
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
Y_hat_df = forecasts.loc['Airline1']
Y_df = AirPassengersPanel[AirPassengersPanel['unique_id']=='Airline1']

plt.plot(Y_df['ds'], Y_df['y'], c='black', label='True')
plt.plot(Y_hat_df['ds'], Y_hat_df['StemGNN'], c='blue', label='Forecast')
ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Year', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
```
