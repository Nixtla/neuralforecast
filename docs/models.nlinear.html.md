---
output-file: models.nlinear.html
title: NLinear
---

NLinear is a simple and fast yet accurate time series forecasting model
for long-horizon forecasting.

The architecture aims to boost the performance when there is a
distribution shift in the dataset: 

1. NLinear first subtracts the input
by the last value of the sequence; 
2. Then, the input goes through a
linear layer, and the subtracted part is added back before making the
final prediction.

**References**

- [Zeng, Ailing, et al. “Are transformers effective
for time series forecasting?.” Proceedings of the AAAI conference on
artificial intelligence. Vol. 37. No. 9.
2023.”](https://ojs.aaai.org/index.php/AAAI/article/view/26317)


![Figure 1. DLinear Architecture.](imgs_models/dlinear.png)
*Figure 1. DLinear
Architecture.*

## NLinear

::: neuralforecast.models.nlinear.NLinear
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
from neuralforecast.models import NLinear
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic, augment_calendar_df

AirPassengersPanel, calendar_cols = augment_calendar_df(df=AirPassengersPanel, freq='M')

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = NLinear(h=12,
                 input_size=24,
                 loss=DistributionLoss(distribution='StudentT', level=[80, 90], return_params=True),
                 scaler_type='robust',
                 learning_rate=1e-3,
                 max_steps=500,
                 val_check_steps=50,
                 early_stop_patience_steps=2)

nf = NeuralForecast(
    models=[model],
    freq='ME'
)
nf.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
forecasts = nf.predict(futr_df=Y_test_df)

Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

if model.loss.is_distribution_output:
    plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
    plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
    plt.plot(plot_df['ds'], plot_df['NLinear-median'], c='blue', label='median')
    plt.fill_between(x=plot_df['ds'][-12:], 
                    y1=plot_df['NLinear-lo-90'][-12:].values, 
                    y2=plot_df['NLinear-hi-90'][-12:].values,
                    alpha=0.4, label='level 90')
    plt.grid()
    plt.legend()
    plt.plot()
else:
    plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
    plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
    plt.plot(plot_df['ds'], plot_df['NLinear'], c='blue', label='Forecast')
    plt.legend()
    plt.grid()
```
