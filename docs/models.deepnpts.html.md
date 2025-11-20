---
output-file: models.deepnpts.html
title: DeepNPTS
---

Deep Non-Parametric Time Series Forecaster
([`DeepNPTS`](./models.deepnpts.html#deepnpts))
is a non-parametric baseline model for time-series forecasting. This
model generates predictions by sampling from the empirical distribution
according to a tunable strategy. This strategy is learned by exploiting
the information across multiple related time series. This model provides
a strong, simple baseline for time series forecasting.

**References**

- [Rangapuram, Syama Sundar, Jan Gasthaus, Lorenzo
Stella, Valentin Flunkert, David Salinas, Yuyang Wang, and Tim
Januschowski (2023). “Deep Non-Parametric Time Series Forecaster”.
arXiv.](https://arxiv.org/abs/2312.14657)


> **Losses**
>
> This implementation differs from the original work in that a weighted
> sum of the empirical distribution is returned as forecast. Therefore,
> it only supports point losses.

## DeepNPTS

::: neuralforecast.models.deepnpts.DeepNPTS
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
from neuralforecast.models import DeepNPTS
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

nf = NeuralForecast(
    models=[DeepNPTS(h=12,
                   input_size=24,
                   stat_exog_list=['airline1'],
                   futr_exog_list=['trend'],
                   max_steps=1000,
                   val_check_steps=10,
                   early_stop_patience_steps=3,
                   scaler_type='robust',
                   enable_progress_bar=True),
    ],
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
plt.plot(plot_df['ds'], plot_df['DeepNPTS'], c='red', label='mean')
plt.grid()
plt.plot()
```