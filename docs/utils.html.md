---
description: >-
  NeuralForecast utility functions and datasets. Includes AirPassengers data, time feature generation, prediction intervals, and synthetic panel data generators.
output-file: utils.html
title: Example Data
---

The `core.NeuralForecast` class allows you to efficiently fit multiple
`NeuralForecast` models for large sets of time series. It operates with pandas DataFrame `df` that identifies individual series and datestamps with the `unique_id` and `ds` columns, and the `y` column denotes the target time
series variable. To assist development, we declare useful datasets that we use throughout all `NeuralForecast`'s unit tests.

## 1. Synthetic Panel Data

::: neuralforecast.utils.generate_series

```python
synthetic_panel = generate_series(n_series=2)
synthetic_panel.groupby('unique_id').head(4)
```

```python
temporal_df, static_df = generate_series(n_series=1000, n_static_features=2,
                                         n_temporal_features=4, equal_ends=False)
static_df.head(2)
```

## 2. AirPassengers Data

The classic Box & Jenkins airline data. Monthly totals of international
airline passengers, 1949 to 1960.

It has been used as a reference on several forecasting libraries, since
it is a series that shows clear trends and seasonalities it offers a
nice opportunity to quickly showcase a model’s predictions performance.

```python
AirPassengersDF.head(12)
```

```python
#We are going to plot the ARIMA predictions, and the prediction intervals.
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
plot_df = AirPassengersDF.set_index('ds')

plot_df[['y']].plot(ax=ax, linewidth=2)
ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
```

```python
import numpy as np
import pandas as pd
```

```python
n_static_features = 3
n_series = 5

static_features = np.random.uniform(low=0.0, high=1.0,
                        size=(n_series, n_static_features))
static_df = pd.DataFrame.from_records(static_features,
                   columns = [f'static_{i}'for i in  range(n_static_features)])
static_df['unique_id'] = np.arange(n_series)
```

```python
static_df
```

## 3. Panel AirPassengers Data

Extension to classic Box & Jenkins airline data. Monthly totals of
international airline passengers, 1949 to 1960.

It includes two series with static, temporal and future exogenous
variables, that can help to explore the performance of models like
[`NBEATSx`](https://nixtlaverse.nixtla.io/neuralforecast/models.nbeatsx.html#nbeatsx)
and
[`TFT`](https://nixtlaverse.nixtla.io/neuralforecast/models.tft.html#tft).

```python
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
plot_df = AirPassengersPanel.set_index('ds')

plot_df.groupby('unique_id')['y'].plot(legend=True)
ax.set_title('AirPassengers Panel Data', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(title='unique_id', prop={'size': 15})
ax.grid()
```

```python
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
plot_df = AirPassengersPanel[AirPassengersPanel.unique_id=='Airline1'].set_index('ds')

plot_df[['y', 'trend', 'y_[lag12]']].plot(ax=ax, linewidth=2)
ax.set_title('Box-Cox AirPassengers Data', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
```

## 4. Time Features

We have developed a utility that generates normalized calendar features
for use as absolute positional embeddings in Transformer-based models.
These embeddings capture seasonal patterns in time series data and can
be easily incorporated into the model architecture. Additionally, the
features can be used as exogenous variables in other models to inform
them of calendar patterns in the data.

### References

- [Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai
Zhang, Jianxin Li, Hui Xiong, Wancai Zhang. “Informer: Beyond Efficient
Transformer for Long Sequence Time-Series
Forecasting”](https://arxiv.org/abs/2012.07436)

------------------------------------------------------------------------

::: neuralforecast.utils.augment_calendar_df

::: neuralforecast.utils.time_features_from_frequency_str

::: neuralforecast.utils.WeekOfYear

::: neuralforecast.utils.MonthOfYear

::: neuralforecast.utils.DayOfYear

::: neuralforecast.utils.DayOfMonth

::: neuralforecast.utils.DayOfWeek

::: neuralforecast.utils.HourOfDay

::: neuralforecast.utils.MinuteOfHour

::: neuralforecast.utils.SecondOfMinute

::: neuralforecast.utils.TimeFeature

```python
AirPassengerPanelCalendar, calendar_cols = augment_calendar_df(df=AirPassengersPanel, freq='M')
AirPassengerPanelCalendar.head()
```

```python
plot_df = AirPassengerPanelCalendar[AirPassengerPanelCalendar.unique_id=='Airline1'].set_index('ds')
plt.plot(plot_df['month'])
plt.grid()
plt.xlabel('Datestamp')
plt.ylabel('Normalized Month')
plt.show()
```

::: neuralforecast.utils.get_indexer_raise_missing

## 5. Prediction Intervals

::: neuralforecast.utils.PredictionIntervals

::: neuralforecast.utils.add_conformal_distribution_intervals

::: neuralforecast.utils.add_conformal_error_intervals

::: neuralforecast.utils.get_prediction_interval_method

::: neuralforecast.utils.quantiles_to_level

::: neuralforecast.utils.level_to_quantiles
