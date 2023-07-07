---
title: Example Data
---

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
%load_ext autoreload
%autoreload 2
```

</details>

:::

> The `core.NeuralForecast` class allows you to efficiently fit multiple
> `NeuralForecast` models for large sets of time series. It operates
> with pandas DataFrame `df` that identifies individual series and
> datestamps with the `unique_id` and `ds` columns, and the `y` column
> denotes the target time series variable. To assist development, we
> declare useful datasets that we use throughout all `NeuralForecast`’s
> unit tests.<br><br>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
import random
from itertools import chain

import numpy as np
import pandas as pd
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
import matplotlib.pyplot as plt

from nbdev.showdoc import add_docs, show_doc
```

</details>

:::

# <span style="color:DarkBlue">1. Synthetic Panel Data </span> {#synthetic-panel-data}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
def generate_series(n_series: int,
                    freq: str = 'D',
                    min_length: int = 50,
                    max_length: int = 500,
                    n_temporal_features: int = 0,
                    n_static_features: int = 0,
                    equal_ends: bool = False,
                    seed: int = 0) -> pd.DataFrame:
    """Generate Synthetic Panel Series.

    Generates `n_series` of frequency `freq` of different lengths in the interval [`min_length`, `max_length`].
    If `n_temporal_features > 0`, then each serie gets temporal features with random values.
    If `n_static_features > 0`, then a static dataframe is returned along the temporal dataframe.
    If `equal_ends == True` then all series end at the same date.

    **Parameters:**<br>
    `n_series`: int, number of series for synthetic panel.<br>
    `min_length`: int, minimal length of synthetic panel's series.<br>
    `max_length`: int, minimal length of synthetic panel's series.<br>
    `n_temporal_features`: int, default=0, number of temporal exogenous variables for synthetic panel's series.<br>
    `n_static_features`: int, default=0, number of static exogenous variables for synthetic panel's series.<br>
    `equal_ends`: bool, if True, series finish in the same date stamp `ds`.<br>
    `freq`: str, frequency of the data, [panda's available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).<br>

    **Returns:**<br>
    `freq`: pandas.DataFrame, synthetic panel with columns [`unique_id`, `ds`, `y`] and exogenous.
    """
    seasonalities = {'D': 7, 'M': 12}
    season = seasonalities[freq]

    rng = np.random.RandomState(seed)
    series_lengths = rng.randint(min_length, max_length + 1, n_series)
    total_length = series_lengths.sum()

    dates = pd.date_range('2000-01-01', periods=max_length, freq=freq).values
    uids = [
        np.repeat(i, serie_length) for i, serie_length in enumerate(series_lengths)
    ]
    if equal_ends:
        ds = [dates[-serie_length:] for serie_length in series_lengths]
    else:
        ds = [dates[:serie_length] for serie_length in series_lengths]

    y = np.arange(total_length) % season + rng.rand(total_length) * 0.5
    temporal_df = pd.DataFrame(dict(unique_id=chain.from_iterable(uids),
                                    ds=chain.from_iterable(ds),
                                    y=y))

    random.seed(seed)
    for i in range(n_temporal_features):
        random.seed(seed)
        temporal_values = [
            [random.randint(0, 100)] * serie_length for serie_length in series_lengths
        ]
        temporal_df[f'temporal_{i}'] = np.hstack(temporal_values)
        temporal_df[f'temporal_{i}'] = temporal_df[f'temporal_{i}'].astype('category')
        if i == 0:
            temporal_df['y'] = temporal_df['y'] * \
                                  (1 + temporal_df[f'temporal_{i}'].cat.codes)

    temporal_df['unique_id'] = temporal_df['unique_id'].astype('category')
    temporal_df['unique_id'] = temporal_df['unique_id'].cat.as_ordered()
    temporal_df = temporal_df.set_index('unique_id')

    if n_static_features > 0:
        static_features = np.random.uniform(low=0.0, high=1.0, 
                        size=(n_series, n_static_features))
        static_df = pd.DataFrame.from_records(static_features, 
                           columns = [f'static_{i}'for i in  range(n_static_features)])
        
        static_df['unique_id'] = np.arange(n_series)
        static_df['unique_id'] = static_df['unique_id'].astype('category')
        static_df['unique_id'] = static_df['unique_id'].cat.as_ordered()
        static_df = static_df.set_index('unique_id')        

        return temporal_df, static_df

    return temporal_df
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(generate_series, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
from neuralforecast.utils import generate_series

synthetic_panel = generate_series(n_series=2)
synthetic_panel.groupby('unique_id').head(4)
```

</details>
<details>
<summary>Code</summary>

``` python
temporal_df, static_df = generate_series(n_series=1000, n_static_features=2,
                                         n_temporal_features=4, equal_ends=False)
static_df.head(2)
```

</details>

# <span style="color:DarkBlue">2. AirPassengers Data </span> {#airpassengers-data}

The classic Box & Jenkins airline data. Monthly totals of international
airline passengers, 1949 to 1960.

It has been used as a reference on several forecasting libraries, since
it is a series that shows clear trends and seasonalities it offers a
nice opportunity to quickly showcase a model’s predictions performance.

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
AirPassengers = np.array([112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104.,
                          118., 115., 126., 141., 135., 125., 149., 170., 170., 158., 133.,
                          114., 140., 145., 150., 178., 163., 172., 178., 199., 199., 184.,
                          162., 146., 166., 171., 180., 193., 181., 183., 218., 230., 242.,
                          209., 191., 172., 194., 196., 196., 236., 235., 229., 243., 264.,
                          272., 237., 211., 180., 201., 204., 188., 235., 227., 234., 264.,
                          302., 293., 259., 229., 203., 229., 242., 233., 267., 269., 270.,
                          315., 364., 347., 312., 274., 237., 278., 284., 277., 317., 313.,
                          318., 374., 413., 405., 355., 306., 271., 306., 315., 301., 356.,
                          348., 355., 422., 465., 467., 404., 347., 305., 336., 340., 318.,
                          362., 348., 363., 435., 491., 505., 404., 359., 310., 337., 360.,
                          342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405.,
                          417., 391., 419., 461., 472., 535., 622., 606., 508., 461., 390.,
                          432.], dtype=np.float32)
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
AirPassengersDF = pd.DataFrame({'unique_id': np.ones(len(AirPassengers)),
                                'ds': pd.date_range(start='1949-01-01',
                                                    periods=len(AirPassengers), freq='M'),
                                'y': AirPassengers})
```

</details>

:::

<details>
<summary>Code</summary>

``` python
from neuralforecast.utils import AirPassengersDF

AirPassengersDF.head(12)
```

</details>
<details>
<summary>Code</summary>

``` python
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

</details>
<details>
<summary>Code</summary>

``` python
import numpy as np
import pandas as pd

n_static_features = 3
n_series = 5

static_features = np.random.uniform(low=0.0, high=1.0, 
                        size=(n_series, n_static_features))
static_df = pd.DataFrame.from_records(static_features, 
                   columns = [f'static_{i}'for i in  range(n_static_features)])
static_df['unique_id'] = np.arange(n_series)
```

</details>
<details>
<summary>Code</summary>

``` python
static_df
```

</details>

# <span style="color:DarkBlue">3. Panel AirPassengers Data </span> {#panel-airpassengers-data}

Extension to classic Box & Jenkins airline data. Monthly totals of
international airline passengers, 1949 to 1960.

It includes two series with static, temporal and future exogenous
variables, that can help to explore the performance of models like
`NBEATSx` and `TFT`.

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
# Declare Panel Data
unique_id = np.concatenate([['Airline1']*len(AirPassengers), ['Airline2']*len(AirPassengers)])
ds = np.concatenate([pd.date_range(start='1949-01-01', 
                                   periods=len(AirPassengers), freq='M').values,
                     pd.date_range(start='1949-01-01', 
                                   periods=len(AirPassengers), freq='M').values])
y = np.concatenate([AirPassengers, AirPassengers+300])

AirPassengersPanel = pd.DataFrame({'unique_id': unique_id, 'ds': ds, 'y': y})

# For future exogenous variables
# Declare SeasonalNaive12 and fill first 12 values with y
snaive = AirPassengersPanel.groupby('unique_id')['y'].shift(periods=12).reset_index(drop=True)
AirPassengersPanel['trend'] = range(len(AirPassengersPanel))
AirPassengersPanel['y_[lag12]'] = snaive
AirPassengersPanel['y_[lag12]'].fillna(AirPassengersPanel['y'], inplace=True)

# Declare Static Data
unique_id = np.array(['Airline1', 'Airline2'])
airline1_dummy = [0, 1]
airline2_dummy = [1, 0]
AirPassengersStatic = pd.DataFrame({'unique_id': unique_id,
                                    'airline1': airline1_dummy,
                                    'airline2': airline2_dummy})

AirPassengersPanel.groupby('unique_id').tail(4)
```

</details>

:::

<details>
<summary>Code</summary>

``` python
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
plot_df = AirPassengersPanel.set_index('ds')

plot_df.groupby('unique_id')['y'].plot(legend=True)
ax.set_title('AirPassengers Panel Data', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(title='unique_id', prop={'size': 15})
ax.grid()
```

</details>
<details>
<summary>Code</summary>

``` python
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
plot_df = AirPassengersPanel[AirPassengersPanel.unique_id=='Airline1'].set_index('ds')

plot_df[['y', 'trend', 'y_[lag12]']].plot(ax=ax, linewidth=2)
ax.set_title('Box-Cox AirPassengers Data', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
```

</details>

# 4. Time Features {#time-features}

We have developed a utility that generates normalized calendar features
for use as absolute positional embeddings in Transformer-based models.
These embeddings capture seasonal patterns in time series data and can
be easily incorporated into the model architecture. Additionally, the
features can be used as exogenous variables in other models to inform
them of calendar patterns in the data.

**References**<br> - [Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai
Zhang, Jianxin Li, Hui Xiong, Wancai Zhang. “Informer: Beyond Efficient
Transformer for Long Sequence Time-Series
Forecasting”](https://arxiv.org/abs/2012.07436)<br>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
from typing import List

class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex):
        return print('Overwrite with corresponding feature')

    def __repr__(self):
        return self.__class__.__name__ + "()"

class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5

class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5

class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5

class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5

class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5

class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5

class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5

class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.week - 1) / 52.0 - 0.5

def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    if freq_str not in ['Q', 'M', 'MS', 'W', 'D', 'B', 'H', 'T', 'S']:
        raise Exception('Frequency not supported')
    
    if freq_str in ['Q','M', 'MS']:
        return [cls() for cls in [MonthOfYear]]
    elif freq_str == 'W':
        return [cls() for cls in [DayOfMonth, WeekOfYear]]
    elif freq_str in ['D','B']:
        return [cls() for cls in [DayOfWeek, DayOfMonth, DayOfYear]]
    elif freq_str == 'H':
        return [cls() for cls in [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]]
    elif freq_str == 'T':
        return [cls() for cls in [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]]
    else:
        return [cls() for cls in [SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]]

def augment_calendar_df(df, freq='H'):
    """
    > * Q - [month]
    > * M - [month]
    > * W - [Day of month, week of year]
    > * D - [Day of week, day of month, day of year]
    > * B - [Day of week, day of month, day of year]
    > * H - [Hour of day, day of week, day of month, day of year]
    > * T - [Minute of hour*, hour of day, day of week, day of month, day of year]
    > * S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]
    *minute returns a number from 0-3 corresponding to the 15 minute period it falls into.
    """
    df = df.copy()

    freq_map = {
        'Q':['month'],
        'M':['month'],
        'MS':['month'],
        'W':['monthday', 'yearweek'],
        'D':['weekday','monthday','yearday'],
        'B':['weekday','monthday','yearday'],
        'H':['dayhour','weekday','monthday','yearday'],
        'T':['hourminute','dayhour','weekday','monthday','yearday'],
        'S':['minutesecond','hourminute','dayhour','weekday','monthday','yearday']
    }

    ds_col = pd.to_datetime(df.ds.values)
    ds_data = np.vstack([feat(ds_col) for feat in time_features_from_frequency_str(freq)]).transpose(1,0)
    ds_data = pd.DataFrame(ds_data, columns=freq_map[freq])
    
    return pd.concat([df, ds_data], axis=1), freq_map[freq]
```

</details>

:::

<details>
<summary>Code</summary>

``` python
AirPassengerPanelCalendar, calendar_cols = augment_calendar_df(df=AirPassengersPanel, freq='M')
AirPassengerPanelCalendar.head()
```

</details>
<details>
<summary>Code</summary>

``` python
plot_df = AirPassengerPanelCalendar[AirPassengerPanelCalendar.unique_id=='Airline1'].set_index('ds')
plt.plot(plot_df['month'])
plt.grid()
plt.xlabel('Datestamp')
plt.ylabel('Normalized Month')
plt.show()
```

</details>

