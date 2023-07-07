---
title: Probabilistic Long-Horizon
---

export const quartoRawHtml =
[`
  <div id="df-354ffb05-28c1-497a-81bb-fe74a34dbbc7">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
</style>
`,`
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-354ffb05-28c1-497a-81bb-fe74a34dbbc7')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }
    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }
    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }
    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }
    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
      <script>
        const buttonEl =
          document.querySelector('#df-354ffb05-28c1-497a-81bb-fe74a34dbbc7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';
        async function convertToInteractive(key) {
          const element = document.querySelector('#df-354ffb05-28c1-497a-81bb-fe74a34dbbc7');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;
          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  `];

Long-horizon forecasting is challenging because of the *volatility* of
the predictions and the *computational complexity*. To solve this
problem we created the [NHITS](https://arxiv.org/abs/2201.12886) model
and made the code available [NeuralForecast
library](https://nixtla.github.io/neuralforecast/models.nhits.html).
`NHITS` specializes its partial outputs in the different frequencies of
the time series through hierarchical interpolation and multi-rate input
processing. We model the target time-series with Student’s
t-distribution. The `NHITS` will output the distribution parameters for
each timestamp.

In this notebook we show how to use `NHITS` on the
[ETTm2](https://github.com/zhouhaoyi/ETDataset) benchmark dataset for
probabilistic forecasting. This data set includes data points for 2
Electricity Transformers at 2 stations, including load, oil temperature.

We will show you how to load data, train, and perform automatic
hyperparameter tuning, **to achieve SoTA performance**, outperforming
even the latest Transformer architectures for a fraction of their
computational cost (50x faster).

You can run these experiments using GPU with Google Colab.

<a href="https://colab.research.google.com/github/Nixtla/neuralforecast/blob/main/nbs/examples/LongHorizon_Probabilistic.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## 1. Libraries {#libraries}

<details>
<summary>Code</summary>

``` python
%%capture
!pip install neuralforecast datasetsforecast
```

</details>
<details>
<summary>Code</summary>

``` python
import torch
import pandas as pd
from datasetsforecast.long_horizon import LongHorizon
```

</details>
<details>
<summary>Code</summary>

``` python
torch.cuda.is_available()
```

</details>

``` text
True
```

## 2. Load ETTm2 Data {#load-ettm2-data}

The `LongHorizon` class will automatically download the complete ETTm2
dataset and process it.

It return three Dataframes: `Y_df` contains the values for the target
variables, `X_df` contains exogenous calendar features and `S_df`
contains static features for each time-series (none for ETTm2). For this
example we will only use `Y_df`.

If you want to use your own data just replace `Y_df`. Be sure to use a
long format and have a simmilar structure than our data set.

<details>
<summary>Code</summary>

``` python
# Change this to your own data to try the model
Y_df, _, _ = LongHorizon.load(directory='./', group='ETTm2')
Y_df['ds'] = pd.to_datetime(Y_df['ds'])

# For this excercise we are going to take 960 timestamps as validation and test
n_time = len(Y_df.ds.unique())
val_size = 96*10
test_size = 96*10

Y_df.groupby('unique_id').head(2)
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[0] }} />

|        | unique_id | ds                  | y         |
|--------|-----------|---------------------|-----------|
| 0      | HUFL      | 2016-07-01 00:00:00 | -0.041413 |
| 1      | HUFL      | 2016-07-01 00:15:00 | -0.185467 |
| 57600  | HULL      | 2016-07-01 00:00:00 | 0.040104  |
| 57601  | HULL      | 2016-07-01 00:15:00 | -0.214450 |
| 115200 | LUFL      | 2016-07-01 00:00:00 | 0.695804  |
| 115201 | LUFL      | 2016-07-01 00:15:00 | 0.434685  |
| 172800 | LULL      | 2016-07-01 00:00:00 | 0.434430  |
| 172801 | LULL      | 2016-07-01 00:15:00 | 0.428168  |
| 230400 | MUFL      | 2016-07-01 00:00:00 | -0.599211 |
| 230401 | MUFL      | 2016-07-01 00:15:00 | -0.658068 |
| 288000 | MULL      | 2016-07-01 00:00:00 | -0.393536 |
| 288001 | MULL      | 2016-07-01 00:15:00 | -0.659338 |
| 345600 | OT        | 2016-07-01 00:00:00 | 1.018032  |
| 345601 | OT        | 2016-07-01 00:15:00 | 0.980124  |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[1] }} />


:::important

DataFrames must include all `['unique_id', 'ds', 'y']` columns. Make
sure `y` column does not have missing or non-numeric values.

:::

Next, plot the `HUFL` variable marking the validation and train splits.

<details>
<summary>Code</summary>

``` python
import matplotlib.pyplot as plt
```

</details>
<details>
<summary>Code</summary>

``` python
# We are going to plot the temperature of the transformer 
# and marking the validation and train splits
u_id = 'HUFL'
x_plot = pd.to_datetime(Y_df[Y_df.unique_id==u_id].ds)
y_plot = Y_df[Y_df.unique_id==u_id].y.values

x_val = x_plot[n_time - val_size - test_size]
x_test = x_plot[n_time - test_size]

fig = plt.figure(figsize=(10, 5))
fig.tight_layout()

plt.plot(x_plot, y_plot)
plt.xlabel('Date', fontsize=17)
plt.ylabel('OT [15 min temperature]', fontsize=17)

plt.axvline(x_val, color='black', linestyle='-.')
plt.axvline(x_test, color='black', linestyle='-.')
plt.text(x_val, 5, '  Validation', fontsize=12)
plt.text(x_test, 3, '  Test', fontsize=12)

plt.grid()
plt.show()
plt.close()
```

</details>

![](LongHorizon_Probabilistic_files/figure-markdown_strict/cell-7-output-1.png)

## 3. Hyperparameter selection and forecasting {#hyperparameter-selection-and-forecasting}

The `AutoNHITS` class will automatically perform hyperparamter tunning
using [Tune library](https://docs.ray.io/en/latest/tune/index.html),
exploring a user-defined or default search space. Models are selected
based on the error on a validation set and the best model is then stored
and used during inference.

The `AutoNHITS.default_config` attribute contains a suggested
hyperparameter space. Here, we specify a different search space
following the paper’s hyperparameters. Notice that *1000 Stochastic
Gradient Steps* are enough to achieve SoTA performance. Feel free to
play around with this space.

<details>
<summary>Code</summary>

``` python
from ray import tune

from neuralforecast.auto import AutoNHITS
from neuralforecast.core import NeuralForecast

from neuralforecast.losses.pytorch import DistributionLoss

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
```

</details>
<details>
<summary>Code</summary>

``` python
horizon = 96 # 24hrs = 4 * 15 min.

# Use your own config or AutoNHITS.default_config
nhits_config = {
       "learning_rate": tune.choice([1e-3]),                                     # Initial Learning rate
       "max_steps": tune.choice([1000]),                                         # Number of SGD steps
       "input_size": tune.choice([5 * horizon]),                                 # input_size = multiplier * horizon
       "batch_size": tune.choice([7]),                                           # Number of series in windows
       "windows_batch_size": tune.choice([256]),                                 # Number of windows in batch
       "n_pool_kernel_size": tune.choice([[2, 2, 2], [16, 8, 1]]),               # MaxPool's Kernelsize
       "n_freq_downsample": tune.choice([[168, 24, 1], [24, 12, 1], [1, 1, 1]]), # Interpolation expressivity ratios
       "activation": tune.choice(['ReLU']),                                      # Type of non-linear activation
       "n_blocks":  tune.choice([[1, 1, 1]]),                                    # Blocks per each 3 stacks
       "mlp_units":  tune.choice([[[512, 512], [512, 512], [512, 512]]]),        # 2 512-Layers per block for each stack
       "interpolation_mode": tune.choice(['linear']),                            # Type of multi-step interpolation
       "random_seed": tune.randint(1, 10),
       "scaler_type": tune.choice(['robust']),
       "val_check_steps": tune.choice([100])
    }
```

</details>

:::tip

Refer to https://docs.ray.io/en/latest/tune/index.html for more
information on the different space options, such as lists and continous
intervals.m

:::

To instantiate `AutoNHITS` you need to define:

-   `h`: forecasting horizon
-   `loss`: training loss. Use the `DistributionLoss` to produce
    probabilistic forecasts.
-   `config`: hyperparameter search space. If `None`, the `AutoNHITS`
    class will use a pre-defined suggested hyperparameter space.
-   `num_samples`: number of configurations explored.

<details>
<summary>Code</summary>

``` python
models = [AutoNHITS(h=horizon,
                    loss=DistributionLoss(distribution='StudentT', level=[80, 90]), 
                    config=nhits_config,
                    num_samples=5)]
```

</details>

Fit the model by instantiating a `NeuralForecast` object with the
following required parameters:

-   `models`: a list of models.

-   `freq`: a string indicating the frequency of the data. (See [panda’s
    available
    frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).)

<details>
<summary>Code</summary>

``` python
# Fit and predict
nf = NeuralForecast(
    models=models,
    freq='15min')
```

</details>

The `cross_validation` method allows you to simulate multiple historic
forecasts, greatly simplifying pipelines by replacing for loops with
`fit` and `predict` methods.

With time series data, cross validation is done by defining a sliding
window across the historical data and predicting the period following
it. This form of cross validation allows us to arrive at a better
estimation of our model’s predictive abilities across a wider range of
temporal instances while also keeping the data in the training set
contiguous as is required by our models.

The `cross_validation` method will use the validation set for
hyperparameter selection, and will then produce the forecasts for the
test set.

<details>
<summary>Code</summary>

``` python
%%capture
Y_hat_df = nf.cross_validation(df=Y_df, val_size=val_size,
                               test_size=test_size, n_windows=None)
```

</details>

## 4. Visualization {#visualization}

Finally, we merge the forecasts with the `Y_df` dataset and plot the
forecasts.

<details>
<summary>Code</summary>

``` python
Y_hat_df = Y_hat_df.reset_index(drop=True)
Y_hat_df = Y_hat_df[(Y_hat_df['unique_id']=='OT') & (Y_hat_df['cutoff']=='2018-02-11 12:00:00')]
Y_hat_df = Y_hat_df.drop(columns=['y','cutoff'])
```

</details>
<details>
<summary>Code</summary>

``` python
plot_df = Y_df.merge(Y_hat_df, on=['unique_id','ds'], how='outer').tail(96*10+50+96*4).head(96*2+96*4)

plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['AutoNHITS-median'], c='blue', label='median')
plt.fill_between(x=plot_df['ds'], 
                    y1=plot_df['AutoNHITS-lo-90.0'], y2=plot_df['AutoNHITS-hi-90.0'],
                    alpha=0.4, label='level 90')
plt.legend()
plt.grid()
plt.plot()
```

</details>

``` text
[]
```

![](LongHorizon_Probabilistic_files/figure-markdown_strict/cell-14-output-2.png)

## References {#references}

[Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza,
Max Mergenthaler-Canseco, Artur Dubrawski (2021). NHITS: Neural
Hierarchical Interpolation for Time Series Forecasting. Accepted at AAAI
2023.](https://arxiv.org/abs/2201.12886)

