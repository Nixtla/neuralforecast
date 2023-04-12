# Nixtla &nbsp; [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Statistical%20Forecasting%20Algorithms%20by%20Nixtla%20&url=https://github.com/Nixtla/neuralforecast&via=nixtlainc&hashtags=StatisticalModels,TimeSeries,Forecasting) &nbsp;[![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlacommunity/shared_invite/zt-1pmhan9j5-F54XR20edHk0UtYAPcW4KQ)

<div align="center">
<!--- <img src="https://raw.githubusercontent.com/Nixtla/neuralforecast1/main/nbs/imgs_indx/logo_mid.png"> --->
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/imgs_indx/logo_mid.png">
<h1 align="center">Neural 🧠 Forecast</h1>
<h3 align="center">User friendly state-of-the-art neural forecasting models.</h3>

[![CI](https://github.com/Nixtla/neuralforecast/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/Nixtla/neuralforecast/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/pypi/pyversions/neuralforecast)](https://pypi.org/project/neuralforecast/)
[![PyPi](https://img.shields.io/pypi/v/neuralforecast?color=blue)](https://pypi.org/project/neuralforecast/)
[![conda-nixtla](https://img.shields.io/conda/vn/conda-forge/neuralforecast?color=seagreen&label=conda)](https://anaconda.org/conda-forge/neuralforecast)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Nixtla/neuralforecast/blob/main/LICENSE)
[![docs](https://img.shields.io/website-up-down-green-red/http/nixtla.github.io/neuralforecast.svg?label=docs)](https://nixtla.github.io/neuralforecast/)  
    
**NeuralForecast** offers a large collection of neural forecasting models focused on their usability, and robustness. The models range from classic networks like `MLP`, `RNN`s to novel proven contributions like `NBEATS`, `TFT` and other architectures.
</div>

## 💻 Installation
<details open>
<summary>PyPI</summary>

You can install `NeuralForecast`'s *released version* from the Python package index [pip](https://pypi.org/project/neuralforecast/) with:

```python
pip install neuralforecast
```

(Installing inside a python virtualenvironment or a conda environment is recommended.)
</details>

<details open>
<summary>Conda</summary>
  
Also you can install `NeuralForecast`'s *released version* from [conda](https://anaconda.org/conda-forge/neuralforecast) with:

```python
conda install -c conda-forge neuralforecast
```

(Installing inside a python virtual environment or a conda environment is recommended.)
</details>

<details>
<summary>Dev Mode</summary>
If you want to make some modifications to the code and see the effects in real time (without reinstalling), follow the steps below:

```bash
git clone https://github.com/Nixtla/neuralforecast.git
cd neuralforecast
pip install -e .
```
</details>

## 🏃🏻‍♀️🏃 Getting Started
To get started follow this [guide](https://colab.research.google.com/github/Nixtla/neuralforecast/blob/main/nbs/examples/Getting_Started.ipynb), where we explore `NBEATS`, extend it towards probabilistic predictions and exogenous variables.

Or follow this simple example where we train the `NBEATS` model and predict the classic Box-Jenkins air passengers dataset.
```python
import numpy as np
import pandas as pd
from IPython.display import display, Markdown

import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
from neuralforecast.utils import AirPassengersDF

# Split data and declare panel dataset
Y_df = AirPassengersDF
Y_train_df = Y_df[Y_df.ds<='1959-12-31'] # 132 train
Y_test_df = Y_df[Y_df.ds>'1959-12-31'] # 12 test

# Fit and predict with NBEATS and NHITS models
horizon = len(Y_test_df)
models = [NBEATS(input_size=2 * horizon, h=horizon, max_epochs=50),
          NHITS(input_size=2 * horizon, h=horizon, max_epochs=50)]
nf = NeuralForecast(models=models, freq='M')
nf.fit(df=Y_train_df)
Y_hat_df = nf.predict().reset_index()

# Plot predictions
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
Y_hat_df = Y_test_df.merge(Y_hat_df, how='left', on=['unique_id', 'ds'])
plot_df = pd.concat([Y_train_df, Y_hat_df]).set_index('ds')

plot_df[['y', 'NBEATS', 'NHITS']].plot(ax=ax, linewidth=2)

ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
```
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/imgs_indx/nbeats_example.png">

## 🎉 New!
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nixtla/neuralforecast/blob/main/nbs/examples/Getting_Started.ipynb) **Quick Start**: Minimal usage example of the NeuralForecast library on Box's AirPassengers Data.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nixtla/neuralforecast/blob/main/nbs/examples/UncertaintyIntervals.ipynb) **Multi Quantile NBEATS Example**: Produce accurate and efficient probabilistic forecasts in long-horizon settings. Outperforming AutoARIMA's accuracy in a fraction of the time.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nixtla/neuralforecast/blob/main/nbs/examples/LongHorizon_with_NHITS.ipynb) **Long Horizon NHITS Example**:  Load, train, and tune hyperparameters, to achieve Long-Horizon SoTA. Outperform Award Winning Transformers by 25% in 50x less time.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nixtla/neuralforecast/blob/main/nbs/examples/HierarchicalNetworks.ipynb) **Hierarchical Forecasting HINT Example**:  Obtain accurate probabilistic coherent predictions for hierarchical datasets.

## 🔥  Highlights

* Unified `StatsForecast` interface `NeuralForecast().fit(Y_df).predict(h=7)`.
* Industry/Competition proven `ESRNN`, `NBEATS`, and `TFT` implementations.
* Improve accuracy and speed over classic `ARIMA`/`ETS` in two lines of code. Check the experiments [here](xXmissingXx).
* Predict Series with little to no history, using Transfer learning. Check the experiments [here](https://github.com/Nixtla/transfer-learning-time-series).

## 🎊 Features 

* **Exogenous Variables**: Static, lagged and future exogenous support.
* **Forecast Interpretability**: Plot trend, seasonality and exogenous `NBEATS`, `NHITS`, `TFT`, `ESRNN` prediction components.
* **Probabilistic Forecasting**: Simple model adapters for quantile losses and parametric distributions.
* **Train and Evaluation Losses** Scale-dependent, percentage and scale independent errors, and parametric likelihoods.
* **Automatic Model Selection** Parallelized automatic hyperparameter tuning, that efficiently searches best validation configuration.
* **Simple Interface** Unified SKLearn Interface for `StatsForecast` and `MLForecast` compatibility.
* **Model Collection**: Out of the box implementation of `MLP`, `LSTM`, `RNN`, `TCN`, `DilatedRNN`, `NBEATS`, `NHITS`, `ESRNN`, `TFT`, `Informer`, `PatchTST` and `HINT`. See the entire [collection here](https://nixtla.github.io/neuralforecast/models.html).

Missing something? Please open an issue or write us in [![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)

## 📖 Why? 

There is a shared belief in Neural forecasting methods' capacity to improve our pipeline's accuracy and efficiency.

Unfortunately, available implementations and published research are yet to realize neural networks' potential. They are hard to use and continuously fail to improve over statistical methods while being computationally prohibitive. For this reason, we created `NeuralForecast`, a library favoring proven accurate and efficient models focusing on their usability.

## 🔬 Accuracy & ⏲ Speed 

### Industry/Competition Proven Methods
An extensive empirical evaluation is critical to generate confidence and promote the adoption and development of novel methods. For this reason, we replicate and verify the results of our implementation of the following industry/competition-proven methods: `ESRNN`, `NBEATS`, `NHITS`, and `TFT`. If you are interested in reproducing the results, check the experiments [here](https://github.com/Nixtla/neuralforecast/blob/main/nbs/experiments/).

### Simple and Efficient Method's Comparison
Like `core.StatsForecast`, the `core.NeuralForecast` wrapper class allows us to easily compare any model in the collection to select or ensemble the best performing methods. Aditionally it offers a high-end interface that operates with (potentially large) sets of time series data stored in pandas DataFrames. The `core.NeuralForecast` efficiently parallelizes computation across GPU resources. Check the experiments [here](https://github.com/Nixtla/neuralforecast/blob/main/nbs/examples/Getting_Started_complete.ipynb).

## 📖 Documentation (WIP)
The [documentation page](https://nixtla.github.io/neuralforecast/) contains the models' code documentation, methods, utils, and other tutorials. Docstrings accompany most code.

## 🔨 How to contribute
If you wish to contribute to the project, please refer to our [contribution guidelines](https://github.com/Nixtla/neuralforecast/blob/main/CONTRIBUTING.md).

## 📚 References
This work is highly influenced by the fantastic work of previous contributors and other scholars on the neural forecasting methods presented here. We want to highlight the work of [Boris Oreshkin](https://arxiv.org/abs/1905.10437), [Slawek Smyl](https://www.sciencedirect.com/science/article/pii/S0169207019301153), [Bryan Lim](https://www.sciencedirect.com/science/article/pii/S0169207021000637), and [David Salinas](https://arxiv.org/abs/1704.04110). We refer to [Benidis et al.](https://arxiv.org/abs/2004.10240) for a comprehensive survey of neural forecasting methods.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/FedericoGarza"><img src="https://avatars.githubusercontent.com/u/10517170?v=4?s=100" width="100px;" alt=""/><br /><sub><b>fede</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/commits?author=FedericoGarza" title="Code">💻</a> <a href="https://github.com/Nixtla/neuralforecast/issues?q=author%3AFedericoGarza" title="Bug reports">🐛</a> <a href="https://github.com/Nixtla/neuralforecast/commits?author=FedericoGarza" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/gdevos010"><img src="https://avatars.githubusercontent.com/u/15316026?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Greg DeVos</b></sub></a><br /><a href="#ideas-gdevos010" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/cchallu"><img src="https://avatars.githubusercontent.com/u/31133398?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Cristian Challu</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/commits?author=cchallu" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/mergenthaler"><img src="https://avatars.githubusercontent.com/u/4086186?v=4?s=100" width="100px;" alt=""/><br /><sub><b>mergenthaler</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/commits?author=mergenthaler" title="Documentation">📖</a> <a href="https://github.com/Nixtla/neuralforecast/commits?author=mergenthaler" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/kdgutier"><img src="https://avatars.githubusercontent.com/u/19935241?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Kin</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/commits?author=kdgutier" title="Code">💻</a> <a href="https://github.com/Nixtla/neuralforecast/issues?q=author%3Akdgutier" title="Bug reports">🐛</a> <a href="#data-kdgutier" title="Data">🔣</a></td>
    <td align="center"><a href="https://github.com/jmoralez"><img src="https://avatars.githubusercontent.com/u/8473587?v=4?s=100" width="100px;" alt=""/><br /><sub><b>José Morales</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/commits?author=jmoralez" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/alejandroxag"><img src="https://avatars.githubusercontent.com/u/64334543?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Alejandro</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/commits?author=alejandroxag" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="http://lavattiata.com"><img src="https://avatars.githubusercontent.com/u/48966177?v=4?s=100" width="100px;" alt=""/><br /><sub><b>stefanialvs</b></sub></a><br /><a href="#design-stefanialvs" title="Design">🎨</a></td>
    <td align="center"><a href="https://bandism.net/"><img src="https://avatars.githubusercontent.com/u/22633385?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ikko Ashimine</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/issues?q=author%3Aeltociear" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/vglaucus"><img src="https://avatars.githubusercontent.com/u/75549033?v=4?s=100" width="100px;" alt=""/><br /><sub><b>vglaucus</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/issues?q=author%3Avglaucus" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/pitmonticone"><img src="https://avatars.githubusercontent.com/u/38562595?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Pietro Monticone</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/issues?q=author%3Apitmonticone" title="Bug reports">🐛</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
