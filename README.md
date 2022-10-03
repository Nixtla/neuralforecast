# Nixtla &nbsp; [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Statistical%20Forecasting%20Algorithms%20by%20Nixtla%20&url=https://github.com/Nixtla/neuralforecast&via=nixtlainc&hashtags=StatisticalModels,TimeSeries,Forecasting) &nbsp;[![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)

<div align="center">
<!--- <img src="https://raw.githubusercontent.com/Nixtla/neuralforecast1/main/nbs/imgs_indx/logo_mid.png"> --->
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/imgs_indx/logo_mid.png">
<h1 align="center">Neural 🧠 Forecast</h1>
<h3 align="center">User friendly state-of-the-art neural forecasting models.</h3>

[![CI](https://github.com/Nixtla/neuralforecast/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/Nixtla/neuralforecast/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/pypi/pyversions/neuralforecast)](https://pypi.org/project/neuralforecast/)
[![PyPi](https://img.shields.io/pypi/v/neuralforecast?color=blue)](https://pypi.org/project/neuralforecast/)
[![conda-nixtla](https://img.shields.io/conda/vn/conda-forge/neuralforecast?color=seagreen&label=conda)](https://anaconda.org/conda-forge/neuralforecast)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Nixtla/neuralforecast1/blob/main/LICENSE)
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
To get started follow this [guide](xXmissingXx), where we explore `NBEATS`, extend it towards probabilistic predictions and exogenous variables.

Or follow this simple example where we train the `NBEATS` model and predict the classic Box-Jenkins air passengers dataset.
```python
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast.utils import AirPassengersDF as Y_df
from neuralforecast.tsdataset import TimeSeriesDataset, TimeSeriesLoader

# Split data and declare panel dataset
Y_train_df = Y_df[Y_df.ds<='1959-12-31'] # 132 train
Y_test_df = Y_df[Y_df.ds>'1959-12-31']   # 12 test

# Fit and predict with N-BEATS model
nforecast = NeuralForecast(models=[NBEATS(input_size=24, h=12)])
y_hat  = nforecast.fit(df=Y_train_df).predict()

Y_test_df['N-BEATS'] = y_hat['N-BEATS'].values
pd.concat([Y_train_df, Y_test_df]).drop('unique_id', axis=1).set_index('ds').plot()
```
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast1/main/nbs/imgs_indx/nbeats_exaple.png">

## 🎉 New!
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](xXmissingXx) **Multi Quantile NBEATS Example**: Produce accurate and efficient probabilistic forecasts in long-horizon settings. Outperforming AutoARIMA's accuracy in a fraction of the time.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](xXmissingXx) **Long Horizon N-HiTS Example**:  Load, train, and tune hyperparameters, to achieve SoTA. Outperform Award Winning Transformers by 25% in 50x less time.

## 🔥  Highlights

* Unified `StatsForecast` interface `NeuralForecast().fit(Y_df).predict(h=7)`.
* Industry/Competition proven `ESRNN`, `NBEATS`, and `TFT` implementations.
* Improve accuracy and speed over classic `ARIMA`/`ETS` in two lines of code. Check the experiments [here](xXmissingXx).
* Predict Series with little to no history, using Transfer learning. Check the experiments [here](xXmissingXx).

## 🎊 Features 

* **Exogenous Variables**: Static, lagged and future exogenous support for models like `TFT`, `NBEATSx` and `ESRNN`.
* **Forecast Interpretability**: Plot trend, seasonality and exogenous `NBEATS`, `NHITS`, `ESRNN` prediction components.
* **Probabilistic Forecasting**: Simple model adapters for quantile losses and parametric distributions.
* **Train and Evaluation Losses** Scale-dependent, percentage and scale independent errors, and parametric likelihoods.
* **Automatic Model Selection** Parallelized automatic hyperparameter tuning, that efficiently searches best validation configuration.
* **Simple Interface** Unified SKLearn Interface for `StatsForecast` and `MLForecast` compatibility.
* **Model Collection**: Out of the box implementation of `MLP`, `LSTM`, `RNN`, `DilatedRNN`, `NBEATS`, `NHITS`, `ESRNN`, `AutoFormer`, `Informer`, `TFT`, `AutoFormer`, `Informer`, and vanilla `Transformer`. See the entire [collection here](https://nixtla.github.io/neuralforecast1/models.mlp.html).

Missing something? Please open an issue or write us in [![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)

## 📖 Why? 

There is a shared belief in Neural forecasting methods' capacity to improve our pipeline's accuracy and efficiency.

Unfortunately, available implementations and published research are yet to realize neural networks' potential. They are hard to use and continuously fail to improve over statistical methods while being computationally prohibitive. For this reason, we created `NeuralForecast`, a library favoring proven accurate and efficient models focusing on their usability.

## 🔬 Accuracy & ⏲ Speed 

### Industry/Competition Proven Methods
An extensive empirical evaluation is critical to generate confidence and promote the adoption and development of novel methods. For this reason, we replicate and verify the results of our implementation of the following industry/competition-proven methods: `ESRNN`, `NBEATS`, `NHITS`, and `TFT`. If you are interested in reproducing the results, check the experiments [here](xXmissingXx).

### Simple and Efficient Method's Comparison
Like `core.StatsForecast`, the `core.NeuralForecast` wrapper class allows us to easily compare any model in the collection to select or ensemble the best performing methods. Aditionally it offers a high-end interface that operates with (potentially large) sets of time series data stored in pandas DataFrames. The `core.NeuralForecast` efficiently parallelizes computation across CPU or GPU resources. Check the experiments [here](xXmissingXx).

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
    <td align="center"><a href="https://github.com/FedericoGarza"><img src="https://avatars.githubusercontent.com/u/10517170?v=4?s=100" width="100px;" alt=""/><br /><sub><b>fede</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/commits?author=FedericoGarza" title="Code">💻</a> <a href="#maintenance-FedericoGarza" title="Maintenance">🚧</a></td>
    <td align="center"><a href="https://github.com/jmoralez"><img src="https://avatars.githubusercontent.com/u/8473587?v=4?s=100" width="100px;" alt=""/><br /><sub><b>José Morales</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/commits?author=jmoralez" title="Code">💻</a> <a href="#maintenance-jmoralez" title="Maintenance">🚧</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/sugatoray/"><img src="https://avatars.githubusercontent.com/u/10201242?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Sugato Ray</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/commits?author=sugatoray" title="Code">💻</a></td>
    <td align="center"><a href="http://www.jefftackes.com"><img src="https://avatars.githubusercontent.com/u/9125316?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jeff Tackes</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/issues?q=author%3Atackes" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/darinkist"><img src="https://avatars.githubusercontent.com/u/62692170?v=4?s=100" width="100px;" alt=""/><br /><sub><b>darinkist</b></sub></a><br /><a href="#ideas-darinkist" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/alech97"><img src="https://avatars.githubusercontent.com/u/22159405?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Alec Helyar</b></sub></a><br /><a href="#question-alech97" title="Answering Questions">💬</a></td>
    <td align="center"><a href="https://dhirschfeld.github.io"><img src="https://avatars.githubusercontent.com/u/881019?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Dave Hirschfeld</b></sub></a><br /><a href="#question-dhirschfeld" title="Answering Questions">💬</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/mergenthaler"><img src="https://avatars.githubusercontent.com/u/4086186?v=4?s=100" width="100px;" alt=""/><br /><sub><b>mergenthaler</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/commits?author=mergenthaler" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/kdgutier"><img src="https://avatars.githubusercontent.com/u/19935241?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Kin</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/commits?author=kdgutier" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/Yasslight90"><img src="https://avatars.githubusercontent.com/u/58293883?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Yasslight90</b></sub></a><br /><a href="#ideas-Yasslight90" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/asinig"><img src="https://avatars.githubusercontent.com/u/99350687?v=4?s=100" width="100px;" alt=""/><br /><sub><b>asinig</b></sub></a><br /><a href="#ideas-asinig" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/guerda"><img src="https://avatars.githubusercontent.com/u/230782?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Philip Gillißen</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/commits?author=guerda" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/shagn"><img src="https://avatars.githubusercontent.com/u/16029092?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Sebastian Hagn</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/issues?q=author%3Ashagn" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/fugue-project/fugue"><img src="https://avatars.githubusercontent.com/u/21092479?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Han Wang</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/commits?author=goodwanghan" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://www.linkedin.com/in/benjamin-jeffrey-218548a8/"><img src="https://avatars.githubusercontent.com/u/36240394?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ben Jeffrey</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/issues?q=author%3Abjeffrey92" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/Beliavsky"><img src="https://avatars.githubusercontent.com/u/38887928?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Beliavsky</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/commits?author=Beliavsky" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/MMenchero"><img src="https://avatars.githubusercontent.com/u/47995617?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Mariana Menchero García </b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/commits?author=MMenchero" title="Code">💻</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/guptanick/"><img src="https://avatars.githubusercontent.com/u/33585645?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Nikhil Gupta</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/issues?q=author%3Angupta23" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/jdegene"><img src="https://avatars.githubusercontent.com/u/17744939?v=4?s=100" width="100px;" alt=""/><br /><sub><b>JD</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/issues?q=author%3Ajdegene" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/jattenberg"><img src="https://avatars.githubusercontent.com/u/924185?v=4?s=100" width="100px;" alt=""/><br /><sub><b>josh attenberg</b></sub></a><br /><a href="https://github.com/Nixtla/neuralforecast/commits?author=jattenberg" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
