# <center>Nixtla &nbsp; [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=SOTA%20Neural%20Forecasting%20Algorithms%20by%20Nixtla%204&url=https://github.com/Nixtla/neuralforecast&via=nixtlainc&hashtags=DeepLearning,TimeSeries,Forecasting) &nbsp;[![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)</center>



<div align="center">
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/indx_imgs/branding/logo_mid.png">
<h1 align="center">Neural 🧠 Forecast</h1>
<h3 align="center">Deep Learning for time series</h3>

[![CI Linux](https://github.com/Nixtla/neuralforecast/actions/workflows/ci-linux.yml/badge.svg?)](https://github.com/Nixtla/neuralforecast/actions/workflows/ci-linux.yml)
[![CI Mac](https://github.com/Nixtla/neuralforecast/actions/workflows/ci-mac.yml/badge.svg?)](https://github.com/Nixtla/neuralforecast/actions/workflows/ci-mac.yml)
[![codecov](https://codecov.io/gh/Nixtla/neuralforecast/branch/main/graph/badge.svg?token=C2P2BJI6S1)](https://codecov.io/gh/Nixtla/neuralforecast)
[![Python](https://img.shields.io/pypi/pyversions/neuralforecast)](https://pypi.org/project/neuralforecast/)
[![PyPi](https://img.shields.io/pypi/v/neuralforecast?color=blue)](https://pypi.org/project/neuralforecast/)
[![conda-nixtla](https://img.shields.io/conda/vn/conda-forge/neuralforecast?color=seagreen&label=conda)](https://anaconda.org/conda-forge/neuralforecast)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Nixtla/neuralforecast/blob/main/LICENSE)
[![docs](https://img.shields.io/website-up-down-green-red/http/nixtla.github.io/neuralforecast.svg?label=docs)](https://nixtla.github.io/neuralforecast/)

State-of-the-art time series forecasting for PyTorch. 

`NeuralForecast` is a Python library for time series forecasting with deep learning models. It includes *benchmark datasets*, *data-loading utilities*, *evaluation functions*, statistical *tests*, univariate model *benchmarks* and *SOTA models* implemented in PyTorch and PyTorchLightning. 



[Getting started](#%F0%9F%A7%AC%20Getting%20Started) •
[Installation](#💻-installation) •
[Models](#forecasting-models)
</div>

## 🎉 New! [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WjBbQzaivQhOldGolzymOtLmo6QX4Ieg#scrollTo=HXKT2-fpUD0Z)  
* **N-HiTS example**: load, train, and tune  hyperparameter, to achieve SoTA. Outperform Transformers by **25% in 50x** less time.




## ⚡ Why Deep Learning on Time Series?
**Accuracy**:
- Global model is fitted simultaneously for several time series.
- Shared information helps with highly parametrized and flexible models.
- Useful for items/skus that have little to no history available.

**Efficiency:**
 - Automatic featurization processes.
 - Fast computations (GPU or TPU).


## 📖 Documentation
Here is a link to the [documentation](https://nixtla.github.io/neuralforecast/).

## 🧬 Getting Started [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nixtla/neuralforecast/blob/main/examples/getting_started.ipynb)

[Example Jupyter Notebook](https://github.com/Nixtla/neuralforecast/blob/main/examples/getting_started.ipynb)

![demo](https://media.giphy.com/media/mCdyxYtLLTNCQ0WhkP/giphy.gif)

## 💻  Installation
<details>
<summary>PyPI</summary>

You can install the *released version* of `NeuralForecast` from the [Python package index](https://pypi.org) with:

```python
pip install neuralforecast
```

(Installing inside a python virtualenvironment or a conda environment is recommended.)
</details>

<details>
<summary>Conda</summary>
  
Also you can install the *released version* of `NeuralForecast` from [conda](https://anaconda.org) with:

```python
conda install -c conda-forge neuralforecast
```

(Installing inside a python virtualenvironment or a conda environment is recommended.)
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



##  Forecasting models

* [Neural Hierarchical Interpolation for Time Series Forecasting (N-HiTS)](https://arxiv.org/abs/2201.12886): A new model for long-horizon forecasting which incorporates novel hierarchical interpolation and multi-rate data sampling techniques to specialize blocks of its architecture to different frequency band of the time-series signal. It achieves SoTA performance on several benchmark datasets, outperforming current Transformer-based models by more than 25%. 

<p align="center">
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/indx_imgs/NHits.jpeg" width="300" title="N-HiTS" align="rigth">
</p>

* [Exponential Smoothing Recurrent Neural Network (ES-RNN)](https://www.sciencedirect.com/science/article/pii/S0169207019301153): A hybrid model that combines the expressivity of non linear models to capture the trends while it normalizes using a Holt-Winters inspired model for the levels and seasonals. This model is the winner of the M4 forecasting competition.

<p align="center">
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/indx_imgs/ESRNN.png" width="300" title="ES-RNN" align="rigth">
</p>

* [Neural Basis Expansion Analysis (N-BEATS)](https://arxiv.org/abs/1905.10437): A model from Element-AI (Yoshua Bengio’s lab) that has proven to achieve state-of-the-art performance on benchmark large scale forecasting datasets like Tourism, M3, and M4. The model is fast to train and has an interpretable configuration.

<p align="center">
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/indx_imgs/NBeats.png" width="300" title="N-BEATS" align="rigth">
</p>

* [Neural Basis Expansion Analysis with Exogenous Variables (N-BEATSx)](https://arxiv.org/abs/2104.05522): The neural basis expansion with exogenous variables is an extension to the original N-BEATS that allows it to include time dependent covariates.

<p align="center">
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/indx_imgs/NBEATSX.png" width="300" title="N-BEATSx" align="rigth">
</p>


* [Transformer-Based Models](https://arxiv.org/abs/1706.03762): Transformer-based framework for unsupervised representation learning of multivariate time series.
  - [Autoformer](https://arxiv.org/abs/2106.13008): Encoder-decoder model with decomposition capabilities and an approximation to attention based on Fourier transform.
  - [Informer](https://arxiv.org/abs/2012.07436): Transformer with MLP based multi-step prediction strategy, that approximates self-attention with sparsity.
  - [Transformer](): Classical vanilla Transformer.


## 📃 License
This project is licensed under the GPLv3 License - see the [LICENSE](https://github.com/Nixtla/neuralforecast/blob/main/LICENSE) file for details.

## 🔨 How to contribute

See [CONTRIBUTING.md](https://github.com/Nixtla/neuralforecast/blob/main/CONTRIBUTING.md).

<!---
## How to cite

If you use `NeuralForecast` in a scientific publication, we encourage you to add
the following references to the related papers:


```bibtex
@article{neuralforecast_arxiv,
  author  = {XXXX},
  title   = {{NeuralForecast: Deep Learning for Time Series Forecasting}},
  journal = {arXiv preprint arXiv:XXX.XXX},
  year    = {2022}
}
```
--->

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
