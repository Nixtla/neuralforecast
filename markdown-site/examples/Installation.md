---
title: Install
---

> Install NeuralForecast with pip or conda

You can install the *released version* of `NeuralForecast` from the
[Python package index](https://pypi.org) with:

``` python
pip install neuralforecast
```

or

``` python
conda install -c conda-forge neuralforecast
```

:::tip

Neural Forecasting methods profit from using GPU computation. Be sure to
have Cuda installed.

:::


:::warning

We are constantly updating neuralforecast, so we suggest fixing the
version to avoid issues. `pip install neuralforecast=="1.0.0"`

:::


:::tip

We recommend installing your libraries inside a python virtual or [conda
environment](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html).

:::

#### User our env (optional) {#user-our-env-optional}

If you don’t have a Conda environment and need tools like Numba, Pandas,
NumPy, Jupyter, Tune, and Nbdev you can use ours by following these
steps:

1.  Clone the NeuralForecast repo:

``` bash
$ git clone https://github.com/Nixtla/neuralforecast.git && cd neuralforecast
```

1.  Create the environment using the `environment.yml` file:

``` bash
$ conda env create -f environment.yml
```

1.  Activate the environment:

``` bash
$ conda activate neuralforecast
```

1.  Install NeuralForecast Dev

``` bash
$ pip install -e ".[dev]"
```

