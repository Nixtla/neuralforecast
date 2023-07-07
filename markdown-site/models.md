---
title: '<span style="color:DarkOrange"> Models </span>'
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

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
from os import cpu_count
import torch

from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator

from neuralforecast.common._base_auto import BaseAuto

from neuralforecast.models.rnn import RNN
from neuralforecast.models.gru import GRU
from neuralforecast.models.tcn import TCN
from neuralforecast.models.lstm import LSTM
from neuralforecast.models.dilated_rnn import DilatedRNN

from neuralforecast.models.mlp import MLP
from neuralforecast.models.nbeats import NBEATS
from neuralforecast.models.nbeatsx import NBEATSx
from neuralforecast.models.nhits import NHITS

from neuralforecast.models.tft import TFT
from neuralforecast.models.vanillatransformer import VanillaTransformer
from neuralforecast.models.informer import Informer
from neuralforecast.models.autoformer import Autoformer
from neuralforecast.models.fedformer import FEDformer
from neuralforecast.models.patchtst import PatchTST

from neuralforecast.models.stemgnn import StemGNN
from neuralforecast.models.hint import HINT

from neuralforecast.losses.pytorch import MAE
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
import matplotlib.pyplot as plt

from fastcore.test import test_eq
from nbdev.showdoc import show_doc

import logging
import warnings

from neuralforecast.losses.pytorch import MSE
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

plt.rcParams["axes.grid"]=True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["figure.figsize"] = (6,4)
```

</details>

:::

> NeuralForecast contains user-friendly implementations of neural
> forecasting models that allow for easy transition of computing
> capabilities (GPU/CPU), computation parallelization, and
> hyperparameter tuning.<br><br> All the NeuralForecast models are
> “global” because we train them with all the series from the input
> pd.DataFrame data `Y_df`, yet the optimization objective is,
> momentarily, “univariate” as it does not consider the interaction
> between the output predictions across time series. Like the
> StatsForecast library, `core.NeuralForecast` allows you to explore
> collections of models efficiently and contains functions for
> convenient wrangling of input and output pd.DataFrames predictions.

# <span style="color:DarkBlue"> 1. Automatic Forecasting </span> {#automatic-forecasting}

## <span style="color:DarkBlue"> A. RNN-Based </span> {#a.-rnn-based}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class AutoRNN(BaseAuto):
    
    default_config = {
        "input_size_multiplier": [-1, 4, 16, 64],
        "inference_input_size_multiplier": [-1],
        "h": None,
        "encoder_hidden_size": tune.choice([50, 100, 200, 300]),
        "encoder_n_layers": tune.randint(1, 4),
        "context_size": tune.choice([5, 10, 50]),
        "decoder_hidden_size": tune.choice([64, 128, 256, 512]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "max_steps": tune.choice([500, 1000]),
        "batch_size": tune.choice([16, 32]),
        "loss": None,
        "random_seed": tune.randint(1, 20)
    }

    def __init__(self,
                 h,
                 loss=MAE(),
                 valid_loss=None,
                 config=None, 
                 search_alg=BasicVariantGenerator(random_state=1),
                 num_samples=10,
                 refit_with_val=False,
                 cpus=cpu_count(),
                 gpus=torch.cuda.device_count(),
                 verbose=False):
        """ Auto RNN
        
        **Parameters:**<br>
        
        """
        # Define search space, input/output sizes
        if config is None:
            config = self.default_config.copy()        
            config['input_size'] = tune.choice([h*x \
                         for x in self.default_config['input_size_multiplier']])
            config['inference_input_size'] = tune.choice([h*x \
                         for x in self.default_config['inference_input_size_multiplier']])
            del config['input_size_multiplier'], config['inference_input_size_multiplier']

        super(AutoRNN, self).__init__(
              cls_model=RNN, 
              h=h,
              loss=loss,
              valid_loss=valid_loss,
              config=config, 
              search_alg=search_alg,
              num_samples=num_samples, 
              refit_with_val=refit_with_val,
              cpus=cpus,
              gpus=gpus,
              verbose=verbose
        )
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(AutoRNN, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengersDF as Y_df

# Split train/test and declare time series dataset
Y_train_df = Y_df[Y_df.ds<='1959-12-31'] # 132 train
Y_test_df = Y_df[Y_df.ds>'1959-12-31']   # 12 test
dataset, *_ = TimeSeriesDataset.from_df(Y_train_df)
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
# Use your own config or AutoRNN.default_config
config = dict(max_steps=2, val_check_steps=1, input_size=-1, encoder_hidden_size=8)
model = AutoRNN(h=12, config=config, num_samples=1, cpus=1)

model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class AutoLSTM(BaseAuto):

    default_config = {
        "input_size_multiplier": [-1, 4, 16, 64],
        "inference_input_size_multiplier": [-1],
        "h": None,
        "encoder_hidden_size": tune.choice([50, 100, 200, 300]),
        "encoder_n_layers": tune.randint(1, 4),
        "context_size": tune.choice([5, 10, 50]),
        "decoder_hidden_size": tune.choice([64, 128, 256, 512]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "max_steps": tune.choice([500, 1000]),
        "batch_size": tune.choice([16, 32]),
        "loss": None,
        "random_seed": tune.randint(1, 20)
    }

    def __init__(self,
                 h,
                 loss=MAE(),
                 valid_loss=None,
                 config=None,
                 search_alg=BasicVariantGenerator(random_state=1),
                 num_samples=10,
                 refit_with_val=False,
                 cpus=cpu_count(),
                 gpus=torch.cuda.device_count(),
                 verbose=False):

        # Define search space, input/output sizes
        if config is None:
            config = self.default_config.copy()        
            config['input_size'] = tune.choice([h*x \
                         for x in self.default_config['input_size_multiplier']])
            config['inference_input_size'] = tune.choice([h*x \
                         for x in self.default_config['inference_input_size_multiplier']])
            del config['input_size_multiplier'], config['inference_input_size_multiplier']

        super(AutoLSTM, self).__init__(
              cls_model=LSTM,
              h=h,
              loss=loss,
              valid_loss=valid_loss,
              config=config,
              search_alg=search_alg,
              num_samples=num_samples,
              refit_with_val=refit_with_val,
              cpus=cpus,
              gpus=gpus,
              verbose=verbose
        )
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(AutoLSTM, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
# Use your own config or AutoLSTM.default_config
config = dict(max_steps=2, val_check_steps=1, input_size=-1, encoder_hidden_size=8)
model = AutoLSTM(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)
```

</details>
<details>
<summary>Code</summary>

``` python
# %%capture
# Use your own config or AutoLSTM.default_config
config = dict(max_steps=2, val_check_steps=1, input_size=-1, encoder_hidden_size=8)
model = AutoLSTM(h=12, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class AutoGRU(BaseAuto):

    default_config = {
        "input_size_multiplier": [-1, 4, 16, 64],
        "inference_input_size_multiplier": [-1],
        "h": None,
        "encoder_hidden_size": tune.choice([50, 100, 200, 300]),
        "encoder_n_layers": tune.randint(1, 4),
        "context_size": tune.choice([5, 10, 50]),
        "decoder_hidden_size": tune.choice([64, 128, 256, 512]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "max_steps": tune.choice([500, 1000]),
        "batch_size": tune.choice([16, 32]),
        "loss": None,
        "random_seed": tune.randint(1, 20)
    }

    def __init__(self,
                 h,
                 loss=MAE(),
                 valid_loss=None,
                 config=None,
                 search_alg=BasicVariantGenerator(random_state=1),
                 num_samples=10,
                 refit_with_val=False,
                 cpus=cpu_count(),
                 gpus=torch.cuda.device_count(),
                 verbose=False,
                 alias=None):
        
        # Define search space, input/output sizes
        if config is None:
            config = self.default_config.copy()        
            config['input_size'] = tune.choice([h*x \
                         for x in self.default_config['input_size_multiplier']])
            config['inference_input_size'] = tune.choice([h*x \
                         for x in self.default_config['inference_input_size_multiplier']])
            del config['input_size_multiplier'], config['inference_input_size_multiplier']

        super(AutoGRU, self).__init__(
              cls_model=GRU,
              h=h,
              loss=loss,
              valid_loss=valid_loss,
              config=config, 
              search_alg=search_alg,
              num_samples=num_samples,
              refit_with_val=refit_with_val,
              cpus=cpus,
              gpus=gpus,
              verbose=verbose,
              alias=alias,
        )
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(AutoGRU, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
# Use your own config or AutoGRU.default_config
config = dict(max_steps=2, val_check_steps=1, input_size=-1, encoder_hidden_size=8)
model = AutoGRU(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class AutoTCN(BaseAuto):

    default_config = {
        "input_size_multiplier": [-1, 4, 16, 64],
        "inference_input_size_multiplier": [-1],
        "h": None,
        "encoder_hidden_size": tune.choice([50, 100, 200, 300]),
        "context_size": tune.choice([5, 10, 50]),
        "decoder_hidden_size": tune.choice([64, 128]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "max_steps": tune.choice([500, 1000]),
        "batch_size": tune.choice([16, 32]),
        "loss": None,
        "random_seed": tune.randint(1, 20)
    }

    def __init__(self,
                 h,
                 loss=MAE(),
                 valid_loss=None,
                 config=None,
                 search_alg=BasicVariantGenerator(random_state=1),
                 num_samples=10,
                 refit_with_val=False,
                 cpus=cpu_count(),
                 gpus=torch.cuda.device_count(),
                 verbose=False,
                 alias=None):
        
        # Define search space, input/output sizes
        if config is None:
            config = self.default_config.copy()        
            config['input_size'] = tune.choice([h*x \
                         for x in self.default_config['input_size_multiplier']])
            config['inference_input_size'] = tune.choice([h*x \
                         for x in self.default_config['inference_input_size_multiplier']])
            del config['input_size_multiplier'], config['inference_input_size_multiplier']

        super(AutoTCN, self).__init__(
              cls_model=TCN,
              h=h,
              loss=loss,
              valid_loss=valid_loss,
              config=config,
              search_alg=search_alg,
              num_samples=num_samples,
              refit_with_val=refit_with_val,
              cpus=cpus,
              gpus=gpus,
              verbose=verbose,
              alias=alias,
        )
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(AutoTCN, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
# Use your own config or AutoTCN.default_config
config = dict(max_steps=2, val_check_steps=1, input_size=-1, encoder_hidden_size=8)
model = AutoTCN(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class AutoDilatedRNN(BaseAuto):

    default_config = {
        "input_size_multiplier": [-1, 4, 16, 64],
        "inference_input_size_multiplier": [-1],
        "h": None,
        "cell_type": tune.choice(['LSTM', 'GRU']),
        "encoder_hidden_size": tune.choice([50, 100, 200, 300]),
        "dilations": tune.choice([ [[1, 2], [4, 8]], [[1, 2, 4, 8]] ]),
        "context_size": tune.choice([5, 10, 50]),
        "decoder_hidden_size": tune.choice([64, 128, 256, 512]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "max_steps": tune.choice([500, 1000]),
        "batch_size": tune.choice([16, 32]),
        "loss": None,
        "random_seed": tune.randint(1, 20)
    }

    def __init__(self,
                 h,
                 loss=MAE(),
                 valid_loss=None,
                 config=None,
                 search_alg=BasicVariantGenerator(random_state=1),
                 num_samples=10,
                 refit_with_val=False,
                 cpus=cpu_count(),
                 gpus=torch.cuda.device_count(),
                 verbose=False,
                 alias=None):
        
        # Define search space, input/output sizes
        if config is None:
            config = self.default_config.copy()        
            config['input_size'] = tune.choice([h*x \
                         for x in self.default_config['input_size_multiplier']])
            config['inference_input_size'] = tune.choice([h*x \
                         for x in self.default_config['inference_input_size_multiplier']])
            del config['input_size_multiplier'], config['inference_input_size_multiplier']

        super(AutoDilatedRNN, self).__init__(
              cls_model=DilatedRNN,
              h=h,
              loss=loss,
              valid_loss=valid_loss,
              config=config,
              search_alg=search_alg,
              num_samples=num_samples, 
              refit_with_val=refit_with_val,
              cpus=cpus,
              gpus=gpus,
              verbose=verbose,
              alias=alias,
         )
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(AutoDilatedRNN, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
# Use your own config or AutoDilatedRNN.default_config
config = dict(max_steps=2, val_check_steps=1, input_size=-1, encoder_hidden_size=8)
model = AutoDilatedRNN(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)
```

</details>

## <span style="color:DarkBlue"> B. MLP-Based </span> {#b.-mlp-based}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class AutoMLP(BaseAuto):

    default_config = {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "hidden_size": tune.choice( [256, 512, 1024] ),
        "num_layers": tune.randint(2, 6),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, 'robust', 'standard']),
        "max_steps": tune.choice([500, 1000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(1, 20),
    }

    def __init__(self,
                 h,
                 loss=MAE(),
                 valid_loss=None,     
                 config=None,
                 search_alg=BasicVariantGenerator(random_state=1),
                 num_samples=10,
                 refit_with_val=False,
                 cpus=cpu_count(),
                 gpus=torch.cuda.device_count(),
                 verbose=False,
                 alias=None):

        # Define search space, input/output sizes
        if config is None:
            config = self.default_config.copy()        
            config['input_size'] = tune.choice([h*x \
                         for x in self.default_config["input_size_multiplier"]])

            # Rolling windows with step_size=1 or step_size=h
            # See `BaseWindows` and `BaseRNN`'s create_windows
            config['step_size'] = tune.choice([1, h])
            del config["input_size_multiplier"]

        super(AutoMLP, self).__init__(
              cls_model=MLP,
              h=h,
              loss=loss,
              valid_loss=valid_loss,
              config=config, 
              search_alg=search_alg,
              num_samples=num_samples, 
              refit_with_val=refit_with_val,
              cpus=cpus,
              gpus=gpus,
              verbose=verbose,
              alias=alias,
        )
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(AutoMLP, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
# Use your own config or AutoMLP.default_config
config = dict(max_steps=2, val_check_steps=1, input_size=12, hidden_size=8)
model = AutoMLP(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class AutoNBEATS(BaseAuto):

    default_config = {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, 'robust', 'standard']),
        "max_steps": tune.choice([500, 1000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(1, 20),
    }

    def __init__(self,
                 h,
                 loss=MAE(),
                 valid_loss=None,
                 config=None, 
                 search_alg=BasicVariantGenerator(random_state=1),
                 num_samples=10,
                 refit_with_val=False,
                 cpus=cpu_count(),
                 gpus=torch.cuda.device_count(),
                 verbose=False,
                 alias=None):
        
        # Define search space, input/output sizes
        if config is None:
            config = self.default_config.copy()        
            config['input_size'] = tune.choice([h*x \
                         for x in self.default_config["input_size_multiplier"]])

            # Rolling windows with step_size=1 or step_size=h
            # See `BaseWindows` and `BaseRNN`'s create_windows
            config['step_size'] = tune.choice([1, h])
            del config["input_size_multiplier"]

        super(AutoNBEATS, self).__init__(
              cls_model=NBEATS, 
              h=h,
              loss=loss,
              valid_loss=valid_loss,
              config=config,
              search_alg=search_alg,
              num_samples=num_samples, 
              refit_with_val=refit_with_val,
              cpus=cpus,
              gpus=gpus,
              verbose=verbose,
              alias=alias,
        )
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(AutoNBEATS, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
# Use your own config or AutoNBEATS.default_config
config = dict(max_steps=2, val_check_steps=1, input_size=12,
              mlp_units=3*[[8, 8]])
model = AutoNBEATS(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class AutoNBEATSx(BaseAuto):

    default_config = {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, 'robust', 'standard']),
        "max_steps": tune.choice([500, 1000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(1, 20),
    }

    def __init__(self,
                 h,
                 loss=MAE(),
                 valid_loss=None,
                 config=None, 
                 search_alg=BasicVariantGenerator(random_state=1),
                 num_samples=10,
                 refit_with_val=False,
                 cpus=cpu_count(),
                 gpus=torch.cuda.device_count(),
                 verbose=False,
                 alias=None):
        
        # Define search space, input/output sizes
        if config is None:
            config = self.default_config.copy()        
            config['input_size'] = tune.choice([h*x \
                         for x in self.default_config["input_size_multiplier"]])

            # Rolling windows with step_size=1 or step_size=h
            # See `BaseWindows` and `BaseRNN`'s create_windows
            config['step_size'] = tune.choice([1, h])
            del config["input_size_multiplier"]

        super(AutoNBEATSx, self).__init__(
              cls_model=NBEATSx,
              h=h,
              loss=loss,
              valid_loss=valid_loss,
              config=config,
              search_alg=search_alg,
              num_samples=num_samples, 
              refit_with_val=refit_with_val,
              cpus=cpus,
              gpus=gpus,
              verbose=verbose,
              alias=alias,
        )
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(AutoNBEATSx, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
# Use your own config or AutoNBEATS.default_config
config = dict(max_steps=2, val_check_steps=1, input_size=12,
              mlp_units=3*[[8, 8]])
model = AutoNBEATSx(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class AutoNHITS(BaseAuto):

    default_config = {
       "input_size_multiplier": [1, 2, 3, 4, 5],
       "h": None,
       "n_pool_kernel_size": tune.choice([[2, 2, 1], 3*[1], 3*[2], 3*[4], 
                                         [8, 4, 1], [16, 8, 1]]),
       "n_freq_downsample": tune.choice([[168, 24, 1], [24, 12, 1], 
                                         [180, 60, 1], [60, 8, 1], 
                                         [40, 20, 1], [1, 1, 1]]),
       "learning_rate": tune.loguniform(1e-4, 1e-1),
       "scaler_type": tune.choice([None, 'robust', 'standard']),
       "max_steps": tune.quniform(lower=500, upper=1500, q=100),
       "batch_size": tune.qloguniform(lower=5, upper=9, base=2, q=1), #[32, 64, 128, 256]
       "windows_batch_size": tune.qloguniform(lower=7, upper=10, base=2, q=1), #[128, 256, 512, 1024]
       "loss": None,
       "random_seed": tune.randint(lower=1, upper=20),
    }

    def __init__(self,
                 h,
                 loss=MAE(),
                 valid_loss=None,
                 config=None, 
                 search_alg=BasicVariantGenerator(random_state=1),
                 num_samples=10,
                 refit_with_val=False,
                 cpus=cpu_count(),
                 gpus=torch.cuda.device_count(),
                 verbose=False,
                 alias=None):

        # Define search space, input/output sizes
        if config is None:
            config = self.default_config.copy()        
            config['input_size'] = tune.choice([h*x \
                         for x in self.default_config["input_size_multiplier"]])
            
            # Rolling windows with step_size=1 or step_size=h
            # See `BaseWindows` and `BaseRNN`'s create_windows
            config['step_size'] = tune.choice([1, h])
            del config["input_size_multiplier"]

        super(AutoNHITS, self).__init__(
              cls_model=NHITS, 
              h=h,
              loss=loss,
              valid_loss=valid_loss,
              config=config,
              search_alg=search_alg,
              num_samples=num_samples,
              refit_with_val=refit_with_val,
              cpus=cpus,
              gpus=gpus,
              verbose=verbose,
              alias=alias,
        )
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(AutoNHITS, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
# Use your own config or AutoNHITS.default_config
config = dict(max_steps=2, val_check_steps=1, input_size=12, 
              mlp_units=3 * [[8, 8]])
model = AutoNHITS(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)
```

</details>

## <span style="color:DarkBlue"> C. Transformer-Based </span> {#c.-transformer-based}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class AutoTFT(BaseAuto):

    default_config = {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "hidden_size": tune.choice([64, 128, 256]),
        "n_head": tune.choice([4, 8]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, 'robust', 'standard']),
        "max_steps": tune.choice([500, 1000, 2000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(1, 20),
    }

    def __init__(self,
                 h,
                 loss=MAE(),
                 valid_loss=None,
                 config=None, 
                 search_alg=BasicVariantGenerator(random_state=1),
                 num_samples=10,
                 refit_with_val=False,
                 cpus=cpu_count(),
                 gpus=torch.cuda.device_count(),
                 verbose=False,
                 alias=None):
        
        # Define search space, input/output sizes
        if config is None:
            config = self.default_config.copy()        
            config['input_size'] = tune.choice([h*x \
                         for x in self.default_config["input_size_multiplier"]])

            # Rolling windows with step_size=1 or step_size=h
            # See `BaseWindows` and `BaseRNN`'s create_windows
            config['step_size'] = tune.choice([1, h])
            del config["input_size_multiplier"]

        super(AutoTFT, self).__init__(
              cls_model=TFT, 
              h=h,
              loss=loss,
              valid_loss=valid_loss,
              config=config,
              search_alg=search_alg,
              num_samples=num_samples, 
              refit_with_val=refit_with_val,
              cpus=cpus,
              gpus=gpus,
              verbose=verbose,
              alias=alias,
        )
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(AutoTFT, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
# Use your own config or AutoNHITS.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, hidden_size=8)
model = AutoTFT(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class AutoVanillaTransformer(BaseAuto):

    default_config = {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "hidden_size": tune.choice([64, 128, 256]),
        "n_head": tune.choice([4, 8]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, 'robust', 'standard']),
        "max_steps": tune.choice([500, 1000, 2000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(1, 20),
    }

    def __init__(self,
                 h,
                 loss=MAE(),
                 valid_loss=None,
                 config=None, 
                 search_alg=BasicVariantGenerator(random_state=1),
                 num_samples=10,
                 refit_with_val=False,
                 cpus=cpu_count(),
                 gpus=torch.cuda.device_count(),
                 verbose=False,
                 alias=None):
        
        # Define search space, input/output sizes
        if config is None:
            config = self.default_config.copy()        
            config['input_size'] = tune.choice([h*x \
                         for x in self.default_config["input_size_multiplier"]])

            # Rolling windows with step_size=1 or step_size=h
            # See `BaseWindows` and `BaseRNN`'s create_windows
            config['step_size'] = tune.choice([1, h])
            del config["input_size_multiplier"]

        super(AutoVanillaTransformer, self).__init__(
              cls_model=VanillaTransformer, 
              h=h,
              loss=loss,
              valid_loss=valid_loss,
              config=config,
              search_alg=search_alg,
              num_samples=num_samples, 
              refit_with_val=refit_with_val,
              cpus=cpus,
              gpus=gpus,
              verbose=verbose,
              alias=alias,
        )
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(AutoVanillaTransformer, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
# Use your own config or AutoNHITS.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, hidden_size=8)
model = AutoVanillaTransformer(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class AutoInformer(BaseAuto):

    default_config = {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "hidden_size": tune.choice([64, 128, 256]),
        "n_head": tune.choice([4, 8]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, 'robust', 'standard']),
        "max_steps": tune.choice([500, 1000, 2000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(1, 20),
    }

    def __init__(self,
                 h,
                 loss=MAE(),
                 valid_loss=None,
                 config=None, 
                 search_alg=BasicVariantGenerator(random_state=1),
                 num_samples=10,
                 refit_with_val=False,
                 cpus=cpu_count(),
                 gpus=torch.cuda.device_count(),
                 verbose=False,
                 alias=None):
        
        # Define search space, input/output sizes
        if config is None:
            config = self.default_config.copy()        
            config['input_size'] = tune.choice([h*x \
                         for x in self.default_config["input_size_multiplier"]])

            # Rolling windows with step_size=1 or step_size=h
            # See `BaseWindows` and `BaseRNN`'s create_windows
            config['step_size'] = tune.choice([1, h])
            del config["input_size_multiplier"]

        super(AutoInformer, self).__init__(
              cls_model=Informer, 
              h=h,
              loss=loss,
              valid_loss=valid_loss,
              config=config,
              search_alg=search_alg,
              num_samples=num_samples, 
              refit_with_val=refit_with_val,
              cpus=cpus,
              gpus=gpus,
              verbose=verbose,
              alias=alias,
        )
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(AutoInformer, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
# Use your own config or AutoNHITS.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, hidden_size=8)
model = AutoInformer(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class AutoAutoformer(BaseAuto):

    default_config = {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "hidden_size": tune.choice([64, 128, 256]),
        "n_head": tune.choice([4, 8]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, 'robust', 'standard']),
        "max_steps": tune.choice([500, 1000, 2000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(1, 20),
    }

    def __init__(self,
                 h,
                 loss=MAE(),
                 valid_loss=None,
                 config=None, 
                 search_alg=BasicVariantGenerator(random_state=1),
                 num_samples=10,
                 refit_with_val=False,
                 cpus=cpu_count(),
                 gpus=torch.cuda.device_count(),
                 verbose=False,
                 alias=None):
        
        # Define search space, input/output sizes
        if config is None:
            config = self.default_config.copy()        
            config['input_size'] = tune.choice([h*x \
                         for x in self.default_config["input_size_multiplier"]])

            # Rolling windows with step_size=1 or step_size=h
            # See `BaseWindows` and `BaseRNN`'s create_windows
            config['step_size'] = tune.choice([1, h])
            del config["input_size_multiplier"]

        super(AutoAutoformer, self).__init__(
              cls_model=Autoformer, 
              h=h,
              loss=loss,
              valid_loss=valid_loss,
              config=config,
              search_alg=search_alg,
              num_samples=num_samples, 
              refit_with_val=refit_with_val,
              cpus=cpus,
              gpus=gpus,
              verbose=verbose,
              alias=alias,
        )
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(AutoAutoformer, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
# Use your own config or AutoNHITS.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, hidden_size=8)
model = AutoAutoformer(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class AutoFEDformer(BaseAuto):

    default_config = {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "hidden_size": tune.choice([64, 128, 256]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, 'robust', 'standard']),
        "max_steps": tune.choice([500, 1000, 2000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(1, 20),
    }

    def __init__(self,
                 h,
                 loss=MAE(),
                 valid_loss=None,
                 config=None, 
                 search_alg=BasicVariantGenerator(random_state=1),
                 num_samples=10,
                 refit_with_val=False,
                 cpus=cpu_count(),
                 gpus=torch.cuda.device_count(),
                 verbose=False,
                 alias=None):
        
        # Define search space, input/output sizes
        if config is None:
            config = self.default_config.copy()        
            config['input_size'] = tune.choice([h*x \
                         for x in self.default_config["input_size_multiplier"]])

            # Rolling windows with step_size=1 or step_size=h
            # See `BaseWindows` and `BaseRNN`'s create_windows
            config['step_size'] = tune.choice([1, h])
            del config["input_size_multiplier"]

        super(AutoFEDformer, self).__init__(
              cls_model=FEDformer, 
              h=h,
              loss=loss,
              valid_loss=valid_loss,
              config=config,
              search_alg=search_alg,
              num_samples=num_samples, 
              refit_with_val=refit_with_val,
              cpus=cpus,
              gpus=gpus,
              verbose=verbose,
              alias=alias,
        )
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(AutoFEDformer, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
# Use your own config or AutoNHITS.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, hidden_size=64)
model = AutoFEDformer(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class AutoPatchTST(BaseAuto):

    default_config = {
        "input_size_multiplier": [1, 2, 3],
        "h": None,
        "hidden_size": tune.choice([16, 128, 256]),
        "n_head": tune.choice([4, 16]),
        "patch_len": tune.choice([16, 24]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, 'robust', 'standard']),
        "revin": tune.choice([False, True]),
        "max_steps": tune.choice([500, 1000, 5000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(1, 20),
    }

    def __init__(self,
                 h,
                 loss=MAE(),
                 valid_loss=None,
                 config=None, 
                 search_alg=BasicVariantGenerator(random_state=1),
                 num_samples=10,
                 refit_with_val=False,
                 cpus=cpu_count(),
                 gpus=torch.cuda.device_count(),
                 verbose=False,
                 alias=None):
        
        # Define search space, input/output sizes
        if config is None:
            config = self.default_config.copy()        
            config['input_size'] = tune.choice([h*x \
                         for x in self.default_config["input_size_multiplier"]])

            # Rolling windows with step_size=1 or step_size=h
            # See `BaseWindows` and `BaseRNN`'s create_windows
            config['step_size'] = tune.choice([1, h])
            del config["input_size_multiplier"]

        super(AutoPatchTST, self).__init__(
              cls_model=PatchTST, 
              h=h,
              loss=loss,
              valid_loss=valid_loss,
              config=config,
              search_alg=search_alg,
              num_samples=num_samples, 
              refit_with_val=refit_with_val,
              cpus=cpus,
              gpus=gpus,
              verbose=verbose,
              alias=alias,
        )
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(AutoPatchTST, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
# Use your own config or AutoNHITS.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12, hidden_size=16)
model = AutoPatchTST(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# Y_plot_df = Y_test_df[['unique_id', 'ds', 'y']].copy()
# Y_plot_df['AutoPatchTST'] = y_hat

# pd.concat([Y_train_df, Y_plot_df]).drop('unique_id', axis=1).set_index('ds').plot()
```

</details>

:::

## <span style="color:DarkBlue"> D. Multivariate </span> {#d.-multivariate}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class AutoStemGNN(BaseAuto):

    default_config = {
        "input_size_multiplier": [1, 2, 3, 4],
        "h": None,
        "n_series": None,
        "n_stacks": tune.choice([2, 3]),
        "multi_layer": tune.choice([3, 5, 7]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, 'robust', 'standard']),
        "max_steps": tune.choice([500, 1000, 2000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "loss": None,
        "random_seed": tune.randint(1, 20),
    }

    def __init__(self,
                 h,
                 n_series,
                 loss=MAE(),
                 valid_loss=None,
                 config=None, 
                 search_alg=BasicVariantGenerator(random_state=1),
                 num_samples=10,
                 refit_with_val=False,
                 cpus=cpu_count(),
                 gpus=torch.cuda.device_count(),
                 verbose=False,
                 alias=None):
        
        # Define search space, input/output sizes
        if config is None:
            config = self.default_config.copy()        
            config['input_size'] = tune.choice([h*x \
                         for x in self.default_config["input_size_multiplier"]])

            # Rolling windows with step_size=1 or step_size=h
            # See `BaseWindows` and `BaseRNN`'s create_windows
            config['step_size'] = tune.choice([1, h])
            del config["input_size_multiplier"]

        # Always use n_series from parameters
        config['n_series'] = n_series

        super(AutoStemGNN, self).__init__(
              cls_model=StemGNN, 
              h=h,
              loss=loss,
              valid_loss=valid_loss,
              config=config,
              search_alg=search_alg,
              num_samples=num_samples, 
              refit_with_val=refit_with_val,
              cpus=cpus,
              gpus=gpus,
              verbose=verbose,
              alias=alias,
        )
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(AutoStemGNN, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
# Use your own config or AutoNHITS.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=12)
model = AutoStemGNN(h=12, n_series=1, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# Y_plot_df = Y_test_df[['unique_id', 'ds', 'y']].copy()
# Y_plot_df['AutoStemGNN'] = y_hat

# pd.concat([Y_train_df, Y_plot_df]).drop('unique_id', axis=1).set_index('ds').plot()
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class AutoHINT(BaseAuto):

    def __init__(self,
                 cls_model,
                 h,
                 loss,
                 valid_loss,
                 S,
                 config,
                 search_alg=BasicVariantGenerator(random_state=1),
                 num_samples=10,
                 cpus=cpu_count(),
                 gpus=torch.cuda.device_count(),
                 refit_with_val=False,
                 verbose=False,
                 alias=None):
        super(AutoHINT, self).__init__(
              cls_model=cls_model, 
              h=h,
              loss=loss,
              valid_loss=valid_loss,
              config=config,
              search_alg=search_alg,
              num_samples=num_samples, 
              refit_with_val=refit_with_val,
              cpus=cpus,
              gpus=gpus,
              verbose=verbose,
              alias=alias,
        )
        # Validate presence of reconciliation strategy
        # parameter in configuration space
        if not ('reconciliation' in config.keys()):
            raise Exception("config needs reconciliation, \
                            try tune.choice(['BottomUp', 'MinTraceOLS', 'MinTraceWLS'])")
        self.S = S

    def _fit_model(self, cls_model, config,
                   dataset, val_size, test_size):
        # Overwrite _fit_model for HINT two-stage instantiation
        reconciliation = config.pop('reconciliation')
        base_model = cls_model(**config)
        model = HINT(h=base_model.h, model=base_model, 
                     S=self.S, reconciliation=reconciliation)
        model.test_size = test_size
        model.fit(
            dataset,
            val_size=val_size, 
            test_size=test_size
        )
        return model
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(AutoHINT, title_level=3)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
def sort_df_hier(Y_df, S_df):
    # NeuralForecast core, sorts unique_id lexicographically
    # by default, this class matches S_df and Y_hat_df order.    
    Y_df.unique_id = Y_df.unique_id.astype('category')
    Y_df.unique_id = Y_df.unique_id.cat.set_categories(S_df.index)
    Y_df = Y_df.sort_values(by=['unique_id', 'ds'])
    return Y_df

# -----Create synthetic dataset-----
np.random.seed(123)
train_steps = 20
num_levels = 7
level = np.arange(0, 100, 0.1)
qs = [[50-lv/2, 50+lv/2] for lv in level]
quantiles = np.sort(np.concatenate(qs)/100)

levels = ['Top', 'Mid1', 'Mid2', 'Bottom1', 'Bottom2', 'Bottom3', 'Bottom4']
unique_ids = np.repeat(levels, train_steps)

S = np.array([[1., 1., 1., 1.],
              [1., 1., 0., 0.],
              [0., 0., 1., 1.],
              [1., 0., 0., 0.],
              [0., 1., 0., 0.],
              [0., 0., 1., 0.],
              [0., 0., 0., 1.]])

S_dict = {col: S[:, i] for i, col in enumerate(levels[3:])}
S_df = pd.DataFrame(S_dict, index=levels)

ds = pd.date_range(start='2018-03-31', periods=train_steps, freq='Q').tolist() * num_levels
# Create Y_df
y_lists = [S @ np.random.uniform(low=100, high=500, size=4) for i in range(train_steps)]
y = [elem for tup in zip(*y_lists) for elem in tup]
Y_df = pd.DataFrame({'unique_id': unique_ids, 'ds': ds, 'y': y})
Y_df = sort_df_hier(Y_df, S_df)

hint_dataset, *_ = TimeSeriesDataset.from_df(df=Y_df)
```

</details>

:::

<details>
<summary>Code</summary>

``` python
%%capture
# Perform a simple hyperparameter optimization with 
# NHITS and then reconcile with HINT
from neuralforecast.losses.pytorch import GMM, sCRPS

base_config = dict(max_steps=1, val_check_steps=1, input_size=8)
base_model = AutoNHITS(h=4, loss=GMM(n_components=2, quantiles=quantiles), 
                       config=base_config, num_samples=1, cpus=1)
model = HINT(h=4, S=S_df.values,
             model=base_model,  reconciliation='MinTraceOLS')

model.fit(dataset=dataset)
y_hat = model.predict(dataset=hint_dataset)

# Perform a conjunct hyperparameter optimization with 
# NHITS + HINT reconciliation configurations
nhits_config = {
       "learning_rate": tune.choice([1e-3]),                                     # Initial Learning rate
       "max_steps": tune.choice([1]),                                            # Number of SGD steps
       "val_check_steps": tune.choice([1]),                                      # Number of steps between validation
       "input_size": tune.choice([5 * 12]),                                      # input_size = multiplier * horizon
       "batch_size": tune.choice([7]),                                           # Number of series in windows
       "windows_batch_size": tune.choice([256]),                                 # Number of windows in batch
       "n_pool_kernel_size": tune.choice([[2, 2, 2], [16, 8, 1]]),               # MaxPool's Kernelsize
       "n_freq_downsample": tune.choice([[168, 24, 1], [24, 12, 1], [1, 1, 1]]), # Interpolation expressivity ratios
       "activation": tune.choice(['ReLU']),                                      # Type of non-linear activation
       "n_blocks":  tune.choice([[1, 1, 1]]),                                    # Blocks per each 3 stacks
       "mlp_units":  tune.choice([[[512, 512], [512, 512], [512, 512]]]),        # 2 512-Layers per block for each stack
       "interpolation_mode": tune.choice(['linear']),                            # Type of multi-step interpolation
       "random_seed": tune.randint(1, 10),
       "reconciliation": tune.choice(['BottomUp', 'MinTraceOLS', 'MinTraceWLS'])
    }
model = AutoHINT(h=4, S=S_df.values,
                 cls_model=NHITS,
                 config=nhits_config,
                 loss=GMM(n_components=2, level=[80, 90]),
                 valid_loss=sCRPS(level=[80, 90]),
                 num_samples=1, cpus=1)
model.fit(dataset=dataset)
y_hat = model.predict(dataset=hint_dataset)
```

</details>

# TESTS {#tests}

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengersDF as Y_df

# Split train/test and declare time series dataset
Y_train_df = Y_df[Y_df.ds<='1959-12-31'] # 132 train
Y_test_df = Y_df[Y_df.ds>'1959-12-31']   # 12 test
dataset, *_ = TimeSeriesDataset.from_df(Y_train_df)

config = dict(max_steps=1, val_check_steps=1, input_size=12)
model = AutoNHITS(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
## TESTS
nhits_config = {
       "learning_rate": tune.choice([1e-3]),                                     # Initial Learning rate
       "max_steps": tune.choice([1]),                                            # Number of SGD steps
       "val_check_steps": tune.choice([1]),                                      # Number of steps between validation
       "input_size": tune.choice([5 * 12]),                                      # input_size = multiplier * horizon
       "batch_size": tune.choice([7]),                                           # Number of series in windows
       "windows_batch_size": tune.choice([256]),                                 # Number of windows in batch
       "n_pool_kernel_size": tune.choice([[2, 2, 2], [16, 8, 1]]),               # MaxPool's Kernelsize
       "n_freq_downsample": tune.choice([[168, 24, 1], [24, 12, 1], [1, 1, 1]]), # Interpolation expressivity ratios
       "activation": tune.choice(['ReLU']),                                      # Type of non-linear activation
       "n_blocks":  tune.choice([[1, 1, 1]]),                                    # Blocks per each 3 stacks
       "mlp_units":  tune.choice([[[512, 512], [512, 512], [512, 512]]]),        # 2 512-Layers per block for each stack
       "interpolation_mode": tune.choice(['linear']),                            # Type of multi-step interpolation
       "random_seed": tune.randint(1, 10),
    }

model = AutoNHITS(h=12, loss=MAE(), valid_loss=MSE(), config=nhits_config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

# Test equality
test_eq(str(type(model.valid_loss)), "<class 'neuralforecast.losses.pytorch.MSE'>")
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
## TODO: Add unit tests for interactions between loss/valid_loss types
## TODO: Unit tests (2 types of networks x 2 types of loss x 2 types of valid loss)
from neuralforecast.losses.pytorch import GMM, sCRPS

## Checking if base recurrent methods run point valid_loss correctly
tcn_config = {
       "learning_rate": tune.choice([1e-3]),                                     # Initial Learning rate
       "max_steps": tune.choice([1]),                                            # Number of SGD steps
       "val_check_steps": tune.choice([1]),                                      # Number of steps between validation
       "input_size": tune.choice([5 * 12]),                                      # input_size = multiplier * horizon
       "batch_size": tune.choice([7]),                                           # Number of series in windows
       "random_seed": tune.randint(1, 10),
    }

model = AutoTCN(h=12, 
                loss=MAE(), 
                valid_loss=MSE(), 
                config=tcn_config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)

## Checking if base recurrent methods run quantile valid_loss correctly
model = AutoTCN(h=12, 
                loss=GMM(n_components=2, level=[80, 90]),
                valid_loss=sCRPS(level=[80, 90]),
                config=tcn_config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
y_hat = model.predict(dataset=dataset)
```

</details>

:::

