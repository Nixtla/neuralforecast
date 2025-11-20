---
output-file: models.timellm.html
title: Time-LLM
---

Time-LLM is a reprogramming framework to repurpose LLMs for general time
series forecasting with the backbone language models kept intact. In
other words, it transforms a forecasting task into a “language task”
that can be tackled by an off-the-shelf LLM.

**References**

- [Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu,
James Y. Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li,
Shirui Pan, Qingsong Wen. “Time-LLM: Time Series Forecasting by
Reprogramming Large Language
Models”](https://arxiv.org/abs/2310.01728)


![Figure 1. Time-LLM Architecture.](imgs_models/timellm.png)
*Figure 1. Time-LLM Architecture.*

## 1. Time-LLM

::: neuralforecast.models.timellm.Time-LLM
    options:
      members:
        - fit
        - predict
      heading_level: 3

### Usage example

```python 
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import TimeLLM
from neuralforecast.utils import AirPassengersPanel

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

prompt_prefix = "The dataset contains data on monthly air passengers. There is a yearly seasonality"

timellm = TimeLLM(h=12,
                 input_size=36,
                 llm='openai-community/gpt2',
                 prompt_prefix=prompt_prefix,
                 batch_size=16,
                 valid_batch_size=16,
                 windows_batch_size=16)

nf = NeuralForecast(
    models=[timellm],
    freq='ME'
)

nf.fit(df=Y_train_df, val_size=12)
forecasts = nf.predict(futr_df=Y_test_df)
```

## 2. Auxiliary Functions

::: neuralforecast.models.timellm.ReprogrammingLayer
    options:
      members: []

::: neuralforecast.models.timellm.FlattenHead
    options:
      members: []

::: neuralforecast.models.timellm.PatchEmbedding
    options:
      members: []

::: neuralforecast.models.timellm.TokenEmbedding
    options:
      members: []

::: neuralforecast.models.timellm.ReplicationPad1d
    options:
      members: []
