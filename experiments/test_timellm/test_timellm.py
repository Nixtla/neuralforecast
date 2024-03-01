import os 

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

import time
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import TimeLLM
from neuralforecast.losses.pytorch import MAE
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengers, AirPassengersPanel, AirPassengersStatic, augment_calendar_df

from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

AirPassengersPanel, calendar_cols = augment_calendar_df(df=AirPassengersPanel, freq='M')

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
gpt2 = GPT2Model.from_pretrained('openai-community/gpt2',config=gpt2_config)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')

prompt_prefix = "The dataset contains data on monthly air passengers. There is a yearly seasonality"

timellm = TimeLLM(h=12,
                 input_size=36,
                 llm=gpt2,
                 llm_config=gpt2_config,
                 llm_tokenizer=gpt2_tokenizer,
                 prompt_prefix=prompt_prefix,
                 max_steps=100,
                 batch_size=24,
                 windows_batch_size=24)

nf = NeuralForecast(
    models=[timellm],
    freq='M'
)

start = time.time()

nf.fit(df=Y_train_df, val_size=12)
forecasts = nf.predict(futr_df=Y_test_df)

end = time.time()

print(f'It took {end-start} seconds to fit and predict with TimeLLM')

print(forecasts)

