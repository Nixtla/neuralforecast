---
description: >-
  One of the simplest neural architectures are Multi Layer Perceptrons (`MLP`)
  composed of stacked Fully Connected Neural Networks trained with
  backpropagation. Each node in the architecture is capable of modeling
  non-linear relationships granted by their activation functions. Novel
  activations like Rectified Linear Units (`ReLU`) have greatly improved the
  ability to fit deeper networks overcoming gradient vanishing problems that
  were associated with `Sigmoid` and `TanH` activations. For the forecasting
  task the last layer is changed to follow a auto-regression problem. This
  version is multivariate, indicating that it will predict all time series of
  the forecasting problem jointly. <br/><br/>**References**<br/>-[Rosenblatt, F.
  (1958). "The perceptron: A probabilistic model for information storage and
  organization in the
  brain."](https://psycnet.apa.org/record/1959-09865-001)<br/>-[Fukushima, K.
  (1975). "Cognitron: A self-organizing multilayered neural
  network."](https://pascal-francis.inist.fr/vibad/index.php?action=getRecordDetail&idt=PASCAL7750396723)<br/>-[Vinod
  Nair, Geoffrey E. Hinton (2010). "Rectified Linear Units Improve Restricted
  Boltzmann Machines"](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)<br/>
output-file: models.mlpmultivariate.html
title: MLPMultivariate
---

![Figure 1. Three layer MLP with autorregresive inputs.](imgs_models/mlp.png)
*Figure 1. Three layer MLP with autorregresive inputs.*

## MLPMultivariate

::: neuralforecast.models.mlpmultivariate.MLPMultivariate
    options:
      members:
        - fit
        - predict
      heading_level: 3

### Usage Example


```python
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import MLPMultivariate
from neuralforecast.losses.pytorch import MAE
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = MLPMultivariate(h=12, 
            input_size=24,
            n_series=2,
            stat_exog_list=['airline1'],
            futr_exog_list=['trend'],            
            loss = MAE(),
            scaler_type='robust',
            learning_rate=1e-3,
            stat_exog_list=['airline1'],
            max_steps=200,
            val_check_steps=10,
            early_stop_patience_steps=2)

fcst = NeuralForecast(
    models=[model],
    freq='ME'
)
fcst.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
forecasts = fcst.predict(futr_df=Y_test_df)

# Plot predictions
Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['MLPMultivariate'], c='blue', label='median')
plt.grid()
plt.legend()
plt.plot()
```
