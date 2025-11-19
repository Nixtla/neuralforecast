---
output-file: models.deepar.html
title: DeepAR
---

The DeepAR model produces probabilistic forecasts based on an
autoregressive recurrent neural network optimized on panel data using
cross-learning. DeepAR obtains its forecast distribution uses a Markov
Chain Monte Carlo sampler with the following conditional probability:
$$\mathbb{P}(\mathbf{y}_{[t+1:t+H]}|\;\mathbf{y}_{[:t]},\; \mathbf{x}^{(f)}_{[:t+H]},\; \mathbf{x}^{(s)})$$

where $\mathbf{x}^{(s)}$ are static exogenous inputs,
$\mathbf{x}^{(f)}_{[:t+H]}$ are future exogenous available at the time
of the prediction. The predictions are obtained by transforming the
hidden states $\mathbf{h}_{t}$ into predictive distribution parameters
$\theta_{t}$, and then generating samples $\mathbf{\hat{y}}_{[t+1:t+H]}$
through Monte Carlo sampling trajectories.

$$

\begin{align}
\mathbf{h}_{t} &= \textrm{RNN}([\mathbf{y}_{t},\mathbf{x}^{(f)}_{t+1},\mathbf{x}^{(s)}], \mathbf{h}_{t-1})\\
\mathbf{\theta}_{t}&=\textrm{Linear}(\mathbf{h}_{t}) \\
\hat{y}_{t+1}&=\textrm{sample}(\;\mathrm{P}(y_{t+1}\;|\;\mathbf{\theta}_{t})\;)
\end{align}

$$

**References**
 - [David Salinas, Valentin Flunkert, Jan Gasthaus,
Tim Januschowski (2020). “DeepAR: Probabilistic forecasting with
autoregressive recurrent networks”. International Journal of
Forecasting.](https://www.sciencedirect.com/science/article/pii/S0169207019301888)
 -
[Alexander Alexandrov et. al (2020). “GluonTS: Probabilistic and Neural
Time Series Modeling in Python”. Journal of Machine Learning
Research.](https://www.jmlr.org/papers/v21/19-820.html)


> **Exogenous Variables, Losses, and Parameters Availability**
>
> Given the sampling procedure during inference, DeepAR only supports
> [`DistributionLoss`](https://nixtlaverse.nixtla.io/neuralforecast/losses.pytorch.html#distributionloss)
> as training loss.
>
> Note that DeepAR generates a non-parametric forecast distribution
> using Monte Carlo. We use this sampling procedure also during
> validation to make it closer to the inference procedure. Therefore,
> only the
> [`MQLoss`](https://nixtlaverse.nixtla.io/neuralforecast/losses.pytorch.html#mqloss)
> is available for validation.
>
> Aditionally, Monte Carlo implies that historic exogenous variables are
> not available for the model.

![Figure 1. DeepAR model, during training the optimization signal comes from likelihood of observations, during inference a recurrent multi-step strategy is used to generate predictive distributions.](imgs_models/deepar.jpeg)
*Figure 1. DeepAR model, during training
the optimization signal comes from likelihood of observations, during
inference a recurrent multi-step strategy is used to generate predictive
distributions.*

## DeepAR

::: neuralforecast.models.deepar.DeepAR
    options:
      members:
        - fit
        - predict
      heading_level: 3

## Usage Example


```python
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import DeepAR
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

nf = NeuralForecast(
    models=[DeepAR(h=12,
                   input_size=24,
                   lstm_n_layers=1,
                   trajectory_samples=100,
                   loss=DistributionLoss(distribution='StudentT', level=[80, 90], return_params=True),
                   valid_loss=MQLoss(level=[80, 90]),
                   learning_rate=0.005,
                   stat_exog_list=['airline1'],
                   futr_exog_list=['trend'],
                   max_steps=100,
                   val_check_steps=10,
                   early_stop_patience_steps=-1,
                   scaler_type='standard',
                   enable_progress_bar=True,
                   ),
    ],
    freq='ME'
)
nf.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
Y_hat_df = nf.predict(futr_df=Y_test_df)

# Plot quantile predictions
Y_hat_df = Y_hat_df.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['DeepAR-median'], c='blue', label='median')
plt.fill_between(x=plot_df['ds'][-12:], 
                 y1=plot_df['DeepAR-lo-90'][-12:].values, 
                 y2=plot_df['DeepAR-hi-90'][-12:].values,
                 alpha=0.4, label='level 90')
plt.legend()
plt.grid()
plt.plot()
```
