---
output-file: models.hint.html
title: HINT
---

The Hierarchical Mixture Networks (HINT) are a highly modular framework
that combines SoTA neural forecast architectures with task-specialized
mixture probability and advanced hierarchical reconciliation strategies.
This powerful combination allows HINT to produce accurate and coherent
probabilistic forecasts.

HINT’s incorporates a `TemporalNorm` module into any neural forecast
architecture, the module normalizes inputs into the network’s
non-linearities operating range and recomposes its output’s scales
through a global skip connection, improving accuracy and training
robustness. HINT ensures the forecast coherence via bootstrap sample
reconciliation that restores the aggregation constraints into its base
samples.

**References**
 - [Kin G. Olivares, David Luo, Cristian Challu,
Stefania La Vattiata, Max Mergenthaler, Artur Dubrawski (2023). “HINT:
Hierarchical Mixture Networks For Coherent Probabilistic Forecasting”.
Neural Information Processing Systems, submitted. Working Paper version
available at arxiv.](https://arxiv.org/abs/2305.07089)
 - [Kin G.
Olivares, O. Nganba Meetei, Ruijun Ma, Rohan Reddy, Mengfei Cao, Lee
Dicker (2022).”Probabilistic Hierarchical Forecasting with Deep Poisson
Mixtures”. International Journal Forecasting, accepted paper available
at arxiv.](https://arxiv.org/pdf/2110.13179.pdf)
 - [Kin G. Olivares,
Federico Garza, David Luo, Cristian Challu, Max Mergenthaler, Souhaib
Ben Taieb, Shanika Wickramasuriya, and Artur Dubrawski (2022).
“HierarchicalForecast: A reference framework for hierarchical
forecasting in python”. Journal of Machine Learning Research, submitted,
abs/2207.03517, 2022b.](https://arxiv.org/abs/2207.03517)

![Figure 1. Hierarchical Mixture Networks (HINT).](imgs_models/hint.png)
*Figure 1. Hierarchical Mixture Networks
(HINT).*

## HINT

::: neuralforecast.models.hint.HINT
    options:
      members:
        - fit
        - predict
      heading_level: 3

## Usage Example

In this example we will use HINT for the hierarchical forecast task, a
multivariate regression problem with aggregation constraints. The
aggregation constraints can be compactcly represented by the summing
matrix $\mathbf{S}_{[i][b]}$, the Figure belows shows an example.

In this example we will make coherent predictions for the TourismL
dataset.

Outline<br/> 1. Import packages<br/> 2. Load hierarchical dataset<br/> 3.
Fit and Predict HINT<br/> 4. Forecast Plot

![](/neuralforecast/imgs_models/hint_notation.png)


```python
import matplotlib.pyplot as plt

from neuralforecast.losses.pytorch import GMM, sCRPS
from datasetsforecast.hierarchical import HierarchicalData

# Auxiliary sorting
def sort_df_hier(Y_df, S_df):
    # NeuralForecast core, sorts unique_id lexicographically
    # by default, this class matches S_df and Y_hat_df order.    
    Y_df.unique_id = Y_df.unique_id.astype('category')
    Y_df.unique_id = Y_df.unique_id.cat.set_categories(S_df.index)
    Y_df = Y_df.sort_values(by=['unique_id', 'ds'])
    return Y_df

# Load TourismSmall dataset
horizon = 12
Y_df, S_df, tags = HierarchicalData.load('./data', 'TourismLarge')
Y_df['ds'] = pd.to_datetime(Y_df['ds'])
Y_df = sort_df_hier(Y_df, S_df)
level = [80,90]

# Instantiate HINT
# BaseNetwork + Distribution + Reconciliation
nhits = NHITS(h=horizon,
              input_size=24,
              loss=GMM(n_components=10, level=level),
              max_steps=2000,
              early_stop_patience_steps=10,
              val_check_steps=50,
              scaler_type='robust',
              learning_rate=1e-3,
              valid_loss=sCRPS(level=level))

model = HINT(h=horizon, S=S_df.values,
             model=nhits,  reconciliation='BottomUp')

# Fit and Predict
nf = NeuralForecast(models=[model], freq='MS')
Y_hat_df = nf.cross_validation(df=Y_df, val_size=12, n_windows=1)
Y_hat_df = Y_hat_df.reset_index()
```


```python
# Plot coherent probabilistic forecast
unique_id = 'TotalAll'
Y_plot_df = Y_df[Y_df.unique_id==unique_id]
plot_df = Y_hat_df[Y_hat_df.unique_id==unique_id]
plot_df = Y_plot_df.merge(plot_df, on=['ds', 'unique_id'], how='left')
n_years = 5

plt.plot(plot_df['ds'][-12*n_years:], plot_df['y_x'][-12*n_years:], c='black', label='True')
plt.plot(plot_df['ds'][-12*n_years:], plot_df['HINT'][-12*n_years:], c='purple', label='mean')
plt.plot(plot_df['ds'][-12*n_years:], plot_df['HINT-median'][-12*n_years:], c='blue', label='median')
plt.fill_between(x=plot_df['ds'][-12*n_years:],
                 y1=plot_df['HINT-lo-90'][-12*n_years:].values,
                 y2=plot_df['HINT-hi-90'][-12*n_years:].values,
                 alpha=0.4, label='level 90')
plt.legend()
plt.grid()
plt.plot()
```
