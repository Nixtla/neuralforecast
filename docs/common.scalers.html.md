---
description: >-
  Temporal normalization has proven to be essential in neural forecasting tasks,
  as it enables network's non-linearities to express themselves. Forecasting
  scaling methods take particular interest in the temporal dimension where most
  of the variance dwells, contrary to other deep learning techniques like
  `BatchNorm` that normalizes across batch and temporal dimensions, and
  `LayerNorm` that normalizes across the feature dimension. Currently we support
  the following techniques: `std`, `median`, `norm`, `norm1`, `invariant`,
  `revin`.
output-file: common.scalers.html
title: TemporalNorm
---

## References

-   [Kin G. Olivares, David Luo, Cristian Challu, Stefania La Vattiata,
    Max Mergenthaler, Artur Dubrawski (2023). "HINT: Hierarchical
    Mixture Networks For Coherent Probabilistic Forecasting". Neural
    Information Processing Systems, submitted. Working Paper version
    available at arxiv.](https://arxiv.org/abs/2305.07089)
-   [Taesung Kim and Jinhee Kim and Yunwon Tae and Cheonbok Park and
    Jang-Ho Choi and Jaegul Choo. "Reversible Instance Normalization for
    Accurate Time-Series Forecasting against Distribution Shift". ICLR
    2022.](https://openreview.net/pdf?id=cGDAkQo1C0p)
-   [David Salinas, Valentin Flunkert, Jan Gasthaus, Tim Januschowski
    (2020). "DeepAR: Probabilistic forecasting with autoregressive
    recurrent networks". International Journal of
    Forecasting.](https://www.sciencedirect.com/science/article/pii/S0169207019301888)


![](imgs_models/temporal_norm.png)
*Figure 1. Illustration of temporal normalization (left), layer normalization (center) and batch normalization (right). The entries in green show the components used to compute the normalizing statistics.*

## 1. Auxiliary Functions

::: neuralforecast.common._scalers.masked_median
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._scalers.masked_mean
    options:
      members: []
      heading_level: 3

## 2. Scalers

::: neuralforecast.common._scalers.minmax_statistics
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._scalers.minmax1_statistics
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._scalers.std_statistics
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._scalers.robust_statistics
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._scalers.invariant_statistics
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._scalers.identity_statistics
    options:
      members: []
      heading_level: 3

## 3. TemporalNorm Module

::: neuralforecast.common._scalers.TemporalNorm
    options:
      members:
        - transform
        - inverse_transform
      heading_level: 3

## Example


```python
import numpy as np
```


```python
# Declare synthetic batch to normalize
x1 = 10**0 * np.arange(36)[:, None]
x2 = 10**1 * np.arange(36)[:, None]

np_x = np.concatenate([x1, x2], axis=1)
np_x = np.repeat(np_x[None, :,:], repeats=2, axis=0)
np_x[0,:,:] = np_x[0,:,:] + 100

np_mask = np.ones(np_x.shape)
np_mask[:, -12:, :] = 0

print(f'x.shape [batch, time, features]={np_x.shape}')
print(f'mask.shape [batch, time, features]={np_mask.shape}')
```


```python
# Validate scalers
x = 1.0*torch.tensor(np_x)
mask = torch.tensor(np_mask)
scaler = TemporalNorm(scaler_type='standard', dim=1)
x_scaled = scaler.transform(x=x, mask=mask)
x_recovered = scaler.inverse_transform(x_scaled)

plt.plot(x[0,:,0], label='x1', color='#78ACA8')
plt.plot(x[0,:,1], label='x2',  color='#E3A39A')
plt.title('Before TemporalNorm')
plt.xlabel('Time')
plt.legend()
plt.show()

plt.plot(x_scaled[0,:,0], label='x1', color='#78ACA8')
plt.plot(x_scaled[0,:,1]+0.1, label='x2+0.1', color='#E3A39A')
plt.title(f'TemporalNorm \'{scaler.scaler_type}\' ')
plt.xlabel('Time')
plt.legend()
plt.show()

plt.plot(x_recovered[0,:,0], label='x1', color='#78ACA8')
plt.plot(x_recovered[0,:,1], label='x2', color='#E3A39A')
plt.title('Recovered')
plt.xlabel('Time')
plt.legend()
plt.show()
```
