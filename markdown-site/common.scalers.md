---
title: TemporalNorm
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

> Temporal normalization has proven to be essential in neural
> forecasting tasks, as it enables network’s non-linearities to express
> themselves. Forecasting scaling methods take particular interest in
> the temporal dimension where most of the variance dwells, contrary to
> other deep learning techniques like `BatchNorm` that normalizes across
> batch and temporal dimensions, and `LayerNorm` that normalizes across
> the feature dimension. Currently we support the following techniques:
> `std`, `median`, `norm`, `norm1`, `invariant`. <br><br>

![Figure 1. Illustration of temporal normalization (left), layer
normalization (center) and batch normalization (right). The entries in
green show the components used to compute the normalizing
statistics.](imgs_models/temporal_norm.png)

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
import torch
import torch.nn as nn
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
from nbdev.showdoc import show_doc
import matplotlib.pyplot as plt

plt.rcParams["axes.grid"]=True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["figure.figsize"] = (4,2)
```

</details>

:::

# <span style="color:DarkBlue"> 1. Auxiliary Functions </span> {#auxiliary-functions}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
def masked_median(x, mask, dim=-1, keepdim=True):
    """ Masked Median

    Compute the median of tensor `x` along dim, ignoring values where 
    `mask` is False. `x` and `mask` need to be broadcastable.

    **Parameters:**<br>
    `x`: torch.Tensor to compute median of along `dim` dimension.<br>
    `mask`: torch Tensor bool with same shape as `x`, where `x` is valid and False
            where `x` should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.<br>
    `dim` (int, optional): Dimension to take median of. Defaults to -1.<br>
    `keepdim` (bool, optional): Keep dimension of `x` or not. Defaults to True.<br>

    **Returns:**<br>
    `x_median`: torch.Tensor with normalized values.
    """
    x_nan = x.float().masked_fill(mask<1, float("nan"))
    x_median, _ = x_nan.nanmedian(dim=dim, keepdim=keepdim)
    x_median = torch.nan_to_num(x_median, nan=0.0)
    return x_median

def masked_mean(x, mask, dim=-1, keepdim=True):
    """ Masked  Mean

    Compute the mean of tensor `x` along dimension, ignoring values where 
    `mask` is False. `x` and `mask` need to be broadcastable.

    **Parameters:**<br>
    `x`: torch.Tensor to compute mean of along `dim` dimension.<br>
    `mask`: torch Tensor bool with same shape as `x`, where `x` is valid and False
            where `x` should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.<br>
    `dim` (int, optional): Dimension to take mean of. Defaults to -1.<br>
    `keepdim` (bool, optional): Keep dimension of `x` or not. Defaults to True.<br>

    **Returns:**<br>
    `x_mean`: torch.Tensor with normalized values.
    """
    x_nan = x.float().masked_fill(mask<1, float("nan"))
    x_mean = x_nan.nanmean(dim=dim, keepdim=keepdim)
    x_mean = torch.nan_to_num(x_mean, nan=0.0)
    return x_mean
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(masked_median, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(masked_mean, title_level=3)
```

</details>

# <span style="color:DarkBlue"> 2. Scalers </span> {#scalers}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
def minmax_scaler(x, mask, eps=1e-6, dim=-1):
    """ MinMax Scaler

    Standardizes temporal features by ensuring its range dweels between
    [0,1] range. This transformation is often used as an alternative 
    to the standard scaler. The scaled features are obtained as:

    $$\mathbf{z} = (\mathbf{x}_{[B,T,C]}-\mathrm{min}({\mathbf{x}})_{[B,1,C]})/
        (\mathrm{max}({\mathbf{x}})_{[B,1,C]}- \mathrm{min}({\mathbf{x}})_{[B,1,C]})$$

    **Parameters:**<br>
    `x`: torch.Tensor input tensor.<br>
    `mask`: torch Tensor bool, same dimension as `x`, indicates where `x` is valid and False
            where `x` should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.<br>
    `eps` (float, optional): Small value to avoid division by zero. Defaults to 1e-6.<br>
    `dim` (int, optional): Dimension over to compute min and max. Defaults to -1.<br>

    **Returns:**<br>
    `z`: torch.Tensor same shape as `x`, except scaled.
    """
    mask = mask.clone()
    mask[mask==0] = torch.inf
    mask[mask==1] = 0
    x_max = torch.max(torch.nan_to_num(x-mask,nan=-torch.inf), dim=dim, keepdim=True)[0]
    x_min = torch.min(torch.nan_to_num(x+mask,nan=torch.inf), dim=dim, keepdim=True)[0]
    x_max = x_max.type(x.dtype)
    x_min = x_min.type(x.dtype)

    # x_range and prevent division by zero
    x_range = x_max - x_min
    x_range[x_range==0] = 1.0
    x_range = x_range + eps

    z = (x - x_min) / x_range
    return z, x_min, x_range
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def inv_minmax_scaler(z, x_min, x_range):
    return z * x_range + x_min
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(minmax_scaler, title_level=3)
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
def minmax1_scaler(x, mask, eps=1e-6, dim=-1):
    """ MinMax1 Scaler

    Standardizes temporal features by ensuring its range dweels between
    [-1,1] range. This transformation is often used as an alternative 
    to the standard scaler or classic Min Max Scaler. 
    The scaled features are obtained as:

    $$\mathbf{z} = 2 (\mathbf{x}_{[B,T,C]}-\mathrm{min}({\mathbf{x}})_{[B,1,C]})/ (\mathrm{max}({\mathbf{x}})_{[B,1,C]}- \mathrm{min}({\mathbf{x}})_{[B,1,C]})-1$$

    **Parameters:**<br>
    `x`: torch.Tensor input tensor.<br>
    `mask`: torch Tensor bool, same dimension as `x`, indicates where `x` is valid and False
            where `x` should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.<br>
    `eps` (float, optional): Small value to avoid division by zero. Defaults to 1e-6.<br>
    `dim` (int, optional): Dimension over to compute min and max. Defaults to -1.<br>

    **Returns:**<br>
    `z`: torch.Tensor same shape as `x`, except scaled.
    """
    # Mask values (set masked to -inf or +inf)
    mask = mask.clone()
    mask[mask==0] = torch.inf
    mask[mask==1] = 0
    x_max = torch.max(torch.nan_to_num(x-mask,nan=-torch.inf), dim=dim, keepdim=True)[0]
    x_min = torch.min(torch.nan_to_num(x+mask,nan=torch.inf), dim=dim, keepdim=True)[0]
    x_max = x_max.type(x.dtype)
    x_min = x_min.type(x.dtype)
    
    # x_range and prevent division by zero
    x_range = x_max - x_min
    x_range[x_range==0] = 1.0
    x_range = x_range + eps

    x = (x - x_min) / x_range
    z = x * (2) - 1
    return z, x_min, x_range
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def inv_minmax1_scaler(z, x_min, x_range):
    z = (z + 1) / 2
    return z * x_range + x_min
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(minmax1_scaler, title_level=3)
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
def std_scaler(x, mask, dim=-1, eps=1e-6):
    """ Standard Scaler

    Standardizes features by removing the mean and scaling
    to unit variance along the `dim` dimension. 

    For example, for `base_windows` models, the scaled features are obtained as (with dim=1):

    $$\mathbf{z} = (\mathbf{x}_{[B,T,C]}-\\bar{\mathbf{x}}_{[B,1,C]})/\hat{\sigma}_{[B,1,C]}$$

    **Parameters:**<br>
    `x`: torch.Tensor.<br>
    `mask`: torch Tensor bool, same dimension as `x`, indicates where `x` is valid and False
            where `x` should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.<br>
    `eps` (float, optional): Small value to avoid division by zero. Defaults to 1e-6.<br>
    `dim` (int, optional): Dimension over to compute mean and std. Defaults to -1.<br>

    **Returns:**<br>
    `z`: torch.Tensor same shape as `x`, except scaled.
    """
    x_means = masked_mean(x=x, mask=mask, dim=dim)
    x_stds = torch.sqrt(masked_mean(x=(x-x_means)**2, mask=mask, dim=dim))
    
    # Protect against division by zero
    x_stds[x_stds==0] = 1.0
    x_stds = x_stds + eps

    z = (x - x_means) / x_stds
    return z, x_means, x_stds
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def inv_std_scaler(z, x_mean, x_std):
    return (z * x_std) + x_mean
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(std_scaler, title_level=3)
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
def robust_scaler(x, mask, dim=-1, eps=1e-6):
    """ Robust Median Scaler

    Standardizes features by removing the median and scaling
    with the mean absolute deviation (mad) a robust estimator of variance.
    This scaler is particularly useful with noisy data where outliers can 
    heavily influence the sample mean / variance in a negative way.
    In these scenarios the median and amd give better results.
    
    For example, for `base_windows` models, the scaled features are obtained as (with dim=1):

    $$\mathbf{z} = (\mathbf{x}_{[B,T,C]}-\\textrm{median}(\mathbf{x})_{[B,1,C]})/\\textrm{mad}(\mathbf{x})_{[B,1,C]}$$
        
    $$\\textrm{mad}(\mathbf{x}) = \\frac{1}{N} \sum_{}|\mathbf{x} - \mathrm{median}(x)|$$

    **Parameters:**<br>
    `x`: torch.Tensor input tensor.<br>
    `mask`: torch Tensor bool, same dimension as `x`, indicates where `x` is valid and False
            where `x` should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.<br>
    `eps` (float, optional): Small value to avoid division by zero. Defaults to 1e-6.<br>
    `dim` (int, optional): Dimension over to compute median and mad. Defaults to -1.<br>

    **Returns:**<br>
    `z`: torch.Tensor same shape as `x`, except scaled.
    """
    x_median = masked_median(x=x, mask=mask, dim=dim)
    x_mad = masked_median(x=torch.abs(x-x_median), mask=mask, dim=dim)

    # Protect x_mad=0 values
    # Assuming normality and relationship between mad and std
    x_means = masked_mean(x=x, mask=mask, dim=dim)
    x_stds = torch.sqrt(masked_mean(x=(x-x_means)**2, mask=mask, dim=dim))  
    x_mad_aux = x_stds * 0.6744897501960817
    x_mad = x_mad * (x_mad>0) + x_mad_aux * (x_mad==0)
    
    # Protect against division by zero
    x_mad[x_mad==0] = 1.0
    x_mad = x_mad + eps

    z = (x - x_median) / x_mad
    return z, x_median, x_mad
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def inv_robust_scaler(z, x_median, x_mad):
    return z * x_mad + x_median
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(robust_scaler, title_level=3)
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
def invariant_scaler(x, mask, dim=-1, eps=1e-6):
    """ Invariant Median Scaler

    Standardizes features by removing the median and scaling
    with the mean absolute deviation (mad) a robust estimator of variance.
    Aditionally it complements the transformation with the arcsinh transformation.

    For example, for `base_windows` models, the scaled features are obtained as (with dim=1):

    $$\mathbf{z} = (\mathbf{x}_{[B,T,C]}-\\textrm{median}(\mathbf{x})_{[B,1,C]})/\\textrm{mad}(\mathbf{x})_{[B,1,C]}$$

    $$\mathbf{z} = \\textrm{arcsinh}(\mathbf{z})$$

    **Parameters:**<br>
    `x`: torch.Tensor input tensor.<br>
    `mask`: torch Tensor bool, same dimension as `x`, indicates where `x` is valid and False
            where `x` should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.<br>
    `eps` (float, optional): Small value to avoid division by zero. Defaults to 1e-6.<br>
    `dim` (int, optional): Dimension over to compute median and mad. Defaults to -1.<br>

    **Returns:**<br>
    `z`: torch.Tensor same shape as `x`, except scaled.
    """
    x_median = masked_median(x=x, mask=mask, dim=dim)
    x_mad = masked_median(x=torch.abs(x-x_median), mask=mask, dim=dim)

    # Protect x_mad=0 values
    # Assuming normality and relationship between mad and std
    x_means = masked_mean(x=x, mask=mask, dim=dim)
    x_stds = torch.sqrt(masked_mean(x=(x-x_means)**2, mask=mask, dim=dim))        
    x_mad_aux = x_stds * 0.6744897501960817
    x_mad = x_mad * (x_mad>0) + x_mad_aux * (x_mad==0)

    # Protect against division by zero
    x_mad[x_mad==0] = 1.0
    x_mad = x_mad + eps
    
    z = torch.arcsinh((x - x_median) / x_mad)
    return z, x_median, x_mad
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def inv_invariant_scaler(z, x_median, x_mad):
    return torch.sinh(z) * x_mad + x_median
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(invariant_scaler, title_level=3)
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
def identity_scaler(x, mask, dim=-1, eps=1e-6):
    """ Identity Scaler

    A placeholder identity scaler, that is argument insensitive.

    **Parameters:**<br>
    `x`: torch.Tensor input tensor.<br>
    `mask`: torch Tensor bool, same dimension as `x`, indicates where `x` is valid and False
            where `x` should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.<br>
    `eps` (float, optional): Small value to avoid division by zero. Defaults to 1e-6.<br>
    `dim` (int, optional): Dimension over to compute median and mad. Defaults to -1.<br>

    **Returns:**<br>
    `x`: original torch.Tensor `x`.
    """
    # Collapse dim dimension
    shape = list(x.shape)
    shape[dim] = 1

    x_shift = torch.zeros(shape)
    x_scale = torch.ones(shape)

    return x, x_shift, x_scale
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def inv_identity_scaler(z, x_shift, x_scale):
    return z
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(identity_scaler, title_level=3)
```

</details>

# <span style="color:DarkBlue"> 3. TemporalNorm Module </span> {#temporalnorm-module}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class TemporalNorm(nn.Module):
    """ Temporal Normalization

    Standardization of the features is a common requirement for many 
    machine learning estimators, and it is commonly achieved by removing 
    the level and scaling its variance. The `TemporalNorm` module applies 
    temporal normalization over the batch of inputs as defined by the type of scaler.

    $$\mathbf{z}_{[B,T,C]} = \\textrm{Scaler}(\mathbf{x}_{[B,T,C]})$$

    **Parameters:**<br>
    `scaler_type`: str, defines the type of scaler used by TemporalNorm.
                    available [`identity`, `standard`, `robust`, `minmax`, `minmax1`, `invariant`].<br>
    `dim` (int, optional): Dimension over to compute scale and shift. Defaults to -1.<br>
    `eps` (float, optional): Small value to avoid division by zero. Defaults to 1e-6.<br>
                    
    """    
    def __init__(self, scaler_type='robust', dim=-1, eps=1e-6):
        super().__init__()
        scalers = {None: identity_scaler,
                   'identity': identity_scaler,
                   'standard': std_scaler,
                   'robust': robust_scaler,
                   'minmax': minmax_scaler,
                   'minmax1': minmax1_scaler,
                   'invariant':invariant_scaler,}
        inverse_scalers = {None: inv_identity_scaler,
                    'identity': inv_identity_scaler,
                    'standard': inv_std_scaler,
                    'robust': inv_robust_scaler,
                    'minmax': inv_minmax_scaler,
                    'minmax1': inv_minmax1_scaler,
                    'invariant': inv_invariant_scaler,}
        assert (scaler_type in scalers.keys()), f'{scaler_type} not defined'

        self.scaler = scalers[scaler_type]
        self.inverse_scaler = inverse_scalers[scaler_type]
        self.scaler_type = scaler_type
        self.dim = dim
        self.eps = eps

    #@torch.no_grad()
    def transform(self, x, mask):
        """ Center and scale the data.

        **Parameters:**<br>
        `x`: torch.Tensor shape [batch, time, channels].<br>
        `mask`: torch Tensor bool, shape  [batch, time] where `x` is valid and False
                where `x` should be masked. Mask should not be all False in any column of
                dimension dim to avoid NaNs from zero division.<br>
        
        **Returns:**<br>
        `z`: torch.Tensor same shape as `x`, except scaled.        
        """
        z, x_shift, x_scale = self.scaler(x=x, mask=mask, dim=self.dim, eps=self.eps)
        self.x_shift = x_shift
        self.x_scale = x_scale
        return z

    #@torch.no_grad()
    def inverse_transform(self, z, x_shift=None, x_scale=None):
        """ Scale back the data to the original representation.

        **Parameters:**<br>
        `z`: torch.Tensor shape [batch, time, channels], scaled.<br>

        **Returns:**<br>
        `x`: torch.Tensor original data.
        """
        if x_shift is None:
            x_shift = self.x_shift
        if x_scale is None:
            x_scale = self.x_scale

        x = self.inverse_scaler(z, x_shift, x_scale)
        return x
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(TemporalNorm, name='TemporalNorm', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(TemporalNorm.transform, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(TemporalNorm.inverse_transform, title_level=3)
```

</details>

# <span style="color:DarkBlue"> Example </span> {#example}

<details>
<summary>Code</summary>

``` python
import numpy as np
```

</details>
<details>
<summary>Code</summary>

``` python
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

</details>
<details>
<summary>Code</summary>

``` python
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

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# Validate scalers
for scaler_type in [None, 'identity', 'standard', 'robust', 'minmax', 'minmax1', 'invariant']:
    x = 1.0*torch.tensor(np_x)
    mask = torch.tensor(np_mask)
    scaler = TemporalNorm(scaler_type=scaler_type, dim=1)
    x_scaled = scaler.transform(x=x, mask=mask)
    x_recovered = scaler.inverse_transform(x_scaled)
    assert torch.allclose(x, x_recovered, atol=1e-5), f'Recovered data is not the same as original with {scaler_type}'
```

</details>

:::

# Test Predict (masked) {#test-predict-masked}

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.utils import AirPassengersDF as Y_df

model = NHITS(h=12,
              input_size=12*2,
              max_steps=1,
              windows_batch_size=None, 
              n_freq_downsample=[1,1,1],
              scaler_type='minmax')

nf = NeuralForecast(models=[model], freq='M')
nf.fit(df=Y_df)
Y_hat = nf.predict(df=Y_df)
assert pd.isnull(Y_hat).sum().sum() == 0, 'Predictions should not have NaNs'
```

</details>

:::

