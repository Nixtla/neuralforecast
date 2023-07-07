---
title: PyTorch Losses
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

> NeuralForecast contains a collection PyTorch Loss classes aimed to be
> used during the models’ optimization. The most important train signal
> is the forecast error, which is the difference between the observed
> value $y_{\tau}$ and the prediction $\hat{y}_{\tau}$, at time
> $y_{\tau}$:$$e_{\tau} = y_{\tau}-\hat{y}_{\tau} \qquad \qquad \tau \in \{t+1,\dots,t+H \}$$
> The train loss summarizes the forecast errors in different train
> optimization objectives.<br><br>All the losses are `torch.nn.modules`
> which helps to automatically moved them across CPU/GPU/TPU devices
> with Pytorch Lightning.

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
from typing import Optional, Union, Tuple

import math
import numpy as np
import torch

import torch.nn.functional as F
from torch.distributions import Distribution
from torch.distributions import (
    Bernoulli,
    Normal, 
    StudentT, 
    Poisson,
    NegativeBinomial
)

from torch.distributions import constraints
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
from neuralforecast.utils import generate_series
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def _divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Auxiliary funtion to handle divide by 0
    """
    div = a / b
    div[div != div] = 0.0
    div[div == float('inf')] = 0.0
    return div
```

</details>

:::

# <span style="color:DarkBlue">1. Scale-dependent Errors </span> {#scale-dependent-errors}

These metrics are on the same scale as the data.

## Mean Absolute Error (MAE) {#mean-absolute-error-mae}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class MAE(torch.nn.Module):
    """Mean Absolute Error

    Calculates Mean Absolute Error between
    `y` and `y_hat`. MAE measures the relative prediction
    accuracy of a forecasting method by calculating the
    deviation of the prediction and the true
    value at a given time and averages these devations
    over the length of the series.

    $$ \mathrm{MAE}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} |y_{\\tau} - \hat{y}_{\\tau}| $$
    """    
    def __init__(self):
        super(MAE, self).__init__()
        self.outputsize_multiplier = 1
        self.output_names = ['']
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor,
                 mask: Union[torch.Tensor, None] = None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `mae`: tensor (single value).
        """
        if mask is None:
            mask = torch.ones_like(y)

        mae = torch.abs(y - y_hat) * mask
        mae = torch.mean(mae)
        return mae
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(MAE, name='MAE.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(MAE.__call__, name='MAE.__call__', title_level=3)
```

</details>

![](imgs_losses/mae_loss.png)

## Mean Squared Error (MSE) {#mean-squared-error-mse}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class MSE(torch.nn.Module):
    """  Mean Squared Error

    Calculates Mean Squared Error between
    `y` and `y_hat`. MSE measures the relative prediction
    accuracy of a forecasting method by calculating the 
    squared deviation of the prediction and the true
    value at a given time, and averages these devations
    over the length of the series.
    
    $$ \mathrm{MSE}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} (y_{\\tau} - \hat{y}_{\\tau})^{2} $$
    """    
    def __init__(self):
        super(MSE, self).__init__()
        self.outputsize_multiplier = 1
        self.output_names = ['']
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor,
                 mask: Union[torch.Tensor, None] = None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `mse`: tensor (single value).
        """
        if mask is None:
            mask = torch.ones_like(y_hat)

        mse = (y - y_hat)**2
        mse = mask * mse
        mse = torch.mean(mse)
        return mse
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(MSE, name='MSE.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(MSE.__call__, name='MSE.__call__', title_level=3)
```

</details>

![](imgs_losses/mse_loss.png)

## Root Mean Squared Error (RMSE) {#root-mean-squared-error-rmse}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class RMSE(torch.nn.Module):
    """ Root Mean Squared Error

    Calculates Root Mean Squared Error between
    `y` and `y_hat`. RMSE measures the relative prediction
    accuracy of a forecasting method by calculating the squared deviation
    of the prediction and the observed value at a given time and
    averages these devations over the length of the series.
    Finally the RMSE will be in the same scale
    as the original time series so its comparison with other
    series is possible only if they share a common scale. 
    RMSE has a direct connection to the L2 norm.
    
    $$ \mathrm{RMSE}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}) = \\sqrt{\\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} (y_{\\tau} - \hat{y}_{\\tau})^{2}} $$
    """
    def __init__(self):
        super(RMSE, self).__init__()
        self.outputsize_multiplier = 1
        self.output_names = ['']
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, 
                 mask: Union[torch.Tensor, None] = None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `rmse`: tensor (single value).
        """
        if mask is None: 
            mask = torch.ones_like(y_hat)

        mse = (y - y_hat)**2
        mse = mask * mse
        mse = torch.mean(mse)
        mse = torch.sqrt(mse)
        return mse
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(RMSE, name='RMSE.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(RMSE.__call__, name='RMSE.__call__', title_level=3)
```

</details>

![](imgs_losses/rmse_loss.png)

# <span style="color:DarkBlue"> 2. Percentage errors </span> {#percentage-errors}

These metrics are unit-free, suitable for comparisons across series.

## Mean Absolute Percentage Error (MAPE) {#mean-absolute-percentage-error-mape}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class MAPE(torch.nn.Module):
    """ Mean Absolute Percentage Error

    Calculates Mean Absolute Percentage Error  between
    `y` and `y_hat`. MAPE measures the relative prediction
    accuracy of a forecasting method by calculating the percentual deviation
    of the prediction and the observed value at a given time and
    averages these devations over the length of the series.
    The closer to zero an observed value is, the higher penalty MAPE loss
    assigns to the corresponding error.

    $$ \mathrm{MAPE}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} \\frac{|y_{\\tau}-\hat{y}_{\\tau}|}{|y_{\\tau}|} $$

    **References:**<br>
    [Makridakis S., "Accuracy measures: theoretical and practical concerns".](https://www.sciencedirect.com/science/article/pii/0169207093900793)    
    """    
    def __init__(self):
        super(MAPE, self).__init__()
        self.outputsize_multiplier = 1
        self.output_names = ['']
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, 
                 mask: Union[torch.Tensor, None] = None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `mape`: tensor (single value).
        """
        if mask is None: 
            mask = torch.ones_like(y_hat)

        mask = _divide_no_nan(mask, torch.abs(y))
        mape = torch.abs(y - y_hat) * mask
        mape = torch.mean(mape)
        return mape
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(MAPE, name='MAPE.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(MAPE.__call__, name='MAPE.__call__', title_level=3)
```

</details>

![](imgs_losses/mape_loss.png)

## Symmetric MAPE (sMAPE) {#symmetric-mape-smape}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class SMAPE(torch.nn.Module):
    """ Symmetric Mean Absolute Percentage Error

    Calculates Symmetric Mean Absolute Percentage Error between
    `y` and `y_hat`. SMAPE measures the relative prediction
    accuracy of a forecasting method by calculating the relative deviation
    of the prediction and the observed value scaled by the sum of the
    absolute values for the prediction and observed value at a
    given time, then averages these devations over the length
    of the series. This allows the SMAPE to have bounds between
    0% and 200% which is desireble compared to normal MAPE that
    may be undetermined when the target is zero.

    $$ \mathrm{sMAPE}_{2}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} \\frac{|y_{\\tau}-\hat{y}_{\\tau}|}{|y_{\\tau}|+|\hat{y}_{\\tau}|} $$

    **References:**<br>
    [Makridakis S., "Accuracy measures: theoretical and practical concerns".](https://www.sciencedirect.com/science/article/pii/0169207093900793)
    """
    def __init__(self):
        super(SMAPE, self).__init__()
        self.outputsize_multiplier = 1
        self.output_names = ['']
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, 
                 mask: Union[torch.Tensor, None] = None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `smape`: tensor (single value).
        """
        if mask is None: 
            mask = torch.ones_like(y_hat)

        delta_y = torch.abs((y - y_hat))
        scale = torch.abs(y) + torch.abs(y_hat)
        smape = _divide_no_nan(delta_y, scale)
        smape = smape * mask
        smape = 2 * torch.mean(smape)
        return smape
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(SMAPE, name='SMAPE.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(SMAPE.__call__, name='SMAPE.__call__', title_level=3)
```

</details>

# <span style="color:DarkBlue"> 3. Scale-independent Errors </span> {#scale-independent-errors}

These metrics measure the relative improvements versus baselines.

## Mean Absolute Scaled Error (MASE) {#mean-absolute-scaled-error-mase}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class MASE(torch.nn.Module):
    """ Mean Absolute Scaled Error 
    Calculates the Mean Absolute Scaled Error between
    `y` and `y_hat`. MASE measures the relative prediction
    accuracy of a forecasting method by comparinng the mean absolute errors
    of the prediction and the observed value against the mean
    absolute errors of the seasonal naive model.
    The MASE partially composed the Overall Weighted Average (OWA), 
    used in the M4 Competition.
    
    $$ \mathrm{MASE}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}, \\mathbf{\hat{y}}^{season}_{\\tau}) = \\frac{1}{H} \sum^{t+H}_{\\tau=t+1} \\frac{|y_{\\tau}-\hat{y}_{\\tau}|}{\mathrm{MAE}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}^{season}_{\\tau})} $$

    **Parameters:**<br>
    `seasonality`: int. Main frequency of the time series; Hourly 24,  Daily 7, Weekly 52, Monthly 12, Quarterly 4, Yearly 1.
    
    **References:**<br>
    [Rob J. Hyndman, & Koehler, A. B. "Another look at measures of forecast accuracy".](https://www.sciencedirect.com/science/article/pii/S0169207006000239)<br>
    [Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos, "The M4 Competition: 100,000 time series and 61 forecasting methods".](https://www.sciencedirect.com/science/article/pii/S0169207019301128)
    """
    def __init__(self, seasonality: int):
        super(MASE, self).__init__()
        self.outputsize_multiplier = 1
        self.seasonality = seasonality
        self.output_names = ['']
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor,  y_insample: torch.Tensor, 
                 mask: Union[torch.Tensor, None] = None):
        """
        **Parameters:**<br>
        `y`: tensor (batch_size, output_size), Actual values.<br>
        `y_hat`: tensor (batch_size, output_size)), Predicted values.<br>
        `y_insample`: tensor (batch_size, input_size), Actual insample Seasonal Naive predictions.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `mase`: tensor (single value).
        """
        if mask is None: 
            mask = torch.ones_like(y_hat)

        delta_y = torch.abs(y - y_hat)
        scale = torch.mean(torch.abs(y_insample[:, self.seasonality:] - \
                                     y_insample[:, :-self.seasonality]), axis=1)
        mase = _divide_no_nan(delta_y, scale[:, None])
        mase = mase * mask
        mase = torch.mean(mase)
        return mase
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(MASE, name='MASE.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(MASE.__call__, name='MASE.__call__', title_level=3)
```

</details>

![](imgs_losses/mase_loss.png)

## Relative Mean Squared Error (relMSE) {#relative-mean-squared-error-relmse}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class relMSE(torch.nn.Module):
    """Relative Mean Squared Error
    Computes Relative Mean Squared Error (relMSE), as proposed by Hyndman & Koehler (2006)
    as an alternative to percentage errors, to avoid measure unstability.
    $$ \mathrm{relMSE}(\\mathbf{y}, \\mathbf{\hat{y}}, \\mathbf{\hat{y}}^{naive1}) =
    \\frac{\mathrm{MSE}(\\mathbf{y}, \\mathbf{\hat{y}})}{\mathrm{MSE}(\\mathbf{y}, \\mathbf{\hat{y}}^{naive1})} $$
    **Parameters:**<br>
    `y_train`: numpy array, Training values.<br>
    **References:**<br>
    - [Hyndman, R. J and Koehler, A. B. (2006).
       "Another look at measures of forecast accuracy",
       International Journal of Forecasting, Volume 22, Issue 4.](https://www.sciencedirect.com/science/article/pii/S0169207006000239)<br>
    - [Kin G. Olivares, O. Nganba Meetei, Ruijun Ma, Rohan Reddy, Mengfei Cao, Lee Dicker. 
       "Probabilistic Hierarchical Forecasting with Deep Poisson Mixtures. 
       Submitted to the International Journal Forecasting, Working paper available at arxiv.](https://arxiv.org/pdf/2110.13179.pdf)
    """
    def __init__(self, y_train):
        super(relMSE, self).__init__()
        self.y_train = y_train
        self.mse = MSE()
        self.is_distribution_output = False
    
    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, 
                 mask: Union[torch.Tensor, None] = None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per series to consider in loss.<br>

        **Returns:**<br>
        `relmse`: tensor (single value).
        """
        if mask is None: 
            mask = torch.ones_like(y)
        n_series, horizon = y.shape

        last_col = self.y_train[:, -1].unsqueeze(1)
        y_naive = last_col.repeat(1, horizon)
            
        norm = self.mse(y=y, y_hat=y_naive)
        loss = self.mse(y=y, y_hat=y_hat, mask=mask)
        loss = loss / (norm + 1e-5)
        return loss
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(relMSE, name='relMSE.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(relMSE.__call__, name='relMSE.__call__', title_level=3)
```

</details>

# <span style="color:DarkBlue"> 4. Probabilistic Errors </span> {#probabilistic-errors}

These methods use statistical approaches for estimating unknown
probability distributions using observed data.

Maximum likelihood estimation involves finding the parameter values that
maximize the likelihood function, which measures the probability of
obtaining the observed data given the parameter values. MLE has good
theoretical properties and efficiency under certain satisfied
assumptions.

On the non-parametric approach, quantile regression measures
non-symmetrically deviation, producing under/over estimation.

## Quantile Loss {#quantile-loss}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class QuantileLoss(torch.nn.Module):
    """ Quantile Loss

    Computes the quantile loss between `y` and `y_hat`.
    QL measures the deviation of a quantile forecast.
    By weighting the absolute deviation in a non symmetric way, the
    loss pays more attention to under or over estimation.
    A common value for q is 0.5 for the deviation from the median (Pinball loss).

    $$ \mathrm{QL}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}^{(q)}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} \Big( (1-q)\,( \hat{y}^{(q)}_{\\tau} - y_{\\tau} )_{+} + q\,( y_{\\tau} - \hat{y}^{(q)}_{\\tau} )_{+} \Big) $$

    **Parameters:**<br>
    `q`: float, between 0 and 1. The slope of the quantile loss, in the context of quantile regression, the q determines the conditional quantile level.<br>

    **References:**<br>
    [Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles".](https://www.jstor.org/stable/1913643)
    """
    def __init__(self, q):
        super(QuantileLoss, self).__init__()
        self.outputsize_multiplier = 1
        self.q = q
        self.output_names = [f'_ql{q}']
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, 
                 mask: Union[torch.Tensor, None] = None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `quantile_loss`: tensor (single value).
        """
        if mask is None: 
            mask = torch.ones_like(y_hat)

        delta_y = y - y_hat
        loss = torch.max(torch.mul(self.q, delta_y), torch.mul((self.q - 1), delta_y))
        loss = loss * mask
        quantile_loss = torch.mean(loss)
        return quantile_loss
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(QuantileLoss, name='QuantileLoss.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(QuantileLoss.__call__, name='QuantileLoss.__call__', title_level=3)
```

</details>

![](imgs_losses/q_loss.png)

## Multi Quantile Loss (MQLoss) {#multi-quantile-loss-mqloss}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def level_to_outputs(level):
    qs = sum([[50-l/2, 50+l/2] for l in level], [])
    output_names = sum([[f'-lo-{l}', f'-hi-{l}'] for l in level], [])

    sort_idx = np.argsort(qs)
    quantiles = np.array(qs)[sort_idx]

    # Add default median
    quantiles = np.concatenate([np.array([50]), quantiles])
    quantiles = torch.Tensor(quantiles) / 100
    output_names = list(np.array(output_names)[sort_idx])
    output_names.insert(0, '-median')
    
    return quantiles, output_names

def quantiles_to_outputs(quantiles):
    output_names = []
    for q in quantiles:
        if q<.50:
            output_names.append(f'-lo-{np.round(100-200*q,2)}')
        elif q>.50:
            output_names.append(f'-hi-{np.round(100-200*(1-q),2)}')
        else:
            output_names.append('-median')
    return quantiles, output_names
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class MQLoss(torch.nn.Module):
    """  Multi-Quantile loss

    Calculates the Multi-Quantile loss (MQL) between `y` and `y_hat`.
    MQL calculates the average multi-quantile Loss for
    a given set of quantiles, based on the absolute 
    difference between predicted quantiles and observed values.
    
    $$ \mathrm{MQL}(\\mathbf{y}_{\\tau},[\\mathbf{\hat{y}}^{(q_{1})}_{\\tau}, ... ,\hat{y}^{(q_{n})}_{\\tau}]) = \\frac{1}{n} \\sum_{q_{i}} \mathrm{QL}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}^{(q_{i})}_{\\tau}) $$
    
    The limit behavior of MQL allows to measure the accuracy 
    of a full predictive distribution $\mathbf{\hat{F}}_{\\tau}$ with 
    the continuous ranked probability score (CRPS). This can be achieved 
    through a numerical integration technique, that discretizes the quantiles 
    and treats the CRPS integral with a left Riemann approximation, averaging over 
    uniformly distanced quantiles.    
    
    $$ \mathrm{CRPS}(y_{\\tau}, \mathbf{\hat{F}}_{\\tau}) = \int^{1}_{0} \mathrm{QL}(y_{\\tau}, \hat{y}^{(q)}_{\\tau}) dq $$

    **Parameters:**<br>
    `level`: int list [0,100]. Probability levels for prediction intervals (Defaults median).
    `quantiles`: float list [0., 1.]. Alternative to level, quantiles to estimate from y distribution.

    **References:**<br>
    [Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles".](https://www.jstor.org/stable/1913643)<br>
    [James E. Matheson and Robert L. Winkler, "Scoring Rules for Continuous Probability Distributions".](https://www.jstor.org/stable/2629907)
    """
    def __init__(self, level=[80, 90], quantiles=None):
        super(MQLoss, self).__init__()
        # Transform level to MQLoss parameters
        qs, self.output_names = level_to_outputs(level)
        qs = torch.Tensor(qs)

        # Transform quantiles to homogeneus output names
        if quantiles is not None:
            _, self.output_names = quantiles_to_outputs(quantiles)
            qs = torch.Tensor(quantiles)

        self.quantiles = torch.nn.Parameter(qs, requires_grad=False)
        self.outputsize_multiplier = len(self.quantiles)
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Identity domain map [B,T,H,Q]/[B,H,Q]
        """
        return y_hat

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, 
                 mask: Union[torch.Tensor, None] = None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `mqloss`: tensor (single value).
        """
        if mask is None: 
            mask = torch.ones_like(y_hat)

        n_q = len(self.quantiles)

        error  = y_hat - y.unsqueeze(-1)
        sq     = torch.maximum(-error, torch.zeros_like(error))
        s1_q   = torch.maximum(error, torch.zeros_like(error))
        mqloss = (self.quantiles * sq + (1 - self.quantiles) * s1_q)

        # Match y/weights dimensions and compute weighted average
        mask = mask / torch.sum(mask)
        mask = mask.unsqueeze(-1)
        mqloss = (1/n_q) * mqloss * mask
        return torch.sum(mqloss)
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(MQLoss, name='MQLoss.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(MQLoss.__call__, name='MQLoss.__call__', title_level=3)
```

</details>

![](imgs_losses/mq_loss.png)

## Weighted MQLoss (wMQLoss) {#weighted-mqloss-wmqloss}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class wMQLoss(torch.nn.Module):
    """ Weighted Multi-Quantile loss
    
    Calculates the Weighted Multi-Quantile loss (WMQL) between `y` and `y_hat`.
    WMQL calculates the weighted average multi-quantile Loss for
    a given set of quantiles, based on the absolute 
    difference between predicted quantiles and observed values.  
        
    $$ \mathrm{wMQL}(\\mathbf{y}_{\\tau},[\\mathbf{\hat{y}}^{(q_{1})}_{\\tau}, ... ,\hat{y}^{(q_{n})}_{\\tau}]) = \\frac{1}{n} \\sum_{q_{i}} \\frac{\mathrm{QL}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}^{(q_{i})}_{\\tau})}{\\sum^{t+H}_{\\tau=t+1} |y_{\\tau}|} $$
    
    **Parameters:**<br>
    `level`: int list [0,100]. Probability levels for prediction intervals (Defaults median).
    `quantiles`: float list [0., 1.]. Alternative to level, quantiles to estimate from y distribution.

    **References:**<br>
    [Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles".](https://www.jstor.org/stable/1913643)<br>
    [James E. Matheson and Robert L. Winkler, "Scoring Rules for Continuous Probability Distributions".](https://www.jstor.org/stable/2629907)
    """
    def __init__(self, level=[80, 90], quantiles=None):
        super(wMQLoss, self).__init__()
        # Transform level to MQLoss parameters
        qs, self.output_names = level_to_outputs(level)
        qs = torch.Tensor(qs)

        # Transform quantiles to homogeneus output names
        if quantiles is not None:
            _, self.output_names = quantiles_to_outputs(quantiles)
            qs = torch.Tensor(quantiles)

        self.quantiles = torch.nn.Parameter(qs, requires_grad=False)
        self.outputsize_multiplier = len(self.quantiles)
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Identity domain map [B,T,H,Q]/[B,H,Q]
        """
        return y_hat

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, 
                 mask: Union[torch.Tensor, None] = None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `mqloss`: tensor (single value).
        """
        if mask is None: 
            mask = torch.ones_like(y_hat)

        error = y_hat - y.unsqueeze(-1)
        
        sq = torch.maximum(-error, torch.zeros_like(error))
        s1_q = torch.maximum(error, torch.zeros_like(error))
        loss = (self.quantiles * sq + (1 - self.quantiles) * s1_q)
        
        mask = mask.unsqueeze(-1)
        wmqloss = _divide_no_nan(torch.sum(loss * mask, axis=-2), 
                                 torch.sum(torch.abs(y.unsqueeze(-1)) * mask, axis=-2))
        return torch.mean(wmqloss)
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(wMQLoss, name='wMQLoss.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(wMQLoss.__call__, name='wMQLoss.__call__', title_level=3)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# Unit tests to check MQLoss' stored quantiles
# attribute is correctly instantiated
check = MQLoss(level=[80, 90])
test_eq(len(check.quantiles), 5)

check = MQLoss(quantiles=[0.0100, 0.1000, 0.5, 0.9000, 0.9900])
print(check.output_names)
print(check.quantiles)
test_eq(len(check.quantiles), 5)

check = MQLoss(quantiles=[0.0100, 0.1000, 0.9000, 0.9900])
test_eq(len(check.quantiles), 4)

check = wMQLoss(quantiles=[0.0100, 0.1000, 0.9000, 0.9900])
test_eq(len(check.quantiles), 4)
```

</details>

:::

## DistributionLoss {#distributionloss}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def weighted_average(x: torch.Tensor, 
                     weights: Optional[torch.Tensor]=None, dim=None) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given dim, masking
    values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    **Parameters:**<br>
    `x`: Input tensor, of which the average must be computed.<br>
    `weights`: Weights tensor, of the same shape as `x`.<br>
    `dim`: The dim along which to average `x`.<br>

    **Returns:**<br>
    `Tensor`: The tensor with values averaged along the specified `dim`.<br>
    """
    if weights is not None:
        weighted_tensor = torch.where(
            weights != 0, x * weights, torch.zeros_like(x)
        )
        sum_weights = torch.clamp(
            weights.sum(dim=dim) if dim else weights.sum(), min=1.0
        )
        return (
            weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()
        ) / sum_weights
    else:
        return x.mean(dim=dim)
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def bernoulli_domain_map(input: torch.Tensor):
    """ Bernoulli Domain Map
    Maps input into distribution constraints, by construction input's 
    last dimension is of matching `distr_args` length.

    **Parameters:**<br>
    `input`: tensor, of dimensions [B,T,H,theta] or [B,H,theta].<br>

    **Returns:**<br>
    `(probs,)`: tuple with tensors of Poisson distribution arguments.<br>
    """
    return (input.squeeze(-1),)

def bernoulli_scale_decouple(output, loc=None, scale=None):
    """ Bernoulli Scale Decouple

    Stabilizes model's output optimization, by learning residual
    variance and residual location based on anchoring `loc`, `scale`.
    Also adds Bernoulli domain protection to the distribution parameters.
    """
    probs = output[0]
    #if (loc is not None) and (scale is not None):
    #    rate = (rate * scale) + loc
    probs = F.sigmoid(probs)#.clone()
    return (probs,)

def student_domain_map(input: torch.Tensor):
    """ Student T Domain Map
    Maps input into distribution constraints, by construction input's 
    last dimension is of matching `distr_args` length.

    **Parameters:**<br>
    `input`: tensor, of dimensions [B,T,H,theta] or [B,H,theta].<br>
    `eps`: float, helps the initialization of scale for easier optimization.<br>

    **Returns:**<br>
    `(df, loc, scale)`: tuple with tensors of StudentT distribution arguments.<br>
    """
    df, loc, scale = torch.tensor_split(input, 3, dim=-1)
    return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)

def student_scale_decouple(output, loc=None, scale=None, eps: float=0.1):
    """ Normal Scale Decouple

    Stabilizes model's output optimization, by learning residual
    variance and residual location based on anchoring `loc`, `scale`.
    Also adds StudentT domain protection to the distribution parameters.
    """
    df, mean, tscale = output
    tscale = F.softplus(tscale)
    if (loc is not None) and (scale is not None):
        mean = (mean * scale) + loc
        tscale = (tscale + eps) * scale
    df = 2.0 + F.softplus(df)
    return (df, mean, tscale)

def normal_domain_map(input: torch.Tensor):
    """ Normal Domain Map
    Maps input into distribution constraints, by construction input's 
    last dimension is of matching `distr_args` length.

    **Parameters:**<br>
    `input`: tensor, of dimensions [B,T,H,theta] or [B,H,theta].<br>
    `eps`: float, helps the initialization of scale for easier optimization.<br>

    **Returns:**<br>
    `(mean, std)`: tuple with tensors of Normal distribution arguments.<br>
    """
    mean, std = torch.tensor_split(input, 2, dim=-1)
    return mean.squeeze(-1), std.squeeze(-1)

def normal_scale_decouple(output, loc=None, scale=None, eps: float=0.2):
    """ Normal Scale Decouple

    Stabilizes model's output optimization, by learning residual
    variance and residual location based on anchoring `loc`, `scale`.
    Also adds Normal domain protection to the distribution parameters.
    """
    mean, std = output
    std = F.softplus(std)
    if (loc is not None) and (scale is not None):
        mean = (mean * scale) + loc
        std = (std + eps) * scale
    return (mean, std)

def poisson_domain_map(input: torch.Tensor):
    """ Poisson Domain Map
    Maps input into distribution constraints, by construction input's 
    last dimension is of matching `distr_args` length.

    **Parameters:**<br>
    `input`: tensor, of dimensions [B,T,H,theta] or [B,H,theta].<br>

    **Returns:**<br>
    `(rate,)`: tuple with tensors of Poisson distribution arguments.<br>
    """
    return (input.squeeze(-1),)

def poisson_scale_decouple(output, loc=None, scale=None):
    """ Poisson Scale Decouple

    Stabilizes model's output optimization, by learning residual
    variance and residual location based on anchoring `loc`, `scale`.
    Also adds Poisson domain protection to the distribution parameters.
    """
    eps  = 1e-10
    rate = output[0]
    if (loc is not None) and (scale is not None):
        rate = (rate * scale) + loc
    rate = F.softplus(rate) + eps
    return (rate,)

def nbinomial_domain_map(input: torch.Tensor):
    """ Negative Binomial Domain Map
    Maps input into distribution constraints, by construction input's 
    last dimension is of matching `distr_args` length.

    **Parameters:**<br>
    `input`: tensor, of dimensions [B,T,H,theta] or [B,H,theta].<br>

    **Returns:**<br>
    `(total_count, alpha)`: tuple with tensors of N.Binomial distribution arguments.<br>
    """
    mu, alpha = torch.tensor_split(input, 2, dim=-1)
    return mu.squeeze(-1), alpha.squeeze(-1)

def nbinomial_scale_decouple(output, loc=None, scale=None):
    """ Negative Binomial Scale Decouple

    Stabilizes model's output optimization, by learning total
    count and logits based on anchoring `loc`, `scale`.
    Also adds Negative Binomial domain protection to the distribution parameters.
    """
    mu, alpha = output
    mu = F.softplus(mu) + 1e-8
    alpha = F.softplus(alpha) + 1e-8    # alpha = 1/total_counts
    if (loc is not None) and (scale is not None):
        mu *= loc
        alpha /= (loc + 1.)

    # mu = total_count * (probs/(1-probs))
    # => probs = mu / (total_count + mu)
    # => probs = mu / [total_count * (1 + mu * (1/total_count))]
    total_count = 1.0 / alpha
    probs = (mu * alpha / (1.0 + mu * alpha)) + 1e-8
    return (total_count, probs)
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def est_lambda(mu, rho):
    return mu ** (2 - rho) / (2 - rho)

def est_alpha(rho):
    return (2 - rho) / (rho - 1)

def est_beta(mu, rho):
    return mu ** (1 - rho) / (rho - 1)


class Tweedie(Distribution):
    """ Tweedie Distribution

    The Tweedie distribution is a compound probability, special case of exponential
    dispersion models EDMs defined by its mean-variance relationship.
    The distribution particularly useful to model sparse series as the probability has
    possitive mass at zero but otherwise is continuous.

    $Y \sim \mathrm{ED}(\\mu,\\sigma^{2}) \qquad
    \mathbb{P}(y|\\mu ,\\sigma^{2})=h(\\sigma^{2},y) \\exp \\left({\\frac {\\theta y-A(\\theta )}{\\sigma^{2}}}\\right)$<br>
    
    $\mu =A'(\\theta ) \qquad \mathrm{Var}(Y) = \\sigma^{2} \\mu^{\\rho}$
    
    Cases of the variance relationship include Normal (`rho` = 0), Poisson (`rho` = 1),
    Gamma (`rho` = 2), inverse Gaussian (`rho` = 3).

    **Parameters:**<br>
    `log_mu`: tensor, with log of means.<br>
    `rho`: float, Tweedie variance power (1,2). Fixed across all observations.<br>
    `sigma2`: tensor, Tweedie variance. Currently fixed in 1.<br>

    **References:**<br>
    - [Tweedie, M. C. K. (1984). An index which distinguishes between some important exponential families. Statistics: Applications and New Directions. 
    Proceedings of the Indian Statistical Institute Golden Jubilee International Conference (Eds. J. K. Ghosh and J. Roy), pp. 579-604. Calcutta: Indian Statistical Institute.]()<br>
    - [Jorgensen, B. (1987). Exponential Dispersion Models. Journal of the Royal Statistical Society. 
       Series B (Methodological), 49(2), 127–162. http://www.jstor.org/stable/2345415](http://www.jstor.org/stable/2345415)<br>
    """
    def __init__(self, log_mu, rho, validate_args=None):
        # TODO: add sigma2 dispersion
        # TODO add constraints
        # arg_constraints = {'log_mu': constraints.real, 'rho': constraints.positive}
        # support = constraints.real
        self.log_mu = log_mu
        self.rho = rho
        assert rho>1 and rho<2, f'rho={rho} parameter needs to be between (1,2).'

        batch_shape = log_mu.size()
        super(Tweedie, self).__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        return torch.exp(self.log_mu)

    @property
    def variance(self):
        return torch.ones_line(self.log_mu) #TODO need to be assigned

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            mu   = self.mean
            rho  = self.rho * torch.ones_like(mu)
            sigma2 = 1 #TODO

            rate  = est_lambda(mu, rho) / sigma2  # rate for poisson
            alpha = est_alpha(rho)                # alpha for Gamma distribution
            beta  = est_beta(mu, rho) / sigma2    # beta for Gamma distribution
            
            # Expand for sample
            rate = rate.expand(shape)
            alpha = alpha.expand(shape)
            beta = beta.expand(shape)

            N = torch.poisson(rate)
            gamma = torch.distributions.gamma.Gamma(N*alpha, beta)
            samples = gamma.sample()
            samples[N==0] = 0

            return samples

    def log_prob(self, y_true):
        rho = self.rho
        y_pred = self.log_mu

        a = y_true * torch.exp((1 - rho) * y_pred) / (1 - rho)
        b = torch.exp((2 - rho) * y_pred) / (2 - rho)

        return a - b

def tweedie_domain_map(input: torch.Tensor):
    """ Tweedie Domain Map
    Maps input into distribution constraints, by construction input's 
    last dimension is of matching `distr_args` length.

    **Parameters:**<br>
    `input`: tensor, of dimensions [B,T,H,theta] or [B,H,theta].<br>

    **Returns:**<br>
    `(log_mu,)`: tuple with tensors of Tweedie distribution arguments.<br>
    """
    # log_mu, probs = torch.tensor_split(input, 2, dim=-1)
    return (input.squeeze(-1),)

def tweedie_scale_decouple(output, loc=None, scale=None):
    """ Tweedie Scale Decouple

    Stabilizes model's output optimization, by learning total
    count and logits based on anchoring `loc`, `scale`.
    Also adds Tweedie domain protection to the distribution parameters.
    """
    log_mu = output[0]
    if (loc is not None) and (scale is not None):
        log_mu += torch.log(loc) # TODO : rho scaling
    return (log_mu,)
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class DistributionLoss(torch.nn.Module):
    """ DistributionLoss

    This PyTorch module wraps the `torch.distribution` classes allowing it to 
    interact with NeuralForecast models modularly. It shares the negative 
    log-likelihood as the optimization objective and a sample method to 
    generate empirically the quantiles defined by the `level` list.

    Additionally, it implements a distribution transformation that factorizes the
    scale-dependent likelihood parameters into a base scale and a multiplier 
    efficiently learnable within the network's non-linearities operating ranges.

    Available distributions:<br>
    - Poisson<br>
    - Normal<br>
    - StudentT<br>
    - NegativeBinomial<br>
    - Tweedie<br>
    - Bernoulli (Temporal Classifiers)

    **Parameters:**<br>
    `distribution`: str, identifier of a torch.distributions.Distribution class.<br>
    `level`: float list [0,100], confidence levels for prediction intervals.<br>
    `quantiles`: float list [0,1], alternative to level list, target quantiles.<br>
    `num_samples`: int=500, number of samples for the empirical quantiles.<br>
    `return_params`: bool=False, wether or not return the Distribution parameters.<br><br>

    **References:**<br>
    - [PyTorch Probability Distributions Package: StudentT.](https://pytorch.org/docs/stable/distributions.html#studentt)<br>
    - [David Salinas, Valentin Flunkert, Jan Gasthaus, Tim Januschowski (2020).
       "DeepAR: Probabilistic forecasting with autoregressive recurrent networks". International Journal of Forecasting.](https://www.sciencedirect.com/science/article/pii/S0169207019301888)<br>
    """
    def __init__(self, distribution, level=[80, 90], quantiles=None,
                 num_samples=1000, return_params=False, **distribution_kwargs):
       super(DistributionLoss, self).__init__()

       available_distributions = dict(
                          Bernoulli=Bernoulli,
                          Normal=Normal,
                          Poisson=Poisson,
                          StudentT=StudentT,
                          NegativeBinomial=NegativeBinomial,
                          Tweedie=Tweedie)
       domain_maps = dict(Bernoulli=bernoulli_domain_map,
                          Normal=normal_domain_map,
                          Poisson=poisson_domain_map,
                          StudentT=student_domain_map,
                          NegativeBinomial=nbinomial_domain_map,
                          Tweedie=tweedie_domain_map)
       scale_decouples = dict(
                          Bernoulli=bernoulli_scale_decouple,
                          Normal=normal_scale_decouple,
                          Poisson=poisson_scale_decouple,
                          StudentT=student_scale_decouple,
                          NegativeBinomial=nbinomial_scale_decouple,
                          Tweedie=tweedie_scale_decouple)
       param_names = dict(Bernoulli=["-logits"],
                          Normal=["-loc", "-scale"],
                          Poisson=["-loc"],
                          StudentT=["-df", "-loc", "-scale"],
                          NegativeBinomial=["-total_count", "-logits"],
                          Tweedie=["-log_mu"])
       assert (distribution in available_distributions.keys()), f'{distribution} not available'

       self.distribution = distribution
       self._base_distribution = available_distributions[distribution]
       self.domain_map = domain_maps[distribution]
       self.scale_decouple = scale_decouples[distribution]
       self.param_names = param_names[distribution]

       self.distribution_kwargs = distribution_kwargs

       qs, self.output_names = level_to_outputs(level)
       qs = torch.Tensor(qs)

        # Transform quantiles to homogeneus output names
       if quantiles is not None:
              _, self.output_names = quantiles_to_outputs(quantiles)
              qs = torch.Tensor(quantiles)
       self.quantiles = torch.nn.Parameter(qs, requires_grad=False)
       self.num_samples = num_samples

       # If True, predict_step will return Distribution's parameters
       self.return_params = return_params
       if self.return_params:
            self.output_names = self.output_names + self.param_names

       # Add first output entry for the sample_mean
       self.output_names.insert(0, "")

       self.outputsize_multiplier = len(self.param_names)
       self.is_distribution_output = True

    def get_distribution(self, distr_args, **distribution_kwargs) -> Distribution:
        """
        Construct the associated Pytorch Distribution, given the collection of
        constructor arguments and, optionally, location and scale tensors.

        **Parameters**<br>
        `distr_args`: Constructor arguments for the underlying Distribution type.<br>

        **Returns**<br>
        `Distribution`: AffineTransformed distribution.<br>
        """
        # TransformedDistribution(distr, [AffineTransform(loc=loc, scale=scale)])
        distr = self._base_distribution(*distr_args, **distribution_kwargs)
        
        if self.distribution =='Poisson':
              distr.support = constraints.nonnegative
        return distr

    def sample(self,
               distr_args: torch.Tensor,
               num_samples: Optional[int] = None):
        """
        Construct the empirical quantiles from the estimated Distribution,
        sampling from it `num_samples` independently.

        **Parameters**<br>
        `distr_args`: Constructor arguments for the underlying Distribution type.<br>
        `loc`: Optional tensor, of the same shape as the batch_shape + event_shape
               of the resulting distribution.<br>
        `scale`: Optional tensor, of the same shape as the batch_shape+event_shape 
               of the resulting distribution.<br>
        `num_samples`: int=500, overwrite number of samples for the empirical quantiles.<br>

        **Returns**<br>
        `samples`: tensor, shape [B,H,`num_samples`].<br>
        `quantiles`: tensor, empirical quantiles defined by `levels`.<br>
        """
        if num_samples is None:
            num_samples = self.num_samples

        B, H = distr_args[0].size()
        Q = len(self.quantiles)

        # Instantiate Scaled Decoupled Distribution
        distr = self.get_distribution(distr_args=distr_args, **self.distribution_kwargs)
        samples = distr.sample(sample_shape=(num_samples,))
        samples = samples.permute(1,2,0) # [samples,B,H] -> [B,H,samples]
        samples = samples.to(distr_args[0].device)
        samples = samples.view(B*H, num_samples)
        sample_mean = torch.mean(samples, dim=-1)

        # Compute quantiles
        quantiles_device = self.quantiles.to(distr_args[0].device)
        quants = torch.quantile(input=samples, 
                                q=quantiles_device, dim=1)
        quants = quants.permute((1,0)) # [Q, B*H] -> [B*H, Q]

        # Final reshapes
        samples = samples.view(B, H, num_samples)
        sample_mean = sample_mean.view(B, H, 1)
        quants  = quants.view(B, H, Q)

        return samples, sample_mean, quants

    def __call__(self,
                 y: torch.Tensor,
                 distr_args: torch.Tensor,
                 mask: Union[torch.Tensor, None] = None):
        """
        Computes the negative log-likelihood objective function. 
        To estimate the following predictive distribution:

        $$\mathrm{P}(\mathbf{y}_{\\tau}\,|\,\\theta) \\quad \mathrm{and} \\quad -\log(\mathrm{P}(\mathbf{y}_{\\tau}\,|\,\\theta))$$

        where $\\theta$ represents the distributions parameters. It aditionally 
        summarizes the objective signal using a weighted average using the `mask` tensor. 

        **Parameters**<br>
        `y`: tensor, Actual values.<br>
        `distr_args`: Constructor arguments for the underlying Distribution type.<br>
        `loc`: Optional tensor, of the same shape as the batch_shape + event_shape
               of the resulting distribution.<br>
        `scale`: Optional tensor, of the same shape as the batch_shape+event_shape 
               of the resulting distribution.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns**<br>
        `loss`: scalar, weighted loss function against which backpropagation will be performed.<br>
        """
        # Instantiate Scaled Decoupled Distribution
        distr = self.get_distribution(distr_args=distr_args, **self.distribution_kwargs)
        loss_values = -distr.log_prob(y)
        loss_weights = mask
        return weighted_average(loss_values, weights=loss_weights)
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(DistributionLoss, name='DistributionLoss.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(DistributionLoss.sample, name='DistributionLoss.sample', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(DistributionLoss.__call__, name='DistributionLoss.__call__', title_level=3)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# Unit tests to check DistributionLoss' stored quantiles
# attribute is correctly instantiated
check = DistributionLoss(distribution='Normal', level=[80, 90])
test_eq(len(check.quantiles), 5)

check = DistributionLoss(distribution='Normal', 
                         quantiles=[0.0100, 0.1000, 0.5, 0.9000, 0.9900])
print(check.output_names)
print(check.quantiles)
test_eq(len(check.quantiles), 5)

check = DistributionLoss(distribution='Normal',
                         quantiles=[0.0100, 0.1000, 0.9000, 0.9900])
test_eq(len(check.quantiles), 4)
```

</details>

:::

## Poisson Mixture Mesh (PMM) {#poisson-mixture-mesh-pmm}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class PMM(torch.nn.Module):
    """ Poisson Mixture Mesh

    This Poisson Mixture statistical model assumes independence across groups of 
    data $\mathcal{G}=\{[g_{i}]\}$, and estimates relationships within the group.

    $$ \mathrm{P}\\left(\mathbf{y}_{[b][t+1:t+H]}\\right) = 
    \prod_{ [g_{i}] \in \mathcal{G}} \mathrm{P} \\left(\mathbf{y}_{[g_{i}][\\tau]} \\right) =
    \prod_{\\beta\in[g_{i}]} 
    \\left(\sum_{k=1}^{K} w_k \prod_{(\\beta,\\tau) \in [g_i][t+1:t+H]} \mathrm{Poisson}(y_{\\beta,\\tau}, \hat{\\lambda}_{\\beta,\\tau,k}) \\right)$$

    **Parameters:**<br>
    `n_components`: int=10, the number of mixture components.<br>
    `level`: float list [0,100], confidence levels for prediction intervals.<br>
    `quantiles`: float list [0,1], alternative to level list, target quantiles.<br>
    `return_params`: bool=False, wether or not return the Distribution parameters.<br>
    `batch_correlation`: bool=False, wether or not model batch correlations.<br>
    `horizon_correlation`: bool=False, wether or not model horizon correlations.<br>

    **References:**<br>
    [Kin G. Olivares, O. Nganba Meetei, Ruijun Ma, Rohan Reddy, Mengfei Cao, Lee Dicker. 
    Probabilistic Hierarchical Forecasting with Deep Poisson Mixtures. Submitted to the International 
    Journal Forecasting, Working paper available at arxiv.](https://arxiv.org/pdf/2110.13179.pdf)
    """
    def __init__(self, n_components=10, level=[80, 90], quantiles=None,
                 num_samples=1000, return_params=False,
                 batch_correlation=False, horizon_correlation=False):
        super(PMM, self).__init__()
        # Transform level to MQLoss parameters
        qs, self.output_names = level_to_outputs(level)
        qs = torch.Tensor(qs)

        # Transform quantiles to homogeneus output names
        if quantiles is not None:
            _, self.output_names = quantiles_to_outputs(quantiles)
            qs = torch.Tensor(quantiles)
        self.quantiles = torch.nn.Parameter(qs, requires_grad=False)
        self.num_samples = num_samples
        self.batch_correlation = batch_correlation
        self.horizon_correlation = horizon_correlation

        # If True, predict_step will return Distribution's parameters
        self.return_params = return_params
        if self.return_params:
            self.param_names = [f"-lambda-{i}" for i in range(1, n_components + 1)]
            self.output_names = self.output_names + self.param_names

        # Add first output entry for the sample_mean
        self.output_names.insert(0, "")

        self.outputsize_multiplier = n_components
        self.is_distribution_output = True

    def domain_map(self, output: torch.Tensor):
        return (output,)#, weights
        
    def scale_decouple(self, 
                       output,
                       loc: Optional[torch.Tensor] = None,
                       scale: Optional[torch.Tensor] = None):
        """ Scale Decouple

        Stabilizes model's output optimization, by learning residual
        variance and residual location based on anchoring `loc`, `scale`.
        Also adds domain protection to the distribution parameters.
        """
        lambdas = output[0]
        if (loc is not None) and (scale is not None):
            loc = loc.view(lambdas.size(dim=0), 1, -1)
            scale = scale.view(lambdas.size(dim=0), 1, -1)
            lambdas = (lambdas * scale) + loc
        lambdas = F.softplus(lambdas)
        return (lambdas,)

    def sample(self, distr_args, num_samples=None):
        """
        Construct the empirical quantiles from the estimated Distribution,
        sampling from it `num_samples` independently.

        **Parameters**<br>
        `distr_args`: Constructor arguments for the underlying Distribution type.<br>
        `loc`: Optional tensor, of the same shape as the batch_shape + event_shape
               of the resulting distribution.<br>
        `scale`: Optional tensor, of the same shape as the batch_shape+event_shape 
               of the resulting distribution.<br>
        `num_samples`: int=500, overwrites number of samples for the empirical quantiles.<br>

        **Returns**<br>
        `samples`: tensor, shape [B,H,`num_samples`].<br>
        `quantiles`: tensor, empirical quantiles defined by `levels`.<br>
        """
        if num_samples is None:
            num_samples = self.num_samples

        lambdas = distr_args[0]
        B, H, K = lambdas.size()
        Q = len(self.quantiles)

        # Sample K ~ Mult(weights)
        # shared across B, H
        # weights = torch.repeat_interleave(input=weights, repeats=H, dim=2)
        weights = (1/K) * torch.ones_like(lambdas).to(lambdas.device)

        # Avoid loop, vectorize
        weights = weights.reshape(-1, K)
        lambdas = lambdas.flatten()        

        # Vectorization trick to recover row_idx
        sample_idxs = torch.multinomial(input=weights, 
                                        num_samples=num_samples,
                                        replacement=True)
        aux_col_idx = torch.unsqueeze(torch.arange(B*H),-1) * K

        # To device
        sample_idxs = sample_idxs.to(lambdas.device)
        aux_col_idx = aux_col_idx.to(lambdas.device)

        sample_idxs = sample_idxs + aux_col_idx
        sample_idxs = sample_idxs.flatten()

        sample_lambdas = lambdas[sample_idxs]

        # Sample y ~ Poisson(lambda) independently
        samples = torch.poisson(sample_lambdas).to(lambdas.device)
        samples = samples.view(B*H, num_samples)
        sample_mean = torch.mean(samples, dim=-1)

        # Compute quantiles
        quantiles_device = self.quantiles.to(lambdas.device)
        quants = torch.quantile(input=samples, q=quantiles_device, dim=1)
        quants = quants.permute((1,0)) # Q, B*H

        # Final reshapes
        samples = samples.view(B, H, num_samples)
        sample_mean = sample_mean.view(B, H, 1)
        quants  = quants.view(B, H, Q)

        return samples, sample_mean, quants
    
    def neglog_likelihood(self,
                          y: torch.Tensor,
                          distr_args: Tuple[torch.Tensor],
                          mask: Union[torch.Tensor, None] = None,):
        if mask is None: 
            mask = (y > 0) * 1
        else:
            mask = mask * ((y > 0) * 1)

        eps  = 1e-10
        lambdas = distr_args[0]
        B, H, K = lambdas.size()

        weights = (1/K) * torch.ones_like(lambdas).to(lambdas.device)

        y = y[:,:,None]
        mask = mask[:,:,None]

        y = y * mask # Protect y negative entries
        
        # Single Poisson likelihood
        log_pi = y.xlogy(lambdas + eps) - lambdas - (y + 1).lgamma()

        if self.batch_correlation:
            log_pi  = torch.sum(log_pi, dim=0, keepdim=True)

        if self.horizon_correlation:
            log_pi  = torch.sum(log_pi, dim=1, keepdim=True)

        # Numerically Stable Mixture loglikelihood
        loglik = torch.logsumexp((torch.log(weights) + log_pi), dim=2, keepdim=True)
        loglik = loglik * mask

        mean   = torch.sum(weights * lambdas, axis=-1, keepdims=True)
        reglrz = torch.mean(torch.square(y - mean) * mask)
        loss   = -torch.mean(loglik) + 0.001 * reglrz
        return loss

    def __call__(self, y: torch.Tensor,
                 distr_args: Tuple[torch.Tensor],
                 mask: Union[torch.Tensor, None] = None):

        return self.neglog_likelihood(y=y, distr_args=distr_args, mask=mask)
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(PMM, name='PMM.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(PMM.sample, name='PMM.sample', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(PMM.__call__, name='PMM.__call__', title_level=3)
```

</details>

![](imgs_losses/pmm.png)

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# Unit tests to check PMM's stored quantiles
# attribute is correctly instantiated
check = PMM(n_components=2, level=[80, 90])
test_eq(len(check.quantiles), 5)

check = PMM(n_components=2, 
            quantiles=[0.0100, 0.1000, 0.5, 0.9000, 0.9900])
print(check.output_names)
print(check.quantiles)
test_eq(len(check.quantiles), 5)

check = PMM(n_components=2,
            quantiles=[0.0100, 0.1000, 0.9000, 0.9900])
test_eq(len(check.quantiles), 4)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# Create single mixture and broadcast to N,H,K
weights = torch.ones((1,3))[None, :, :]
lambdas = torch.Tensor([[5,10,15], [10,20,30]])[None, :, :]

# Create repetitions for the batch dimension N.
N=2
weights = torch.repeat_interleave(input=weights, repeats=N, dim=0)
lambdas = torch.repeat_interleave(input=lambdas, repeats=N, dim=0)

print('weights.shape (N,H,K) \t', weights.shape)
print('lambdas.shape (N,H,K) \t', lambdas.shape)

distr = PMM(quantiles=[0.1, 0.40, 0.5, 0.60, 0.9])
distr_args = (lambdas,)
samples, sample_mean, quants = distr.sample(distr_args)

print('samples.shape (N,H,num_samples) ', samples.shape)
print('sample_mean.shape (N,H) ', sample_mean.shape)
print('quants.shape  (N,H,Q) \t\t', quants.shape)

# Plot synthethic data
x_plot = range(quants.shape[1]) # H length
y_plot_hat = quants[0,:,:]  # Filter N,G,T -> H,Q
samples_hat = samples[0,:,:]  # Filter N,G,T -> H,num_samples

# Kernel density plot for single forecast horizon \tau = t+1
fig, ax = plt.subplots(figsize=(3.7, 2.9))

ax.hist(samples_hat[0,:], alpha=0.5, label=r'Horizon $\tau+1$')
ax.hist(samples_hat[1,:], alpha=0.5, label=r'Horizon $\tau+2$')
ax.set(xlabel='Y values', ylabel='Probability')
plt.title('Single horizon Distributions')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
plt.grid()
plt.show()
plt.close()

# Plot simulated trajectory
fig, ax = plt.subplots(figsize=(3.7, 2.9))
plt.plot(x_plot, y_plot_hat[:,2], color='black', label='median [q50]')
plt.fill_between(x_plot,
                 y1=y_plot_hat[:,1], y2=y_plot_hat[:,3],
                 facecolor='blue', alpha=0.4, label='[p25-p75]')
plt.fill_between(x_plot,
                 y1=y_plot_hat[:,0], y2=y_plot_hat[:,4],
                 facecolor='blue', alpha=0.2, label='[p1-p99]')
ax.set(xlabel='Horizon', ylabel='Y values')
plt.title('PMM Probabilistic Predictions')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
plt.grid()
plt.show()
plt.close()
```

</details>

:::

## Gaussian Mixture Mesh (GMM) {#gaussian-mixture-mesh-gmm}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class GMM(torch.nn.Module):
    """ Gaussian Mixture Mesh

    This Gaussian Mixture statistical model assumes independence across groups of 
    data $\mathcal{G}=\{[g_{i}]\}$, and estimates relationships within the group.

    $$ \mathrm{P}\\left(\mathbf{y}_{[b][t+1:t+H]}\\right) = 
    \prod_{ [g_{i}] \in \mathcal{G}} \mathrm{P}\left(\mathbf{y}_{[g_{i}][\\tau]}\\right)=
    \prod_{\\beta\in[g_{i}]}
    \\left(\sum_{k=1}^{K} w_k \prod_{(\\beta,\\tau) \in [g_i][t+1:t+H]} 
    \mathrm{Gaussian}(y_{\\beta,\\tau}, \hat{\mu}_{\\beta,\\tau,k}, \sigma_{\\beta,\\tau,k})\\right)$$

    **Parameters:**<br>
    `n_components`: int=10, the number of mixture components.<br>
    `level`: float list [0,100], confidence levels for prediction intervals.<br>
    `quantiles`: float list [0,1], alternative to level list, target quantiles.<br>
    `return_params`: bool=False, wether or not return the Distribution parameters.<br>
    `batch_correlation`: bool=False, wether or not model batch correlations.<br>
    `horizon_correlation`: bool=False, wether or not model horizon correlations.<br><br>

    **References:**<br>
    [Kin G. Olivares, O. Nganba Meetei, Ruijun Ma, Rohan Reddy, Mengfei Cao, Lee Dicker. 
    Probabilistic Hierarchical Forecasting with Deep Poisson Mixtures. Submitted to the International 
    Journal Forecasting, Working paper available at arxiv.](https://arxiv.org/pdf/2110.13179.pdf)
    """
    def __init__(self, n_components=1, level=[80, 90], quantiles=None, 
                 num_samples=1000, return_params=False,
                 batch_correlation=False, horizon_correlation=False):
        super(GMM, self).__init__()
        # Transform level to MQLoss parameters
        qs, self.output_names = level_to_outputs(level)
        qs = torch.Tensor(qs)

        # Transform quantiles to homogeneus output names
        if quantiles is not None:
            _, self.output_names = quantiles_to_outputs(quantiles)
            qs = torch.Tensor(quantiles)
        self.quantiles = torch.nn.Parameter(qs, requires_grad=False)
        self.num_samples = num_samples
        self.batch_correlation = batch_correlation
        self.horizon_correlation = horizon_correlation        

        # If True, predict_step will return Distribution's parameters
        self.return_params = return_params
        if self.return_params:
            mu_names = [f"-mu-{i}" for i in range(1, n_components + 1)]
            std_names = [f"-std-{i}" for i in range(1, n_components + 1)]
            mu_std_names = [i for j in zip(mu_names, std_names) for i in j]
            self.output_names = self.output_names + mu_std_names

        # Add first output entry for the sample_mean
        self.output_names.insert(0, "")

        self.outputsize_multiplier = 2 * n_components
        self.is_distribution_output = True

    def domain_map(self, output: torch.Tensor):
        means, stds = torch.tensor_split(output, 2, dim=-1)
        return (means, stds)

    def scale_decouple(self, 
                       output,
                       loc: Optional[torch.Tensor] = None,
                       scale: Optional[torch.Tensor] = None,
                       eps: float=0.2):
        """ Scale Decouple

        Stabilizes model's output optimization, by learning residual
        variance and residual location based on anchoring `loc`, `scale`.
        Also adds domain protection to the distribution parameters.
        """
        means, stds = output
        stds = F.softplus(stds)
        if (loc is not None) and (scale is not None):
            loc = loc.view(means.size(dim=0), 1, -1)
            scale = scale.view(means.size(dim=0), 1, -1)            
            means = (means * scale) + loc
            stds = (stds + eps) * scale
        return (means, stds)

    def sample(self, distr_args, num_samples=None):
        """
        Construct the empirical quantiles from the estimated Distribution,
        sampling from it `num_samples` independently.

        **Parameters**<br>
        `distr_args`: Constructor arguments for the underlying Distribution type.<br>
        `loc`: Optional tensor, of the same shape as the batch_shape + event_shape
               of the resulting distribution.<br>
        `scale`: Optional tensor, of the same shape as the batch_shape+event_shape 
               of the resulting distribution.<br>
        `num_samples`: int=500, number of samples for the empirical quantiles.<br>

        **Returns**<br>
        `samples`: tensor, shape [B,H,`num_samples`].<br>
        `quantiles`: tensor, empirical quantiles defined by `levels`.<br>
        """
        if num_samples is None:
            num_samples = self.num_samples
            
        means, stds = distr_args
        B, H, K = means.size()
        Q = len(self.quantiles)
        assert means.shape == stds.shape

        # Sample K ~ Mult(weights)
        # shared across B, H
        # weights = torch.repeat_interleave(input=weights, repeats=H, dim=2)
        
        weights = (1/K) * torch.ones_like(means).to(means.device)
        
        # Avoid loop, vectorize
        weights = weights.reshape(-1, K)
        means = means.flatten()
        stds = stds.flatten()

        # Vectorization trick to recover row_idx
        sample_idxs = torch.multinomial(input=weights, 
                                        num_samples=num_samples,
                                        replacement=True)
        aux_col_idx = torch.unsqueeze(torch.arange(B*H),-1) * K

        # To device
        sample_idxs = sample_idxs.to(means.device)
        aux_col_idx = aux_col_idx.to(means.device)

        sample_idxs = sample_idxs + aux_col_idx
        sample_idxs = sample_idxs.flatten()

        sample_means = means[sample_idxs]
        sample_stds  = stds[sample_idxs]

        # Sample y ~ Normal(mu, std) independently
        samples = torch.normal(sample_means, sample_stds).to(means.device)
        samples = samples.view(B*H, num_samples)
        sample_mean = torch.mean(samples, dim=-1)

        # Compute quantiles
        quantiles_device = self.quantiles.to(means.device)
        quants = torch.quantile(input=samples, q=quantiles_device, dim=1)
        quants = quants.permute((1,0)) # Q, B*H

        # Final reshapes
        samples = samples.view(B, H, num_samples)
        sample_mean = sample_mean.view(B, H, 1)
        quants  = quants.view(B, H, Q)

        return samples, sample_mean, quants

    def neglog_likelihood(self,
                          y: torch.Tensor,
                          distr_args: Tuple[torch.Tensor, torch.Tensor],
                          mask: Union[torch.Tensor, None] = None):

        if mask is None: 
            mask = torch.ones_like(y)
            
        means, stds = distr_args
        B, H, K = means.size()
        
        weights = (1/K) * torch.ones_like(means).to(means.device)
        
        y = y[:,:, None]
        mask = mask[:,:,None]
        
        var = stds ** 2
        log_stds = torch.log(stds)
        log_pi = - ((y - means) ** 2 / (2 * var)) - log_stds \
                 - math.log(math.sqrt(2 * math.pi))

        if self.batch_correlation:
            log_pi  = torch.sum(log_pi, dim=0, keepdim=True)

        if self.horizon_correlation:    
            log_pi  = torch.sum(log_pi, dim=1, keepdim=True)

        # Numerically Stable Mixture loglikelihood
        loglik = torch.logsumexp((torch.log(weights) + log_pi), dim=2, keepdim=True)
        loglik  = loglik * mask

        loss = -torch.mean(loglik)
        return loss
    
    def __call__(self, y: torch.Tensor,
                 distr_args: Tuple[torch.Tensor, torch.Tensor],
                 mask: Union[torch.Tensor, None] = None,):

        return self.neglog_likelihood(y=y, distr_args=distr_args, mask=mask)
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(GMM, name='GMM.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(GMM.sample, name='GMM.sample', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(GMM.__call__, name='GMM.__call__', title_level=3)
```

</details>

![](imgs_losses/gmm.png)

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# Unit tests to check PMM's stored quantiles
# attribute is correctly instantiated
check = GMM(n_components=2, level=[80, 90])
test_eq(len(check.quantiles), 5)

check = GMM(n_components=2, 
            quantiles=[0.0100, 0.1000, 0.5, 0.9000, 0.9900])
print(check.output_names)
print(check.quantiles)
test_eq(len(check.quantiles), 5)

check = GMM(n_components=2,
            quantiles=[0.0100, 0.1000, 0.9000, 0.9900])
test_eq(len(check.quantiles), 4)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# Create single mixture and broadcast to N,H,K
means   = torch.Tensor([[5,10,15], [10,20,30]])[None, :, :]

# # Create repetitions for the batch dimension N.
N=2
means = torch.repeat_interleave(input=means, repeats=N, dim=0)
weights = torch.ones_like(means)
stds  = torch.ones_like(means)

print('weights.shape (N,H,K) \t', weights.shape)
print('means.shape (N,H,K) \t', means.shape)
print('stds.shape (N,H,K) \t', stds.shape)

distr = GMM(quantiles=[0.1, 0.40, 0.5, 0.60, 0.9])
distr_args = (means, stds)
samples, sample_mean, quants = distr.sample(distr_args)

print('samples.shape (N,H,num_samples) ', samples.shape)
print('sample_mean.shape (N,H) ', sample_mean.shape)
print('quants.shape  (N,H,Q) \t\t', quants.shape)

# Plot synthethic data
x_plot = range(quants.shape[1]) # H length
y_plot_hat = quants[0,:,:]  # Filter N,G,T -> H,Q
samples_hat = samples[0,:,:]  # Filter N,G,T -> H,num_samples

# Kernel density plot for single forecast horizon \tau = t+1
fig, ax = plt.subplots(figsize=(3.7, 2.9))

ax.hist(samples_hat[0,:], alpha=0.5, bins=50,
        label=r'Horizon $\tau+1$')
ax.hist(samples_hat[1,:], alpha=0.5, bins=50,
        label=r'Horizon $\tau+2$')
ax.set(xlabel='Y values', ylabel='Probability')
plt.title('Single horizon Distributions')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
plt.grid()
plt.show()
plt.close()

# Plot simulated trajectory
fig, ax = plt.subplots(figsize=(3.7, 2.9))
plt.plot(x_plot, y_plot_hat[:,2], color='black', label='median [q50]')
plt.fill_between(x_plot,
                 y1=y_plot_hat[:,1], y2=y_plot_hat[:,3],
                 facecolor='blue', alpha=0.4, label='[p25-p75]')
plt.fill_between(x_plot,
                 y1=y_plot_hat[:,0], y2=y_plot_hat[:,4],
                 facecolor='blue', alpha=0.2, label='[p1-p99]')
ax.set(xlabel='Horizon', ylabel='Y values')
plt.title('GMM Probabilistic Predictions')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
plt.grid()
plt.show()
plt.close()
```

</details>

:::

## Negative Binomial Mixture Mesh (NBMM) {#negative-binomial-mixture-mesh-nbmm}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class NBMM(torch.nn.Module):
    """ Negative Binomial Mixture Mesh

    This N. Binomial Mixture statistical model assumes independence across groups of 
    data $\mathcal{G}=\{[g_{i}]\}$, and estimates relationships within the group.

    $$ \mathrm{P}\\left(\mathbf{y}_{[b][t+1:t+H]}\\right) = 
    \prod_{ [g_{i}] \in \mathcal{G}} \mathrm{P}\left(\mathbf{y}_{[g_{i}][\\tau]}\\right)=
    \prod_{\\beta\in[g_{i}]}
    \\left(\sum_{k=1}^{K} w_k \prod_{(\\beta,\\tau) \in [g_i][t+1:t+H]} 
    \mathrm{NBinomial}(y_{\\beta,\\tau}, \hat{r}_{\\beta,\\tau,k}, \hat{p}_{\\beta,\\tau,k})\\right)$$

    **Parameters:**<br>
    `n_components`: int=10, the number of mixture components.<br>
    `level`: float list [0,100], confidence levels for prediction intervals.<br>
    `quantiles`: float list [0,1], alternative to level list, target quantiles.<br>
    `return_params`: bool=False, wether or not return the Distribution parameters.<br><br>

    **References:**<br>
    [Kin G. Olivares, O. Nganba Meetei, Ruijun Ma, Rohan Reddy, Mengfei Cao, Lee Dicker. 
    Probabilistic Hierarchical Forecasting with Deep Poisson Mixtures. Submitted to the International 
    Journal Forecasting, Working paper available at arxiv.](https://arxiv.org/pdf/2110.13179.pdf)
    """
    def __init__(self, n_components=1, level=[80, 90], quantiles=None, 
                 num_samples=1000, return_params=False):
        super(NBMM, self).__init__()
        # Transform level to MQLoss parameters
        qs, self.output_names = level_to_outputs(level)
        qs = torch.Tensor(qs)

        # Transform quantiles to homogeneus output names
        if quantiles is not None:
            _, self.output_names = quantiles_to_outputs(quantiles)
            qs = torch.Tensor(quantiles)
        self.quantiles = torch.nn.Parameter(qs, requires_grad=False)
        self.num_samples = num_samples

        # If True, predict_step will return Distribution's parameters
        self.return_params = return_params
        if self.return_params:
            total_count_names = [f"-total_count-{i}" for i in range(1, n_components + 1)]
            probs_names = [f"-probs-{i}" for i in range(1, n_components + 1)]
            param_names = [i for j in zip(total_count_names, probs_names) for i in j]
            self.output_names = self.output_names + param_names

        # Add first output entry for the sample_mean
        self.output_names.insert(0, "")            

        self.outputsize_multiplier = 2 * n_components
        self.is_distribution_output = True

    def domain_map(self, output: torch.Tensor):
        mu, alpha = torch.tensor_split(output, 2, dim=-1)
        return (mu, alpha)

    def scale_decouple(self, 
                       output,
                       loc: Optional[torch.Tensor] = None,
                       scale: Optional[torch.Tensor] = None,
                       eps: float=0.2):
        """ Scale Decouple

        Stabilizes model's output optimization, by learning residual
        variance and residual location based on anchoring `loc`, `scale`.
        Also adds domain protection to the distribution parameters.
        """
        # Efficient NBinomial parametrization
        mu, alpha = output
        mu = F.softplus(mu) + 1e-8
        alpha = F.softplus(alpha) + 1e-8    # alpha = 1/total_counts
        if (loc is not None) and (scale is not None):
            loc = loc.view(mu.size(dim=0), 1, -1)
            mu *= loc
            alpha /= (loc + 1.)

        # mu = total_count * (probs/(1-probs))
        # => probs = mu / (total_count + mu)
        # => probs = mu / [total_count * (1 + mu * (1/total_count))]
        total_count = 1.0 / alpha
        probs = (mu * alpha / (1.0 + mu * alpha)) + 1e-8 
        return (total_count, probs)

    def sample(self, distr_args, num_samples=None):
        """
        Construct the empirical quantiles from the estimated Distribution,
        sampling from it `num_samples` independently.

        **Parameters**<br>
        `distr_args`: Constructor arguments for the underlying Distribution type.<br>
        `loc`: Optional tensor, of the same shape as the batch_shape + event_shape
               of the resulting distribution.<br>
        `scale`: Optional tensor, of the same shape as the batch_shape+event_shape 
               of the resulting distribution.<br>
        `num_samples`: int=500, number of samples for the empirical quantiles.<br>

        **Returns**<br>
        `samples`: tensor, shape [B,H,`num_samples`].<br>
        `quantiles`: tensor, empirical quantiles defined by `levels`.<br>
        """
        if num_samples is None:
            num_samples = self.num_samples
            
        total_count, probs = distr_args
        B, H, K = total_count.size()
        Q = len(self.quantiles)
        assert total_count.shape == probs.shape

        # Sample K ~ Mult(weights)
        # shared across B, H
        # weights = torch.repeat_interleave(input=weights, repeats=H, dim=2)
        
        weights = (1/K) * torch.ones_like(probs).to(probs.device)
        
        # Avoid loop, vectorize
        weights = weights.reshape(-1, K)
        total_count = total_count.flatten()
        probs = probs.flatten()

        # Vectorization trick to recover row_idx
        sample_idxs = torch.multinomial(input=weights, 
                                        num_samples=num_samples,
                                        replacement=True)
        aux_col_idx = torch.unsqueeze(torch.arange(B*H),-1) * K

        # To device
        sample_idxs = sample_idxs.to(probs.device)
        aux_col_idx = aux_col_idx.to(probs.device)

        sample_idxs = sample_idxs + aux_col_idx
        sample_idxs = sample_idxs.flatten()

        sample_total_count = total_count[sample_idxs]
        sample_probs  = probs[sample_idxs]

        # Sample y ~ NBinomial(total_count, probs) independently
        dist = NegativeBinomial(total_count=sample_total_count, 
                                probs=sample_probs)
        samples = dist.sample(sample_shape=(1,)).to(probs.device)[0]
        samples = samples.view(B*H, num_samples)
        sample_mean = torch.mean(samples, dim=-1)

        # Compute quantiles
        quantiles_device = self.quantiles.to(probs.device)
        quants = torch.quantile(input=samples, q=quantiles_device, dim=1)
        quants = quants.permute((1,0)) # Q, B*H

        # Final reshapes
        samples = samples.view(B, H, num_samples)
        sample_mean = sample_mean.view(B, H, 1)
        quants  = quants.view(B, H, Q)

        return samples, sample_mean, quants

    def neglog_likelihood(self,
                          y: torch.Tensor,
                          distr_args: Tuple[torch.Tensor, torch.Tensor],
                          mask: Union[torch.Tensor, None] = None):

        if mask is None: 
            mask = torch.ones_like(y)
            
        total_count, probs = distr_args
        B, H, K = total_count.size()
        
        weights = (1/K) * torch.ones_like(probs).to(probs.device)
        
        y = y[:,:, None]
        mask = mask[:,:,None]

        log_unnormalized_prob = (total_count * torch.log(1.-probs) + y * torch.log(probs))
        log_normalization = (-torch.lgamma(total_count + y) + torch.lgamma(1. + y) +
                             torch.lgamma(total_count))
        log_normalization[total_count + y == 0.] = 0.
        log =  log_unnormalized_prob - log_normalization

        #log  = torch.sum(log, dim=0, keepdim=True) # Joint within batch/group
        #log  = torch.sum(log, dim=1, keepdim=True) # Joint within horizon

        # Numerical stability mixture and loglik
        log_max = torch.amax(log, dim=2, keepdim=True) # [1,1,K] (collapsed joints)
        lik     = weights * torch.exp(log-log_max)     # Take max
        loglik  = torch.log(torch.sum(lik, dim=2, keepdim=True)) + log_max # Return max
        
        loglik  = loglik * mask #replace with mask

        loss = -torch.mean(loglik)
        return loss
    
    def __call__(self, y: torch.Tensor,
                 distr_args: Tuple[torch.Tensor, torch.Tensor],
                 mask: Union[torch.Tensor, None] = None,):

        return self.neglog_likelihood(y=y, distr_args=distr_args, mask=mask)
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(NBMM, name='NBMM.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(NBMM.sample, name='NBMM.sample', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(NBMM.__call__, name='NBMM.__call__', title_level=3)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# Create single mixture and broadcast to N,H,K
counts   = torch.Tensor([[10,20,30], [20,40,60]])[None, :, :]

# # Create repetitions for the batch dimension N.
N=2
counts = torch.repeat_interleave(input=counts, repeats=N, dim=0)
weights = torch.ones_like(counts)
probs  = torch.ones_like(counts) * 0.5

print('weights.shape (N,H,K) \t', weights.shape)
print('counts.shape (N,H,K) \t', counts.shape)
print('probs.shape (N,H,K) \t', probs.shape)

model = NBMM(quantiles=[0.1, 0.40, 0.5, 0.60, 0.9])
distr_args = (counts, probs)
samples, sample_mean, quants = model.sample(distr_args, num_samples=2000)

print('samples.shape (N,H,num_samples) ', samples.shape)
print('sample_mean.shape (N,H) ', sample_mean.shape)
print('quants.shape  (N,H,Q) \t\t', quants.shape)

# Plot synthethic data
x_plot = range(quants.shape[1]) # H length
y_plot_hat = quants[0,:,:]  # Filter N,G,T -> H,Q
samples_hat = samples[0,:,:]  # Filter N,G,T -> H,num_samples

# Kernel density plot for single forecast horizon \tau = t+1
fig, ax = plt.subplots(figsize=(3.7, 2.9))

ax.hist(samples_hat[0,:], alpha=0.5, bins=30,
        label=r'Horizon $\tau+1$')
ax.hist(samples_hat[1,:], alpha=0.5, bins=30,
        label=r'Horizon $\tau+2$')
ax.set(xlabel='Y values', ylabel='Probability')
plt.title('Single horizon Distributions')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
plt.grid()
plt.show()
plt.close()

# Plot simulated trajectory
fig, ax = plt.subplots(figsize=(3.7, 2.9))
plt.plot(x_plot, y_plot_hat[:,2], color='black', label='median [q50]')
plt.fill_between(x_plot,
                 y1=y_plot_hat[:,1], y2=y_plot_hat[:,3],
                 facecolor='blue', alpha=0.4, label='[p25-p75]')
plt.fill_between(x_plot,
                 y1=y_plot_hat[:,0], y2=y_plot_hat[:,4],
                 facecolor='blue', alpha=0.2, label='[p1-p99]')
ax.set(xlabel='Horizon', ylabel='Y values')
plt.title('NBM Probabilistic Predictions')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
plt.grid()
plt.show()
plt.close()
```

</details>

:::

# <span style="color:DarkBlue"> 5. Robustified Errors </span> {#robustified-errors}

This type of errors from robust statistic focus on methods resistant to
outliers and violations of assumptions, providing reliable estimates and
inferences. Robust estimators are used to reduce the impact of outliers,
offering more stable results.

## Huber Loss {#huber-loss}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class HuberLoss(torch.nn.Module):
    """ Huber Loss

    The Huber loss, employed in robust regression, is a loss function that 
    exhibits reduced sensitivity to outliers in data when compared to the 
    squared error loss. This function is also refered as SmoothL1.

    The Huber loss function is quadratic for small errors and linear for large 
    errors, with equal values and slopes of the different sections at the two 
    points where $(y_{\\tau}-\hat{y}_{\\tau})^{2}$=$|y_{\\tau}-\hat{y}_{\\tau}|$.

    $$ L_{\delta}(y_{\\tau},\; \hat{y}_{\\tau})
    =\\begin{cases}{\\frac{1}{2}}(y_{\\tau}-\hat{y}_{\\tau})^{2}\;{\\text{for }}|y_{\\tau}-\hat{y}_{\\tau}|\leq \delta \\\ 
    \\delta \ \cdot \left(|y_{\\tau}-\hat{y}_{\\tau}|-{\\frac {1}{2}}\delta \\right),\;{\\text{otherwise.}}\end{cases}$$

    where $\\delta$ is a threshold parameter that determines the point at which the loss transitions from quadratic to linear,
    and can be tuned to control the trade-off between robustness and accuracy in the predictions.

    **Parameters:**<br>
    `delta`: float=1.0, Specifies the threshold at which to change between delta-scaled L1 and L2 loss.

    **References:**<br>
    [Huber Peter, J (1964). "Robust Estimation of a Location Parameter". Annals of Statistics](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full)
    """
    def __init__(self, delta: float=1.):
        super(HuberLoss, self).__init__()
        self.outputsize_multiplier = 1
        self.delta = delta
        self.output_names = ['']
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, 
                 mask: Union[torch.Tensor, None] = None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `huber_loss`: tensor (single value).
        """
        if mask is None: 
            mask = torch.ones_like(y_hat)

        huber_loss = F.huber_loss(y * mask, y_hat * mask,
                                  reduction='mean', delta=self.delta)
        return huber_loss
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(HuberLoss, name='HuberLoss.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(HuberLoss.__call__, name='HuberLoss.__call__', title_level=3)
```

</details>

![](imgs_losses/huber_loss.png)

## Tukey Loss {#tukey-loss}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class TukeyLoss(torch.nn.Module):
    """ Tukey Loss

    The Tukey loss function, also known as Tukey's biweight function, is a 
    robust statistical loss function used in robust statistics. Tukey's loss exhibits
    quadratic behavior near the origin, like the Huber loss; however, it is even more
    robust to outliers as the loss for large residuals remains constant instead of 
    scaling linearly.

    The parameter $c$ in Tukey's loss determines the ''saturation'' point
    of the function: Higher values of $c$ enhance sensitivity, while lower values 
    increase resistance to outliers.

    $$ L_{c}(y_{\\tau},\; \hat{y}_{\\tau})
    =\\begin{cases}{
    \\frac{c^{2}}{6}} \\left[1-(\\frac{y_{\\tau}-\hat{y}_{\\tau}}{c})^{2} \\right]^{3}    \;\\text{for } |y_{\\tau}-\hat{y}_{\\tau}|\leq c \\\ 
    \\frac{c^{2}}{6} \qquad \\text{otherwise.}  \end{cases}$$

    Please note that the Tukey loss function assumes the data to be stationary or
    normalized beforehand. If the error values are excessively large, the algorithm
    may need help to converge during optimization. It is advisable to employ small learning rates.

    **Parameters:**<br>
    `c`: float=4.685, Specifies the Tukey loss' threshold on which residuals are no longer considered.<br>
    `normalize`: bool=True, Wether normalization is performed within Tukey loss' computation.<br>

    **References:**<br>
    [Beaton, A. E., and Tukey, J. W. (1974). "The Fitting of Power Series, Meaning Polynomials, Illustrated on Band-Spectroscopic Data."](https://www.jstor.org/stable/1267936)
    """
    def __init__(self, c: float=4.685, normalize: bool=True):
        super(TukeyLoss, self).__init__()
        self.outputsize_multiplier = 1
        self.c = c
        self.normalize = normalize
        self.output_names = ['']
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def masked_mean(self, x, mask, dim):
        x_nan = x.masked_fill(mask < 1, float("nan"))
        x_mean = x_nan.nanmean(dim=dim, keepdim=True)
        x_mean = torch.nan_to_num(x_mean, nan=0.0)
        return x_mean

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, 
                 mask: Union[torch.Tensor, None] = None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `tukey_loss`: tensor (single value).
        """
        if mask is None:
            mask = torch.ones_like(y_hat)

        # We normalize the Tukey loss, to satisfy 4.685 normal outlier bounds
        if self.normalize:
            y_mean = self.masked_mean(x=y, mask=mask, dim=-1)
            y_std = torch.sqrt(self.masked_mean(x=(y - y_mean) ** 2, mask=mask, dim=-1)) + 1e-2
        else:
            y_std = 1.
        delta_y = torch.abs(y - y_hat) / y_std

        tukey_mask = torch.greater_equal(self.c * torch.ones_like(delta_y), delta_y)
        tukey_loss = tukey_mask * mask * (1-(delta_y/(self.c))**2)**3 + (1-(tukey_mask * 1))
        tukey_loss = (self.c**2 / 6) * torch.mean(tukey_loss)
        return tukey_loss
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(TukeyLoss, name='TukeyLoss.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(TukeyLoss.__call__, name='TukeyLoss.__call__', title_level=3)
```

</details>

![](imgs_losses/tukey_loss.png)

## Huberized Quantile Loss {#huberized-quantile-loss}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class HuberQLoss(torch.nn.Module):
    """ Huberized Quantile Loss

    The Huberized quantile loss is a modified version of the quantile loss function that
    combines the advantages of the quantile loss and the Huber loss. It is commonly used
    in regression tasks, especially when dealing with data that contains outliers or heavy tails.

    The Huberized quantile loss between `y` and `y_hat` measure the Huber Loss in a non-symmetric way.
    The loss pays more attention to under/over-estimation depending on the quantile parameter $q$; 
    and controls the trade-off between robustness and accuracy in the predictions with the parameter $delta$.

    $$ \mathrm{HuberQL}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}^{(q)}_{\\tau}) = 
    (1-q)\, L_{\delta}(y_{\\tau},\; \hat{y}^{(q)}_{\\tau}) \mathbb{1}\{ \hat{y}^{(q)}_{\\tau} \geq y_{\\tau} \} + 
    q\, L_{\delta}(y_{\\tau},\; \hat{y}^{(q)}_{\\tau}) \mathbb{1}\{ \hat{y}^{(q)}_{\\tau} < y_{\\tau} \} $$

    **Parameters:**<br>
    `delta`: float=1.0, Specifies the threshold at which to change between delta-scaled L1 and L2 loss.<br>
    `q`: float, between 0 and 1. The slope of the quantile loss, in the context of quantile regression, the q determines the conditional quantile level.<br>

    **References:**<br>
    [Huber Peter, J (1964). "Robust Estimation of a Location Parameter". Annals of Statistics](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full)<br>
    [Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles".](https://www.jstor.org/stable/1913643)
    """
    def __init__(self, q:float, delta: float=1.):
        super(HuberQLoss, self).__init__()
        self.q = q
        self.delta = delta
        self.outputsize_multiplier = 1
        self.output_names = [f'_q{q}_d{delta}']
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, 
                 mask: Union[torch.Tensor, None] = None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `huber_qloss`: tensor (single value).
        """
        if mask is None: 
            mask = torch.ones_like(y_hat)
        
        error  = y_hat - y
        zero_error = torch.zeros_like(error)
        sq     = torch.maximum(-error, zero_error)
        s1_q   = torch.maximum(error, zero_error)
        hqloss = self.q * F.huber_loss(sq * mask, zero_error * mask, 
                                        reduction='mean', delta=self.delta) + \
                 (1 - self.q) * F.huber_loss(s1_q * mask, zero_error * mask, 
                                        reduction='mean', delta=self.delta)
        return hqloss
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(HuberQLoss, name='HuberQLoss.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(HuberQLoss.__call__, name='HuberQLoss.__call__', title_level=3)
```

</details>

![](imgs_losses/huber_qloss.png)

## Huberized MQLoss {#huberized-mqloss}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class HuberMQLoss(torch.nn.Module):
    """  Huberized Multi-Quantile loss

    The Huberized Multi-Quantile loss (HuberMQL) is a modified version of the multi-quantile loss function 
    that combines the advantages of the quantile loss and the Huber loss. HuberMQL is commonly used in regression 
    tasks, especially when dealing with data that contains outliers or heavy tails. The loss function pays 
    more attention to under/over-estimation depending on the quantile list $[q_{1},q_{2},\dots]$ parameter. 
    It controls the trade-off between robustness and prediction accuracy with the parameter $\\delta$.

    $$ \mathrm{HuberMQL}_{\delta}(\\mathbf{y}_{\\tau},[\\mathbf{\hat{y}}^{(q_{1})}_{\\tau}, ... ,\hat{y}^{(q_{n})}_{\\tau}]) = 
    \\frac{1}{n} \\sum_{q_{i}} \mathrm{HuberQL}_{\\delta}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}^{(q_{i})}_{\\tau}) $$

    **Parameters:**<br>
    `level`: int list [0,100]. Probability levels for prediction intervals (Defaults median).
    `quantiles`: float list [0., 1.]. Alternative to level, quantiles to estimate from y distribution.
    `delta`: float=1.0, Specifies the threshold at which to change between delta-scaled L1 and L2 loss.<br>    

    **References:**<br>
    [Huber Peter, J (1964). "Robust Estimation of a Location Parameter". Annals of Statistics](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full)<br>
    [Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles".](https://www.jstor.org/stable/1913643)
    """
    def __init__(self, level=[80, 90], quantiles=None, delta: float=1.0):
        super(HuberMQLoss, self).__init__()
        # Transform level to MQLoss parameters
        qs, self.output_names = level_to_outputs(level)
        qs = torch.Tensor(qs)

        # Transform quantiles to homogeneus output names
        if quantiles is not None:
            _, self.output_names = quantiles_to_outputs(quantiles)
            qs = torch.Tensor(quantiles)

        self.delta = delta
        self.quantiles = torch.nn.Parameter(qs, requires_grad=False)
        self.outputsize_multiplier = len(self.quantiles)
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Identity domain map [B,T,H,Q]/[B,H,Q]
        """
        return y_hat

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, 
                 mask: Union[torch.Tensor, None] = None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `mqloss`: tensor (single value).
        """
        if mask is None: 
            mask = torch.ones_like(y_hat)

        n_q = len(self.quantiles)

        mask = mask.unsqueeze(-1)
        error  = y_hat - y.unsqueeze(-1)
        zero_error = torch.zeros_like(error)        
        sq     = torch.maximum(-error, torch.zeros_like(error))
        s1_q   = torch.maximum(error, torch.zeros_like(error))
        hmqloss = F.huber_loss(self.quantiles * sq * mask, zero_error * mask, 
                                        reduction='mean', delta=self.delta) + \
                  F.huber_loss((1 - self.quantiles) * s1_q * mask, zero_error * mask, 
                                reduction='mean', delta=self.delta)

        # Match y/weights dimensions and compute weighted average
        mask = mask / torch.sum(mask)
        
        hmqloss = (1/n_q) * hmqloss * mask
        return torch.sum(hmqloss)
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(HuberMQLoss, name='HuberMQLoss.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(HuberMQLoss.__call__, name='HuberMQLoss.__call__', title_level=3)
```

</details>

![](imgs_losses/hmq_loss.png)

# <span style="color:DarkBlue"> 6. Others </span> {#others}

## Accuracy {#accuracy}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class Accuracy(torch.nn.Module):
    """ Accuracy

    Computes the accuracy between categorical `y` and `y_hat`.
    This evaluation metric is only meant for evalution, as it
    is not differentiable.

    $$ \mathrm{Accuracy}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} \mathrm{1}\{\\mathbf{y}_{\\tau}==\\mathbf{\hat{y}}_{\\tau}\} $$

    """
    def __init__(self,):
        super(Accuracy, self).__init__()
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, 
                 mask: Union[torch.Tensor, None] = None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `accuracy`: tensor (single value).
        """
        if mask is None:
            mask = torch.ones_like(y_hat)

        measure = (y.unsqueeze(-1) == y_hat) * mask.unsqueeze(-1)
        accuracy = torch.mean(measure)
        return accuracy
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(Accuracy, name='Accuracy.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(Accuracy.__call__, name='Accuracy.__call__', title_level=3)
```

</details>

## Scaled Continuous Ranked Probability Score (sCRPS) {#scaled-continuous-ranked-probability-score-scrps}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class sCRPS(torch.nn.Module):
    """Scaled Continues Ranked Probability Score

    Calculates a scaled variation of the CRPS, as proposed by Rangapuram (2021),
    to measure the accuracy of predicted quantiles `y_hat` compared to the observation `y`.

    This metric averages percentual weighted absolute deviations as 
    defined by the quantile losses.

    $$ \mathrm{sCRPS}(\\mathbf{\hat{y}}^{(q)}_{\\tau}, \mathbf{y}_{\\tau}) = \\frac{2}{N} \sum_{i}
    \int^{1}_{0}
    \\frac{\mathrm{QL}(\\mathbf{\hat{y}}^{(q}_{\\tau} y_{i,\\tau})_{q}}{\sum_{i} | y_{i,\\tau} |} dq $$

    where $\\mathbf{\hat{y}}^{(q}_{\\tau}$ is the estimated quantile, and $y_{i,\\tau}$
    are the target variable realizations.

    **Parameters:**<br>
    `level`: int list [0,100]. Probability levels for prediction intervals (Defaults median).
    `quantiles`: float list [0., 1.]. Alternative to level, quantiles to estimate from y distribution.

    **References:**<br>
    - [Gneiting, Tilmann. (2011). \"Quantiles as optimal point forecasts\". 
    International Journal of Forecasting.](https://www.sciencedirect.com/science/article/pii/S0169207010000063)<br>
    - [Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos, Zhi Chen, Anil Gaba, Ilia Tsetlin, Robert L. Winkler. (2022). 
    \"The M5 uncertainty competition: Results, findings and conclusions\". 
    International Journal of Forecasting.](https://www.sciencedirect.com/science/article/pii/S0169207021001722)<br>
    - [Syama Sundar Rangapuram, Lucien D Werner, Konstantinos Benidis, Pedro Mercado, Jan Gasthaus, Tim Januschowski. (2021). 
    \"End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series\". 
    Proceedings of the 38th International Conference on Machine Learning (ICML).](https://proceedings.mlr.press/v139/rangapuram21a.html)
    """
    def __init__(self, level=[80, 90], quantiles=None):
        super(sCRPS, self).__init__()
        self.mql = MQLoss(level=level, quantiles=quantiles)
        self.is_distribution_output = False
    
    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, 
                 mask: Union[torch.Tensor, None] = None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per series to consider in loss.<br>

        **Returns:**<br>
        `scrps`: tensor (single value).
        """
        mql = self.mql(y=y, y_hat=y_hat, mask=mask)
        norm = torch.sum(torch.abs(y))
        unmean = torch.sum(mask)
        scrps = 2 * mql * unmean / (norm + 1e-5)
        return scrps
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(sCRPS, name='sCRPS.__init__', title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(sCRPS.__call__, name='sCRPS.__call__', title_level=3)
```

</details>

