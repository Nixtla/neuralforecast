


__all__ = ['mae', 'mse', 'rmse', 'mape', 'smape', 'mase', 'rmae', 'quantile_loss', 'mqloss',
           'excess_volatility', 'forecast_percentage_change', 'cross_validation_to_windows']


import re
from typing import List, NamedTuple, Optional, Tuple, Union

import numpy as np


def _divide_no_nan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Auxiliary function to handle divide by 0
    """
    div = a / b
    div[div != div] = 0.0
    div[div == float("inf")] = 0.0
    return div


def _metric_protections(
    y: np.ndarray, y_hat: np.ndarray, weights: Optional[np.ndarray]
) -> None:
    assert (weights is None) or (np.sum(weights) > 0), "Sum of weights cannot be 0"
    assert (weights is None) or (
        weights.shape == y.shape
    ), f"Wrong weight dimension weights.shape {weights.shape}, y.shape {y.shape}"


def mae(
    y: np.ndarray,
    y_hat: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    r"""Mean Absolute Error

    Calculates Mean Absolute Error between
    `y` and `y_hat`. MAE measures the relative prediction
    accuracy of a forecasting method by calculating the
    deviation of the prediction and the true
    value at a given time and averages these devations
    over the length of the series.

    ```math
    \mathrm{MAE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}_{\tau}) = \frac{1}{H} \sum^{t+H}_{\tau=t+1} |y_{\tau} - \hat{y}_{\tau}|
    ```


    Args:
        y (np.ndarray): Actual values.
        y_hat (np.ndarray): Predicted values.
        mask (np.ndarray, optional): Specifies date stamps per serie to consider in loss. Defaults to None.

    Returns:
        float: MAE.
    """
    _metric_protections(y, y_hat, weights)

    delta_y = np.abs(y - y_hat)
    if weights is not None:
        mae = np.average(
            delta_y[~np.isnan(delta_y)], weights=weights[~np.isnan(delta_y)], axis=axis
        )
    else:
        mae = np.nanmean(delta_y, axis=axis)

    return mae


def mse(
    y: np.ndarray,
    y_hat: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    r"""Mean Squared Error

    Calculates Mean Squared Error between
    `y` and `y_hat`. MSE measures the relative prediction
    accuracy of a forecasting method by calculating the
    squared deviation of the prediction and the true
    value at a given time, and averages these devations
    over the length of the series.

    ```math
    \mathrm{MSE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}_{\tau}) = \frac{1}{H} \sum^{t+H}_{\tau=t+1} (y_{\tau} - \hat{y}_{\tau})^{2}
    ```

    Args:
        y (np.ndarray): Actual values.
        y_hat (np.ndarray): Predicted values.
        mask (np.ndarray, optional): Specifies date stamps per serie to consider in loss. Defaults to None.

    Returns:
        float: MSE.
    """
    _metric_protections(y, y_hat, weights)

    delta_y = np.square(y - y_hat)
    if weights is not None:
        mse = np.average(
            delta_y[~np.isnan(delta_y)], weights=weights[~np.isnan(delta_y)], axis=axis
        )
    else:
        mse = np.nanmean(delta_y, axis=axis)

    return mse


def rmse(
    y: np.ndarray,
    y_hat: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    r"""Root Mean Squared Error

    Calculates Root Mean Squared Error between
    `y` and `y_hat`. RMSE measures the relative prediction
    accuracy of a forecasting method by calculating the squared deviation
    of the prediction and the observed value at a given time and
    averages these devations over the length of the series.
    Finally the RMSE will be in the same scale
    as the original time series so its comparison with other
    series is possible only if they share a common scale.
    RMSE has a direct connection to the L2 norm.

    ```math
    \mathrm{RMSE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}_{\tau}) = \sqrt{\frac{1}{H} \sum^{t+H}_{\tau=t+1} (y_{\tau} - \hat{y}_{\tau})^{2}}
    ```


    Args:
        y (np.ndarray): Actual values.
        y_hat (np.ndarray): Predicted values.
        mask (np.ndarray, optional): Specifies date stamps per serie to consider in loss. Defaults to None.

    Returns:
        float: RMSE.
    """
    return np.sqrt(mse(y, y_hat, weights, axis))


def mape(
    y: np.ndarray,
    y_hat: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    r"""Mean Absolute Percentage Error

    Calculates Mean Absolute Percentage Error  between
    `y` and `y_hat`. MAPE measures the relative prediction
    accuracy of a forecasting method by calculating the percentual deviation
    of the prediction and the observed value at a given time and
    averages these devations over the length of the series.
    The closer to zero an observed value is, the higher penalty MAPE loss
    assigns to the corresponding error.

    ```math
    \mathrm{MAPE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}_{\tau}) = \frac{1}{H} \sum^{t+H}_{\tau=t+1} \frac{|y_{\tau}-\hat{y}_{\tau}|}{|y_{\tau}|}
    ```


    Args:
        y (np.ndarray): Actual values.
        y_hat (np.ndarray): Predicted values.
        mask (np.ndarray, optional): Specifies date stamps per serie to consider in loss. Defaults to None.

    Returns:
        float: MAPE.
    """
    _metric_protections(y, y_hat, weights)

    delta_y = np.abs(y - y_hat)
    scale = np.abs(y)
    mape = _divide_no_nan(delta_y, scale)
    mape = np.average(mape, weights=weights, axis=axis)

    return mape


def smape(
    y: np.ndarray,
    y_hat: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    r"""Symmetric Mean Absolute Percentage Error

    Calculates Symmetric Mean Absolute Percentage Error between
    `y` and `y_hat`. SMAPE measures the relative prediction
    accuracy of a forecasting method by calculating the relative deviation
    of the prediction and the observed value scaled by the sum of the
    absolute values for the prediction and observed value at a
    given time, then averages these devations over the length
    of the series. This allows the SMAPE to have bounds between
    0% and 200% which is desirable compared to normal MAPE that
    may be undetermined when the target is zero.

    ```math
    \mathrm{sMAPE}_{2}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}_{\tau}) = \frac{1}{H} \sum^{t+H}_{\tau=t+1} \frac{|y_{\tau}-\hat{y}_{\tau}|}{|y_{\tau}|+|\hat{y}_{\tau}|}
    ```


    Args:
        y (np.ndarray): Actual values.
        y_hat (np.ndarray): Predicted values.
        mask (np.ndarray, optional): Specifies date stamps per serie to consider in loss. Defaults to None.

    Returns:
        float: SMAPE.

    References:
        - [Makridakis S., "Accuracy measures: theoretical and practical concerns".](https://www.sciencedirect.com/science/article/pii/0169207093900793)
    """
    _metric_protections(y, y_hat, weights)

    delta_y = np.abs(y - y_hat)
    scale = np.abs(y) + np.abs(y_hat)
    smape = _divide_no_nan(delta_y, scale)
    smape = 2 * np.average(smape, weights=weights, axis=axis)

    if np.isscalar(smape):
        assert smape <= 2, "SMAPE should be lower than 200"
    else:
        assert all(smape <= 2), "SMAPE should be lower than 200"

    return smape


def mase(
    y: np.ndarray,
    y_hat: np.ndarray,
    y_train: np.ndarray,
    seasonality: int,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    r"""Mean Absolute Scaled Error
    Calculates the Mean Absolute Scaled Error between
    `y` and `y_hat`. MASE measures the relative prediction
    accuracy of a forecasting method by comparinng the mean absolute errors
    of the prediction and the observed value against the mean
    absolute errors of the seasonal naive model.
    The MASE partially composed the Overall Weighted Average (OWA),
    used in the M4 Competition.

    ```math
    \mathrm{MASE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}_{\tau}, \mathbf{\hat{y}}^{season}_{\tau}) = \frac{1}{H} \sum^{t+H}_{\tau=t+1} \frac{|y_{\tau}-\hat{y}_{\tau}|}{\mathrm{MAE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}^{season}_{\tau})}
    ```


    Args:
        y (np.ndarray): Actual values.
        y_hat (np.ndarray): Predicted values.
        y_insample (np.ndarray): Actual insample Seasonal Naive predictions.
        seasonality (int): Main frequency of the time series; Hourly 24,  Daily 7, Weekly 52, Monthly 12, Quarterly 4, Yearly 1.
        mask (np.ndarray, optional): Specifies date stamps per serie to consider in loss. Defaults to None.

    Returns:
        float: MASE.

    References:
        - [Rob J. Hyndman, & Koehler, A. B. "Another look at measures of forecast accuracy".](https://www.sciencedirect.com/science/article/pii/S0169207006000239)
        - [Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos, "The M4 Competition: 100,000 time series and 61 forecasting methods".](https://www.sciencedirect.com/science/article/pii/S0169207019301128)
    """
    delta_y = np.abs(y - y_hat)
    delta_y = np.average(delta_y, weights=weights, axis=axis)

    scale = np.abs(y_train[:-seasonality] - y_train[seasonality:])
    scale = np.average(scale, axis=axis)

    mase = delta_y / scale

    return mase


def rmae(
    y: np.ndarray,
    y_hat1: np.ndarray,
    y_hat2: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    r"""RMAE

    Calculates Relative Mean Absolute Error (RMAE) between
    two sets of forecasts (from two different forecasting methods).
    A number smaller than one implies that the forecast in the
    numerator is better than the forecast in the denominator.

    ```math
    \mathrm{rMAE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}_{\tau}, \mathbf{\hat{y}}^{base}_{\tau}) = \frac{1}{H} \sum^{t+H}_{\tau=t+1} \frac{|y_{\tau}-\hat{y}_{\tau}|}{\mathrm{MAE}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}^{base}_{\tau})}
    ```

    Args:
        y (np.ndarray): observed values.
        y_hat1 (np.ndarray): Predicted values of first model.
        y_hat2 (np.ndarray): Predicted values of baseline model.
        weights (np.ndarray, optional): Weights for weighted average. Defaults to None.
        axis (Optional[int], optional): Axis or axes along which to average a. Defaults to None.
        The default, axis=None, will average over all of the elements of
        the input array.

    Returns:
        float: RMAE.

    References:
        - [Rob J. Hyndman, & Koehler, A. B. "Another look at measures of forecast accuracy".](https://www.sciencedirect.com/science/article/pii/S0169207006000239)
    """
    numerator = mae(y=y, y_hat=y_hat1, weights=weights, axis=axis)
    denominator = mae(y=y, y_hat=y_hat2, weights=weights, axis=axis)
    rmae = numerator / denominator

    return rmae


def quantile_loss(
    y: np.ndarray,
    y_hat: np.ndarray,
    q: float = 0.5,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    r"""Quantile Loss

    Computes the quantile loss between `y` and `y_hat`.
    QL measures the deviation of a quantile forecast.
    By weighting the absolute deviation in a non symmetric way, the
    loss pays more attention to under or over estimation.
    A common value for q is 0.5 for the deviation from the median (Pinball loss).

    ```math
    \mathrm{QL}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}^{(q)}_{\tau}) = \frac{1}{H} \sum^{t+H}_{\tau=t+1} \Big( (1-q)\,( \hat{y}^{(q)}_{\tau} - y_{\tau} )_{+} + q\,( y_{\tau} - \hat{y}^{(q)}_{\tau} )_{+} \Big)
    ```


    Args:
        y (np.ndarray): Actual values.
        y_hat (np.ndarray): Predicted values.
        q (float, optional): The slope of the quantile loss, in the context of quantile regression, the q determines the conditional quantile level. Defaults to 0.5.
        mask (np.ndarray, optional): Specifies date stamps per serie to consider in loss. Defaults to None.

    Returns:
        float: Quantile loss.

    References:
        - [Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles".](https://www.jstor.org/stable/1913643)
    """
    _metric_protections(y, y_hat, weights)

    delta_y = y - y_hat
    loss = np.maximum(q * delta_y, (q - 1) * delta_y)

    if weights is not None:
        quantile_loss = np.average(
            loss[~np.isnan(loss)], weights=weights[~np.isnan(loss)], axis=axis
        )
    else:
        quantile_loss = np.nanmean(loss, axis=axis)

    return quantile_loss


def mqloss(
    y: np.ndarray,
    y_hat: np.ndarray,
    quantiles: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    r"""Multi-Quantile loss

    Calculates the Multi-Quantile loss (MQL) between `y` and `y_hat`.
    MQL calculates the average multi-quantile Loss for
    a given set of quantiles, based on the absolute
    difference between predicted quantiles and observed values.

    ```math
    \mathrm{MQL}(\mathbf{y}_{\tau},[\mathbf{\hat{y}}^{(q_{1})}_{\tau}, ... ,\hat{y}^{(q_{n})}_{\tau}]) = \frac{1}{n} \sum_{q_{i}} \mathrm{QL}(\mathbf{y}_{\tau}, \mathbf{\hat{y}}^{(q_{i})}_{\tau})
    ```


    The limit behavior of MQL allows to measure the accuracy
    of a full predictive distribution $\mathbf{\hat{F}}_{\tau}$ with
    the continuous ranked probability score (CRPS). This can be achieved
    through a numerical integration technique, that discretizes the quantiles
    and treats the CRPS integral with a left Riemann approximation, averaging over
    uniformly distanced quantiles.

    ```math
    \mathrm{CRPS}(y_{\tau}, \mathbf{\hat{F}}_{\tau}) = \int^{1}_{0} \mathrm{QL}(y_{\tau}, \hat{y}^{(q)}_{\tau}) dq
    ```


    Args:
        y (np.ndarray): Actual values.
        y_hat (np.ndarray): Predicted values.
        quantiles (np.ndarray): Quantiles to estimate from the distribution of y.
        mask (np.ndarray, optional): Specifies date stamps per serie to consider in loss. Defaults to None.

    Returns:
        float: MQLoss.

    References:
        - [Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles".](https://www.jstor.org/stable/1913643)
    [James E. Matheson and Robert L. Winkler, "Scoring Rules for Continuous Probability Distributions".](https://www.jstor.org/stable/2629907)
    """
    if weights is None:
        weights = np.ones(y.shape)

    _metric_protections(y, y_hat, weights)
    n_q = len(quantiles)

    y_rep = np.expand_dims(y, axis=-1)
    error = y_hat - y_rep
    sq = np.maximum(-error, np.zeros_like(error))
    s1_q = np.maximum(error, np.zeros_like(error))
    mqloss = quantiles * sq + (1 - quantiles) * s1_q

    # Match y/weights dimensions and compute weighted average
    weights = np.repeat(np.expand_dims(weights, axis=-1), repeats=n_q, axis=-1)
    mqloss = np.average(mqloss, weights=weights, axis=axis)

    return mqloss


def _validate_stride(stride: int, h: int) -> None:
    """Reject strides for which forecast volatility is undefined.

    Args:
        stride (int): Step size between consecutive forecast windows.
        h (int): Forecast horizon.

    Raises:
        ValueError: If `stride` is not positive, or if `stride >= h`.
    """
    if stride < 1:
        raise ValueError(f"stride={stride} must be a positive integer.")
    if stride == h:
        raise ValueError(
            f"stride={stride} equals h={h}: windows are non-overlapping so each "
            "target date has exactly one forecast. Forecast volatility is undefined "
            "in this case."
        )
    if stride > h:
        raise ValueError(
            f"stride={stride} > h={h}: some target dates have no forecast coverage. "
            "Please review the step size used to generate the forecast windows."
        )


def _reshape_windows_by_date(
    x: np.ndarray,
    stride: int,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Regroup overlapping forecast windows by the date they target.

    Rearranges `[B, T, H, C]` into `[B, (T-1)*stride + H, H, C]`, collecting into a
    single row every forecast that targets the same date.

    In a rolling forecast setup multiple windows overlap on the same target date: the
    1-step-ahead prediction from window `t` and the 2-step-ahead prediction from window
    `t-1` both target date `t`. Grouping them enables a direct comparison of forecasts
    that share a target.

    The output has `(T-1)*stride + H` rows (one per unique target date) and `H` columns
    (one per horizon that could reach that date). Edge dates are only partially
    observed and hold `np.nan` for the horizons that cannot reach them.

    Args:
        x (np.ndarray): Array of shape `[B, T, H, C]` to regroup.
        stride (int): Step size between consecutive forecast windows.
        mask (np.ndarray, optional): Array of shape `[B, T, H, C]`. Masked positions
            (`mask == 0`) are set to `np.nan` before regrouping. Defaults to None.

    Returns:
        np.ndarray: Array of shape `[B, (T-1)*stride + H, H, C]`.
    """
    b, t, h, c = x.shape
    s = (t - 1) * stride + h

    if mask is not None:
        x = np.where(mask == 0, np.nan, x)

    t_grid, h_grid = np.meshgrid(np.arange(t), np.arange(h), indexing="ij")
    d_grid = t_grid * stride + h_grid  # [T, H] target date of each (t, h) pair
    out = np.full((b, s, h, c), np.nan, dtype=float)
    out[:, d_grid, h_grid, :] = x  # scatter [B, T, H, C] -> [B, S, H, C]
    return out


def _pinball_loss(
    y: np.ndarray,
    y_hat: np.ndarray,
    quantiles: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Element-wise pinball loss, without aggregation.

    Unlike `neuralforecast.losses.numpy.quantile_loss`, this helper returns the
    unreduced loss so that excess volatility can combine three loss terms before any
    averaging, and it accepts a per-quantile target so that one quantile forecast can
    be scored against another.

    Args:
        y (np.ndarray): Target values, either `[..., C]` (a point target, broadcast
            across quantiles) or `[..., C, Q]` (a per-quantile target).
        y_hat (np.ndarray): Quantile predictions of shape `[..., C, Q]`.
        quantiles (np.ndarray): Quantile levels of shape `[Q]`, each in `(0, 1)`.
        mask (np.ndarray, optional): Array of shape `[..., C]`. Positions where the
            mask is 0 contribute 0 to the loss. Defaults to None.

    Returns:
        np.ndarray: Unreduced loss of shape `[..., C, Q]`.
    """
    if y.ndim < y_hat.ndim:
        errors = y[..., np.newaxis] - y_hat  # point target, shared across quantiles
    else:
        errors = y - y_hat  # per-quantile pairing

    q = np.asarray(quantiles, dtype=errors.dtype)
    loss = np.maximum(q * errors, (q - 1) * errors)

    if mask is not None:
        loss = loss * np.broadcast_to(mask[..., np.newaxis], loss.shape)
    return loss


def excess_volatility(
    y: np.ndarray,
    y_hat: np.ndarray,
    quantiles: List[float],
    stride: int = 1,
    scaling: bool = True,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> float:
    r"""Scaled Excess Volatility (sEV)

    Measures *harmful* forecast volatility, by charging a forecast revision for its
    cost and crediting it for the accuracy it buys. A multi-horizon system run on a
    schedule issues several overlapping forecasts for the same target date, one per
    forecast creation date (FCD); this penalises only the revisions that move a
    forecast away from the truth, or that overshoot it, rewarding accuracy-improving
    revisions while separating them from harmful volatility.

    Indexing by series $b$, FCD $t$ and horizon step $h$, the forecast
    $\hat{\mathbf{y}}_{b,t,h+1}$ issued at FCD $t$ and the forecast
    $\hat{\mathbf{y}}_{b,t+1,h}$ issued one FCD later land on the same target date,
    the second being a revision of the first:

    $$ \mathrm{sEV}\left(\mathbf{y}_{[b][t][h]},\; \hat{\mathbf{y}}_{[b][t][h]} \right) = \frac{\sum_{b,t,h}\mathrm{EV}(y_{b,t,h},\;\mathbf{\hat{y}}_{b,t,h+1},\;\mathbf{\hat{y}}_{b,t+1,h})}{\sum_{b,t,h}|y_{b,t,h}|} $$

    $$ \mathrm{EV}(y,\;\mathbf{\hat{y}}_1,\; \mathbf{\hat{y}}_2) = \mathrm{QL}(\mathbf{\hat{y}}_2,\mathbf{\hat{y}}_1) - (\mathrm{QL}(y,\mathbf{\hat{y}}_1)-\mathrm{QL}(y,\mathbf{\hat{y}}_2)) $$

    where $\mathrm{QL}$ is the quantile loss at level $q \in \mathcal{Q}$:

    $$ \mathrm{QL}_q(y, \hat{y}^{(q)}) = q(y-\hat{y}^{(q)})_+ + (1-q)(\hat{y}^{(q)}-y)_+ $$

    The first EV term is the *revision cost*, how far the older forecast sits from the
    newer one; the bracketed term is the *accuracy improvement* the revision produced.
    Setting `scaling=False` returns the unscaled numerator, $\sum \mathrm{EV}$.

    Args:
        y (np.ndarray): Target values of shape `[B, T, H, C]`, for `B` series, `T`
            forecast windows, horizon `H` and `C` target channels.
        y_hat (np.ndarray): Quantile predictions of shape `[B, T, H, C, Q]`.
        quantiles (list of floats): Quantile levels, e.g. `[0.1, 0.5, 0.9]`. Must match
            the size of the last axis of `y_hat`.
        stride (int): Step size between consecutive forecast windows. Must be smaller
            than `H`, otherwise a target date has at most one forecast and volatility
            is undefined. Defaults to 1.
        scaling (bool): If True, divide by the sum of `|y|` over the compared pairs, so
            the result is scale independent and comparable across series. Defaults to
            True.
        mask (np.ndarray, optional): Array of shape `[B, T, H, C]`, 1 for a real
            timestep and 0 for a padded or missing one. Independently of this, the
            structurally unreachable horizons at the edge dates are always excluded.
            Defaults to None.
        eps (float): Floor added to the scaling denominator. Defaults to 1e-8.

    Returns:
        float: Excess volatility. Values above 0 indicate revisions that cost more than
        the accuracy they bought.

    Raises:
        ValueError: If `stride >= H`, or if `quantiles` does not match `y_hat`.

    References:
        - [Willa Potosnak, Malcolm Wolff, Mengfei Cao, Ruijun Ma, Tatiana
          Konstantinova, Dmitry Efimov, Michael W. Mahoney, Boris Oreshkin, Kin G.
          Olivares, "Forking-Sequences: Statistically and Computationally Efficient
          Multi-Horizon Forecasting with Reduced Volatility". Transactions on
          Machine Learning Research (2026).](https://openreview.net/forum?id=dXdycy7WCX)
    """
    b, t, h, c, q_size = y_hat.shape
    _validate_stride(stride, h)

    if len(quantiles) != q_size:
        raise ValueError(
            f"len(quantiles)={len(quantiles)} does not match the last axis of y_hat "
            f"(Q={q_size})."
        )
    if y.shape != (b, t, h, c):
        raise ValueError(
            f"y.shape={y.shape} is incompatible with y_hat.shape={y_hat.shape}; "
            f"expected {(b, t, h, c)}."
        )

    s = (t - 1) * stride + h
    quantiles_arr = np.asarray(quantiles, dtype=float)

    reshaped_y_hat = _reshape_windows_by_date(
        x=y_hat.reshape(b, t, h, c * q_size),
        stride=stride,
        mask=np.repeat(mask, q_size, axis=-1) if mask is not None else None,
    )
    reshaped_y = _reshape_windows_by_date(x=y, stride=stride)
    reshaped_mask = _reshape_windows_by_date(
        x=mask if mask is not None else np.ones_like(y, dtype=float),
        stride=stride,
    )

    # A larger horizon index means the forecast was made from an earlier window.
    y_hat_before = reshaped_y_hat[:, :, stride:, :]  # [B, S, K, C*Q]
    y_hat_update = reshaped_y_hat[:, :, :-stride, :]  # [B, S, K, C*Q]
    reshaped_y = reshaped_y[:, :, stride:, :]  # [B, S, K, C]

    # Only compare a date when both members of the pair are present.
    pair_mask = np.logical_and(
        np.nan_to_num(reshaped_mask[:, :, stride:, :], nan=0.0),
        np.nan_to_num(reshaped_mask[:, :, :-stride, :], nan=0.0),
    ).astype(float)

    k = h - stride
    y_hat_before = np.nan_to_num(y_hat_before, nan=0.0).reshape(b, s, k, c, q_size)
    y_hat_update = np.nan_to_num(y_hat_update, nan=0.0).reshape(b, s, k, c, q_size)
    reshaped_y = np.nan_to_num(reshaped_y, nan=0.0)

    # Divide by Q so each term is a mean over quantiles rather than a sum.
    revision_cost = (
        _pinball_loss(
            y=y_hat_update,  # the newer forecast acts as the target
            y_hat=y_hat_before,
            quantiles=quantiles_arr,
            mask=pair_mask,
        )
        / q_size
    )
    accuracy_before = (
        _pinball_loss(
            y=reshaped_y,
            y_hat=y_hat_before,
            quantiles=quantiles_arr,
            mask=pair_mask,
        )
        / q_size
    )
    accuracy_update = (
        _pinball_loss(
            y=reshaped_y,
            y_hat=y_hat_update,
            quantiles=quantiles_arr,
            mask=pair_mask,
        )
        / q_size
    )

    ev = float((revision_cost - (accuracy_before - accuracy_update)).sum())

    if not scaling:
        return ev
    denom = float(np.sum(np.abs(reshaped_y) * pair_mask)) + eps
    return ev / denom


def forecast_percentage_change(
    y_hat: np.ndarray,
    stride: int = 1,
    symmetric: bool = True,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> float:
    r"""Scaled Forecast Percentage Change (sFPC)

    Measures the relative change in predicted quantiles across consecutive forecast
    creation dates (FCDs), giving a quantitative view of the forecast revision rate.
    Unlike `excess_volatility` it treats every revision as equally undesirable, even
    ones that improve accuracy, so the two are best read together.

    Indexing by series $b$, FCD $t$ and horizon step $h$, the forecasts
    $\hat{Y}^{(q)}_{b,t,h+1}$ and $\hat{Y}^{(q)}_{b,t+1,h}$ land on the same target
    date, the second being a revision of the first:

    $$ \mathrm{sFPC}_{q}\left(\hat{\mathbf{Y}}^{(q)}_{{[b][t][h]}} \right) = \frac{200}{B \times T \times H} \sum_{b,t,h} \frac{|\hat{Y}^{(q)}_{b,t+1,h}-\hat{Y}^{(q)}_{b,t,h+1}|}{|\hat{Y}^{(q)}_{b,t+1,h}| + |\hat{Y}^{(q)}_{b,t,h+1}|} $$

    Inspired by sMAPE, the denominator is symmetric in the two forecasts. This keeps
    the metric well behaved when predicted values are small and avoids the
    division-by-zero problems common to traditional percentage-based metrics. A small
    `eps`, omitted from the equation above, is added to the denominator as a guard.

    Setting `symmetric=False` instead returns the one-sided variant, which divides by
    $|\hat{Y}^{(q)}_{b,t,h+1}|$ alone and applies no factor of 200. The two are
    different quantities, not two scalings of one: sFPC is bounded in `[0, 200]` and
    symmetric in the two forecasts, while FPC is unbounded and measures revisions
    relative to the earlier forecast only. Higher values mean a more volatile
    forecaster in both cases.

    Args:
        y_hat (np.ndarray): Point predictions of shape `[B, T, H, C]`, for `B` series,
            `T` forecast windows, horizon `H` and `C` target channels. To score a
            quantile forecast, pass the median slice.
        stride (int): Step size between consecutive forecast windows. Must be smaller
            than `H`, otherwise a target date has at most one forecast and volatility
            is undefined. Defaults to 1.
        symmetric (bool): If True, compute sFPC, using the symmetric denominator and
            scaling by 200. If False, compute FPC, dividing by `|y_hat_before|` alone
            and applying no scaling. Defaults to True.
        mask (np.ndarray, optional): Array of shape `[B, T, H, C]`, 1 for a real
            timestep and 0 for a padded or missing one. Masked positions are dropped
            from the mean. Defaults to None.
        eps (float): Floor added to the denominator, guarding against division by zero
            on series that approach zero. Defaults to 1e-6.

    Returns:
        float: sFPC when `symmetric` is True, otherwise FPC. Returns `np.nan` when no
        pair of forecasts survives the mask.

    Raises:
        ValueError: If `stride >= H`.

    References:
        - [Willa Potosnak, Malcolm Wolff, Mengfei Cao, Ruijun Ma, Tatiana
          Konstantinova, Dmitry Efimov, Michael W. Mahoney, Boris Oreshkin, Kin G.
          Olivares, "Forking-Sequences: Statistically and Computationally Efficient
          Multi-Horizon Forecasting with Reduced Volatility". Transactions on
          Machine Learning Research (2026).](https://openreview.net/forum?id=dXdycy7WCX)
    """
    if y_hat.ndim != 4:
        raise ValueError(
            f"y_hat must have shape [B, T, H, C]; got {y_hat.ndim} dimensions."
        )
    h = y_hat.shape[2]
    _validate_stride(stride, h)

    reshaped_y_hat = _reshape_windows_by_date(x=y_hat, stride=stride, mask=mask)

    # A larger horizon index means the forecast was made from an earlier window.
    y_hat_before = reshaped_y_hat[:, :, stride:, :]  # [B, S, H-stride, C]
    y_hat_update = reshaped_y_hat[:, :, :-stride, :]  # [B, S, H-stride, C]

    num = np.abs(y_hat_update - y_hat_before)
    if symmetric:
        den = np.abs(y_hat_update) + np.abs(y_hat_before) + eps
        scale = 200.0
    else:
        den = np.abs(y_hat_before) + eps
        scale = 1.0

    ratio = num / den
    if np.isnan(ratio).all():
        # every pair was masked out or structurally unreachable
        return float("nan")
    return float(scale * np.nanmean(ratio))


class ForecastWindows(NamedTuple):
    """Dense forecast windows extracted from a cross validation DataFrame.

    Attributes:
        y (np.ndarray): Target values of shape `[B, T, H, 1]`.
        y_hat (np.ndarray): Quantile predictions of shape `[B, T, H, 1, Q]`.
        quantiles (list of floats): Quantile levels matching the last axis of `y_hat`.
        mask (np.ndarray): Array of shape `[B, T, H, 1]`, 1 where the target and every
            quantile prediction are present and 0 elsewhere.
        stride (int): Step size between consecutive forecast windows.
    """

    y: np.ndarray
    y_hat: np.ndarray
    quantiles: List[float]
    mask: np.ndarray
    stride: int


_LO_HI_RE = re.compile(r"^-(lo|hi)-(\d+(?:\.\d+)?)$")


def _parse_quantile_columns(
    columns: List[str], model: str
) -> Tuple[List[float], List[str]]:
    """Recover quantile levels from a model's cross validation output columns.

    `NeuralForecast` names probabilistic outputs `{model}-median`, `{model}-lo-{level}`
    and `{model}-hi-{level}`; this inverts that naming.

    Args:
        columns (list of str): Columns of the cross validation DataFrame.
        model (str): Model name, as it appears in those columns.

    Returns:
        tuple: The sorted quantile levels and their matching column names.

    Raises:
        ValueError: If no quantile column is found for `model`.
    """
    found = []
    for col in columns:
        if not col.startswith(model):
            continue
        suffix = col[len(model) :]
        if suffix == "-median":
            found.append((0.5, col))
            continue
        match = _LO_HI_RE.match(suffix)
        if match is None:
            continue
        side, level = match.group(1), float(match.group(2))
        # level_to_outputs maps level L to the quantile pair (100-L)/200, (100+L)/200
        found.append((((100 - level) if side == "lo" else (100 + level)) / 200, col))

    if not found:
        raise ValueError(
            f"No quantile columns found for model '{model}'. Excess volatility needs a "
            f"probabilistic forecast: run cross_validation with `level=` or "
            f"`quantiles=`, or with a probabilistic loss such as MQLoss."
        )

    found.sort(key=lambda pair: pair[0])
    return [q for q, _ in found], [col for _, col in found]


def _dense_rank_within_group(group: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Rank each value among the distinct values seen within its own group.

    Args:
        group (np.ndarray): Integer group code per row.
        value (np.ndarray): Integer value code per row.

    Returns:
        np.ndarray: For each row, the 0-based rank of its value among the distinct
        values of its group.
    """
    n_values = int(value.max()) + 1
    pair_codes = group.astype(np.int64) * n_values + value.astype(np.int64)
    uniq_pairs, inverse = np.unique(pair_codes, return_inverse=True)
    # np.unique sorts, so the distinct pairs are ordered by group and then by value.
    pair_group = uniq_pairs // n_values
    group_start = np.searchsorted(pair_group, np.arange(int(group.max()) + 1))
    return inverse - group_start[group]


def cross_validation_to_windows(
    df,
    model: str,
    step_size: Optional[int] = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> ForecastWindows:
    """Reshape `NeuralForecast.cross_validation` output into dense forecast windows.

    `cross_validation` returns a long DataFrame with one row per (series, cutoff,
    timestamp). The volatility metrics need those rows as dense
    `[B, T, H, C, Q]` arrays, with one row per forecast window, so that the forecasts
    targeting a shared date can be compared. This performs that reshape and recovers
    the quantile levels from the model's output column names.

    Because cross validation forecasts a single target column, the channel axis `C` is
    always 1. Pass multivariate forecasts to the metrics directly as arrays.

    Args:
        df (pandas or polars DataFrame): Output of `NeuralForecast.cross_validation`,
            containing `id_col`, `time_col`, `cutoff`, `target_col` and the model's
            quantile columns.
        model (str): Model name, as it appears in the forecast columns.
        step_size (int, optional): The `step_size` used for cross validation. If None,
            it is inferred from the spacing of the cutoffs. Defaults to None.
        id_col (str): Column identifying each series. Defaults to 'unique_id'.
        time_col (str): Column identifying each timestep. Defaults to 'ds'.
        target_col (str): Column containing the target. Defaults to 'y'.

    Returns:
        ForecastWindows: The dense `y`, `y_hat`, `quantiles`, `mask` and `stride`, ready
        to pass to `excess_volatility` or `forecast_percentage_change`.

    Raises:
        ValueError: If a required column is missing, if no quantile column is found for
            `model`, if the windows are not evenly spaced, or if a passed `step_size`
            disagrees with the cutoffs.

    Example:
        >>> from neuralforecast.losses.numpy import (
        ...     cross_validation_to_windows,
        ...     excess_volatility,
        ...     forecast_percentage_change,
        ... )
        >>> cv_df = nf.cross_validation(df, n_windows=10, step_size=1)  # doctest: +SKIP
        >>> w = cross_validation_to_windows(cv_df, model='NHITS')  # doctest: +SKIP
        >>> excess_volatility(  # doctest: +SKIP
        ...     y=w.y, y_hat=w.y_hat, quantiles=w.quantiles,
        ...     stride=w.stride, mask=w.mask,
        ... )
        >>> forecast_percentage_change(  # doctest: +SKIP
        ...     y_hat=w.y_hat[..., w.quantiles.index(0.5)],
        ...     stride=w.stride, mask=w.mask,
        ... )
    """
    columns = list(df.columns)
    required = [id_col, time_col, "cutoff", target_col]
    missing = [col for col in required if col not in columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) {missing} in the cross validation DataFrame. "
            f"Expected the output of NeuralForecast.cross_validation."
        )

    quantiles, quantile_cols = _parse_quantile_columns(columns, model)

    # Rows are placed by index rather than by position, so order does not matter.
    ids = np.asarray(df[id_col].to_numpy())
    times = np.asarray(df[time_col].to_numpy())
    cutoffs = np.asarray(df["cutoff"].to_numpy())

    _, b_idx = np.unique(ids, return_inverse=True)
    b_idx = b_idx.reshape(-1)
    _, time_codes = np.unique(times, return_inverse=True)
    _, cutoff_codes = np.unique(cutoffs, return_inverse=True)

    n_series = int(b_idx.max()) + 1
    t_idx = _dense_rank_within_group(b_idx, cutoff_codes.reshape(-1))
    date_rank = _dense_rank_within_group(b_idx, time_codes.reshape(-1))
    n_windows = int(t_idx.max()) + 1

    # The horizon index is the offset of a row from the first date of its own window.
    window_start = np.full(
        (n_series, n_windows), np.iinfo(np.int64).max, dtype=np.int64
    )
    np.minimum.at(window_start, (b_idx, t_idx), date_rank)
    h_idx = date_rank - window_start[b_idx, t_idx]
    horizon = int(h_idx.max()) + 1

    # Series shorter than the rest yield fewer windows, leaving trailing empty cells.
    has_window = np.zeros((n_series, n_windows), dtype=bool)
    has_window[b_idx, t_idx] = True

    stride = _infer_stride(window_start, has_window, step_size)

    y = np.full((n_series, n_windows, horizon, 1), np.nan, dtype=float)
    y[b_idx, t_idx, h_idx, 0] = np.asarray(df[target_col].to_numpy(), dtype=float)

    y_hat = np.full(
        (n_series, n_windows, horizon, 1, len(quantiles)), np.nan, dtype=float
    )
    for q_pos, col in enumerate(quantile_cols):
        y_hat[b_idx, t_idx, h_idx, 0, q_pos] = np.asarray(
            df[col].to_numpy(), dtype=float
        )

    mask = (~np.isnan(y) & ~np.isnan(y_hat).any(axis=-1)).astype(float)
    return ForecastWindows(
        y=np.nan_to_num(y, nan=0.0),
        y_hat=np.nan_to_num(y_hat, nan=0.0),
        quantiles=quantiles,
        mask=mask,
        stride=stride,
    )


def _infer_stride(
    window_start: np.ndarray, has_window: np.ndarray, step_size: Optional[int]
) -> int:
    """Derive the window step size from the position of each window's first date.

    Args:
        window_start (np.ndarray): Array of shape `[B, T]` holding the date rank of the
            first timestamp of each window.
        has_window (np.ndarray): Boolean array of shape `[B, T]`, True where a series
            actually has that window. Series shorter than the rest yield fewer windows.
        step_size (int, optional): A step size supplied by the caller, validated
            against the cutoffs when given. Defaults to None.

    Returns:
        int: The step size between consecutive forecast windows.

    Raises:
        ValueError: If the windows are unevenly spaced, or if `step_size` disagrees
            with the cutoffs.
    """
    # Only consecutive windows that both exist say anything about the spacing.
    adjacent = has_window[:, :-1] & has_window[:, 1:]
    if not adjacent.any():
        if step_size is not None:
            return step_size
        raise ValueError(
            "Cannot infer step_size: no series has two consecutive forecast windows. "
            "Re-run cross_validation with n_windows > 1, or pass step_size explicitly."
        )

    gaps = np.unique(np.diff(window_start, axis=1)[adjacent])
    if gaps.size != 1:
        raise ValueError(
            f"Forecast windows are not evenly spaced; found gaps {sorted(gaps)} "
            "between consecutive cutoffs. The volatility metrics assume a constant "
            "step size between windows."
        )

    inferred = int(gaps[0])
    if step_size is not None and step_size != inferred:
        raise ValueError(
            f"Passed step_size={step_size} disagrees with the cutoffs in the "
            f"DataFrame, which are spaced {inferred} timesteps apart."
        )
    return inferred
