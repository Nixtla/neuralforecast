__all__ = ['BasePointLoss', 'MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE', 'MASE', 'relMSE', 'QuantileLoss', 'MQLoss', 'QuantileLayer',
           'IQLoss', 'DistributionLoss', 'PMM', 'GMM', 'NBMM', 'HuberLoss', 'TukeyLoss', 'HuberQLoss', 'HuberMQLoss',
           'HuberIQLoss', 'Accuracy', 'sCRPS']


from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    AffineTransform,
    Bernoulli,
    Beta,
    Categorical,
    Distribution,
    Gamma,
    MixtureSameFamily,
    NegativeBinomial,
    Normal,
    Poisson,
    StudentT,
    TransformedDistribution,
    constraints,
)


def _divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Auxiliary funtion to handle divide by 0
    """
    div = a / b
    return torch.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0)


def _weighted_mean(losses, weights):
    """
    Compute weighted mean of losses per datapoint.
    """
    return _divide_no_nan(torch.sum(losses * weights), torch.sum(weights))


class BasePointLoss(torch.nn.Module):
    """Base class for point loss functions.

    Args:
        horizon_weight (Optional[torch.Tensor]): Tensor of size h, weight for each timestamp of the forecasting window. Defaults to None.
        outputsize_multiplier (Optional[int]): Multiplier for the output size. Defaults to None.
        output_names (Optional[List[str]]): Names of the outputs. Defaults to None.
    """

    def __init__(
        self, horizon_weight=None, outputsize_multiplier=None, output_names=None
    ):
        super(BasePointLoss, self).__init__()
        if horizon_weight is not None:
            horizon_weight = torch.Tensor(horizon_weight.flatten())
        self.horizon_weight = horizon_weight
        self.outputsize_multiplier = outputsize_multiplier
        self.output_names = output_names
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """Domain mapping for predicted values.

        Args:
            y_hat (torch.Tensor): Predicted values tensor.
                - Univariate: [B, H, 1]
                - Multivariate: [B, H, N]

        Returns:
            torch.Tensor: Mapped values tensor with shape [B, H, N].
        """
        return y_hat

    def _compute_weights(self, y, mask):
        """Compute final weights for each datapoint based on all weights and masks.

        Set horizon_weight to a ones[H] tensor if not set.
        If set, check that it has the same length as the horizon in x.

        Args:
            y (torch.Tensor): Target values tensor.
            mask (torch.Tensor, optional): Mask tensor specifying datapoints to consider.

        Returns:
            torch.Tensor: Final weights tensor for each datapoint.
        """
        if mask is None:
            mask = torch.ones_like(y)

        if self.horizon_weight is None:
            weights = torch.ones_like(mask)
        else:
            assert mask.shape[1] == len(
                self.horizon_weight
            ), "horizon_weight must have same length as Y"
            weights = self.horizon_weight.clone()
            weights = weights[None, :, None].to(mask.device)
            weights = torch.ones_like(mask, device=mask.device) * weights

        return weights * mask


class MAE(BasePointLoss):
    """Mean Absolute Error.

    Calculates Mean Absolute Error between `y` and `y_hat`. MAE measures the relative prediction
    accuracy of a forecasting method by calculating the deviation of the prediction and the true
    value at a given time and averages these devations over the length of the series.

    $$ \mathrm{MAE}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} |y_{\\tau} - \hat{y}_{\\tau}| $$

    Args:
        horizon_weight (Optional[torch.Tensor]): Tensor of size h, weight for each timestamp of the forecasting window. Defaults to None.
    """

    def __init__(self, horizon_weight=None):
        super(MAE, self).__init__(
            horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""]
        )

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
        y_insample: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """Calculate Mean Absolute Error between actual and predicted values.

        Args:
            y (torch.Tensor): Actual values.
            y_hat (torch.Tensor): Predicted values.
            mask (Union[torch.Tensor, None], optional): Specifies datapoints to consider in loss. Defaults to None.
            y_insample (Union[torch.Tensor, None], optional): Actual insample values. Defaults to None.

        Returns:
            torch.Tensor: MAE (single value).
        """
        losses = torch.abs(y - y_hat)
        weights = self._compute_weights(y=y, mask=mask)
        return _weighted_mean(losses=losses, weights=weights)


class MSE(BasePointLoss):
    """Mean Squared Error.

    Calculates Mean Squared Error between `y` and `y_hat`. MSE measures the relative prediction
    accuracy of a forecasting method by calculating the squared deviation of the prediction and the true
    value at a given time, and averages these devations over the length of the series.

    $$ \mathrm{MSE}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} (y_{\\tau} - \hat{y}_{\\tau})^{2} $$

    Args:
        horizon_weight (Optional[torch.Tensor]): Tensor of size h, weight for each timestamp of the forecasting window. Defaults to None.
    """

    def __init__(self, horizon_weight=None):
        super(MSE, self).__init__(
            horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""]
        )

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: Union[torch.Tensor, None] = None,
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """Calculate Mean Squared Error between actual and predicted values.

        Args:
            y (torch.Tensor): Actual values.
            y_hat (torch.Tensor): Predicted values.
            y_insample (Union[torch.Tensor, None], optional): Actual insample values. Defaults to None.
            mask (Union[torch.Tensor, None], optional): Specifies datapoints to consider in loss. Defaults to None.

        Returns:
            torch.Tensor: MSE (single value).
        """
        losses = (y - y_hat) ** 2
        weights = self._compute_weights(y=y, mask=mask)
        return _weighted_mean(losses=losses, weights=weights)


class RMSE(BasePointLoss):
    """Root Mean Squared Error.

    Calculates Root Mean Squared Error between `y` and `y_hat`. RMSE measures the relative prediction
    accuracy of a forecasting method by calculating the squared deviation of the prediction and the observed value at
    a given time and averages these devations over the length of the series.
    Finally the RMSE will be in the same scale as the original time series so its comparison with other
    series is possible only if they share a common scale. RMSE has a direct connection to the L2 norm.

    $$ \mathrm{RMSE}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}) = \\sqrt{\\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} (y_{\\tau} - \hat{y}_{\\tau})^{2}} $$

    Args:
        horizon_weight (Optional[torch.Tensor]): Tensor of size h, weight for each timestamp of the forecasting window. Defaults to None.
    """

    def __init__(self, horizon_weight=None):
        super(RMSE, self).__init__(
            horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""]
        )

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
        y_insample: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Args:
            y: Tensor, Actual values.
            y_hat: Tensor, Predicted values.
            mask: Tensor, Specifies datapoints to consider in loss.

        Returns:
            rmse: Tensor (single value).
        """
        losses = (y - y_hat) ** 2
        weights = self._compute_weights(y=y, mask=mask)
        losses = _weighted_mean(losses=losses, weights=weights)
        return torch.sqrt(losses)


class MAPE(BasePointLoss):
    """Mean Absolute Percentage Error

    Calculates Mean Absolute Percentage Error  between
    `y` and `y_hat`. MAPE measures the relative prediction
    accuracy of a forecasting method by calculating the percentual deviation
    of the prediction and the observed value at a given time and
    averages these devations over the length of the series.
    The closer to zero an observed value is, the higher penalty MAPE loss
    assigns to the corresponding error.

    $$ \mathrm{MAPE}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} \\frac{|y_{\\tau}-\hat{y}_{\\tau}|}{|y_{\\tau}|} $$

    Args:
        horizon_weight: Tensor of size h, weight for each timestamp of the forecasting window.

    References:
        - [Makridakis S., "Accuracy measures: theoretical and practical concerns".](https://www.sciencedirect.com/science/article/pii/0169207093900793)
    """

    def __init__(self, horizon_weight=None):
        super(MAPE, self).__init__(
            horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""]
        )

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: Union[torch.Tensor, None] = None,
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Args:
            y: Tensor, Actual values.
            y_hat: Tensor, Predicted values.
            mask: Tensor, Specifies date stamps per serie to consider in loss.

        Returns:
            mape: Tensor (single value).
        """
        scale = _divide_no_nan(torch.ones_like(y, device=y.device), torch.abs(y))
        losses = torch.abs(y - y_hat) * scale
        weights = self._compute_weights(y=y, mask=mask)
        mape = _weighted_mean(losses=losses, weights=weights)
        return mape


class SMAPE(BasePointLoss):
    """Symmetric Mean Absolute Percentage Error

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

    Args:
        horizon_weight: Tensor of size h, weight for each timestamp of the forecasting window.

    References:
        - [Makridakis S., "Accuracy measures: theoretical and practical concerns".](https://www.sciencedirect.com/science/article/pii/0169207093900793)
    """

    def __init__(self, horizon_weight=None):
        super(SMAPE, self).__init__(
            horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""]
        )

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
        y_insample: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Args:
            y: Tensor, Actual values.
            y_hat: Tensor, Predicted values.
            mask: Tensor, Specifies date stamps per serie to consider in loss.

        Returns:
            smape: Tensor (single value).
        """
        delta_y = torch.abs((y - y_hat))
        scale = torch.abs(y) + torch.abs(y_hat)
        losses = _divide_no_nan(delta_y, scale)
        weights = self._compute_weights(y=y, mask=mask)
        return 2 * _weighted_mean(losses=losses, weights=weights)


class MASE(BasePointLoss):
    """Mean Absolute Scaled Error
    Calculates the Mean Absolute Scaled Error between
    `y` and `y_hat`. MASE measures the relative prediction
    accuracy of a forecasting method by comparinng the mean absolute errors
    of the prediction and the observed value against the mean
    absolute errors of the seasonal naive model.
    The MASE partially composed the Overall Weighted Average (OWA),
    used in the M4 Competition.

    $$ \mathrm{MASE}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}, \\mathbf{\hat{y}}^{season}_{\\tau}) = \\frac{1}{H} \sum^{t+H}_{\\tau=t+1} \\frac{|y_{\\tau}-\hat{y}_{\\tau}|}{\mathrm{MAE}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}^{season}_{\\tau})} $$

    Args:
        seasonality: Int. Main frequency of the time series; Hourly 24,  Daily 7, Weekly 52, Monthly 12, Quarterly 4, Yearly 1.
        horizon_weight: Tensor of size h, weight for each timestamp of the forecasting window.

    References:
        [Rob J. Hyndman, & Koehler, A. B. "Another look at measures of forecast accuracy".](https://www.sciencedirect.com/science/article/pii/S0169207006000239)<br>
        [Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos, "The M4 Competition: 100,000 time series and 61 forecasting methods".](https://www.sciencedirect.com/science/article/pii/S0169207019301128)
    """

    def __init__(self, seasonality: int, horizon_weight=None):
        super(MASE, self).__init__(
            horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""]
        )
        self.seasonality = seasonality

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Args:
            y: Tensor (batch_size, output_size), Actual values.
            y_hat: Tensor (batch_size, output_size)), Predicted values.
            y_insample: Tensor (batch_size, input_size), Actual insample values.
            mask: Tensor, Specifies date stamps per serie to consider in loss.

        Returns:
            mase: Tensor (single value).
        """
        delta_y = torch.abs(y - y_hat)
        scale = torch.mean(
            torch.abs(
                y_insample[:, self.seasonality :] - y_insample[:, : -self.seasonality]
            ),
            axis=1,
        )
        losses = _divide_no_nan(delta_y, scale[:, None, None])
        weights = self._compute_weights(y=y, mask=mask)
        return _weighted_mean(losses=losses, weights=weights)


class relMSE(BasePointLoss):
    """Relative Mean Squared Error
    Computes Relative Mean Squared Error (relMSE), as proposed by Hyndman & Koehler (2006)
    as an alternative to percentage errors, to avoid measure unstability.
    $$
    \mathrm{relMSE}(\mathbf{y}, \mathbf{\hat{y}}, \mathbf{\hat{y}}^{benchmark}) =
    \frac{\mathrm{MSE}(\mathbf{y}, \mathbf{\hat{y}})}{\mathrm{MSE}(\mathbf{y}, \mathbf{\hat{y}}^{benchmark})}
    $$

    Args:
        y_train: Numpy array, deprecated.
        horizon_weight: Tensor of size h, weight for each timestamp of the forecasting window.

    References:
        - [Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.](https://www.sciencedirect.com/science/article/pii/S0169207006000239)<br>
        - [Kin G. Olivares, O. Nganba Meetei, Ruijun Ma, Rohan Reddy, Mengfei Cao, Lee Dicker. "Probabilistic Hierarchical Forecasting with Deep Poisson Mixtures. Submitted to the International Journal Forecasting, Working paper available at arxiv.](https://arxiv.org/pdf/2110.13179.pdf)
    """

    def __init__(self, y_train=None, horizon_weight=None):
        super(relMSE, self).__init__(
            horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""]
        )
        if y_train is not None:
            raise DeprecationWarning("y_train will be deprecated in a future release.")
        self.mse = MSE(horizon_weight=horizon_weight)

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_benchmark: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Args:
            y: Tensor (batch_size, output_size), Actual values.
            y_hat: Tensor (batch_size, output_size)), Predicted values.
            y_benchmark: Tensor (batch_size, output_size), Benchmark predicted values.
            mask: Tensor, Specifies date stamps per serie to consider in loss.

        Returns:
            relMSE: Tensor (single value).
        """
        norm = self.mse(y=y, y_hat=y_benchmark, mask=mask)  # Already weighted
        norm = norm + 1e-5  # Numerical stability
        loss = self.mse(y=y, y_hat=y_hat, mask=mask)  # Already weighted
        loss = _divide_no_nan(loss, norm)
        return loss


class QuantileLoss(BasePointLoss):
    """Quantile Loss.

    Computes the quantile loss between `y` and `y_hat`.
    QL measures the deviation of a quantile forecast.
    By weighting the absolute deviation in a non symmetric way, the
    loss pays more attention to under or over estimation.
    A common value for q is 0.5 for the deviation from the median (Pinball loss).

    $$
    \mathrm{QL}(\mathbf{y}_{\\tau}, \mathbf{\hat{y}}^{(q)}_{\\tau}) = \\frac{1}{H} \sum^{t+H}_{\\tau=t+1} \Big( (1-q)\,( \hat{y}^{(q)}_{\\tau} - y_{\\tau} )_{+} + q\,( y_{\\tau} - \hat{y}^{(q)}_{\\tau} )_{+} \Big)
    $$

    Args:
        q (float): Between 0 and 1. The slope of the quantile loss, in the context of quantile regression, the q determines the conditional quantile level.
        horizon_weight (Optional[torch.Tensor]): Tensor of size h, weight for each timestamp of the forecasting window. Defaults to None.

    References:
        [Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles".](https://www.jstor.org/stable/1913643)
    """

    def __init__(self, q, horizon_weight=None):
        super(QuantileLoss, self).__init__(
            horizon_weight=horizon_weight,
            outputsize_multiplier=1,
            output_names=[f"_ql{q}"],
        )
        self.q = q

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: Union[torch.Tensor, None] = None,
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """Calculate quantile loss between actual and predicted values.

        Args:
            y (torch.Tensor): Actual values.
            y_hat (torch.Tensor): Predicted values.
            y_insample (Union[torch.Tensor, None], optional): Actual insample values. Defaults to None.
            mask (Union[torch.Tensor, None], optional): Specifies datapoints to consider in loss. Defaults to None.

        Returns:
            torch.Tensor: Quantile loss (single value).
        """
        delta_y = y - y_hat
        losses = torch.max(torch.mul(self.q, delta_y), torch.mul((self.q - 1), delta_y))
        weights = self._compute_weights(y=y, mask=mask)
        return _weighted_mean(losses=losses, weights=weights)


def level_to_outputs(level):
    qs = sum([[50 - l / 2, 50 + l / 2] for l in level], [])
    output_names = sum([[f"-lo-{l}", f"-hi-{l}"] for l in level], [])

    sort_idx = np.argsort(qs)
    quantiles = np.array(qs)[sort_idx]

    # Add default median
    quantiles = np.concatenate([np.array([50]), quantiles])
    quantiles = torch.Tensor(quantiles) / 100
    output_names = list(np.array(output_names)[sort_idx])
    output_names.insert(0, "-median")

    return quantiles, output_names


def quantiles_to_outputs(quantiles):
    output_names = []
    for q in quantiles:
        if q < 0.50:
            output_names.append(f"-lo-{np.round(100-200*q,2)}")
        elif q > 0.50:
            output_names.append(f"-hi-{np.round(100-200*(1-q),2)}")
        else:
            output_names.append("-median")
    return quantiles, output_names


class MQLoss(BasePointLoss):
    """Multi-Quantile loss

    Calculates the Multi-Quantile loss (MQL) between `y` and `y_hat`.
    MQL calculates the average multi-quantile Loss for
    a given set of quantiles, based on the absolute
    difference between predicted quantiles and observed values.

    $$
    \mathrm{MQL}(\mathbf{y}_{\\tau},[\mathbf{\hat{y}}^{(q_{1})}_{\\tau}, ... ,\hat{y}^{(q_{n})}_{\\tau}]) = \\frac{1}{n} \sum_{q_{i}} \mathrm{QL}(\mathbf{y}_{\\tau}, \mathbf{\hat{y}}^{(q_{i})}_{\\tau})
    $$

    The limit behavior of MQL allows to measure the accuracy
    of a full predictive distribution $\mathbf{\hat{F}}_{\\tau}$ with
    the continuous ranked probability score (CRPS). This can be achieved
    through a numerical integration technique, that discretizes the quantiles
    and treats the CRPS integral with a left Riemann approximation, averaging over
    uniformly distanced quantiles.

    $$
    \mathrm{CRPS}(y_{\\tau}, \mathbf{\hat{F}}_{\\tau}) = \int^{1}_{0} \mathrm{QL}(y_{\\tau}, \hat{y}^{(q)}_{\\tau}) dq
    $$

    Args:
        level (List[int], optional): Probability levels for prediction intervals. Defaults to [80, 90].
        quantiles (Optional[List[float]]): Alternative to level, quantiles to estimate from y distribution. Defaults to None.
        horizon_weight (Optional[torch.Tensor]): Tensor of size h, weight for each timestamp of the forecasting window. Defaults to None.

    References:
        [Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles".](https://www.jstor.org/stable/1913643)
        [James E. Matheson and Robert L. Winkler, "Scoring Rules for Continuous Probability Distributions".](https://www.jstor.org/stable/2629907)
    """

    def __init__(self, level=[80, 90], quantiles=None, horizon_weight=None):

        qs, output_names = level_to_outputs(level)
        qs = torch.Tensor(qs)
        # Transform quantiles to homogeneous output names
        if quantiles is not None:
            _, output_names = quantiles_to_outputs(quantiles)
            qs = torch.Tensor(quantiles)

        super(MQLoss, self).__init__(
            horizon_weight=horizon_weight,
            outputsize_multiplier=len(qs),
            output_names=output_names,
        )

        self.quantiles = torch.nn.Parameter(qs, requires_grad=False)

    def domain_map(self, y_hat: torch.Tensor):
        """Reshapes input tensor to match the expected output format.

        Args:
            y_hat (torch.Tensor): Input tensor.
                - Univariate: [B, H, 1 * Q]
                - Multivariate: [B, H, N * Q]

        Returns:
            torch.Tensor: Reshaped tensor with shape [B, H, N, Q].
        """
        output = y_hat.reshape(
            y_hat.shape[0], y_hat.shape[1], -1, self.outputsize_multiplier
        )

        return output

    def _compute_weights(self, y, mask):
        """Compute final weights for each datapoint based on all weights and masks.

        Set horizon_weight to a ones[H] tensor if not set.
        If set, check that it has the same length as the horizon in x.

        Args:
            y: Tensor with shape [B, h, N, 1].
            mask: Tensor with shape [B, h, N, 1].
        """

        if self.horizon_weight is None:
            weights = torch.ones_like(mask)
        else:
            assert mask.shape[1] == len(
                self.horizon_weight
            ), "horizon_weight must have same length as Y"
            weights = self.horizon_weight.clone()
            weights = weights[None, :, None, None]
            weights = weights.to(mask.device)
            weights = torch.ones_like(mask, device=mask.device) * weights

        return weights * mask

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: Union[torch.Tensor, None] = None,
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """Computes the multi-quantile loss.

        Args:
            y (torch.Tensor): Actual values.
            y_hat (torch.Tensor): Predicted values.
            y_insample (Union[torch.Tensor, None], optional): In-sample values. Defaults to None.
            mask (Union[torch.Tensor, None], optional): Specifies date stamps per serie to consider in loss. Defaults to None.

        Returns:
            torch.Tensor: Multi-quantile loss (single value).
        """
        # [B, h, N] -> [B, h, N, 1]
        if y_hat.ndim == 3:
            y_hat = y_hat.unsqueeze(-1)

        y = y.unsqueeze(-1)
        if mask is not None:
            mask = mask.unsqueeze(-1)
        else:
            mask = torch.ones_like(y, device=y.device)

        error = y_hat - y

        sq = torch.maximum(-error, torch.zeros_like(error))
        s1_q = torch.maximum(error, torch.zeros_like(error))

        quantiles = self.quantiles[None, None, None, :]
        losses = (1 / len(quantiles)) * (quantiles * sq + (1 - quantiles) * s1_q)
        weights = self._compute_weights(y=losses, mask=mask)  # Use losses for extra dim

        return _weighted_mean(losses=losses, weights=weights)


class QuantileLayer(nn.Module):
    """Implicit Quantile Layer from the paper IQN for Distributional Reinforcement Learning.

    Code from GluonTS: https://github.com/awslabs/gluonts/blob/61133ef6e2d88177b32ace4afc6843ab9a7bc8cd/src/gluonts/torch/distributions/implicit_quantile_network.py

    References:
        Dabney et al. 2018. https://arxiv.org/abs/1806.06923
    """

    def __init__(self, num_output: int, cos_embedding_dim: int = 128):
        super().__init__()

        self.output_layer = nn.Sequential(
            nn.Linear(cos_embedding_dim, cos_embedding_dim),
            nn.PReLU(),
            nn.Linear(cos_embedding_dim, num_output),
        )

        self.register_buffer("integers", torch.arange(0, cos_embedding_dim))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        cos_emb_tau = torch.cos(tau * self.integers * torch.pi)
        return self.output_layer(cos_emb_tau)


class IQLoss(QuantileLoss):
    """Implicit Quantile Loss.

    Computes the quantile loss between `y` and `y_hat`, with the quantile `q` provided as an input to the network.
    IQL measures the deviation of a quantile forecast.
    By weighting the absolute deviation in a non symmetric way, the
    loss pays more attention to under or over estimation.

    $$
    \mathrm{QL}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}^{(q)}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} \Big( (1-q)\,( \hat{y}^{(q)}_{\\tau} - y_{\\tau} )_{+} + q\,( y_{\\tau} - \hat{y}^{(q)}_{\\tau} )_{+} \Big)
    $$

    Args:
        cos_embedding_dim (int, optional): Cosine embedding dimension. Defaults to 64.
        concentration0 (float, optional): Beta distribution concentration parameter. Defaults to 1.0.
        concentration1 (float, optional): Beta distribution concentration parameter. Defaults to 1.0.
        horizon_weight (Optional[torch.Tensor]): Tensor of size h, weight for each timestamp of the forecasting window. Defaults to None.

    References:
        Gouttes, Adèle, Kashif Rasul, Mateusz Koren, Johannes Stephan, and Tofigh Naghibi, "Probabilistic Time Series Forecasting with Implicit Quantile Networks". http://arxiv.org/abs/2107.03743
    """

    def __init__(
        self,
        cos_embedding_dim=64,
        concentration0=1.0,
        concentration1=1.0,
        horizon_weight=None,
    ):
        self.update_quantile()
        super(IQLoss, self).__init__(q=self.q, horizon_weight=horizon_weight)

        self.cos_embedding_dim = cos_embedding_dim
        self.concentration0 = concentration0
        self.concentration1 = concentration1
        self.has_sampled = False
        self.has_predicted = False

        self.quantile_layer = QuantileLayer(
            num_output=1, cos_embedding_dim=self.cos_embedding_dim
        )
        self.output_layer = nn.Sequential(nn.Linear(1, 1), nn.PReLU())

    def _sample_quantiles(self, sample_size, device):
        if not self.has_sampled:
            self._init_sampling_distribution(device)

        quantiles = self.sampling_distr.sample(sample_size)
        self.q = quantiles.squeeze(-1)
        self.has_sampled = True
        self.has_predicted = False

        return quantiles

    def _init_sampling_distribution(self, device):
        concentration0 = torch.tensor(
            [self.concentration0], device=device, dtype=torch.float32
        )
        concentration1 = torch.tensor(
            [self.concentration1], device=device, dtype=torch.float32
        )
        self.sampling_distr = Beta(
            concentration0=concentration0, concentration1=concentration1
        )

    def update_quantile(self, q: List[float] = [0.5]):
        self.q = q[0]
        self.output_names = [f"_ql{q[0]}"]
        self.has_predicted = True

    def domain_map(self, y_hat):
        """Adds IQN network to output of network.

        Args:
            y_hat (torch.Tensor): Input tensor.
                - Univariate: [B, h, 1]
                - Multivariate: [B, h, N]

        Returns:
            torch.Tensor: Domain mapped tensor.
        """
        if self.eval() and self.has_predicted:
            quantiles = torch.full(
                size=y_hat.shape,
                fill_value=self.q,
                device=y_hat.device,
                dtype=y_hat.dtype,
            )
            quantiles = quantiles.unsqueeze(-1)
        else:
            quantiles = self._sample_quantiles(
                sample_size=y_hat.shape, device=y_hat.device
            )

        # Embed the quantiles and add to y_hat
        emb_taus = self.quantile_layer(quantiles)
        emb_inputs = y_hat.unsqueeze(-1) * (1.0 + emb_taus)
        emb_outputs = self.output_layer(emb_inputs)

        # Domain map
        y_hat = emb_outputs.squeeze(-1)

        return y_hat


def weighted_average(
    x: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None
) -> torch.Tensor:
    """Computes the weighted average of a given tensor across a given dim.

    Masks values associated with weight zero, meaning instead of `nan * 0 = nan`
    you will get `0 * 0 = 0`.

    Args:
        x (torch.Tensor): Input tensor, of which the average must be computed.
        weights (Optional[torch.Tensor], optional): Weights tensor, of the same shape as `x`. Defaults to None.
        dim (optional): The dim along which to average `x`. Defaults to None.

    Returns:
        torch.Tensor: The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, x * weights, torch.zeros_like(x))
        sum_weights = torch.clamp(
            weights.sum(dim=dim) if dim else weights.sum(), min=1.0
        )
        return (
            weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()
        ) / sum_weights
    else:
        return x.mean(dim=dim)


def bernoulli_scale_decouple(output, loc=None, scale=None):
    """Bernoulli Scale Decouple.

    Stabilizes model's output optimization, by learning residual
    variance and residual location based on anchoring `loc`, `scale`.
    Also adds Bernoulli domain protection to the distribution parameters.

    Args:
        output: Model output tensor.
        loc (optional): Location parameter. Defaults to None.
        scale (optional): Scale parameter. Defaults to None.

    Returns:
        tuple: Processed probabilities.
    """
    probs = output[0]
    # if (loc is not None) and (scale is not None):
    #    rate = (rate * scale) + loc
    probs = F.sigmoid(probs)  # .clone()
    return (probs,)


def student_scale_decouple(output, loc=None, scale=None, eps: float = 0.1):
    """Student-T Scale Decouple.

    Stabilizes model's output optimization, by learning residual
    variance and residual location based on anchoring `loc`, `scale`.
    Also adds StudentT domain protection to the distribution parameters.

    Args:
        output: Model output tensor.
        loc (optional): Location parameter. Defaults to None.
        scale (optional): Scale parameter. Defaults to None.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 0.1.

    Returns:
        tuple: Processed degrees of freedom, mean, and scale parameters.
    """
    df, mean, tscale = output
    tscale = F.softplus(tscale)
    if (loc is not None) and (scale is not None):
        mean = (mean * scale) + loc
        tscale = (tscale + eps) * scale
    df = 3.0 + F.softplus(df)
    return (df, mean, tscale)


def normal_scale_decouple(output, loc=None, scale=None, eps: float = 0.2):
    """Normal Scale Decouple.

    Stabilizes model's output optimization, by learning residual
    variance and residual location based on anchoring `loc`, `scale`.
    Also adds Normal domain protection to the distribution parameters.

    Args:
        output: Model output tensor.
        loc (optional): Location parameter. Defaults to None.
        scale (optional): Scale parameter. Defaults to None.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 0.2.

    Returns:
        tuple: Processed mean and standard deviation parameters.
    """
    mean, std = output
    std = F.softplus(std)
    if (loc is not None) and (scale is not None):
        mean = (mean * scale) + loc
        std = (std + eps) * scale
    return (mean, std)


def poisson_scale_decouple(output, loc=None, scale=None):
    """Poisson Scale Decouple

    Stabilizes model's output optimization, by learning residual
    variance and residual location based on anchoring `loc`, `scale`.
    Also adds Poisson domain protection to the distribution parameters.
    """
    eps = 1e-10
    rate = output[0]
    if (loc is not None) and (scale is not None):
        rate = (rate * scale) + loc
    rate = F.softplus(rate) + eps
    return (rate,)


def nbinomial_scale_decouple(output, loc=None, scale=None):
    """Negative Binomial Scale Decouple

    Stabilizes model's output optimization, by learning total
    count and logits based on anchoring `loc`, `scale`.
    Also adds Negative Binomial domain protection to the distribution parameters.
    """
    mu, alpha = output
    mu = F.softplus(mu) + 1e-8
    alpha = F.softplus(alpha) + 1e-8  # alpha = 1/total_counts
    if (loc is not None) and (scale is not None):
        mu = mu * scale + loc
        alpha /= scale + 1.0

    # mu = total_count * (probs/(1-probs))
    # => probs = mu / (total_count + mu)
    # => probs = mu / [total_count * (1 + mu * (1/total_count))]
    total_count = 1.0 / alpha
    probs = (mu * alpha / (1.0 + mu * alpha)) + 1e-8
    return (total_count, probs)


def est_lambda(mu, rho):
    return mu ** (2 - rho) / (2 - rho)


def est_alpha(rho):
    return (2 - rho) / (rho - 1)


def est_beta(mu, rho):
    return mu ** (1 - rho) / (rho - 1)


class Tweedie(Distribution):
    """Tweedie Distribution.

    The Tweedie distribution is a compound probability, special case of exponential
    dispersion models EDMs defined by its mean-variance relationship.
    The distribution particularly useful to model sparse series as the probability has
    possitive mass at zero but otherwise is continuous.

    $$
    Y \sim \mathrm{ED}(\\mu,\\sigma^{2}) \qquad
    \mathbb{P}(y|\\mu ,\\sigma^{2})=h(\\sigma^{2},y) \\exp \\left({\\frac {\\theta y-A(\\theta )}{\\sigma^{2}}}\\right)
    $$

    $$
    \mu =A'(\\theta ) \qquad \mathrm{Var}(Y) = \\sigma^{2} \\mu^{\\rho}
    $$

    Cases of the variance relationship include Normal (`rho` = 0), Poisson (`rho` = 1),
    Gamma (`rho` = 2), inverse Gaussian (`rho` = 3).

    Args:
        log_mu (torch.Tensor): Tensor with log of means.
        rho (float): Tweedie variance power (1,2). Fixed across all observations.
        validate_args (optional): Validation arguments. Defaults to None.

    Note:
        sigma2: Tweedie variance. Currently fixed in 1.

    References:
        - Tweedie, M. C. K. (1984). An index which distinguishes between some important exponential families. Statistics: Applications and New Directions. Proceedings of the Indian Statistical Institute Golden Jubilee International Conference (Eds. J. K. Ghosh and J. Roy), pp. 579-604. Calcutta: Indian Statistical Institute.
        - Jorgensen, B. (1987). Exponential Dispersion Models. Journal of the Royal Statistical Society. Series B (Methodological), 49(2), 127–162. http://www.jstor.org/stable/2345415
    """

    arg_constraints = {"log_mu": constraints.real}
    support = constraints.nonnegative

    def __init__(self, log_mu, rho, validate_args=None):
        # TODO: add sigma2 dispersion
        # TODO add constraints
        # support = constraints.real
        self.log_mu = log_mu
        self.rho = rho
        assert rho > 1 and rho < 2, f"rho={rho} parameter needs to be between (1,2)."

        batch_shape = log_mu.size()
        super(Tweedie, self).__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        return torch.exp(self.log_mu)

    @property
    def variance(self):
        return torch.ones_line(self.log_mu)  # TODO need to be assigned

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            mu = self.mean
            rho = self.rho * torch.ones_like(mu)
            sigma2 = 1  # TODO

            rate = est_lambda(mu, rho) / sigma2  # rate for poisson
            alpha = est_alpha(rho)  # alpha for Gamma distribution
            beta = est_beta(mu, rho) / sigma2  # beta for Gamma distribution

            # Expand for sample
            rate = rate.expand(shape)
            alpha = alpha.expand(shape)
            beta = beta.expand(shape)

            N = torch.poisson(rate) + 1e-5
            gamma = Gamma(N * alpha, beta)
            samples = gamma.sample()
            samples[N == 0] = 0

            return samples

    def log_prob(self, y_true):
        rho = self.rho
        y_pred = self.log_mu

        a = y_true * torch.exp((1 - rho) * y_pred) / (1 - rho)
        b = torch.exp((2 - rho) * y_pred) / (2 - rho)

        return a - b


def tweedie_domain_map(input: torch.Tensor, rho: float = 1.5):
    """
    Maps output of neural network to domain of distribution loss

    """
    return (input, rho)


def tweedie_scale_decouple(output, loc=None, scale=None):
    """Tweedie Scale Decouple

    Stabilizes model's output optimization, by learning total
    count and logits based on anchoring `loc`, `scale`.
    Also adds Tweedie domain protection to the distribution parameters.
    """
    log_mu, rho = output
    log_mu = F.softplus(log_mu)
    log_mu = torch.clamp(log_mu, 1e-9, 37)
    if (loc is not None) and (scale is not None):
        log_mu += torch.log(loc)

    log_mu = torch.clamp(log_mu, 1e-9, 37)
    return (log_mu, rho)


# Code adapted from: https://github.com/awslabs/gluonts/blob/61133ef6e2d88177b32ace4afc6843ab9a7bc8cd/src/gluonts/torch/distributions/isqf.py


class ISQF(TransformedDistribution):
    """Distribution class for the Incremental (Spline) Quantile Function.

    Args:
        spline_knots (torch.Tensor): Tensor parametrizing the x-positions of the spline knots. Shape: (*batch_shape, (num_qk-1), num_pieces)
        spline_heights (torch.Tensor): Tensor parametrizing the y-positions of the spline knots. Shape: (*batch_shape, (num_qk-1), num_pieces)
        beta_l (torch.Tensor): Tensor containing the non-negative learnable parameter of the left tail. Shape: (*batch_shape,)
        beta_r (torch.Tensor): Tensor containing the non-negative learnable parameter of the right tail. Shape: (*batch_shape,)
        qk_y (torch.Tensor): Tensor containing the increasing y-positions of the quantile knots. Shape: (*batch_shape, num_qk)
        qk_x (torch.Tensor): Tensor containing the increasing x-positions of the quantile knots. Shape: (*batch_shape, num_qk)
        loc (torch.Tensor): Tensor containing the location in case of a transformed random variable. Shape: (*batch_shape,)
        scale (torch.Tensor): Tensor containing the scale in case of a transformed random variable. Shape: (*batch_shape,)

    References:
        Park, Youngsuk, Danielle Maddix, François-Xavier Aubet, Kelvin Kan, Jan Gasthaus, and Yuyang Wang (2022). "Learning Quantile Functions without Quantile Crossing for Distribution-free Time Series Forecasting". https://proceedings.mlr.press/v151/park22a.html
    """

    def __init__(
        self,
        spline_knots: torch.Tensor,
        spline_heights: torch.Tensor,
        beta_l: torch.Tensor,
        beta_r: torch.Tensor,
        qk_y: torch.Tensor,
        qk_x: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
        validate_args=None,
    ) -> None:
        base_distribution = BaseISQF(
            spline_knots=spline_knots,
            spline_heights=spline_heights,
            beta_l=beta_l,
            beta_r=beta_r,
            qk_y=qk_y,
            qk_x=qk_x,
            validate_args=validate_args,
        )
        transforms = AffineTransform(loc=loc, scale=scale)
        super().__init__(base_distribution, transforms, validate_args=validate_args)

    def crps(self, y: torch.Tensor) -> torch.Tensor:
        z = y
        scale = 1.0
        t = self.transforms[0]
        z = t._inverse(z)
        scale *= t.scale
        p = self.base_dist.crps(z)
        return p * scale

    @property
    def mean(self):
        """
        Function used to compute the empirical mean
        """
        samples = self.sample([1000])
        return samples.mean(dim=0)


class BaseISQF(Distribution):
    """Base distribution class for the Incremental (Spline) Quantile Function.

    Args:
        spline_knots (torch.Tensor): Tensor parametrizing the x-positions of the spline knots. Shape: (*batch_shape, (num_qk-1), num_pieces)
        spline_heights (torch.Tensor): Tensor parametrizing the y-positions of the spline knots. Shape: (*batch_shape, (num_qk-1), num_pieces)
        beta_l (torch.Tensor): Tensor containing the non-negative learnable parameter of the left tail. (*batch_shape,)
        beta_r (torch.Tensor): Tensor containing the non-negative learnable parameter of the right tail. (*batch_shape,)
        qk_y (torch.Tensor): Tensor containing the increasing y-positions of the quantile knots. Shape: (*batch_shape, num_qk)
        qk_x (torch.Tensor): Tensor containing the increasing x-positions of the quantile knots. Shape: (*batch_shape, num_qk)
        tol (float, optional): Tolerance hyperparameter for numerical stability. Defaults to 1e-4.
        validate_args (bool, optional): Whether to validate arguments. Defaults to False.

    References:
        Park, Youngsuk, Danielle Maddix, François-Xavier Aubet, Kelvin Kan, Jan Gasthaus, and Yuyang Wang (2022). "Learning Quantile Functions without Quantile Crossing for Distribution-free Time Series Forecasting". https://proceedings.mlr.press/v151/park22a.html
    """

    def __init__(
        self,
        spline_knots: torch.Tensor,
        spline_heights: torch.Tensor,
        beta_l: torch.Tensor,
        beta_r: torch.Tensor,
        qk_y: torch.Tensor,
        qk_x: torch.Tensor,
        tol: float = 1e-4,
        validate_args: bool = False,
    ) -> None:
        self.num_qk, self.num_pieces = qk_y.shape[-1], spline_knots.shape[-1]
        self.spline_knots, self.spline_heights = spline_knots, spline_heights
        self.beta_l, self.beta_r = beta_l, beta_r
        self.qk_y_all = qk_y
        self.tol = tol

        super().__init__(batch_shape=self.batch_shape, validate_args=validate_args)

        # Get quantile knots (qk) parameters
        (
            self.qk_x,
            self.qk_x_plus,
            self.qk_x_l,
            self.qk_x_r,
        ) = BaseISQF.parameterize_qk(qk_x)
        (
            self.qk_y,
            self.qk_y_plus,
            self.qk_y_l,
            self.qk_y_r,
        ) = BaseISQF.parameterize_qk(qk_y)

        # Get spline knots (sk) parameters
        self.sk_y, self.delta_sk_y = BaseISQF.parameterize_spline(
            self.spline_heights,
            self.qk_y,
            self.qk_y_plus,
            self.tol,
        )
        self.sk_x, self.delta_sk_x = BaseISQF.parameterize_spline(
            self.spline_knots,
            self.qk_x,
            self.qk_x_plus,
            self.tol,
        )

        if self.num_pieces > 1:
            self.sk_x_plus = torch.cat(
                [self.sk_x[..., 1:], self.qk_x_plus.unsqueeze(dim=-1)], dim=-1
            )
        else:
            self.sk_x_plus = self.qk_x_plus.unsqueeze(dim=-1)

        # Get tails parameters
        self.tail_al, self.tail_bl = BaseISQF.parameterize_tail(
            self.beta_l, self.qk_x_l, self.qk_y_l
        )
        self.tail_ar, self.tail_br = BaseISQF.parameterize_tail(
            -self.beta_r, 1 - self.qk_x_r, self.qk_y_r
        )

    @staticmethod
    def parameterize_qk(
        quantile_knots: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Function to parameterize the x or y positions of the num_qk quantile knots.

        Args:
            quantile_knots (torch.Tensor): x or y positions of the quantile knots. Shape: (*batch_shape, num_qk)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - qk: x or y positions of the quantile knots (qk), with index=1, ..., num_qk-1. Shape: (*batch_shape, num_qk-1)
                - qk_plus: x or y positions of the quantile knots (qk), with index=2, ..., num_qk. Shape: (*batch_shape, num_qk-1)
                - qk_l: x or y positions of the left-most quantile knot (qk). Shape: (*batch_shape)
                - qk_r: x or y positions of the right-most quantile knot (qk). Shape: (*batch_shape)
        """

        qk, qk_plus = quantile_knots[..., :-1], quantile_knots[..., 1:]
        qk_l, qk_r = quantile_knots[..., 0], quantile_knots[..., -1]

        return qk, qk_plus, qk_l, qk_r

    @staticmethod
    def parameterize_spline(
        spline_knots: torch.Tensor,
        qk: torch.Tensor,
        qk_plus: torch.Tensor,
        tol: float = 1e-4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function to parameterize the x or y positions of the spline knots.

        Args:
            spline_knots (torch.Tensor): Variable that parameterizes the spline knot positions.
            qk (torch.Tensor): x or y positions of the quantile knots (qk), with index=1, ..., num_qk-1. Shape: (*batch_shape, num_qk-1)
            qk_plus (torch.Tensor): x or y positions of the quantile knots (qk), with index=2, ..., num_qk. Shape: (*batch_shape, num_qk-1)
            tol (float, optional): Tolerance hyperparameter for numerical stability. Defaults to 1e-4.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - sk: x or y positions of the spline knots (sk). Shape: (*batch_shape, num_qk-1, num_pieces)
                - delta_sk: difference of x or y positions of the spline knots (sk). Shape: (*batch_shape, num_qk-1, num_pieces)
        """

        # The spacing between spline knots is parameterized
        # by softmax function (in [0,1] and sum to 1)
        # We add tol to prevent overflow in computing 1/spacing in spline CRPS
        # After adding tol, it is normalized by
        # (1 + num_pieces * tol) to keep the sum-to-1 property

        num_pieces = spline_knots.shape[-1]

        delta_x = (F.softmax(spline_knots, dim=-1) + tol) / (1 + num_pieces * tol)

        zero_tensor = torch.zeros_like(delta_x[..., 0:1])  # 0:1 for keeping dimension
        x = torch.cat([zero_tensor, torch.cumsum(delta_x, dim=-1)[..., :-1]], dim=-1)

        qk, qk_plus = qk.unsqueeze(dim=-1), qk_plus.unsqueeze(dim=-1)
        sk = x * (qk_plus - qk) + qk
        delta_sk = delta_x * (qk_plus - qk)

        return sk, delta_sk

    @staticmethod
    def parameterize_tail(
        beta: torch.Tensor, qk_x: torch.Tensor, qk_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function to parameterize the tail parameters.

        Note that the exponential tails are given by:
        q(alpha) = a_l log(alpha) + b_l if left tail
        q(alpha) = a_r log(1-alpha) + b_r if right tail

        Where:
        a_l=1/beta_l, b_l=-a_l*log(qk_x_l)+q(qk_x_l)
        a_r=1/beta_r, b_r=a_r*log(1-qk_x_r)+q(qk_x_r)

        Args:
            beta (torch.Tensor): Parameterizes the left or right tail. Shape: (*batch_shape,)
            qk_x (torch.Tensor): Left- or right-most x-positions of the quantile knots. Shape: (*batch_shape,)
            qk_y (torch.Tensor): Left- or right-most y-positions of the quantile knots. Shape: (*batch_shape,)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - tail_a: a_l or a_r as described above
                - tail_b: b_l or b_r as described above
        """

        tail_a = 1 / beta
        tail_b = -tail_a * torch.log(qk_x) + qk_y

        return tail_a, tail_b

    def quantile(self, alpha: torch.Tensor) -> torch.Tensor:
        return self.quantile_internal(alpha, dim=0)

    def quantile_internal(
        self, alpha: torch.Tensor, dim: Optional[int] = None
    ) -> torch.Tensor:
        """Evaluates the quantile function at the quantile levels input_alpha.

        Args:
            alpha (torch.Tensor): Tensor of shape = (*batch_shape,) if axis=None, or containing an additional axis on the specified position, otherwise.
            dim (Optional[int], optional): Index of the axis containing the different quantile levels which are to be computed. Read the description below for detailed information. Defaults to None.

        Returns:
            torch.Tensor: Quantiles tensor, of the same shape as alpha.
        """

        qk_x, qk_x_l, qk_x_plus = self.qk_x, self.qk_x_l, self.qk_x_plus

        # The following describes the parameters reshaping in
        # quantile_internal, quantile_spline and quantile_tail

        # tail parameters: tail_al, tail_ar, tail_bl, tail_br,
        # shape = (*batch_shape,)
        # spline parameters: sk_x, sk_x_plus, sk_y, sk_y_plus,
        # shape = (*batch_shape, num_qk-1, num_pieces)
        # quantile knots parameters: qk_x, qk_x_plus, qk_y, qk_y_plus,
        # shape = (*batch_shape, num_qk-1)

        # dim=None - passed at inference when num_samples is None
        # shape of input_alpha = (*batch_shape,), will be expanded to
        # (*batch_shape, 1, 1) to perform operation
        # The shapes of parameters are as described above,
        # no reshaping is needed

        # dim=0 - passed at inference when num_samples is not None
        # shape of input_alpha = (num_samples, *batch_shape)
        # it will be expanded to
        # (num_samples, *batch_shape, 1, 1) to perform operation
        #
        # The shapes of tail parameters
        # should be (num_samples, *batch_shape)
        #
        # The shapes of spline parameters
        # should be (num_samples, *batch_shape, num_qk-1, num_pieces)
        #
        # The shapes of quantile knots parameters
        # should be (num_samples, *batch_shape, num_qk-1)
        #
        # We expand at dim=0 for all of them

        # dim=-2 - passed at training when we evaluate quantiles at
        # spline knots in order to compute alpha_tilde
        #
        # This is only for the quantile_spline function
        # shape of input_alpha = (*batch_shape, num_qk-1, num_pieces)
        # it will be expanded to
        # (*batch_shape, num_qk-1, num_pieces, 1) to perform operation
        #
        # The shapes of spline and quantile knots parameters should be
        # (*batch_shape, num_qk-1, 1, num_pieces)
        # and (*batch_shape, num_qk-1, 1), respectively
        #
        # We expand at dim=-2 and dim=-1 for
        # spline and quantile knots parameters, respectively

        if dim is not None:
            qk_x_l = qk_x_l.unsqueeze(dim=dim)
            qk_x = qk_x.unsqueeze(dim=dim)
            qk_x_plus = qk_x_plus.unsqueeze(dim=dim)

        quantile = torch.where(
            alpha < qk_x_l,
            self.quantile_tail(alpha, dim=dim, left_tail=True),
            self.quantile_tail(alpha, dim=dim, left_tail=False),
        )

        spline_val = self.quantile_spline(alpha, dim=dim)

        for spline_idx in range(self.num_qk - 1):
            is_in_between = torch.logical_and(
                qk_x[..., spline_idx] <= alpha,
                alpha < qk_x_plus[..., spline_idx],
            )

            quantile = torch.where(
                is_in_between,
                spline_val[..., spline_idx],
                quantile,
            )

        return quantile

    def quantile_spline(
        self,
        alpha: torch.Tensor,
        dim: Optional[int] = None,
    ) -> torch.Tensor:
        # Refer to the description in quantile_internal

        qk_y = self.qk_y
        sk_x, delta_sk_x, delta_sk_y = (
            self.sk_x,
            self.delta_sk_x,
            self.delta_sk_y,
        )

        if dim is not None:
            qk_y = qk_y.unsqueeze(dim=0 if dim == 0 else -1)
            sk_x = sk_x.unsqueeze(dim=dim)
            delta_sk_x = delta_sk_x.unsqueeze(dim=dim)
            delta_sk_y = delta_sk_y.unsqueeze(dim=dim)

        if dim is None or dim == 0:
            alpha = alpha.unsqueeze(dim=-1)

        alpha = alpha.unsqueeze(dim=-1)

        spline_val = (alpha - sk_x) / delta_sk_x
        spline_val = torch.maximum(
            torch.minimum(spline_val, torch.ones_like(spline_val)),
            torch.zeros_like(spline_val),
        )

        return qk_y + torch.sum(spline_val * delta_sk_y, dim=-1)

    def quantile_tail(
        self,
        alpha: torch.Tensor,
        dim: Optional[int] = None,
        left_tail: bool = True,
    ) -> torch.Tensor:
        # Refer to the description in quantile_internal

        if left_tail:
            tail_a, tail_b = self.tail_al, self.tail_bl
        else:
            tail_a, tail_b = self.tail_ar, self.tail_br
            alpha = 1 - alpha

        if dim is not None:
            tail_a, tail_b = tail_a.unsqueeze(dim=dim), tail_b.unsqueeze(dim=dim)

        return tail_a * torch.log(alpha) + tail_b

    def cdf_spline(self, z: torch.Tensor) -> torch.Tensor:
        """For observations z and splines defined in [qk_x[k], qk_x[k+1]].

        Computes the quantile level alpha_tilde such that:
        - alpha_tilde = q^{-1}(z) if z is in-between qk_x[k] and qk_x[k+1]
        - alpha_tilde = qk_x[k] if z< qk_x[k]
        - alpha_tilde = qk_x[k+1] if z>qk_x[k+1]

        Args:
            z (torch.Tensor): Observation. Shape: (*batch_shape,)

        Returns:
            torch.Tensor: Corresponding quantile level alpha_tilde. Shape: (*batch_shape, num_qk-1)
        """

        qk_y, qk_y_plus = self.qk_y, self.qk_y_plus
        qk_x, qk_x_plus = self.qk_x, self.qk_x_plus
        sk_x, delta_sk_x, delta_sk_y = (
            self.sk_x,
            self.delta_sk_x,
            self.delta_sk_y,
        )

        z_expand = z.unsqueeze(dim=-1)

        if self.num_pieces > 1:
            qk_y_expand = qk_y.unsqueeze(dim=-1)
            z_expand_twice = z_expand.unsqueeze(dim=-1)

            knots_eval = self.quantile_spline(sk_x, dim=-2)

            # Compute \sum_{s=0}^{s_0-1} \Delta sk_y[s],
            # where \Delta sk_y[s] = (sk_y[s+1]-sk_y[s])
            mask_sum_s0 = torch.lt(knots_eval, z_expand_twice)
            mask_sum_s0_minus = torch.cat(
                [
                    mask_sum_s0[..., 1:],
                    torch.zeros_like(qk_y_expand, dtype=torch.bool),
                ],
                dim=-1,
            )
            sum_delta_sk_y = torch.sum(mask_sum_s0_minus * delta_sk_y, dim=-1)

            mask_s0_only = torch.logical_and(
                mask_sum_s0, torch.logical_not(mask_sum_s0_minus)
            )
            # Compute (sk_x[s_0+1]-sk_x[s_0])/(sk_y[s_0+1]-sk_y[s_0])
            frac_s0 = torch.sum((mask_s0_only * delta_sk_x) / delta_sk_y, dim=-1)

            # Compute sk_x_{s_0}
            sk_x_s0 = torch.sum(mask_s0_only * sk_x, dim=-1)

            # Compute alpha_tilde
            alpha_tilde = sk_x_s0 + (z_expand - qk_y - sum_delta_sk_y) * frac_s0

        else:
            # num_pieces=1, ISQF reduces to IQF
            alpha_tilde = qk_x + (z_expand - qk_y) / (qk_y_plus - qk_y) * (
                qk_x_plus - qk_x
            )

        alpha_tilde = torch.minimum(torch.maximum(alpha_tilde, qk_x), qk_x_plus)

        return alpha_tilde

    def cdf_tail(self, z: torch.Tensor, left_tail: bool = True) -> torch.Tensor:
        """Computes the quantile level alpha_tilde such that:

        - alpha_tilde = q^{-1}(z) if z is in the tail region
        - alpha_tilde = qk_x_l or qk_x_r if z is in the non-tail region

        Args:
            z (torch.Tensor): Observation. Shape: (*batch_shape,)
            left_tail (bool, optional): If True, compute alpha_tilde for the left tail. Otherwise, compute alpha_tilde for the right tail. Defaults to True.

        Returns:
            torch.Tensor: Corresponding quantile level alpha_tilde. Shape: (*batch_shape,)
        """

        if left_tail:
            tail_a, tail_b, qk_x = self.tail_al, self.tail_bl, self.qk_x_l
        else:
            tail_a, tail_b, qk_x = self.tail_ar, self.tail_br, 1 - self.qk_x_r

        log_alpha_tilde = torch.minimum((z - tail_b) / tail_a, torch.log(qk_x))
        alpha_tilde = torch.exp(log_alpha_tilde)
        return alpha_tilde if left_tail else 1 - alpha_tilde

    def crps_tail(self, z: torch.Tensor, left_tail: bool = True) -> torch.Tensor:
        """Compute CRPS in analytical form for left/right tails.

        Args:
            z (torch.Tensor): Observation to evaluate. Shape: (*batch_shape,)
            left_tail (bool, optional): If True, compute CRPS for the left tail. Otherwise, compute CRPS for the right tail. Defaults to True.

        Returns:
            torch.Tensor: Tensor containing the CRPS, of the same shape as z.
        """

        alpha_tilde = self.cdf_tail(z, left_tail=left_tail)

        if left_tail:
            tail_a, tail_b, qk_x, qk_y = (
                self.tail_al,
                self.tail_bl,
                self.qk_x_l,
                self.qk_y_l,
            )
            term1 = (z - tail_b) * (qk_x**2 - 2 * qk_x + 2 * alpha_tilde)
            term2 = qk_x**2 * tail_a * (-torch.log(qk_x) + 0.5)
            term2 = term2 + 2 * torch.where(
                z < qk_y,
                qk_x * tail_a * (torch.log(qk_x) - 1)
                + alpha_tilde * (-z + tail_b + tail_a),
                torch.zeros_like(qk_x),
            )
        else:
            tail_a, tail_b, qk_x, qk_y = (
                self.tail_ar,
                self.tail_br,
                self.qk_x_r,
                self.qk_y_r,
            )
            term1 = (z - tail_b) * (-1 - qk_x**2 + 2 * alpha_tilde)
            term2 = tail_a * (
                -0.5 * (qk_x + 1) ** 2
                + (qk_x**2 - 1) * torch.log(1 - qk_x)
                + 2 * alpha_tilde
            )
            term2 = term2 + 2 * torch.where(
                z > qk_y,
                (1 - alpha_tilde) * (z - tail_b),
                tail_a * (1 - qk_x) * torch.log(1 - qk_x),
            )

        return term1 + term2

    def crps_spline(self, z: torch.Tensor) -> torch.Tensor:
        """Compute CRPS in analytical form for the spline.

        Args:
            z (torch.Tensor): Observation to evaluate.

        Returns:
            torch.Tensor: CRPS value for the spline.
        """

        qk_x, qk_x_plus, qk_y = self.qk_x, self.qk_x_plus, self.qk_y
        sk_x, sk_x_plus = self.sk_x, self.sk_x_plus
        delta_sk_x, delta_sk_y = self.delta_sk_x, self.delta_sk_y

        z_expand = z.unsqueeze(dim=-1)
        qk_x_plus_expand = qk_x_plus.unsqueeze(dim=-1)

        alpha_tilde = self.cdf_spline(z)
        alpha_tilde_expand = alpha_tilde.unsqueeze(dim=-1)

        r = torch.minimum(torch.maximum(alpha_tilde_expand, sk_x), sk_x_plus)

        coeff1 = (
            -2 / 3 * sk_x_plus**3
            + sk_x * sk_x_plus**2
            + sk_x_plus**2
            - (1 / 3) * sk_x**3
            - 2 * sk_x * sk_x_plus
            - r**2
            + 2 * sk_x * r
        )

        coeff2 = (
            -2 * torch.maximum(alpha_tilde_expand, sk_x_plus)
            + sk_x_plus**2
            + 2 * qk_x_plus_expand
            - qk_x_plus_expand**2
        )

        result = (
            (qk_x_plus**2 - qk_x**2) * (z_expand - qk_y)
            + 2 * (qk_x_plus - alpha_tilde) * (qk_y - z_expand)
            + torch.sum((delta_sk_y / delta_sk_x) * coeff1, dim=-1)
            + torch.sum(delta_sk_y * coeff2, dim=-1)
        )

        return torch.sum(result, dim=-1)

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.crps(z))

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return -F.softplus(self.crps(z))

    def crps(self, z: torch.Tensor) -> torch.Tensor:
        """Compute CRPS in analytical form.

        Args:
            z (torch.Tensor): Observation to evaluate.

        Returns:
            torch.Tensor: CRPS value.
        """

        crps_lt = self.crps_tail(z, left_tail=True)
        crps_rt = self.crps_tail(z, left_tail=False)

        return crps_lt + crps_rt + self.crps_spline(z)

    def cdf(self, z: torch.Tensor) -> torch.Tensor:
        """Computes the quantile level alpha_tilde such that q(alpha_tilde) = z.

        Args:
            z (torch.Tensor): Tensor of shape = (*batch_shape,)

        Returns:
            torch.Tensor: Quantile level alpha_tilde.
        """

        qk_y, qk_y_l, qk_y_plus = self.qk_y, self.qk_y_l, self.qk_y_plus

        alpha_tilde = torch.where(
            z < qk_y_l,
            self.cdf_tail(z, left_tail=True),
            self.cdf_tail(z, left_tail=False),
        )

        spline_alpha_tilde = self.cdf_spline(z)

        for spline_idx in range(self.num_qk - 1):
            is_in_between = torch.logical_and(
                qk_y[..., spline_idx] <= z, z < qk_y_plus[..., spline_idx]
            )

            alpha_tilde = torch.where(
                is_in_between, spline_alpha_tilde[..., spline_idx], alpha_tilde
            )

        return alpha_tilde

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Function used to draw random samples

        Args:
            sample_shape (torch.Size, optional): Shape of the sample. Defaults to torch.Size().

        Returns:
            torch.Tensor: Random samples.

        """

        # if sample_shape=()) then input_alpha should have the same shape
        # as beta_l, i.e., (*batch_shape,)
        # else u should be (*sample_shape, *batch_shape)
        target_shape = (
            self.beta_l.shape
            if sample_shape == torch.Size()
            else torch.Size(sample_shape) + self.beta_l.shape
        )

        alpha = torch.rand(
            target_shape,
            dtype=self.beta_l.dtype,
            device=self.beta_l.device,
            layout=self.beta_l.layout,
        )

        sample = self.quantile(alpha)

        if sample_shape == torch.Size():
            sample = sample.squeeze(dim=0)

        return sample

    @property
    def batch_shape(self) -> torch.Size:
        return self.beta_l.shape


def isqf_domain_map(
    input: torch.Tensor,
    tol: float = 1e-4,
    quantiles: torch.Tensor = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32),
    num_pieces: int = 5,
):
    """ISQF Domain Map
    Maps input into distribution constraints, by construction input's
    last dimension is of matching `distr_args` length.

    Args:
        input (torch.Tensor): Tensor of dimensions [B, H, N * n_outputs].
        tol (float, optional): Tolerance. Defaults to 1e-4.
        quantiles (torch.Tensor, optional): Quantiles used for ISQF (i.e. x-positions for the knots). Defaults to torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32).
        num_pieces (int, optional): Number of pieces used for each quantile spline. Defaults to 5.

    Returns:
        tuple: Tuple with tensors of ISQF distribution arguments.
    """

    # Add tol to prevent the y-distance of
    # two quantile knots from being too small
    #
    # Because in this case the spline knots could be squeezed together
    # and cause overflow in spline CRPS computation
    num_qk = len(quantiles)
    knots_pieces = (num_qk - 1) * num_pieces
    n_outputs = 2 * knots_pieces + 2 + num_qk

    # Reshape: [B, h, N * n_outputs] -> [B, h, N, n_outputs]
    input_reshaped = input.reshape(input.shape[0], input.shape[1], -1, n_outputs)
    spline_knots, spline_heights, beta_l, beta_r, quantile_knots = torch.split(
        input_reshaped, [knots_pieces, knots_pieces, 1, 1, num_qk], dim=-1
    )
    quantile_knots = torch.cat(
        [quantile_knots[..., :1], F.softplus(quantile_knots[..., 1:]) + tol], dim=-1
    )

    qk_y = torch.cumsum(quantile_knots, dim=-1)

    # Prevent overflow when we compute 1/beta
    beta_l = F.softplus(beta_l.squeeze(-1)) + tol
    beta_r = F.softplus(beta_r.squeeze(-1)) + tol

    # Reshape spline arguments
    batch_shape = spline_knots.shape[:-1]

    # repeat qk_x from (num_qk,) to (*batch_shape, num_qk)
    qk_x_repeat = quantiles.repeat(*batch_shape, 1).to(input.device)

    # knots and heights have shape (*batch_shape, (num_qk-1)*num_pieces)
    # reshape them to (*batch_shape, (num_qk-1), num_pieces)
    spline_knots_reshape = spline_knots.reshape(*batch_shape, (num_qk - 1), num_pieces)
    spline_heights_reshape = spline_heights.reshape(
        *batch_shape, (num_qk - 1), num_pieces
    )

    return (
        spline_knots_reshape,
        spline_heights_reshape,
        beta_l,
        beta_r,
        qk_y,
        qk_x_repeat,
    )


def isqf_scale_decouple(output, loc=None, scale=None):
    """ISQF Scale Decouple

    Stabilizes model's output optimization. We simply pass through
    the location and the scale to the (transformed) distribution constructor
    """
    spline_knots, spline_heights, beta_l, beta_r, qk_y, qk_x_repeat = output
    if loc is None:
        loc = torch.zeros_like(beta_l)
    if scale is None:
        scale = torch.ones_like(beta_l)

    return (spline_knots, spline_heights, beta_l, beta_r, qk_y, qk_x_repeat, loc, scale)


class DistributionLoss(torch.nn.Module):
    """DistributionLoss

    This PyTorch module wraps the `torch.distribution` classes allowing it to
    interact with NeuralForecast models modularly. It shares the negative
    log-likelihood as the optimization objective and a sample method to
    generate empirically the quantiles defined by the `level` list.

    Additionally, it implements a distribution transformation that factorizes the
    scale-dependent likelihood parameters into a base scale and a multiplier
    efficiently learnable within the network's non-linearities operating ranges.

    Available distributions:
    - Poisson
    - Normal
    - StudentT
    - NegativeBinomial
    - Tweedie
    - Bernoulli (Temporal Classifiers)
    - ISQF (Incremental Spline Quantile Function)

    Args:
        distribution (str): Identifier of a torch.distributions.Distribution class.
        level (float list): Confidence levels for prediction intervals.
        quantiles (float list): Alternative to level list, target quantiles.
        num_samples (int): Number of samples for the empirical quantiles.
        return_params (bool): Whether or not return the Distribution parameters.
        horizon_weight (Tensor): Tensor of size h, weight for each timestamp of the forecasting window.

    Returns:
        tuple: Tuple with tensors of ISQF distribution arguments.

    References:
        - [PyTorch Probability Distributions Package: StudentT.](https://pytorch.org/docs/stable/distributions.html#studentt)
        - [David Salinas, Valentin Flunkert, Jan Gasthaus, Tim Januschowski (2020). "DeepAR: Probabilistic forecasting with autoregressive recurrent networks". International Journal of Forecasting.](https://www.sciencedirect.com/science/article/pii/S0169207019301888)
        - [Park, Youngsuk, Danielle Maddix, François-Xavier Aubet, Kelvin Kan, Jan Gasthaus, and Yuyang Wang (2022). "Learning Quantile Functions without Quantile Crossing for Distribution-free Time Series Forecasting".](https://proceedings.mlr.press/v151/park22a.html)

    """

    def __init__(
        self,
        distribution,
        level=[80, 90],
        quantiles=None,
        num_samples=1000,
        return_params=False,
        horizon_weight=None,
        **distribution_kwargs,
    ):
        super(DistributionLoss, self).__init__()

        qs, self.output_names = level_to_outputs(level)
        qs = torch.Tensor(qs)

        # Transform quantiles to homogeneous output names
        if quantiles is not None:
            quantiles = sorted(quantiles)
            _, self.output_names = quantiles_to_outputs(quantiles)
            qs = torch.Tensor(quantiles)
        self.quantiles = torch.nn.Parameter(qs, requires_grad=False)
        num_qk = len(self.quantiles)

        # Generate a horizon weight tensor from the array
        if horizon_weight is not None:
            horizon_weight = torch.Tensor(horizon_weight.flatten())
        self.horizon_weight = horizon_weight

        if "num_pieces" not in distribution_kwargs:
            num_pieces = 5
        else:
            num_pieces = distribution_kwargs.pop("num_pieces")

        available_distributions = dict(
            Bernoulli=Bernoulli,
            Normal=Normal,
            Poisson=Poisson,
            StudentT=StudentT,
            NegativeBinomial=NegativeBinomial,
            Tweedie=Tweedie,
            ISQF=ISQF,
        )
        scale_decouples = dict(
            Bernoulli=bernoulli_scale_decouple,
            Normal=normal_scale_decouple,
            Poisson=poisson_scale_decouple,
            StudentT=student_scale_decouple,
            NegativeBinomial=nbinomial_scale_decouple,
            Tweedie=tweedie_scale_decouple,
            ISQF=isqf_scale_decouple,
        )
        param_names = dict(
            Bernoulli=["-logits"],
            Normal=["-loc", "-scale"],
            Poisson=["-loc"],
            StudentT=["-df", "-loc", "-scale"],
            NegativeBinomial=["-total_count", "-logits"],
            Tweedie=["-log_mu"],
            ISQF=[f"-spline_knot_{i + 1}" for i in range((num_qk - 1) * num_pieces)]
            + [f"-spline_height_{i + 1}" for i in range((num_qk - 1) * num_pieces)]
            + ["-beta_l", "-beta_r"]
            + [f"-quantile_knot_{i + 1}" for i in range(num_qk)],
        )
        assert (
            distribution in available_distributions.keys()
        ), f"{distribution} not available"
        if distribution == "ISQF":
            quantiles = torch.sort(qs).values
            self.domain_map = partial(
                isqf_domain_map, quantiles=quantiles, num_pieces=num_pieces
            )
            if return_params:
                raise Exception("ISQF does not support 'return_params=True'")
        elif distribution == "Tweedie":
            rho = distribution_kwargs.pop("rho")
            self.domain_map = partial(tweedie_domain_map, rho=rho)
            if return_params:
                raise Exception("Tweedie does not support 'return_params=True'")
        else:
            self.domain_map = self._domain_map

        self.distribution = distribution
        self._base_distribution = available_distributions[distribution]
        self.scale_decouple = scale_decouples[distribution]
        self.distribution_kwargs = distribution_kwargs
        self.num_samples = num_samples
        self.param_names = param_names[distribution]

        # If True, predict_step will return Distribution's parameters
        self.return_params = return_params
        if self.return_params:
            self.output_names = self.output_names + self.param_names

        # Add first output entry for the sample_mean
        self.output_names.insert(0, "")

        self.outputsize_multiplier = len(self.param_names)
        self.is_distribution_output = True
        self.has_predicted = False

    def _domain_map(self, input: torch.Tensor):
        """
        Maps output of neural network to domain of distribution loss

        """
        output = torch.tensor_split(input, self.outputsize_multiplier, dim=2)

        return output

    def get_distribution(self, distr_args, **distribution_kwargs) -> Distribution:
        """
        Construct the associated Pytorch Distribution, given the collection of
        constructor arguments and, optionally, location and scale tensors.

        Args:
            distr_args (torch.Tensor): Constructor arguments for the underlying Distribution type.

        Returns:
            Distribution: AffineTransformed distribution.
        """
        distr = self._base_distribution(*distr_args, **distribution_kwargs)
        self.distr_mean = distr.mean

        if self.distribution in ("Poisson", "NegativeBinomial"):
            distr.support = constraints.nonnegative
        return distr

    def sample(self, distr_args: torch.Tensor, num_samples: Optional[int] = None):
        """
        Construct the empirical quantiles from the estimated Distribution,
        sampling from it `num_samples` independently.

        Args:
            distr_args (torch.Tensor): Constructor arguments for the underlying Distribution type.
            num_samples (int, optional): Overwrite number of samples for the empirical quantiles. Defaults to None.

        Returns:
            tuple: Tuple with samples, sample mean, and quantiles.
        """
        if num_samples is None:
            num_samples = self.num_samples

        # Instantiate Scaled Decoupled Distribution
        distr = self.get_distribution(distr_args=distr_args, **self.distribution_kwargs)
        samples = distr.sample(sample_shape=(num_samples,))
        samples = samples.permute(
            1, 2, 3, 0
        )  # [samples, B, H, N] -> [B, H, N, samples]

        sample_mean = torch.mean(samples, dim=-1, keepdim=True)

        # Compute quantiles
        quantiles_device = self.quantiles.to(distr_args[0].device)
        quants = torch.quantile(input=samples, q=quantiles_device, dim=-1)
        quants = quants.permute(1, 2, 3, 0)  # [Q, B, H, N] -> [B, H, N, Q]

        return samples, sample_mean, quants

    def update_quantile(self, q: Optional[List[float]] = None):
        if q is not None:
            self.quantiles = nn.Parameter(
                torch.tensor(q, dtype=torch.float32), requires_grad=False
            )
            self.output_names = (
                [""]
                + [f"_ql{q_i}" for q_i in q]
                + self.return_params * self.param_names
            )
            self.has_predicted = True
        elif q is None and self.has_predicted:
            self.quantiles = nn.Parameter(
                torch.tensor([0.5], dtype=torch.float32), requires_grad=False
            )
            self.output_names = ["", "-median"] + self.return_params * self.param_names

    def _compute_weights(self, y, mask):
        """
        Compute final weights for each datapoint (based on all weights and all masks)
        Set horizon_weight to a ones[H] tensor if not set.
        If set, check that it has the same length as the horizon in x.
        """
        if mask is None:
            mask = torch.ones_like(y)

        if self.horizon_weight is None:
            weights = torch.ones_like(mask)
        else:
            assert mask.shape[1] == len(
                self.horizon_weight
            ), "horizon_weight must have same length as Y"
            weights = self.horizon_weight.clone()
            weights = weights[None, :, None].to(mask.device)
            weights = torch.ones_like(mask, device=mask.device) * weights

        return weights * mask

    def __call__(
        self,
        y: torch.Tensor,
        distr_args: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
        """
        Computes the negative log-likelihood objective function.
        To estimate the following predictive distribution:

        $$
        \mathrm{P}(\mathbf{y}_{\\tau}\,|\,\\theta) \\quad \mathrm{and} \\quad -\log(\mathrm{P}(\mathbf{y}_{\\tau}\,|\,\\theta))
        $$

        where $\\theta$ represents the distributions parameters. It aditionally
        summarizes the objective signal using a weighted average using the `mask` tensor.

        Args:
            y (torch.Tensor): Actual values.
            distr_args (torch.Tensor): Constructor arguments for the underlying Distribution type.
            loc (Optional[torch.Tensor], optional): Optional tensor, of the same shape as the batch_shape + event_shape. Defaults to None.
               of the resulting distribution.<br>
            scale (Optional[torch.Tensor], optional): Optional tensor, of the same shape as the batch_shape+event_shape
               of the resulting distribution. Defaults to None.
            mask (Union[torch.Tensor, None], optional): Specifies date stamps per serie to consider in loss. Defaults to None.

        Returns:
            float: Weighted loss function against which backpropagation will be performed.
        """
        # Instantiate Scaled Decoupled Distribution
        distr = self.get_distribution(distr_args=distr_args, **self.distribution_kwargs)
        loss_values = -distr.log_prob(y)
        loss_weights = self._compute_weights(y=y, mask=mask)
        return weighted_average(loss_values, weights=loss_weights)


class PMM(torch.nn.Module):
    """Poisson Mixture Mesh

    This Poisson Mixture statistical model assumes independence across groups of
    data $\mathcal{G}=\{[g_{i}]\}$, and estimates relationships within the group.

    $$
    \mathrm{P}\\left(\mathbf{y}_{[b][t+1:t+H]}\\right) =
    \prod_{ [g_{i}] \in \mathcal{G}} \mathrm{P} \\left(\mathbf{y}_{[g_{i}][\\tau]} \\right) =
    \prod_{\\beta\in[g_{i}]}
    \\left(\sum_{k=1}^{K} w_k \prod_{(\\beta,\\tau) \in [g_i][t+1:t+H]} \mathrm{Poisson}(y_{\\beta,\\tau}, \hat{\\lambda}_{\\beta,\\tau,k}) \\right)
    $$

    Args:
        n_components (int, optional): The number of mixture components. Defaults to 10.
        level (float list, optional): Confidence levels for prediction intervals. Defaults to [80, 90].
        quantiles (float list, optional): Alternative to level list, target quantiles. Defaults to None.
        return_params (bool, optional): Whether or not return the Distribution parameters. Defaults to False.
        batch_correlation (bool, optional): Whether or not model batch correlations. Defaults to False.
        horizon_correlation (bool, optional): Whether or not model horizon correlations. Defaults to False.

    References:
        - [Kin G. Olivares, O. Nganba Meetei, Ruijun Ma, Rohan Reddy, Mengfei Cao, Lee Dicker. Probabilistic Hierarchical Forecasting with Deep Poisson Mixtures. Submitted to the International Journal Forecasting, Working paper available at arxiv.](https://arxiv.org/pdf/2110.13179.pdf)
    """

    def __init__(
        self,
        n_components=10,
        level=[80, 90],
        quantiles=None,
        num_samples=1000,
        return_params=False,
        batch_correlation=False,
        horizon_correlation=False,
        weighted=False,
    ):
        super(PMM, self).__init__()
        # Transform level to MQLoss parameters
        qs, self.output_names = level_to_outputs(level)
        qs = torch.Tensor(qs)

        # Transform quantiles to homogeneous output names
        if quantiles is not None:
            _, self.output_names = quantiles_to_outputs(quantiles)
            qs = torch.Tensor(quantiles)
        self.quantiles = torch.nn.Parameter(qs, requires_grad=False)
        self.num_samples = num_samples
        self.batch_correlation = batch_correlation
        self.horizon_correlation = horizon_correlation
        self.weighted = weighted

        # If True, predict_step will return Distribution's parameters
        self.return_params = return_params

        lambda_names = [f"-lambda-{i}" for i in range(1, n_components + 1)]
        if weighted:
            weight_names = [f"-weight-{i}" for i in range(1, n_components + 1)]
            self.param_names = [i for j in zip(lambda_names, weight_names) for i in j]
        else:
            self.param_names = lambda_names

        if self.return_params:
            self.output_names = self.output_names + self.param_names

        # Add first output entry for the sample_mean
        self.output_names.insert(0, "")

        self.n_outputs = 1 + weighted
        self.n_components = n_components
        self.outputsize_multiplier = self.n_outputs * n_components
        self.is_distribution_output = True
        self.has_predicted = False

    def domain_map(self, output: torch.Tensor):
        output = output.reshape(
            output.shape[0], output.shape[1], -1, self.outputsize_multiplier
        )

        return torch.tensor_split(output, self.n_outputs, dim=-1)

    def scale_decouple(
        self,
        output,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ):
        """Scale Decouple

        Stabilizes model's output optimization, by learning residual
        variance and residual location based on anchoring `loc`, `scale`.
        Also adds domain protection to the distribution parameters.
        """
        if self.weighted:
            lambdas, weights = output
            weights = F.softmax(weights, dim=-1)
        else:
            lambdas = output[0]

        if (loc is not None) and (scale is not None):
            if loc.ndim == 3:
                loc = loc.unsqueeze(-1)
                scale = scale.unsqueeze(-1)
            lambdas = (lambdas * scale) + loc

        lambdas = F.softplus(lambdas) + 1e-3

        if self.weighted:
            return (lambdas, weights)
        else:
            return (lambdas,)

    def get_distribution(self, distr_args) -> Distribution:
        """
        Construct the associated Pytorch Distribution, given the collection of
        constructor arguments and, optionally, location and scale tensors.

        Args:
            distr_args (torch.Tensor): Constructor arguments for the underlying Distribution type.

        Returns:
            Distribution: AffineTransformed distribution.
        """
        if self.weighted:
            lambdas, weights = distr_args
        else:
            lambdas = distr_args[0]
            weights = torch.full_like(lambdas, fill_value=1 / self.n_components)

        mix = Categorical(weights)
        components = Poisson(rate=lambdas)
        components.support = constraints.nonnegative
        distr = MixtureSameFamily(
            mixture_distribution=mix, component_distribution=components
        )

        self.distr_mean = distr.mean

        return distr

    def sample(self, distr_args: torch.Tensor, num_samples: Optional[int] = None):
        """
        Construct the empirical quantiles from the estimated Distribution,
        sampling from it `num_samples` independently.

        Args:
            distr_args (torch.Tensor): Constructor arguments for the underlying Distribution type.
            num_samples (int, optional): Overwrite number of samples for the empirical quantiles. Defaults to None.

        Returns:
            tuple: Tuple with samples, sample mean, and quantiles.
        """
        if num_samples is None:
            num_samples = self.num_samples

        # Instantiate Scaled Decoupled Distribution
        distr = self.get_distribution(distr_args=distr_args)
        samples = distr.sample(sample_shape=(num_samples,))
        samples = samples.permute(
            1, 2, 3, 0
        )  # [samples, B, H, N] -> [B, H, N, samples]

        sample_mean = torch.mean(samples, dim=-1, keepdim=True)

        # Compute quantiles
        quantiles_device = self.quantiles.to(distr_args[0].device)
        quants = torch.quantile(input=samples, q=quantiles_device, dim=-1)
        quants = quants.permute(1, 2, 3, 0)  # [Q, B, H, N] -> [B, H, N, Q]

        return samples, sample_mean, quants

    def update_quantile(self, q: Optional[List[float]] = None):
        if q is not None:
            self.quantiles = nn.Parameter(
                torch.tensor(q, dtype=torch.float32), requires_grad=False
            )
            self.output_names = (
                [""]
                + [f"_ql{q_i}" for q_i in q]
                + self.return_params * self.param_names
            )
            self.has_predicted = True
        elif q is None and self.has_predicted:
            self.quantiles = nn.Parameter(
                torch.tensor([0.5], dtype=torch.float32), requires_grad=False
            )
            self.output_names = ["", "-median"] + self.return_params * self.param_names

    def __call__(
        self,
        y: torch.Tensor,
        distr_args: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
        """
        Computes the negative log-likelihood objective function.
        To estimate the following predictive distribution:

        $$
        \mathrm{P}(\mathbf{y}_{\\tau}\,|\,\\theta) \\quad \mathrm{and} \\quad -\log(\mathrm{P}(\mathbf{y}_{\\tau}\,|\,\\theta))
        $$

        where $\\theta$ represents the distributions parameters. It aditionally
        summarizes the objective signal using a weighted average using the `mask` tensor.

        Args:
            y (torch.Tensor): Actual values.
            distr_args (torch.Tensor): Constructor arguments for the underlying Distribution type.
            mask (Union[torch.Tensor, None], optional): Specifies date stamps per serie to consider in loss. Defaults to None.

        Returns:
            float: Weighted loss function against which backpropagation will be performed.
        """
        # Instantiate Scaled Decoupled Distribution
        distr = self.get_distribution(distr_args=distr_args)
        x = distr._pad(y)
        log_prob_x = distr.component_distribution.log_prob(x)
        log_mix_prob = torch.log_softmax(distr.mixture_distribution.logits, dim=-1)
        if self.batch_correlation:
            log_prob_x = torch.sum(log_prob_x, dim=0, keepdim=True)
        if self.horizon_correlation:
            log_prob_x = torch.sum(log_prob_x, dim=1, keepdim=True)

        loss_values = -torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)

        return weighted_average(loss_values, weights=mask)


class GMM(torch.nn.Module):
    """Gaussian Mixture Mesh

    This Gaussian Mixture statistical model assumes independence across groups of
    data $\mathcal{G}=\{[g_{i}]\}$, and estimates relationships within the group.

    $$
    \mathrm{P}\\left(\mathbf{y}_{[b][t+1:t+H]}\\right) =
    \prod_{ [g_{i}] \in \mathcal{G}} \mathrm{P}\left(\mathbf{y}_{[g_{i}][\\tau]}\\right)=
    \prod_{\\beta\in[g_{i}]}
    \\left(\sum_{k=1}^{K} w_k \prod_{(\\beta,\\tau) \in [g_i][t+1:t+H]}
    \mathrm{Gaussian}(y_{\\beta,\\tau}, \hat{\mu}_{\\beta,\\tau,k}, \sigma_{\\beta,\\tau,k})\\right)
    $$

    Args:
        n_components (int, optional): The number of mixture components. Defaults to 10.
        level (float list, optional): Confidence levels for prediction intervals. Defaults to [80, 90].
        quantiles (float list, optional): Alternative to level list, target quantiles. Defaults to None.
        return_params (bool, optional): Whether or not return the Distribution parameters. Defaults to False.
        batch_correlation (bool, optional): Whether or not model batch correlations. Defaults to False.
        horizon_correlation (bool, optional): Whether or not model horizon correlations. Defaults to False.
        weighted (bool, optional): Whether or not model weighted components. Defaults to False.
        num_samples (int, optional): Number of samples for the empirical quantiles. Defaults to 1000.

    References:
        - [Kin G. Olivares, O. Nganba Meetei, Ruijun Ma, Rohan Reddy, Mengfei Cao, Lee Dicker.
            Probabilistic Hierarchical Forecasting with Deep Poisson Mixtures. Submitted to the International
            Journal Forecasting, Working paper available at arxiv.](https://arxiv.org/pdf/2110.13179.pdf)
    """

    def __init__(
        self,
        n_components=1,
        level=[80, 90],
        quantiles=None,
        num_samples=1000,
        return_params=False,
        batch_correlation=False,
        horizon_correlation=False,
        weighted=False,
    ):
        super(GMM, self).__init__()
        # Transform level to MQLoss parameters
        qs, self.output_names = level_to_outputs(level)
        qs = torch.Tensor(qs)

        # Transform quantiles to homogeneous output names
        if quantiles is not None:
            _, self.output_names = quantiles_to_outputs(quantiles)
            qs = torch.Tensor(quantiles)
        self.quantiles = torch.nn.Parameter(qs, requires_grad=False)
        self.num_samples = num_samples
        self.batch_correlation = batch_correlation
        self.horizon_correlation = horizon_correlation
        self.weighted = weighted

        # If True, predict_step will return Distribution's parameters
        self.return_params = return_params

        mu_names = [f"-mu-{i}" for i in range(1, n_components + 1)]
        std_names = [f"-std-{i}" for i in range(1, n_components + 1)]
        if weighted:
            weight_names = [f"-weight-{i}" for i in range(1, n_components + 1)]
            self.param_names = [
                i for j in zip(mu_names, std_names, weight_names) for i in j
            ]
        else:
            self.param_names = [i for j in zip(mu_names, std_names) for i in j]

        if self.return_params:
            self.output_names = self.output_names + self.param_names

        # Add first output entry for the sample_mean
        self.output_names.insert(0, "")

        self.n_outputs = 2 + weighted
        self.n_components = n_components
        self.outputsize_multiplier = self.n_outputs * n_components
        self.is_distribution_output = True
        self.has_predicted = False

    def domain_map(self, output: torch.Tensor):
        output = output.reshape(
            output.shape[0], output.shape[1], -1, self.outputsize_multiplier
        )

        return torch.tensor_split(output, self.n_outputs, dim=-1)

    def scale_decouple(
        self,
        output,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        eps: float = 0.2,
    ):
        """Scale Decouple

        Stabilizes model's output optimization, by learning residual
        variance and residual location based on anchoring `loc`, `scale`.
        Also adds domain protection to the distribution parameters.
        """
        if self.weighted:
            means, stds, weights = output
            weights = F.softmax(weights, dim=-1)
        else:
            means, stds = output

        stds = F.softplus(stds)
        if (loc is not None) and (scale is not None):
            if loc.ndim == 3:
                loc = loc.unsqueeze(-1)
                scale = scale.unsqueeze(-1)
            means = (means * scale) + loc
            stds = (stds + eps) * scale

        if self.weighted:
            return (means, stds, weights)
        else:
            return (means, stds)

    def get_distribution(self, distr_args) -> Distribution:
        """
        Construct the associated Pytorch Distribution, given the collection of
        constructor arguments and, optionally, location and scale tensors.

        Args:
            distr_args (torch.Tensor): Constructor arguments for the underlying Distribution type.

        Returns:
            Distribution: AffineTransformed distribution.
        """
        if self.weighted:
            means, stds, weights = distr_args
        else:
            means, stds = distr_args
            weights = torch.full_like(means, fill_value=1 / self.n_components)

        mix = Categorical(weights)
        components = Normal(loc=means, scale=stds)
        distr = MixtureSameFamily(
            mixture_distribution=mix, component_distribution=components
        )

        self.distr_mean = distr.mean

        return distr

    def sample(self, distr_args: torch.Tensor, num_samples: Optional[int] = None):
        """
        Construct the empirical quantiles from the estimated Distribution,
        sampling from it `num_samples` independently.

        Args:
            distr_args (torch.Tensor): Constructor arguments for the underlying Distribution type.
            num_samples (int, optional): Overwrite number of samples for the empirical quantiles. Defaults to None.

        Returns:
            tuple: Tuple with samples, sample mean, and quantiles.
        """
        if num_samples is None:
            num_samples = self.num_samples

        # Instantiate Scaled Decoupled Distribution
        distr = self.get_distribution(distr_args=distr_args)
        samples = distr.sample(sample_shape=(num_samples,))
        samples = samples.permute(
            1, 2, 3, 0
        )  # [samples, B, H, N] -> [B, H, N, samples]

        sample_mean = torch.mean(samples, dim=-1, keepdim=True)

        # Compute quantiles
        quantiles_device = self.quantiles.to(distr_args[0].device)
        quants = torch.quantile(input=samples, q=quantiles_device, dim=-1)
        quants = quants.permute(1, 2, 3, 0)  # [Q, B, H, N] -> [B, H, N, Q]

        return samples, sample_mean, quants

    def update_quantile(self, q: Optional[List[float]] = None):
        if q is not None:
            self.quantiles = nn.Parameter(
                torch.tensor(q, dtype=torch.float32), requires_grad=False
            )
            self.output_names = (
                [""]
                + [f"_ql{q_i}" for q_i in q]
                + self.return_params * self.param_names
            )
            self.has_predicted = True
        elif q is None and self.has_predicted:
            self.quantiles = nn.Parameter(
                torch.tensor([0.5], dtype=torch.float32), requires_grad=False
            )
            self.output_names = ["", "-median"] + self.return_params * self.param_names

    def __call__(
        self,
        y: torch.Tensor,
        distr_args: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
        """
        Computes the negative log-likelihood objective function.
        To estimate the following predictive distribution:

        $$\mathrm{P}(\mathbf{y}_{\\tau}\,|\,\\theta) \\quad \mathrm{and} \\quad -\log(\mathrm{P}(\mathbf{y}_{\\tau}\,|\,\\theta))$$

        where $\\theta$ represents the distributions parameters. It aditionally
        summarizes the objective signal using a weighted average using the `mask` tensor.

        Args:
            y (torch.Tensor): Actual values.
            distr_args (torch.Tensor): Constructor arguments for the underlying Distribution type.
            mask (Union[torch.Tensor, None], optional): Specifies date stamps per serie to consider in loss. Defaults to None.

        Returns:
            float: Weighted loss function against which backpropagation will be performed.
        """
        # Instantiate Scaled Decoupled Distribution
        distr = self.get_distribution(distr_args=distr_args)
        x = distr._pad(y)
        log_prob_x = distr.component_distribution.log_prob(x)
        log_mix_prob = torch.log_softmax(distr.mixture_distribution.logits, dim=-1)
        if self.batch_correlation:
            log_prob_x = torch.sum(log_prob_x, dim=0, keepdim=True)
        if self.horizon_correlation:
            log_prob_x = torch.sum(log_prob_x, dim=1, keepdim=True)
        loss_values = -torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)

        return weighted_average(loss_values, weights=mask)


class NBMM(torch.nn.Module):
    """Negative Binomial Mixture Mesh

    This N. Binomial Mixture statistical model assumes independence across groups of
    data $\mathcal{G}=\{[g_{i}]\}$, and estimates relationships within the group.

    $$
    \mathrm{P}\\left(\mathbf{y}_{[b][t+1:t+H]}\\right) =
    \prod_{ [g_{i}] \in \mathcal{G}} \mathrm{P}\left(\mathbf{y}_{[g_{i}][\\tau]}\\right)=
    \prod_{\\beta\in[g_{i}]}
    \\left(\sum_{k=1}^{K} w_k \prod_{(\\beta,\\tau) \in [g_i][t+1:t+H]}
    \mathrm{NBinomial}(y_{\\beta,\\tau}, \hat{r}_{\\beta,\\tau,k}, \hat{p}_{\\beta,\\tau,k})\\right)
    $$

    Args:
        n_components (int, optional): The number of mixture components. Defaults to 10.
        level (float list, optional): Confidence levels for prediction intervals. Defaults to [80, 90].
        quantiles (float list, optional): Alternative to level list, target quantiles. Defaults to None.
        return_params (bool, optional): Whether or not return the Distribution parameters. Defaults to False.
        weighted (bool, optional): Whether or not model weighted components. Defaults to False.
        num_samples (int, optional): Number of samples for the empirical quantiles. Defaults to 1000.

    References:
        - [Kin G. Olivares, O. Nganba Meetei, Ruijun Ma, Rohan Reddy, Mengfei Cao, Lee Dicker.
            Probabilistic Hierarchical Forecasting with Deep Poisson Mixtures. Submitted to the International
            Journal Forecasting, Working paper available at arxiv.](https://arxiv.org/pdf/2110.13179.pdf)
    """

    def __init__(
        self,
        n_components=1,
        level=[80, 90],
        quantiles=None,
        num_samples=1000,
        return_params=False,
        weighted=False,
    ):
        super(NBMM, self).__init__()
        # Transform level to MQLoss parameters
        qs, self.output_names = level_to_outputs(level)
        qs = torch.Tensor(qs)

        # Transform quantiles to homogeneous output names
        if quantiles is not None:
            _, self.output_names = quantiles_to_outputs(quantiles)
            qs = torch.Tensor(quantiles)
        self.quantiles = torch.nn.Parameter(qs, requires_grad=False)
        self.num_samples = num_samples
        self.weighted = weighted

        # If True, predict_step will return Distribution's parameters
        self.return_params = return_params

        total_count_names = [f"-total_count-{i}" for i in range(1, n_components + 1)]
        probs_names = [f"-probs-{i}" for i in range(1, n_components + 1)]
        if weighted:
            weight_names = [f"-weight-{i}" for i in range(1, n_components + 1)]
            self.param_names = [
                i for j in zip(total_count_names, probs_names, weight_names) for i in j
            ]
        else:
            self.param_names = [
                i for j in zip(total_count_names, probs_names) for i in j
            ]

        if self.return_params:
            self.output_names = self.output_names + self.param_names

        # Add first output entry for the sample_mean
        self.output_names.insert(0, "")

        self.n_outputs = 2 + weighted
        self.n_components = n_components
        self.outputsize_multiplier = self.n_outputs * n_components
        self.is_distribution_output = True
        self.has_predicted = False

    def domain_map(self, output: torch.Tensor):
        output = output.reshape(
            output.shape[0], output.shape[1], -1, self.outputsize_multiplier
        )

        return torch.tensor_split(output, self.n_outputs, dim=-1)

    def scale_decouple(
        self,
        output,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        eps: float = 0.2,
    ):
        """Scale Decouple

        Stabilizes model's output optimization, by learning residual
        variance and residual location based on anchoring `loc`, `scale`.
        Also adds domain protection to the distribution parameters.
        """
        # Efficient NBinomial parametrization
        if self.weighted:
            mu, alpha, weights = output
            weights = F.softmax(weights, dim=-1)
        else:
            mu, alpha = output

        mu = F.softplus(mu) + 1e-8
        alpha = F.softplus(alpha) + 1e-8  # alpha = 1/total_counts
        if (loc is not None) and (scale is not None):
            if loc.ndim == 3:
                loc = loc.unsqueeze(-1)
                scale = scale.unsqueeze(-1)
            mu *= loc
            alpha /= loc + 1.0

        # mu = total_count * (probs/(1-probs))
        # => probs = mu / (total_count + mu)
        # => probs = mu / [total_count * (1 + mu * (1/total_count))]
        total_count = 1.0 / alpha
        probs = (mu * alpha / (1.0 + mu * alpha)) + 1e-8
        if self.weighted:
            return (total_count, probs, weights)
        else:
            return (total_count, probs)

    def get_distribution(self, distr_args) -> Distribution:
        """
        Construct the associated Pytorch Distribution, given the collection of
        constructor arguments and, optionally, location and scale tensors.

        Args:
            distr_args (torch.Tensor): Constructor arguments for the underlying Distribution type.

        Returns:
            Distribution: AffineTransformed distribution.
        """
        if self.weighted:
            total_count, probs, weights = distr_args
        else:
            total_count, probs = distr_args
            weights = torch.full_like(total_count, fill_value=1 / self.n_components)

        mix = Categorical(weights)
        components = NegativeBinomial(total_count, probs)
        components.support = constraints.nonnegative
        distr = MixtureSameFamily(
            mixture_distribution=mix, component_distribution=components
        )

        self.distr_mean = distr.mean

        return distr

    def sample(self, distr_args: torch.Tensor, num_samples: Optional[int] = None):
        """
        Construct the empirical quantiles from the estimated Distribution,
        sampling from it `num_samples` independently.

        Args:
            distr_args (torch.Tensor): Constructor arguments for the underlying Distribution type.
            num_samples (int, optional): Overwrite number of samples for the empirical quantiles. Defaults to None.

        Returns:
            tuple: Tuple with samples, sample mean, and quantiles.
        """
        if num_samples is None:
            num_samples = self.num_samples

        # Instantiate Scaled Decoupled Distribution
        distr = self.get_distribution(distr_args=distr_args)
        samples = distr.sample(sample_shape=(num_samples,))
        samples = samples.permute(
            1, 2, 3, 0
        )  # [samples, B, H, N] -> [B, H, N, samples]

        sample_mean = torch.mean(samples, dim=-1, keepdim=True)

        # Compute quantiles
        quantiles_device = self.quantiles.to(distr_args[0].device)
        quants = torch.quantile(input=samples, q=quantiles_device, dim=-1)
        quants = quants.permute(1, 2, 3, 0)  # [Q, B, H, N] -> [B, H, N, Q]

        return samples, sample_mean, quants

    def update_quantile(self, q: Optional[List[float]] = None):
        if q is not None:
            self.quantiles = nn.Parameter(
                torch.tensor(q, dtype=torch.float32), requires_grad=False
            )
            self.output_names = (
                [""]
                + [f"_ql{q_i}" for q_i in q]
                + self.return_params * self.param_names
            )
            self.has_predicted = True
        elif q is None and self.has_predicted:
            self.quantiles = nn.Parameter(
                torch.tensor([0.5], dtype=torch.float32), requires_grad=False
            )
            self.output_names = ["", "-median"] + self.return_params * self.param_names

    def __call__(
        self,
        y: torch.Tensor,
        distr_args: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
        """
        Computes the negative log-likelihood objective function.
        To estimate the following predictive distribution:

        $$
        \mathrm{P}(\mathbf{y}_{\\tau}\,|\,\\theta) \\quad \mathrm{and} \\quad -\log(\mathrm{P}(\mathbf{y}_{\\tau}\,|\,\\theta))
        $$

        where $\\theta$ represents the distributions parameters. It aditionally
        summarizes the objective signal using a weighted average using the `mask` tensor.

        Args:
            y (torch.Tensor): Actual values.
            distr_args (torch.Tensor): Constructor arguments for the underlying Distribution type.
            mask (Union[torch.Tensor, None], optional): Specifies date stamps per serie to consider in loss. Defaults to None.

        Returns:
            float: Weighted loss function against which backpropagation will be performed.
        """
        # Instantiate Scaled Decoupled Distribution
        distr = self.get_distribution(distr_args=distr_args)
        loss_values = -distr.log_prob(y)
        loss_weights = mask

        return weighted_average(loss_values, weights=loss_weights)


class HuberLoss(BasePointLoss):
    """ Huber Loss

    The Huber loss, employed in robust regression, is a loss function that
    exhibits reduced sensitivity to outliers in data when compared to the
    squared error loss. This function is also refered as SmoothL1.

    The Huber loss function is quadratic for small errors and linear for large
    errors, with equal values and slopes of the different sections at the two
    points where $(y_{\\tau}-\hat{y}_{\\tau})^{2}$=$|y_{\\tau}-\hat{y}_{\\tau}|$.

    $$
    L_{\delta}(y_{\\tau},\; \hat{y}_{\\tau})
    =\\begin{cases}{\\frac{1}{2}}(y_{\\tau}-\hat{y}_{\\tau})^{2}\;{\\text{for }}|y_{\\tau}-\hat{y}_{\\tau}|\leq \delta \\\
    \\delta \ \cdot \left(|y_{\\tau}-\hat{y}_{\\tau}|-{\\frac {1}{2}}\delta \\right),\;{\\text{otherwise.}}\end{cases}
    $$

    where $\\delta$ is a threshold parameter that determines the point at which the loss transitions from quadratic to linear,
    and can be tuned to control the trade-off between robustness and accuracy in the predictions.

    Args:
        delta (float, optional): Specifies the threshold at which to change between delta-scaled L1 and L2 loss. Defaults to 1.0.
        horizon_weight (Union[torch.Tensor, None], optional): Tensor of size h, weight for each timestamp of the forecasting window. Defaults to None.

    References:
        - [Huber Peter, J (1964). "Robust Estimation of a Location Parameter". Annals of Statistics](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full)
    """

    def __init__(self, delta: float = 1.0, horizon_weight=None):
        super(HuberLoss, self).__init__(
            horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""]
        )
        self.delta = delta

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: Union[torch.Tensor, None] = None,
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Args:
            y (torch.Tensor): Actual values.
            y_hat (torch.Tensor): Predicted values.
            mask (Union[torch.Tensor, None], optional): Specifies date stamps per serie to consider in loss. Defaults to None.

        Returns:
            float: Huber loss.
        """
        losses = F.huber_loss(y, y_hat, reduction="none", delta=self.delta)
        weights = self._compute_weights(y=y, mask=mask)
        return _weighted_mean(losses=losses, weights=weights)


class TukeyLoss(BasePointLoss):
    """ Tukey Loss

    The Tukey loss function, also known as Tukey's biweight function, is a
    robust statistical loss function used in robust statistics. Tukey's loss exhibits
    quadratic behavior near the origin, like the Huber loss; however, it is even more
    robust to outliers as the loss for large residuals remains constant instead of
    scaling linearly.

    The parameter $c$ in Tukey's loss determines the ''saturation'' point
    of the function: Higher values of $c$ enhance sensitivity, while lower values
    increase resistance to outliers.

    $$
    L_{c}(y_{\\tau},\; \hat{y}_{\\tau})
    =\\begin{cases}{
    \\frac{c^{2}}{6}} \\left[1-(\\frac{y_{\\tau}-\hat{y}_{\\tau}}{c})^{2} \\right]^{3}    \;\\text{for } |y_{\\tau}-\hat{y}_{\\tau}|\leq c \\\
    \\frac{c^{2}}{6} \qquad \\text{otherwise.}  \end{cases}
    $$

    Please note that the Tukey loss function assumes the data to be stationary or
    normalized beforehand. If the error values are excessively large, the algorithm
    may need help to converge during optimization. It is advisable to employ small learning rates.

    Args:
        c (float, optional): Specifies the Tukey loss' threshold on which residuals are no longer considered. Defaults to 4.685.
        normalize (bool, optional): Wether normalization is performed within Tukey loss' computation. Defaults to True.

    References:
        - [Beaton, A. E., and Tukey, J. W. (1974). "The Fitting of Power Series, Meaning Polynomials, Illustrated on Band-Spectroscopic Data."](https://www.jstor.org/stable/1267936)
    """

    def __init__(self, c: float = 4.685, normalize: bool = True):
        super(TukeyLoss, self).__init__()
        self.outputsize_multiplier = 1
        self.c = c
        self.normalize = normalize
        self.output_names = [""]
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Args:
            y_hat (torch.Tensor): Predicted values
            - shape: [B, H, 1] for univariate
            - shape: [B, H, N] for multivariate

        Returns:
            torch.Tensor: Transformed values.
            - shape: [B, H, 1] for univariate
            - shape: [B, H, N] for multivariate
        """

        return y_hat

    def masked_mean(self, x, mask, dim):
        x_nan = x.masked_fill(mask < 1, float("nan"))
        x_mean = x_nan.nanmean(dim=dim, keepdim=True)
        x_mean = torch.nan_to_num(x_mean, nan=0.0)
        return x_mean

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: Union[torch.Tensor, None] = None,
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Args:
            y (torch.Tensor): Actual values.
            y_hat (torch.Tensor): Predicted values.
            mask (Union[torch.Tensor, None], optional): Specifies date stamps per serie to consider in loss. Defaults to None.

        Returns:
            float: Tukey loss.
        """
        if mask is None:
            mask = torch.ones_like(y_hat)

        # We normalize the Tukey loss, to satisfy 4.685 normal outlier bounds
        if self.normalize:
            y_mean = self.masked_mean(x=y, mask=mask, dim=-1)
            y_std = (
                torch.sqrt(self.masked_mean(x=(y - y_mean) ** 2, mask=mask, dim=-1))
                + 1e-2
            )
        else:
            y_std = 1.0
        delta_y = torch.abs(y - y_hat) / y_std

        tukey_mask = torch.greater_equal(self.c * torch.ones_like(delta_y), delta_y)
        tukey_loss = tukey_mask * mask * (1 - (delta_y / (self.c)) ** 2) ** 3 + (
            1 - (tukey_mask * 1)
        )
        tukey_loss = (self.c**2 / 6) * torch.mean(tukey_loss)
        return tukey_loss


class HuberQLoss(BasePointLoss):
    """Huberized Quantile Loss

    The Huberized quantile loss is a modified version of the quantile loss function that
    combines the advantages of the quantile loss and the Huber loss. It is commonly used
    in regression tasks, especially when dealing with data that contains outliers or heavy tails.

    The Huberized quantile loss between `y` and `y_hat` measure the Huber Loss in a non-symmetric way.
    The loss pays more attention to under/over-estimation depending on the quantile parameter $q$;
    and controls the trade-off between robustness and accuracy in the predictions with the parameter $delta$.

    $$
    \mathrm{HuberQL}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}^{(q)}_{\\tau}) =
    (1-q)\, L_{\delta}(y_{\\tau},\; \hat{y}^{(q)}_{\\tau}) \mathbb{1}\{ \hat{y}^{(q)}_{\\tau} \geq y_{\\tau} \} +
    q\, L_{\delta}(y_{\\tau},\; \hat{y}^{(q)}_{\\tau}) \mathbb{1}\{ \hat{y}^{(q)}_{\\tau} < y_{\\tau} \}
    $$

    Args:
        delta (float, optional): Specifies the threshold at which to change between delta-scaled L1 and L2 loss. Defaults to 1.0.
        q (float, optional): The slope of the quantile loss, in the context of quantile regression, the q determines the conditional quantile level. Defaults to 0.5.
        horizon_weight (Union[torch.Tensor, None], optional): Tensor of size h, weight for each timestamp of the forecasting window. Defaults to None.

    References:
        - [Huber Peter, J (1964). "Robust Estimation of a Location Parameter". Annals of Statistics](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full)
        - [Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles".](https://www.jstor.org/stable/1913643)
    """

    def __init__(self, q, delta: float = 1.0, horizon_weight=None):
        super(HuberQLoss, self).__init__(
            horizon_weight=horizon_weight,
            outputsize_multiplier=1,
            output_names=[f"_q{q}_d{delta}"],
        )
        self.q = q
        self.delta = delta

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: Union[torch.Tensor, None] = None,
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Args:
            y (torch.Tensor): Actual values.
            y_hat (torch.Tensor): Predicted values.
            mask (Union[torch.Tensor, None], optional): Specifies date stamps per serie to consider in loss. Defaults to None.

        Returns:
            float: HuberQLoss.
        """

        error = y_hat - y
        zero_error = torch.zeros_like(error)
        sq = torch.maximum(-error, zero_error)
        s1_q = torch.maximum(error, zero_error)
        losses = self.q * F.huber_loss(
            sq, zero_error, reduction="none", delta=self.delta
        ) + (1 - self.q) * F.huber_loss(
            s1_q, zero_error, reduction="none", delta=self.delta
        )

        weights = self._compute_weights(y=y, mask=mask)
        return _weighted_mean(losses=losses, weights=weights)


class HuberMQLoss(BasePointLoss):
    """Huberized Multi-Quantile loss

    The Huberized Multi-Quantile loss (HuberMQL) is a modified version of the multi-quantile loss function
    that combines the advantages of the quantile loss and the Huber loss. HuberMQL is commonly used in regression
    tasks, especially when dealing with data that contains outliers or heavy tails. The loss function pays
    more attention to under/over-estimation depending on the quantile list $[q_{1},q_{2},\dots]$ parameter.
    It controls the trade-off between robustness and prediction accuracy with the parameter $\\delta$.

    $$
    \mathrm{HuberMQL}_{\delta}(\\mathbf{y}_{\\tau},[\\mathbf{\hat{y}}^{(q_{1})}_{\\tau}, ... ,\hat{y}^{(q_{n})}_{\\tau}]) =
    \\frac{1}{n} \\sum_{q_{i}} \mathrm{HuberQL}_{\\delta}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}^{(q_{i})}_{\\tau})
    $$

    Args:
        level (int list, optional): Probability levels for prediction intervals (Defaults median). Defaults to [80, 90].
        quantiles (float list, optional): Alternative to level, quantiles to estimate from y distribution. Defaults to None.
        delta (float, optional): Specifies the threshold at which to change between delta-scaled L1 and L2 loss. Defaults to 1.0.
        horizon_weight (Union[torch.Tensor, None], optional): Tensor of size h, weight for each timestamp of the forecasting window. Defaults to None.

    References:
        - [Huber Peter, J (1964). "Robust Estimation of a Location Parameter". Annals of Statistics](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full)
        - [Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles".](https://www.jstor.org/stable/1913643)
    """

    def __init__(
        self, level=[80, 90], quantiles=None, delta: float = 1.0, horizon_weight=None
    ):

        qs, output_names = level_to_outputs(level)
        qs = torch.Tensor(qs)
        # Transform quantiles to homogeneous output names
        if quantiles is not None:
            _, output_names = quantiles_to_outputs(quantiles)
            qs = torch.Tensor(quantiles)

        super(HuberMQLoss, self).__init__(
            horizon_weight=horizon_weight,
            outputsize_multiplier=len(qs),
            output_names=output_names,
        )

        self.quantiles = torch.nn.Parameter(qs, requires_grad=False)
        self.delta = delta

    def domain_map(self, y_hat: torch.Tensor):
        """
        Args:
            y_hat (torch.Tensor): Predicted values.

        Returns:
            torch.Tensor: Transformed values.
            - shape: [B, H, 1 * Q] for univariate
            - shape: [B, H, N * Q] for multivariate
        """
        output = y_hat.reshape(
            y_hat.shape[0], y_hat.shape[1], -1, self.outputsize_multiplier
        )

        return output

    def _compute_weights(self, y, mask):
        """
        Compute final weights for each datapoint (based on all weights and all masks)
        Set horizon_weight to a ones[H] tensor if not set.
        If set, check that it has the same length as the horizon in x.
        """

        if self.horizon_weight is None:
            weights = torch.ones_like(mask)
        else:
            assert mask.shape[1] == len(
                self.horizon_weight
            ), "horizon_weight must have same length as Y"
            weights = self.horizon_weight.clone()
            weights = weights[None, :, None, None].to(mask.device)
            weights = torch.ones_like(mask, device=mask.device) * weights

        return weights * mask

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: Union[torch.Tensor, None] = None,
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Args:
            y (torch.Tensor): Actual values.
            y_hat (torch.Tensor): Predicted values.
            mask (Union[torch.Tensor, None], optional): Specifies date stamps per serie to consider in loss. Defaults to None.

        Returns:
            float: HuberMQLoss.
        """
        # [B, h, N] -> [B, h, N, 1]
        if y_hat.ndim == 3:
            y_hat = y_hat.unsqueeze(-1)

        y = y.unsqueeze(-1)
        if mask is not None:
            mask = mask.unsqueeze(-1)
        else:
            mask = torch.ones_like(y, device=y.device)

        error = y_hat - y

        zero_error = torch.zeros_like(error)
        sq = torch.maximum(-error, torch.zeros_like(error))
        s1_q = torch.maximum(error, torch.zeros_like(error))

        quantiles = self.quantiles[None, None, None, :]
        losses = F.huber_loss(
            quantiles * sq, zero_error, reduction="none", delta=self.delta
        ) + F.huber_loss(
            (1 - quantiles) * s1_q, zero_error, reduction="none", delta=self.delta
        )
        losses = (1 / len(quantiles)) * losses

        weights = self._compute_weights(y=losses, mask=mask)

        return _weighted_mean(losses=losses, weights=weights)


class HuberIQLoss(HuberQLoss):
    """Implicit Huber Quantile Loss

    Computes the huberized quantile loss between `y` and `y_hat`, with the quantile `q` provided as an input to the network.
    HuberIQLoss measures the deviation of a huberized quantile forecast.
    By weighting the absolute deviation in a non symmetric way, the
    loss pays more attention to under or over estimation.

    $$
    \mathrm{HuberIQL}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}^{(q)}_{\\tau}) =
    (1-q)\, L_{\delta}(y_{\\tau},\; \hat{y}^{(q)}_{\\tau}) \mathbb{1}\{ \hat{y}^{(q)}_{\\tau} \geq y_{\\tau} \} +
    q\, L_{\delta}(y_{\\tau},\; \hat{y}^{(q)}_{\\tau}) \mathbb{1}\{ \hat{y}^{(q)}_{\\tau} < y_{\\tau} \}
    $$

    Args:
        quantile_sampling (str, optional): Sampling distribution used to sample the quantiles during training. Choose from ['uniform', 'beta']. Defaults to 'uniform'.
        horizon_weight (Union[torch.Tensor, None], optional): Tensor of size h, weight for each timestamp of the forecasting window. Defaults to None.
        delta (float, optional): Specifies the threshold at which to change between delta-scaled L1 and L2 loss. Defaults to 1.0.

    References:
        - [Gouttes, Adèle, Kashif Rasul, Mateusz Koren, Johannes Stephan, and Tofigh Naghibi, "Probabilistic Time Series Forecasting with Implicit Quantile Networks".](http://arxiv.org/abs/2107.03743)
        - [Huber Peter, J (1964). "Robust Estimation of a Location Parameter". Annals of Statistics](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full)
        - [Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles".](https://www.jstor.org/stable/1913643)
    """

    def __init__(
        self,
        cos_embedding_dim=64,
        concentration0=1.0,
        concentration1=1.0,
        delta=1.0,
        horizon_weight=None,
    ):
        self.update_quantile()
        super(HuberIQLoss, self).__init__(
            q=self.q, delta=delta, horizon_weight=horizon_weight
        )

        self.cos_embedding_dim = cos_embedding_dim
        self.concentration0 = concentration0
        self.concentration1 = concentration1
        self.has_sampled = False
        self.has_predicted = False

        self.quantile_layer = QuantileLayer(
            num_output=1, cos_embedding_dim=self.cos_embedding_dim
        )
        self.output_layer = nn.Sequential(nn.Linear(1, 1), nn.PReLU())

    def _sample_quantiles(self, sample_size, device):
        if not self.has_sampled:
            self._init_sampling_distribution(device)

        quantiles = self.sampling_distr.sample(sample_size)
        self.q = quantiles.squeeze(-1)
        self.has_sampled = True
        self.has_predicted = False

        return quantiles

    def _init_sampling_distribution(self, device):
        concentration0 = torch.tensor(
            [self.concentration0], device=device, dtype=torch.float32
        )
        concentration1 = torch.tensor(
            [self.concentration1], device=device, dtype=torch.float32
        )
        self.sampling_distr = Beta(
            concentration0=concentration0, concentration1=concentration1
        )

    def update_quantile(self, q: List[float] = [0.5]):
        self.q = q[0]
        self.output_names = [f"_ql{q[0]}"]
        self.has_predicted = True

    def domain_map(self, y_hat):
        """
        Adds IQN network to output of network

        Args:
            y_hat (torch.Tensor): Predicted values.

            - shape: [B, h, 1] for univariate
            - shape: [B, h, N] for multivariate
        """
        if self.eval() and self.has_predicted:
            quantiles = torch.full(
                size=y_hat.shape,
                fill_value=self.q,
                device=y_hat.device,
                dtype=y_hat.dtype,
            )
            quantiles = quantiles.unsqueeze(-1)
        else:
            quantiles = self._sample_quantiles(
                sample_size=y_hat.shape, device=y_hat.device
            )

        # Embed the quantiles and add to y_hat
        emb_taus = self.quantile_layer(quantiles)
        emb_inputs = y_hat.unsqueeze(-1) * (1.0 + emb_taus)
        emb_outputs = self.output_layer(emb_inputs)

        # Domain map
        y_hat = emb_outputs.squeeze(-1)

        return y_hat


class Accuracy(BasePointLoss):
    """Accuracy

    Computes the accuracy between categorical `y` and `y_hat`.
    This evaluation metric is only meant for evalution, as it
    is not differentiable.

    $$
    \mathrm{Accuracy}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} \mathrm{1}\{\\mathbf{y}_{\\tau}==\\mathbf{\hat{y}}_{\\tau}\}
    $$

    """

    def __init__(
        self,
    ):
        super(Accuracy, self).__init__()
        self.is_distribution_output = False
        self.outputsize_multiplier = 1

    def domain_map(self, y_hat: torch.Tensor):
        """
        Args:
            y_hat (torch.Tensor): Predicted values.

            - shape: [B, H, 1] for univariate
            - shape: [B, H, N] for multivariate

        Returns:
            torch.Tensor: Transformed values.
            - shape: [B, H, 1] for univariate
            - shape: [B, H, N] for multivariate
        """

        return y_hat

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Args:
            y (torch.Tensor): Actual values.
            y_hat (torch.Tensor): Predicted values.
            mask (Union[torch.Tensor, None], optional): Specifies date stamps per serie to consider in loss. Defaults to None.

        Returns:
            float: Accuracy.
        """

        if mask is None:
            mask = torch.ones_like(y_hat)

        measure = (y == y_hat) * mask
        accuracy = torch.mean(measure)
        return accuracy


class sCRPS(BasePointLoss):
    """Scaled Continues Ranked Probability Score

    Calculates a scaled variation of the CRPS, as proposed by Rangapuram (2021),
    to measure the accuracy of predicted quantiles `y_hat` compared to the observation `y`.

    This metric averages percentual weighted absolute deviations as
    defined by the quantile losses.

    $$
    \mathrm{sCRPS}(\\mathbf{\hat{y}}^{(q)}_{\\tau}, \mathbf{y}_{\\tau}) = \\frac{2}{N} \sum_{i}
    \int^{1}_{0}
    \\frac{\mathrm{QL}(\\mathbf{\hat{y}}^{(q}_{\\tau} y_{i,\\tau})_{q}}{\sum_{i} | y_{i,\\tau} |} dq
    $$

    where $\\mathbf{\hat{y}}^{(q}_{\\tau}$ is the estimated quantile, and $y_{i,\\tau}$
    are the target variable realizations.

    Args:
        level (int list, optional): Probability levels for prediction intervals (Defaults median). Defaults to [80, 90].
        quantiles (float list, optional): Alternative to level, quantiles to estimate from y distribution. Defaults to None.

    References:
        - [Gneiting, Tilmann. (2011). "Quantiles as optimal point forecasts". International Journal of Forecasting.](https://www.sciencedirect.com/science/article/pii/S0169207010000063)
        - [Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos, Zhi Chen, Anil Gaba, Ilia Tsetlin, Robert L. Winkler. (2022). "The M5 uncertainty competition: Results, findings and conclusions". International Journal of Forecasting.](https://www.sciencedirect.com/science/article/pii/S0169207021001722)
        - [Syama Sundar Rangapuram, Lucien D Werner, Konstantinos Benidis, Pedro Mercado, Jan Gasthaus, Tim Januschowski. (2021). "End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series". Proceedings of the 38th International Conference on Machine Learning (ICML).](https://proceedings.mlr.press/v139/rangapuram21a.html)
    """

    def __init__(self, level=[80, 90], quantiles=None):
        super(sCRPS, self).__init__()
        self.mql = MQLoss(level=level, quantiles=quantiles)
        self.is_distribution_output = False

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Args:
            y (torch.Tensor): Actual values.
            y_hat (torch.Tensor): Predicted values.
            mask (Union[torch.Tensor, None], optional): Specifies date stamps per series to consider in loss. Defaults to None.

        Returns:
            float: sCRPS.
        """
        mql = self.mql(y=y, y_hat=y_hat, mask=mask, y_insample=y_insample)
        norm = torch.sum(torch.abs(y))
        unmean = torch.sum(mask)
        scrps = 2 * mql * unmean / (norm + 1e-5)
        return scrps
