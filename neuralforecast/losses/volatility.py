__all__ = [
    "excess_volatility",
    "forecast_percentage_change",
    "cross_validation_to_windows",
]


import re
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import utilsforecast.processing as ufp
from utilsforecast.compat import DataFrame


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
    r"""Excess Volatility

    Measures *harmful* forecast instability, by charging a forecast revision for its
    cost and crediting it for the accuracy it buys. A forecaster that revises its
    predictions only when the revision improves accuracy scores at or below zero; one
    that churns its predictions without becoming more accurate scores above zero.

    For each pair of overlapping windows that target the same date, let
    $\hat{y}^{before}$ be the forecast made from the earlier window and
    $\hat{y}^{update}$ the forecast made from the later one. Then

    $$ \mathrm{EV} = \mathrm{QL}(\mathbf{\hat{y}}^{update}, \mathbf{\hat{y}}^{before}) - \Big( \mathrm{QL}(\mathbf{y}, \mathbf{\hat{y}}^{before}) - \mathrm{QL}(\mathbf{y}, \mathbf{\hat{y}}^{update}) \Big) $$

    where the first term is the *revision cost* (how far the older forecast sits from
    the newer one) and the bracketed term is the *accuracy improvement* the revision
    produced. $\mathrm{QL}$ is the pinball loss averaged over quantiles.

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
        - [Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles".](https://www.jstor.org/stable/1913643)
        - [Rakshitha Godahewa et al., "A Strong Baseline for Forecast Stability".](https://arxiv.org/abs/2310.17332)
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
    r"""Forecast Percentage Change

    Measures the relative size of the revisions a forecaster makes as new data arrives,
    ignoring whether those revisions were justified. Use it alongside
    `excess_volatility`, which accounts for the accuracy a revision bought.

    For each pair of overlapping windows targeting the same date, with
    $\hat{y}^{before}$ the forecast from the earlier window and $\hat{y}^{update}$ the
    forecast from the later one:

    $$ \mathrm{sFPC} = 200 \cdot \mathrm{mean}\left( \frac{|\hat{y}^{update} - \hat{y}^{before}|}{|\hat{y}^{update}| + |\hat{y}^{before}| + \epsilon} \right) $$

    $$ \mathrm{FPC} = \mathrm{mean}\left( \frac{|\hat{y}^{update} - \hat{y}^{before}|}{|\hat{y}^{before}| + \epsilon} \right) $$

    These are two different quantities, not two scalings of one: sFPC is bounded in
    `[0, 200]` and symmetric in the two forecasts, while FPC is unbounded and measures
    revisions relative to the earlier forecast only. Higher values mean a more volatile
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
        - [Rakshitha Godahewa et al., "A Strong Baseline for Forecast Stability".](https://arxiv.org/abs/2310.17332)
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
    df: DataFrame,
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
        >>> from neuralforecast.losses.volatility import (
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

    df = ufp.sort(df, by=[id_col, "cutoff", time_col])
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
