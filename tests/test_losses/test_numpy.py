import numpy as np
import pytest

from neuralforecast.losses.numpy import mase


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("seasonality", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
def test_mase_differences_along_axis(axis, seasonality, weighted):
    y_train = np.array([[1., 2., 3., 4., 5., 6.], [10., 12., 14., 16., 18., 20.]])
    y = np.array([[7., 8.], [22., 24.]])
    y_hat = y - np.array([[1., 3.], [2., 4.]])
    weights = np.array([[1., 2.], [3., 1.]]) if weighted else None
    expected = [
        mase(actual, pred, train, seasonality, weights=weight)
        for actual, pred, train, weight in zip(
            y, y_hat, y_train, weights if weighted else [None, None]
        )
    ]
    if axis == 0:
        y_train, y, y_hat = y_train.T, y.T, y_hat.T
        if weighted:
            weights = weights.T

    result = mase(y, y_hat, y_train, seasonality, weights=weights, axis=axis)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("seasonality", [1, 2])
def test_mase_default_axis_keeps_first_axis_differencing(seasonality):
    y_train = np.array([[1., 10.], [2., 12.], [3., 14.], [4., 16.]])
    y = np.array([[5., 18.], [6., 20.]])
    y_hat = y - 1
    expected_scale = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))

    result = mase(y, y_hat, y_train, seasonality)

    np.testing.assert_allclose(result, 1 / expected_scale)
