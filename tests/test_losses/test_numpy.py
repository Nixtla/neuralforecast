import numpy as np

from neuralforecast.losses.numpy import mae


def test_a_weighted_mean_keeps_each_series():
    """A weight does not mix two series into one score."""
    y = np.zeros((2, 2))
    y_hat = np.array([[1.0, 3.0], [5.0, 7.0]])
    weights = np.ones_like(y_hat)
    per_series = mae(y, y_hat, weights, axis=-1)
    assert np.array_equal(per_series, np.array([2.0, 6.0]))
    assert mae(y, y_hat, weights) == 4.0
    assert np.array_equal(mae(y, y_hat, axis=-1), np.array([2.0, 6.0]))
    # A missing point is left out of that series only.
    y_missing = np.array([[1.0, np.nan], [3.0, 5.0]])
    skipped = mae(y_missing, np.zeros_like(y_missing), weights, axis=-1)
    assert np.array_equal(skipped, np.array([1.0, 4.0]))
