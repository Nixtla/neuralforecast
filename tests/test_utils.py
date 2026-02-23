import pytest

from neuralforecast.utils import (
    PredictionIntervals,
    level_to_quantiles,
    quantiles_to_level,
)


# Test level_to_quantiles
def test_level_to_quantiles():
    level_base = [80, 90]
    quantiles_base = [0.05, 0.1, 0.9, 0.95]
    quantiles = level_to_quantiles(level_base)
    level = quantiles_to_level(quantiles_base)

    assert quantiles == quantiles_base
    assert level == level_base


@pytest.mark.parametrize("step_size", [0, -1])
def test_prediction_intervals_step_size_validation(step_size):
    with pytest.raises(ValueError, match="step_size must be at least 1"):
        PredictionIntervals(step_size=step_size)
