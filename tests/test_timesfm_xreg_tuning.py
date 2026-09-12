import pytest

from neuralforecast.inference_tuning import get_inference_tuning_config
from neuralforecast.models.timesfm import TimesFM


def test_timesfm_preserves_previous_xreg_ridge_default():
    model = TimesFM(h=2, input_size=16)
    assert model.xreg_ridge == 1e-3


def test_timesfm_rejects_invalid_xreg_ridge():
    with pytest.raises(ValueError, match="positive finite"):
        TimesFM(h=2, input_size=16, xreg_ridge=0)


def test_timesfm_inference_space_tunes_xreg_ridge():
    config = get_inference_tuning_config("TimesFM", h=12)
    assert "xreg_ridge" in config
