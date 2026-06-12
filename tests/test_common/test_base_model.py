import pytest

from neuralforecast.common._base_model import _num_requested_devices
from neuralforecast.models import MLP, NHITS, SOFTS, MLPMultivariate, TSMixer

MULTIVARIATE_MODELS = [TSMixer, MLPMultivariate, SOFTS]
UNIVARIATE_MODELS = [NHITS, MLP]
GUARD_MATCH = "Single-device execution is the supported configuration"


def _model_kwargs(model_cls):
    kwargs = {"h": 2, "input_size": 4, "max_steps": 1}
    if model_cls in MULTIVARIATE_MODELS:
        kwargs["n_series"] = 3
    return kwargs


class TestMultivariateMultiDeviceGuard:
    """Multivariate models require all series in every batch, so data-parallel
    multi-device training is not supported and must fail fast (issue #1100)."""

    @pytest.mark.parametrize("model_cls", MULTIVARIATE_MODELS)
    @pytest.mark.parametrize("devices", [2, [0, 1], "2"], ids=["int", "list", "str"])
    def test_multivariate_with_multiple_devices_raises(self, model_cls, devices):
        with pytest.raises(ValueError, match=GUARD_MATCH):
            model_cls(**_model_kwargs(model_cls), accelerator="cpu", devices=devices)

    @pytest.mark.parametrize("model_cls", MULTIVARIATE_MODELS)
    def test_multivariate_with_multiple_nodes_raises(self, model_cls):
        with pytest.raises(ValueError, match=GUARD_MATCH):
            model_cls(
                **_model_kwargs(model_cls),
                accelerator="cpu",
                devices=1,
                num_nodes=2,
            )

    @pytest.mark.parametrize("model_cls", MULTIVARIATE_MODELS)
    @pytest.mark.parametrize("devices", [None, 1, [0]], ids=["default", "int", "list"])
    def test_multivariate_with_single_device_is_allowed(self, model_cls, devices):
        kwargs = _model_kwargs(model_cls)
        if devices is not None:
            kwargs.update(accelerator="cpu", devices=devices)
        model_cls(**kwargs)

    @pytest.mark.parametrize("model_cls", UNIVARIATE_MODELS)
    def test_univariate_with_multiple_devices_is_allowed(self, model_cls):
        model_cls(**_model_kwargs(model_cls), accelerator="cpu", devices=2)

    def test_error_message_is_actionable(self):
        with pytest.raises(ValueError) as exc_info:
            TSMixer(**_model_kwargs(TSMixer), accelerator="cpu", devices=2)
        message = str(exc_info.value)
        assert "TSMixer" in message
        assert "all series" in message
        assert "devices=1" in message
        assert "issues/1100" in message


@pytest.mark.parametrize(
    "trainer_kwargs, expected",
    [
        ({}, 1),
        ({"devices": None}, 1),
        ({"devices": 1}, 1),
        ({"devices": 4}, 4),
        ({"devices": [0, 1]}, 2),
        ({"devices": "2"}, 2),
        ({"devices": "auto", "accelerator": "cpu"}, 1),
        ({"devices": -1, "accelerator": "cpu"}, 1),
        ({"devices": "bogus"}, 1),
    ],
)
def test_num_requested_devices(trainer_kwargs, expected):
    assert _num_requested_devices(trainer_kwargs) == expected
