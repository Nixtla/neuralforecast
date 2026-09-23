"""Security-relevant behaviour of `BaseModel.load`.

These assert the guarantees, not the implementation: a caller asking for a
restricted load gets one, and an unrestricted load is never silent.
"""

import warnings

import pytest
import torch

from neuralforecast.models import NLinear


@pytest.fixture
def ckpt(tmp_path):
    path = tmp_path / "nlinear.ckpt"
    NLinear(h=2, input_size=4, max_steps=1).save(str(path))
    return str(path)


def test_round_trip_still_works(ckpt):
    model = NLinear.load(ckpt)
    assert isinstance(model, NLinear)
    assert model.h == 2


def test_caller_weights_only_is_honored(ckpt, monkeypatch):
    seen = {}
    real_load = torch.load

    def spy(f, **kwargs):
        seen.update(kwargs)
        return real_load(f, weights_only=False)

    monkeypatch.setattr(torch, "load", spy)
    NLinear.load(ckpt, weights_only=True)
    assert seen["weights_only"] is True


def test_defaults_to_unrestricted_load_for_backwards_compat(ckpt, monkeypatch):
    seen = {}
    real_load = torch.load

    def spy(f, **kwargs):
        seen.update(kwargs)
        return real_load(f, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    NLinear.load(ckpt)
    assert seen["weights_only"] is False


def test_unrestricted_load_warns(ckpt):
    with pytest.warns(UserWarning, match="pickle"):
        NLinear.load(ckpt)


def test_restricted_load_does_not_warn(ckpt, monkeypatch):
    real_load = torch.load
    monkeypatch.setattr(torch, "load", lambda f, **kw: real_load(f, weights_only=False))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        NLinear.load(ckpt, weights_only=True)


def test_warnings_from_torch_load_are_not_suppressed(ckpt, monkeypatch):
    """The old code wrapped `torch.load` in a filter that hid FutureWarning."""
    real_load = torch.load

    def noisy(f, **kwargs):
        warnings.warn("from torch.load", FutureWarning)
        return real_load(f, **kwargs)

    monkeypatch.setattr(torch, "load", noisy)
    with pytest.warns(FutureWarning, match="from torch.load"):
        NLinear.load(ckpt)
