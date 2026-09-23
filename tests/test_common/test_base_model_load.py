"""Security-relevant behaviour of `BaseModel.save` / `BaseModel.load`.

These assert the guarantees, not the implementation: a v2 checkpoint never
reaches pickle, a legacy checkpoint is unrestricted only on request, and the
restricted reader never falls back to the unrestricted one.
"""

import json
import warnings

import pytest
import torch

from neuralforecast._serialization import looks_like_safetensors, register_loss
from neuralforecast.losses.pytorch import MAE, MQLoss
from neuralforecast.models import NLinear, TCN


class CustomLoss(MAE):
    """Module-level so a v1 fixture can pickle it."""


def _model(**kwargs):
    return NLinear(h=2, input_size=4, max_steps=1, **kwargs)


def _write_v1(model, path):
    """A legacy artifact, in the format shipped before the migration."""
    hparams = {k: v for k, v in dict(model.hparams).items() if k != "callbacks"}
    torch.save({"hyper_parameters": hparams, "state_dict": model.state_dict()}, path)
    return str(path)


@pytest.fixture
def v2_ckpt(tmp_path):
    path = tmp_path / "v2.ckpt"
    _model().save(str(path))
    return str(path)


@pytest.fixture
def v1_ckpt(tmp_path):
    return _write_v1(_model(), tmp_path / "v1.ckpt")


# --------------------------------------------------------------------------
# v2 format
# --------------------------------------------------------------------------


def test_save_writes_safetensors(v2_ckpt):
    with open(v2_ckpt, "rb") as f:
        assert looks_like_safetensors(f.read())


def test_v2_roundtrip_is_bit_identical(tmp_path):
    model = _model(loss=MQLoss(level=[80, 90]))
    path = str(tmp_path / "m.ckpt")
    model.save(path)

    loaded = NLinear.load(path)
    assert isinstance(loaded.loss, MQLoss)
    assert loaded.h == model.h
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key


def test_v2_roundtrip_restores_tied_tensors(tmp_path):
    model = TCN(h=2, input_size=4, max_steps=1)
    path = str(tmp_path / "tcn.ckpt")
    model.save(path)

    loaded = TCN.load(path)
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key


def test_v2_never_reaches_torch_load(v2_ckpt, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("torch.load must not be reached on the v2 path")

    monkeypatch.setattr(torch, "load", forbidden)
    NLinear.load(v2_ckpt)


def test_v2_does_not_warn_about_pickle(v2_ckpt):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        NLinear.load(v2_ckpt)


def test_v2_refuses_a_checkpoint_of_another_class(tmp_path):
    path = str(tmp_path / "tcn.ckpt")
    TCN(h=2, input_size=4, max_steps=1).save(path)
    with pytest.raises(ValueError, match="holds a TCN checkpoint"):
        NLinear.load(path)


def test_v2_metadata_names_an_unregistered_class_and_is_refused(tmp_path, monkeypatch):
    import builtins

    from neuralforecast import _serialization

    path = str(tmp_path / "m.ckpt")
    _model().save(path)

    with open(path, "rb") as f:
        data = f.read()
    metadata = _serialization.read_metadata(data)
    hparams = json.loads(metadata["hyper_parameters"])
    hparams["loss"] = {"__nf__": "loss", "cls": "os.system", "args": {}}

    tensors, _ = _serialization.load_tensors(data)
    tampered = _serialization.save_tensors(
        tensors, {**metadata, "hyper_parameters": json.dumps(hparams)}
    )
    with open(path, "wb") as f:
        f.write(tampered)

    imported = []
    real_import = builtins.__import__

    def spy(name, *args, **kwargs):
        imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", spy)
    with pytest.raises(_serialization.SerializationError, match="not registered"):
        NLinear.load(path)
    assert not [n for n in imported if n.split(".")[0] == "os"]


def test_runtime_only_hparams_are_stripped(tmp_path):
    from neuralforecast import _serialization

    model = _model()
    model.hparams["logger"] = object()
    path = str(tmp_path / "m.ckpt")
    model.save(path)

    with open(path, "rb") as f:
        metadata = _serialization.read_metadata(f.read())
    assert "logger" not in json.loads(metadata["hyper_parameters"])


def test_tensor_valued_hyperparameter_roundtrips(tmp_path):
    import numpy as np

    model = _model(loss=MAE(horizon_weight=np.array([0.25, 0.75])))
    path = str(tmp_path / "m.ckpt")
    model.save(path)

    loaded = NLinear.load(path)
    assert torch.equal(loaded.loss.horizon_weight, model.loss.horizon_weight)


# --------------------------------------------------------------------------
# Legacy v1 format
# --------------------------------------------------------------------------


def test_v1_loads_with_explicit_pickle_consent(v1_ckpt):
    with pytest.warns(UserWarning, match="pickle"):
        assert isinstance(NLinear.load(v1_ckpt, allow_pickle=True), NLinear)


def test_v1_unrestricted_read_passes_weights_only_false(v1_ckpt, monkeypatch):
    seen = {}
    real_load = torch.load

    def spy(f, **kwargs):
        seen.update(kwargs)
        return real_load(f, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    with pytest.warns(UserWarning):
        NLinear.load(v1_ckpt, allow_pickle=True)
    assert seen["weights_only"] is False


def test_v1_restricted_read_uses_weights_only_true(v1_ckpt, monkeypatch):
    seen = {}
    real_load = torch.load

    def spy(f, **kwargs):
        seen.update(kwargs)
        return real_load(f, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    NLinear.load(v1_ckpt, allow_pickle=False)
    assert seen["weights_only"] is True


def test_caller_weights_only_forces_the_restricted_read(v1_ckpt, monkeypatch):
    seen = []
    real_load = torch.load

    def spy(f, **kwargs):
        seen.append(kwargs["weights_only"])
        return real_load(f, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    NLinear.load(v1_ckpt, allow_pickle=True, weights_only=True)
    assert seen == [True], "allow_pickle must not override an explicit request"


def test_restricted_read_never_falls_back_to_pickle(tmp_path, monkeypatch):
    """A checkpoint the allowlist refuses must raise, not load unrestricted."""

    register_loss(CustomLoss)
    path = _write_v1(_model(loss=CustomLoss()), tmp_path / "custom.ckpt")

    seen = []
    real_load = torch.load

    def spy(f, **kwargs):
        seen.append(kwargs.get("weights_only"))
        return real_load(f, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    monkeypatch.setattr(
        "neuralforecast.common._base_model.registered_classes", lambda kind: ()
    )
    with pytest.raises(ValueError, match="Cannot safely load"):
        NLinear.load(path, allow_pickle=False)
    assert False not in seen, "must never retry with weights_only=False"


def test_restricted_read_error_names_the_refused_global(tmp_path):
    path = _write_v1(_model(loss=MQLoss(level=[80, 90])), tmp_path / "mq.ckpt")
    with pytest.raises(ValueError, match=r"numpy\._core\.multiarray\.scalar"):
        NLinear.load(path, allow_pickle=False)


def test_load_from_bytes_is_not_on_the_allowlist():
    """`torch.storage._load_from_bytes` calls `torch.load(weights_only=False)`.

    Allowlisting it would hand an attacker an unrestricted load through the
    allowlist. Deleting this assertion is a security decision.
    """
    from neuralforecast.common._base_model import _V1_EXTRA_SAFE_GLOBALS

    names = {f"{c.__module__}.{c.__qualname__}" for c in _V1_EXTRA_SAFE_GLOBALS}
    assert "torch.storage._load_from_bytes" not in names
    assert not any("_unpickle_block" in n or "__pyx_unpickle" in n for n in names)


def test_numpy_globals_are_withheld_pending_sign_off():
    """Remove this test only together with the review it is waiting on."""
    from neuralforecast.common._base_model import _V1_EXTRA_SAFE_GLOBALS

    assert _V1_EXTRA_SAFE_GLOBALS == ()


def test_warnings_from_torch_load_are_not_suppressed(v1_ckpt, monkeypatch):
    """The old code wrapped `torch.load` in a filter that hid FutureWarning."""
    real_load = torch.load

    def noisy(f, **kwargs):
        warnings.warn("from torch.load", FutureWarning)
        return real_load(f, **kwargs)

    monkeypatch.setattr(torch, "load", noisy)
    with pytest.warns(FutureWarning, match="from torch.load"):
        NLinear.load(v1_ckpt, allow_pickle=True)
