"""An artifact that tries to execute code must be refused, and must not run.

The payloads here are real: unpickling one calls `_Detonate.__reduce__`, which
runs `_witness.append`. Tests that only check for an error can pass while the
payload has already executed, so every test asserts the witness is empty.
"""

import pickle
import shutil

import pytest
import torch

from neuralforecast import NeuralForecast
from neuralforecast._serialization import SerializationError
from neuralforecast.models import NLinear

V1_FIXTURE = "tests/backward_comp/data/NLinear"

_witness = []


def _detonate():
    """Stands in for whatever an exploit's `__reduce__` would call."""
    _witness.append("detonated")
    return "detonated"


class _Detonate:
    """Runs on unpickle, the way a real exploit's `__reduce__` would.

    `_detonate` is pickled by reference, so unpickling calls this module's
    function; pickling a bound method of a list would detonate a copy instead.
    """

    def __reduce__(self):
        return (_detonate, ())


@pytest.fixture(autouse=True)
def clear_witness():
    _witness.clear()
    yield
    _witness.clear()


def _payload():
    return pickle.dumps({"hyper_parameters": _Detonate(), "state_dict": {}})


def test_the_payload_really_executes_when_unpickled():
    """Guards the other tests: a harmless payload would make them vacuous."""
    pickle.loads(_payload())
    assert _witness == ["detonated"]


def test_a_malicious_checkpoint_is_refused_by_default(tmp_path):
    path = tmp_path / "evil.ckpt"
    path.write_bytes(_payload())

    with pytest.raises(ValueError):
        NLinear.load(str(path))
    assert _witness == []


def test_a_malicious_checkpoint_still_runs_with_explicit_consent(tmp_path):
    """`allow_pickle=True` is documented as executing the file; prove it does."""
    path = tmp_path / "evil.ckpt"
    path.write_bytes(_payload())

    with pytest.warns(UserWarning, match="executes any code"):
        with pytest.raises(Exception):
            NLinear.load(str(path), allow_pickle=True)
    assert _witness == ["detonated"], "the opt-in path is meant to be unsafe"


@pytest.mark.parametrize("sidecar", ["alias_to_model.pkl", "configuration.pkl"])
def test_a_malicious_sidecar_is_refused_by_default(tmp_path, sidecar):
    directory = tmp_path / "evil"
    shutil.copytree(V1_FIXTURE, directory)
    (directory / sidecar).write_bytes(pickle.dumps(_Detonate()))

    with pytest.raises(ValueError):
        NeuralForecast.load(str(directory))
    assert _witness == []


def test_a_malicious_dataset_is_refused_before_anything_is_read(tmp_path):
    directory = tmp_path / "evil"
    shutil.copytree(V1_FIXTURE, directory)
    (directory / "dataset.pkl").write_bytes(pickle.dumps(_Detonate()))

    with pytest.raises(ValueError, match="Refusing to load the legacy dataset"):
        NeuralForecast.load(str(directory))
    assert _witness == []


def test_a_pickle_payload_smuggled_into_a_v2_file_is_inert(tmp_path):
    """Bytes the metadata does not reference are never interpreted."""
    from neuralforecast import _serialization

    path = tmp_path / "m.safetensors"
    NLinear(h=2, input_size=4, max_steps=1).save(str(path))

    data = path.read_bytes()
    metadata = _serialization.read_metadata(data)
    tensors, _ = _serialization.load_tensors(data)
    tensors["__hparams__.evil"] = torch.frombuffer(
        bytearray(_payload()), dtype=torch.uint8
    )
    path.write_bytes(_serialization.save_tensors(tensors, metadata))

    assert NLinear.load(str(path)) is not None
    assert _witness == []


def test_v2_metadata_naming_a_class_outside_the_registry_is_refused(tmp_path):
    import json

    from neuralforecast import _serialization

    path = tmp_path / "m.safetensors"
    NLinear(h=2, input_size=4, max_steps=1).save(str(path))

    data = path.read_bytes()
    metadata = _serialization.read_metadata(data)
    hparams = json.loads(metadata["hyper_parameters"])
    hparams["loss"] = {
        "__nf__": "loss",
        "cls": "tests.test_malicious_artifacts._detonate",
        "args": {},
    }
    tensors, _ = _serialization.load_tensors(data)
    path.write_bytes(
        _serialization.save_tensors(
            tensors, {**metadata, "hyper_parameters": json.dumps(hparams)}
        )
    )

    with pytest.raises(SerializationError, match="not registered"):
        NLinear.load(str(path))
    assert _witness == []


def test_torch_load_is_never_reached_unrestricted_by_default(tmp_path, monkeypatch):
    """Assert by patching, not by trusting the control flow."""
    seen = []
    real_load = torch.load

    def spy(*args, **kwargs):
        seen.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    path = tmp_path / "evil.ckpt"
    path.write_bytes(_payload())

    with pytest.raises(ValueError):
        NLinear.load(str(path))
    assert False not in seen and None not in seen
    assert _witness == []


def test_pickle_load_is_never_reached_by_default(tmp_path, monkeypatch):
    directory = tmp_path / "evil"
    shutil.copytree(V1_FIXTURE, directory)
    (directory / "configuration.pkl").write_bytes(pickle.dumps(_Detonate()))

    monkeypatch.setattr(
        pickle, "load", lambda *a, **k: pytest.fail("pickle.load was reached")
    )
    with pytest.raises(ValueError):
        NeuralForecast.load(str(directory))
    assert _witness == []
