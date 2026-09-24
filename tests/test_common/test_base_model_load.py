"""Security-relevant behaviour of `BaseModel.save` / `BaseModel.load`.

Asserts the guarantees, not the implementation.
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
    """A DistributionLoss checkpoint asks for `getattr`, which stays refused."""
    from neuralforecast.losses.pytorch import DistributionLoss
    from neuralforecast.models import DeepAR

    model = DeepAR(h=2, input_size=4, max_steps=1, loss=DistributionLoss("Normal"))
    path = _write_v1(model, tmp_path / "deepar.ckpt")
    with pytest.raises(ValueError, match="`getattr`"):
        DeepAR.load(path, allow_pickle=False)


def test_load_from_bytes_is_not_on_the_allowlist():
    """`_load_from_bytes` is an unrestricted load; deleting this is a decision."""
    from neuralforecast.common._base_model import _V1_EXTRA_SAFE_GLOBALS

    names = {f"{c.__module__}.{c.__qualname__}" for c in _V1_EXTRA_SAFE_GLOBALS}
    assert "torch.storage._load_from_bytes" not in names
    assert not any("_unpickle_block" in n or "__pyx_unpickle" in n for n in names)


def test_allowlist_is_exactly_what_was_signed_off():
    """Widening this list is a security review. Adding an entry fails here first."""
    from neuralforecast.common._base_model import _V1_EXTRA_SAFE_GLOBALS

    names = {f"{c.__module__}.{c.__qualname__}" for c in _V1_EXTRA_SAFE_GLOBALS}
    assert names == {
        "numpy._core.multiarray.scalar",
        "numpy.dtype",
        "numpy.dtypes.StrDType",
        "_codecs.encode",
    }


def test_getattr_is_never_allowlisted():
    """A general attribute reader defeats any allowlist; those checkpoints migrate."""
    from neuralforecast.common._base_model import _V1_EXTRA_SAFE_GLOBALS

    assert getattr not in _V1_EXTRA_SAFE_GLOBALS


def test_quantile_loss_checkpoint_loads_restricted(tmp_path):
    """What the signed-off numpy entries buy."""
    path = _write_v1(_model(loss=MQLoss(level=[80, 90])), tmp_path / "mq.ckpt")
    loaded = NLinear.load(path, allow_pickle=False)
    assert isinstance(loaded.loss, MQLoss)


def test_legacy_default_is_now_restricted(v1_ckpt, monkeypatch):
    seen = {}
    real_load = torch.load

    def spy(f, **kwargs):
        seen.update(kwargs)
        return real_load(f, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    NLinear.load(v1_ckpt)
    assert seen["weights_only"] is True, "pickle must not be reached without consent"


def test_timellm_cannot_be_saved_or_loaded(tmp_path):
    """Refused at both ends: an attacker writes the artifact, not our `save`."""
    from neuralforecast.models import TimeLLM

    path = str(tmp_path / "m.safetensors")
    with pytest.raises(ValueError, match="cannot be saved or loaded"):
        TimeLLM.load(path)

    model = NLinear.__new__(TimeLLM)
    with pytest.raises(ValueError, match="cannot be saved or loaded"):
        TimeLLM.save(model, path)


def test_artifact_claiming_to_be_timellm_is_refused(tmp_path):
    from neuralforecast import _serialization

    path = str(tmp_path / "m.safetensors")
    _model().save(path)
    with open(path, "rb") as f:
        data = f.read()
    metadata = _serialization.read_metadata(data)
    tensors, _ = _serialization.load_tensors(data)
    with open(path, "wb") as f:
        f.write(
            _serialization.save_tensors(tensors, {**metadata, "model_class": "TimeLLM"})
        )

    with pytest.raises(ValueError, match="cannot be saved or loaded"):
        NLinear.load(path)


def test_warnings_from_torch_load_are_not_suppressed(v1_ckpt, monkeypatch):
    """The old code wrapped `torch.load` in a filter that hid FutureWarning."""
    real_load = torch.load

    def noisy(f, **kwargs):
        warnings.warn("from torch.load", FutureWarning)
        return real_load(f, **kwargs)

    monkeypatch.setattr(torch, "load", noisy)
    with pytest.warns(FutureWarning, match="from torch.load"):
        NLinear.load(v1_ckpt, allow_pickle=True)


# --------------------------------------------------------------------------
# Live loss state (PR #1625 review)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("level", [[80], [80, 90]])
def test_quantiles_updated_by_predict_survive_a_round_trip(tmp_path, level):
    """Init args alone gave a size mismatch, or the wrong output_names."""
    from neuralforecast.models import DeepAR

    model = DeepAR(h=2, input_size=4, max_steps=1)
    model.loss.update_quantile(q=[q / 100 for q in level])
    path = str(tmp_path / "deepar.safetensors")
    model.save(path)

    loaded = DeepAR.load(path)
    assert torch.equal(loaded.loss.quantiles, model.loss.quantiles)
    assert loaded.loss.output_names == model.loss.output_names


def test_aliased_valid_loss_stays_aliased(tmp_path):
    """With valid_loss=None the two are the same object; that must survive."""
    from neuralforecast.losses.pytorch import DistributionLoss
    from neuralforecast.models import NHITS

    model = NHITS(h=2, input_size=4, max_steps=1, loss=DistributionLoss("Normal"))
    assert model.valid_loss is model.loss
    model.loss.update_quantile(q=[0.8])

    path = str(tmp_path / "nhits.safetensors")
    model.save(path)
    loaded = NHITS.load(path)
    assert loaded.valid_loss is loaded.loss
    assert torch.equal(loaded.loss.quantiles, model.loss.quantiles)


def test_distinct_valid_loss_keeps_its_own_quantiles(tmp_path):
    model = _model(loss=MQLoss(level=[80, 90]), valid_loss=MQLoss(level=[50]))
    path = str(tmp_path / "vl.safetensors")
    model.save(path)

    loaded = NLinear.load(path)
    assert loaded.loss.output_names == model.loss.output_names
    assert loaded.valid_loss.output_names == model.valid_loss.output_names
    assert not torch.equal(loaded.loss.quantiles, loaded.valid_loss.quantiles)


def test_serialize_does_not_touch_disk(tmp_path):
    """`save` is split so callers can encode before removing anything."""
    blob = _model().serialize()
    assert looks_like_safetensors(blob)
    assert not list(tmp_path.iterdir())


# --------------------------------------------------------------------------
# Runtime settings an artifact must not choose (PR #1625 review)
# --------------------------------------------------------------------------


def _tamper(path, **hparams):
    from neuralforecast import _serialization

    with open(path, "rb") as f:
        data = f.read()
    metadata = _serialization.read_metadata(data)
    stored = json.loads(metadata["hyper_parameters"])
    stored.update(hparams)
    tensors, _ = _serialization.load_tensors(data)
    with open(path, "wb") as f:
        f.write(
            _serialization.save_tensors(
                tensors, {**metadata, "hyper_parameters": json.dumps(stored)}
            )
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("default_root_dir", "s3://attacker/exfil"),
        ("strategy", "ddp"),
        ("num_nodes", 8),
        ("profiler", "advanced"),
        ("detect_anomaly", True),
    ],
)
def test_artifact_cannot_choose_trainer_runtime_settings(tmp_path, key, value):
    """`default_root_dir` alone would send checkpoints to an attacker's bucket."""
    path = str(tmp_path / "m.safetensors")
    _model().save(path)
    _tamper(path, **{key: value})

    with pytest.warns(UserWarning, match="Ignoring runtime settings"):
        loaded = NLinear.load(path)
    assert loaded.trainer_kwargs.get(key) is None


@pytest.mark.parametrize(
    "key,value",
    [("num_workers", 64), ("prefetch_factor", 9999), ("multiprocessing_context", "fork")],
)
def test_artifact_cannot_choose_dataloader_runtime_settings(tmp_path, key, value):
    path = str(tmp_path / "m.safetensors")
    _model(dataloader_kwargs={"drop_last": True}).save(path)
    _tamper(path, dataloader_kwargs={"drop_last": True, key: value})

    with pytest.warns(UserWarning, match="Ignoring runtime settings"):
        loaded = NLinear.load(path)
    assert loaded.dataloader_kwargs == {"drop_last": True}


@pytest.mark.parametrize("model", [NLinear, TCN])
def test_an_untampered_model_warns_about_nothing(tmp_path, model):
    """Model hyperparameters span the whole init chain, not just cls.__init__."""
    path = str(tmp_path / "m.safetensors")
    model(h=2, input_size=4, max_steps=1).save(path)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model.load(path)


def test_model_describing_trainer_kwargs_still_round_trip(tmp_path):
    path = str(tmp_path / "m.safetensors")
    _model(enable_checkpointing=True, accelerator="cpu", devices=1).save(path)

    loaded = NLinear.load(path)
    assert loaded.trainer_kwargs["enable_checkpointing"] is True
    assert loaded.trainer_kwargs["accelerator"] == "cpu"
