"""Unit tests for the v2 serialization primitives.

Nothing in `neuralforecast` calls this module yet; these cover it on its own.
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from neuralforecast._serialization import (
    TAG,
    SerializationError,
    decode_mapping,
    decode_value,
    encode_mapping,
    encode_value,
    load_state_dict_exact,
    load_tensors,
    looks_like_safetensors,
    register_loss,
    register_optimizer,
    save_tensors,
)
from neuralforecast.losses.pytorch import MAE, MQLoss, DistributionLoss
from neuralforecast.models import TCN

polars = pytest.importorskip("polars", reason="polars parity is optional")


def roundtrip(value):
    tensors = {}
    encoded = encode_value(value, tensors, "v")
    json.dumps(encoded)  # must be JSON-serializable, not merely dict-shaped
    return decode_value(encoded, tensors)


# --------------------------------------------------------------------------
# Registry is the security boundary
# --------------------------------------------------------------------------


def test_unregistered_class_is_refused_not_imported():
    for payload in (
        {TAG: "loss", "cls": "os.system", "args": {}},
        {TAG: "loss", "cls": "builtins.eval", "args": {}},
        {TAG: "torch_cls", "kind": "optimizer", "name": "subprocess.Popen"},
        {TAG: "torch_cls", "kind": "lr_scheduler", "name": "posix.system"},
    ):
        with pytest.raises(SerializationError, match="not registered"):
            decode_value(payload)


def test_unknown_registry_is_refused():
    with pytest.raises(SerializationError, match="Unknown registry"):
        decode_value({TAG: "torch_cls", "kind": "builtins", "name": "eval"})


def test_unknown_tag_is_an_error_not_a_passthrough():
    with pytest.raises(SerializationError, match="Unknown tag"):
        decode_value({TAG: "callable", "name": "anything"})


def test_unknown_tag_nested_inside_a_dict_is_still_refused():
    with pytest.raises(SerializationError, match="Unknown tag"):
        decode_value({"a": {"b": [{TAG: "callable", "name": "anything"}]}})


def test_registry_holds_no_dynamic_import():
    """A dynamic-import fallback here would undo the whole point of the registry."""
    import ast

    import neuralforecast._serialization as mod

    tree = ast.parse(open(mod.__file__).read())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "importlib" not in imported

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called & {"eval", "exec", "__import__", "getattr_static"}


# --------------------------------------------------------------------------
# Loss capture and round-trip
# --------------------------------------------------------------------------


def test_loss_roundtrip():
    loss = MQLoss(level=[80, 90])
    decoded = roundtrip(loss)
    assert isinstance(decoded, MQLoss)
    assert torch.equal(decoded.quantiles, loss.quantiles)
    assert decoded.output_names == loss.output_names


def test_distribution_loss_captures_num_pieces_before_it_is_popped():
    """`DistributionLoss.__init__` pops `num_pieces` out of distribution_kwargs.

    Capture happens before that, so the reconstructed loss must match the
    original rather than silently falling back to the default of 5.
    """
    loss = DistributionLoss(distribution="ISQF", num_pieces=7)
    assert loss._nf_init_kwargs["num_pieces"] == 7

    default = DistributionLoss(distribution="ISQF")
    assert loss.outputsize_multiplier != default.outputsize_multiplier

    decoded = roundtrip(loss)
    assert isinstance(decoded, DistributionLoss)
    assert decoded.outputsize_multiplier == loss.outputsize_multiplier


def test_var_keyword_is_flattened_not_nested():
    loss = DistributionLoss(distribution="Normal")
    assert "distribution_kwargs" not in loss._nf_init_kwargs
    assert isinstance(roundtrip(loss), DistributionLoss)


def test_tensor_valued_loss_argument_moves_to_the_payload():
    weight = np.array([0.1, 0.9])
    loss = MAE(horizon_weight=weight)
    tensors = {}
    encoded = encode_value(loss, tensors, "loss")
    assert encoded["args"]["horizon_weight"][TAG] == "tensor"
    assert list(tensors) == ["__hparams__.loss.args.horizon_weight"]
    decoded = decode_value(encoded, tensors)
    assert torch.equal(decoded.horizon_weight, loss.horizon_weight)


def test_custom_loss_needs_registration():
    class MyLoss(MAE):
        pass

    with pytest.raises(SerializationError, match="register_loss"):
        encode_value(MyLoss(), {}, "loss")

    register_loss(MyLoss)
    assert isinstance(roundtrip(MyLoss()), MyLoss)


def test_registration_takes_a_class_not_a_name():
    with pytest.raises(TypeError):
        register_loss("neuralforecast.losses.pytorch.MAE")
    with pytest.raises(TypeError):
        register_optimizer(MAE)


def test_registration_rejects_a_conflicting_name():
    class OtherLoss(MAE):
        pass

    register_loss(OtherLoss, name="dup")
    register_loss(OtherLoss, name="dup")  # idempotent
    with pytest.raises(ValueError, match="already registered"):
        register_loss(MAE, name="dup")


# --------------------------------------------------------------------------
# Tagged values
# --------------------------------------------------------------------------


def test_primitives_and_containers_pass_through():
    value = {"a": 1, "b": [1.5, "x", None, True], "c": {"d": []}}
    assert roundtrip(value) == value


def test_optimizer_and_scheduler_classes():
    assert roundtrip(torch.optim.Adam) is torch.optim.Adam
    assert (
        roundtrip(torch.optim.lr_scheduler.StepLR) is torch.optim.lr_scheduler.StepLR
    )


def test_numpy_scalars_become_primitives():
    encoded, _ = encode_mapping(
        {"i": np.int64(3), "f": np.float32(1.5), "b": np.bool_(True), "s": np.str_("q")}
    )
    assert encoded == {"i": 3, "f": 1.5, "b": True, "s": "q"}


def test_numpy_array_roundtrips_with_dtype():
    array = np.arange(6, dtype="float32").reshape(2, 3)
    decoded = roundtrip(array)
    assert decoded.dtype == array.dtype
    np.testing.assert_array_equal(decoded, array)


def test_datetime64_roundtrip_numpy_and_pandas():
    array = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    np.testing.assert_array_equal(roundtrip(array), array)

    index = pd.DatetimeIndex(array, name="ds")
    decoded = roundtrip(index)
    assert isinstance(decoded, pd.DatetimeIndex)
    assert decoded.name == "ds"
    pd.testing.assert_index_equal(decoded, index)


def test_pandas_index_roundtrip():
    index = pd.Index(["a", "b"], name="unique_id")
    pd.testing.assert_index_equal(roundtrip(index), index)


def test_polars_series_roundtrip():
    series = polars.Series("unique_id", ["a", "b"])
    assert roundtrip(series).to_list() == series.to_list()
    dates = polars.Series("ds", np.array(["2020-01-01"], dtype="datetime64[ns]"))
    assert roundtrip(dates).to_list() == dates.to_list()


def test_unsupported_polars_dtype_fails_at_save_time():
    with pytest.raises(SerializationError, match="no JSON encoding"):
        encode_value(polars.Series("x", [[1, 2]]), {}, "x")


def test_unencodable_value_names_the_key():
    with pytest.raises(SerializationError, match="hp.logger"):
        encode_mapping({"hp": {"logger": object()}})


def test_non_string_dict_key_is_refused():
    with pytest.raises(SerializationError, match="string dict keys"):
        encode_mapping({"hp": {1: "x"}})


def test_tensor_pointer_to_a_missing_payload_is_refused():
    with pytest.raises(SerializationError, match="missing from the payload"):
        decode_value({TAG: "tensor", "key": "__hparams__.gone", "kind": "torch"}, {})


def test_encode_decode_mapping_roundtrip():
    mapping = {"h": 12, "loss": MQLoss(level=[80]), "optimizer": torch.optim.Adam}
    encoded, tensors = encode_mapping(mapping)
    decoded = decode_mapping(json.loads(json.dumps(encoded)), tensors)
    assert decoded["h"] == 12
    assert isinstance(decoded["loss"], MQLoss)
    assert decoded["optimizer"] is torch.optim.Adam


# --------------------------------------------------------------------------
# Tensor payload
# --------------------------------------------------------------------------


def test_shared_tensors_are_dropped_and_recorded():
    model = TCN(h=2, input_size=4, max_steps=1)
    state_dict = model.state_dict()
    blob = save_tensors(state_dict, {"model_class": "TCN"})
    tensors, metadata = load_tensors(blob)

    shared = json.loads(metadata["shared"])
    dropped = {n for names in shared.values() for n in names}
    assert dropped, "TCN is expected to have tied tensors"
    assert set(tensors) == set(state_dict) - dropped
    assert metadata["nf_format"] == "2"


def test_load_state_dict_exact_restores_ties():
    model = TCN(h=2, input_size=4, max_steps=1)
    blob = save_tensors(model.state_dict(), {})
    tensors, metadata = load_tensors(blob)

    fresh = TCN(h=2, input_size=4, max_steps=1)
    load_state_dict_exact(fresh, tensors, metadata)
    for key, value in model.state_dict().items():
        assert torch.equal(fresh.state_dict()[key], value), key


def test_load_state_dict_exact_refuses_a_truncated_payload():
    model = TCN(h=2, input_size=4, max_steps=1)
    blob = save_tensors(model.state_dict(), {})
    tensors, metadata = load_tensors(blob)
    tensors.pop(sorted(tensors)[0])

    with pytest.raises(SerializationError, match="does not match"):
        load_state_dict_exact(TCN(h=2, input_size=4, max_steps=1), tensors, metadata)


def test_complex_dtype_roundtrips():
    blob = save_tensors({"w": torch.randn(4, dtype=torch.complex64)}, {})
    tensors, _ = load_tensors(blob)
    assert tensors["w"].dtype == torch.complex64


def test_format_sniffing():
    blob = save_tensors({"w": torch.zeros(2)}, {})
    assert looks_like_safetensors(blob)
    assert not looks_like_safetensors(b"\x80\x05\x95pickled-bytes")
    assert not looks_like_safetensors(b"")
    assert not looks_like_safetensors(b"\xff" * 8 + b"junk")


def test_upstream_canary():
    """`_remove_duplicate_names` is private; fail here, not at a user's load."""
    from safetensors.torch import _remove_duplicate_names

    shared = _remove_duplicate_names(TCN(h=2, input_size=4, max_steps=1).state_dict())
    assert shared, "upstream no longer reports TCN's tied tensors"
    assert all(isinstance(v, list) for v in shared.values())


# --------------------------------------------------------------------------
# Recovering constructor arguments from an unpickled loss
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loss",
    [
        MAE(),
        MQLoss(level=[80, 90]),
        DistributionLoss(distribution="Normal"),
        DistributionLoss(distribution="ISQF", num_pieces=7),
    ],
    ids=["MAE", "MQLoss", "Normal", "ISQF"],
)
def test_loss_args_are_recovered_when_init_never_ran(loss):
    """Unpickling restores an object without calling `__init__`."""
    del loss._nf_init_kwargs

    decoded = roundtrip(loss)
    assert type(decoded) is type(loss)
    assert decoded.output_names == loss.output_names
    assert decoded.outputsize_multiplier == loss.outputsize_multiplier


def test_recovery_keeps_num_pieces_out_of_the_default():
    """ISQF's `num_pieces` survives only via the `domain_map` partial."""
    loss = DistributionLoss(distribution="ISQF", num_pieces=7)
    default = DistributionLoss(distribution="ISQF")
    assert loss.outputsize_multiplier != default.outputsize_multiplier

    del loss._nf_init_kwargs
    assert roundtrip(loss).outputsize_multiplier == loss.outputsize_multiplier


def test_recovery_fails_loudly_when_it_would_change_the_loss(monkeypatch):
    from neuralforecast import _serialization

    loss = MQLoss(level=[80, 90])
    del loss._nf_init_kwargs
    monkeypatch.setattr(_serialization, "_LOSS_DERIVED_STATE", ())
    monkeypatch.setattr(
        _serialization, "_LOSS_INVARIANTS", ("output_names", "outputsize_multiplier")
    )
    with pytest.raises(SerializationError, match="changes `output_names`"):
        encode_value(loss, {}, "loss")


def test_decode_refuses_unexpected_loss_state():
    payload = {
        TAG: "loss",
        "cls": "MAE",
        "args": {"horizon_weight": None},
        "state": {"forward": "anything"},
    }
    with pytest.raises(SerializationError, match="unexpected loss attribute"):
        decode_value(payload)
