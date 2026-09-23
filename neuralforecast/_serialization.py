"""Non-executable serialization primitives for the v2 artifact format.

Tensors go to safetensors, everything else to JSON. The JSON carries tagged
values for the handful of non-primitive things a checkpoint has to hold.

Security boundary: every class named inside an artifact is resolved through a
closed registry in this module. Nothing here ever calls `importlib`, `eval`, or
`getattr` on a module path taken from an artifact, and nothing here may be
changed to do so. A name that is not in a registry is a load error, never a
dynamic-import fallback -- the registry is the only reason JSON is safer than
pickle here, since JSON is just a transport.
"""

import copy
import functools
import inspect
import json
import struct
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from safetensors.torch import load as _st_load
from safetensors.torch import save as _st_save

TAG = "__nf__"
HPARAM_TENSOR_PREFIX = "__hparams__"
FORMAT_VERSION = "2"

# safetensors files start with a little-endian uint64 header length followed by
# that many bytes of JSON. There is no magic number, so this is the sniff.
_SAFETENSORS_HEADER_SIZE = 8
_MAX_HEADER_BYTES = 100_000_000


class SerializationError(Exception):
    """Raised when a value cannot be encoded, or an artifact cannot be decoded."""


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

_LOSSES: Dict[str, type] = {}
_USER_LOSSES: Dict[str, type] = {}
_USER_OPTIMIZERS: Dict[str, type] = {}
_USER_LR_SCHEDULERS: Dict[str, type] = {}


def _builtin_losses() -> Dict[str, type]:
    """Built-in losses, resolved lazily to avoid a circular import."""
    if not _LOSSES:
        import neuralforecast.losses.pytorch as losses

        for name in losses.__all__:
            obj = getattr(losses, name)
            if isinstance(obj, type) and issubclass(obj, _SerializableLoss):
                _LOSSES[name] = obj
    return _LOSSES


def _torch_members(module, base: type) -> Dict[str, type]:
    return {
        name: obj
        for name, obj in vars(module).items()
        if isinstance(obj, type) and issubclass(obj, base) and obj is not base
    }


@functools.lru_cache(maxsize=None)
def _builtin_optimizers() -> Dict[str, type]:
    return _torch_members(torch.optim, torch.optim.Optimizer)


@functools.lru_cache(maxsize=None)
def _builtin_lr_schedulers() -> Dict[str, type]:
    return _torch_members(
        torch.optim.lr_scheduler, torch.optim.lr_scheduler.LRScheduler
    )


@functools.lru_cache(maxsize=None)
def _datasets() -> Dict[str, type]:
    from neuralforecast.tsdataset import (
        LocalFilesTimeSeriesDataset,
        TimeSeriesDataset,
    )

    return {
        "TimeSeriesDataset": TimeSeriesDataset,
        "LocalFilesTimeSeriesDataset": LocalFilesTimeSeriesDataset,
    }


def register_loss(cls: type, name: Optional[str] = None) -> type:
    """Allow a user-defined loss to be saved and loaded in the v2 format.

    Registration is an explicit in-process call made by the user. It is never
    driven by data read from an artifact -- doing so would defeat the registry.

    Args:
        cls (type): The loss class, a subclass of a `neuralforecast` loss base.
        name (Optional[str]): Name to register under. Defaults to `cls.__name__`.

    Returns:
        type: `cls`, so this can be used as a decorator.
    """
    return _register(cls, name, _USER_LOSSES, _SerializableLoss, "loss")


def register_optimizer(cls: type, name: Optional[str] = None) -> type:
    """Allow a user-defined optimizer to be saved and loaded. See `register_loss`."""
    return _register(cls, name, _USER_OPTIMIZERS, torch.optim.Optimizer, "optimizer")


def register_lr_scheduler(cls: type, name: Optional[str] = None) -> type:
    """Allow a user-defined LR scheduler to be saved and loaded. See `register_loss`."""
    return _register(
        cls,
        name,
        _USER_LR_SCHEDULERS,
        torch.optim.lr_scheduler.LRScheduler,
        "lr_scheduler",
    )


def _register(cls, name, target, base, kind):
    if not isinstance(cls, type):
        raise TypeError(f"register_{kind} expects a class, got {type(cls).__name__}.")
    if not issubclass(cls, base):
        raise TypeError(f"{cls.__name__} is not a subclass of {base.__name__}.")
    name = name or cls.__name__
    existing = target.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"A different {kind} is already registered as {name!r}.")
    target[name] = cls
    return cls


def _registries(kind: str) -> Tuple[Dict[str, type], Dict[str, type]]:
    empty: Dict[str, type] = {}
    registries: Dict[str, Tuple[Dict[str, type], Dict[str, type]]] = {
        "loss": (_builtin_losses(), _USER_LOSSES),
        "optimizer": (_builtin_optimizers(), _USER_OPTIMIZERS),
        "lr_scheduler": (_builtin_lr_schedulers(), _USER_LR_SCHEDULERS),
        "dataset": (_datasets(), empty),
    }
    if kind not in registries:
        raise SerializationError(f"Unknown registry {kind!r} in artifact.")
    return registries[kind]


def _resolve(name: str, kind: str) -> type:
    builtin, user = _registries(kind)
    cls = builtin.get(name) or user.get(name)
    if cls is None:
        raise SerializationError(
            f"{kind} {name!r} is not registered. Artifacts only resolve names "
            f"through a closed registry; loading is refused rather than "
            f"importing {name!r}."
        )
    return cls


def _name_of(cls: type, kind: str) -> str:
    builtin, user = _registries(kind)
    for registry in (builtin, user):
        for name, registered in registry.items():
            if registered is cls:
                return name
    raise SerializationError(
        f"{cls.__name__} is not a registered {kind}, so it cannot be saved. "
        f"Register it with neuralforecast.register_{kind}({cls.__name__}) before "
        f"saving, or supply it again at load time."
    )


# ---------------------------------------------------------------------------
# Loss constructor-argument capture
# ---------------------------------------------------------------------------


class _SerializableLoss:
    """Records the arguments a loss was constructed with, as `_nf_init_kwargs`.

    `save_hyperparameters()` stores the loss *object*, not how it was built, so
    without this there is nothing to write into a JSON artifact.
    """

    _nf_init_kwargs: Dict[str, Any]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        orig = cls.__dict__.get("__init__")
        if orig is None or getattr(orig, "_nf_wrapped", False):
            return

        signature = inspect.signature(orig)
        var_keyword = next(
            (
                p.name
                for p in signature.parameters.values()
                if p.kind is inspect.Parameter.VAR_KEYWORD
            ),
            None,
        )

        @functools.wraps(orig)
        def __init__(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            captured = dict(list(bound.arguments.items())[1:])
            if var_keyword is not None:
                # `bind` nests **kwargs under the parameter's own name; flatten
                # it or the round-tripped call gets `distribution_kwargs=...`.
                captured.update(captured.pop(var_keyword, {}))
            # Capture before the original runs: DistributionLoss.__init__ pops
            # `num_pieces` out of distribution_kwargs, mutating the caller's dict.
            captured = copy.deepcopy(captured)
            orig(self, *args, **kwargs)
            self._nf_init_kwargs = captured

        __init__._nf_wrapped = True
        cls.__init__ = __init__


# ---------------------------------------------------------------------------
# Tagged JSON encoding
# ---------------------------------------------------------------------------

_POLARS_DTYPES = ("String", "Int32", "Int64", "Float32", "Float64", "Datetime", "Date")


def encode_value(value: Any, tensors: Dict[str, torch.Tensor], path: str) -> Any:
    """Encode one value to JSON-safe data, moving tensors into `tensors`.

    Args:
        value: The value to encode.
        tensors (dict): Collects tensors keyed by their pointer; mutated in place.
        path (str): Dotted path of `value`, used for tensor keys and error messages.

    Returns:
        A JSON-serializable representation of `value`.

    Raises:
        SerializationError: If `value` has no encoding. Raised at save time so
            the person who can fix it finds out immediately.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [encode_value(v, tensors, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise SerializationError(
                    f"{path}: only string dict keys can be encoded, got {key!r}."
                )
        return {k: encode_value(v, tensors, f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.str_):
        return str(value)

    tagged = _encode_tagged(value, tensors, path)
    if tagged is not None:
        return tagged

    raise SerializationError(
        f"{path}: cannot encode a value of type {type(value).__name__} without "
        f"pickle. Remove it before saving, or register it if it is a loss, "
        f"optimizer or scheduler."
    )


def _encode_tagged(value, tensors, path):
    if isinstance(value, _SerializableLoss):
        args = getattr(value, "_nf_init_kwargs", None)
        if args is None:
            raise SerializationError(
                f"{path}: {type(value).__name__} was built before its constructor "
                f"arguments could be recorded, so it cannot be saved."
            )
        return {
            TAG: "loss",
            "cls": _name_of(type(value), "loss"),
            "args": encode_value(args, tensors, f"{path}.args"),
        }
    if isinstance(value, type) and issubclass(value, torch.optim.Optimizer):
        return {TAG: "torch_cls", "kind": "optimizer", "name": _name_of(value, "optimizer")}
    if isinstance(value, type) and issubclass(
        value, torch.optim.lr_scheduler.LRScheduler
    ):
        return {
            TAG: "torch_cls",
            "kind": "lr_scheduler",
            "name": _name_of(value, "lr_scheduler"),
        }
    if isinstance(value, torch.Tensor):
        key = f"{HPARAM_TENSOR_PREFIX}.{path}"
        tensors[key] = value.detach().cpu().contiguous()
        return {TAG: "tensor", "key": key, "kind": "torch"}
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "M":
            return _encode_datetime64(value)
        if value.dtype.kind in "US":
            return {TAG: "index", "kind": "numpy", "dtype": "str", "values": value.tolist()}
        key = f"{HPARAM_TENSOR_PREFIX}.{path}"
        tensors[key] = torch.as_tensor(np.ascontiguousarray(value))
        return {TAG: "tensor", "key": key, "kind": "numpy", "dtype": str(value.dtype)}

    pandas_index = _as_pandas_index(value)
    if pandas_index is not None:
        return pandas_index
    polars_series = _as_polars_series(value, path)
    if polars_series is not None:
        return polars_series
    return None


def _encode_datetime64(values, kind="numpy"):
    unit = np.datetime_data(values.dtype)[0]
    return {
        TAG: "datetime64",
        "kind": kind,
        "unit": unit,
        "values": values.astype("int64").tolist(),
    }


def _as_pandas_index(value):
    import pandas as pd

    if not isinstance(value, pd.Index):
        return None
    if value.dtype.kind == "M":
        encoded = _encode_datetime64(value.to_numpy(), kind="pandas")
        encoded["name"] = value.name
        return encoded
    return {
        TAG: "index",
        "kind": "pandas",
        "dtype": str(value.dtype),
        "name": value.name,
        "values": value.tolist(),
    }


def _as_polars_series(value, path):
    polars = _polars_or_none()
    if polars is None or not isinstance(value, polars.Series):
        return None
    dtype = str(value.dtype)
    base = dtype.split("(")[0]
    if base not in _POLARS_DTYPES:
        raise SerializationError(
            f"{path}: polars dtype {dtype!r} has no JSON encoding. Supported: "
            f"{', '.join(_POLARS_DTYPES)}."
        )
    if base in ("Datetime", "Date"):
        return _encode_datetime64(value.to_numpy(), kind="polars")
    return {
        TAG: "index",
        "kind": "polars",
        "dtype": base,
        "name": value.name,
        "values": value.to_list(),
    }


def _polars_or_none():
    try:
        import polars
    except ImportError:
        return None
    return polars


def decode_value(value: Any, tensors: Optional[Dict[str, torch.Tensor]] = None) -> Any:
    """Inverse of `encode_value`.

    An unrecognised tag is an error, never passed through as a plain dict: a
    decoder that silently accepts unknown tags is how a future format extension
    turns into a bypass.
    """
    tensors = tensors if tensors is not None else {}
    if isinstance(value, list):
        return [decode_value(v, tensors) for v in value]
    if not isinstance(value, dict):
        return value
    if TAG not in value:
        return {k: decode_value(v, tensors) for k, v in value.items()}

    kind = value[TAG]
    if kind == "loss":
        cls = _resolve(value["cls"], "loss")
        return cls(**decode_value(value["args"], tensors))
    if kind == "torch_cls":
        return _resolve(value["name"], value["kind"])
    if kind == "tensor":
        return _decode_tensor(value, tensors)
    if kind == "datetime64":
        return _decode_datetime64(value)
    if kind == "index":
        return _decode_index(value)
    raise SerializationError(
        f"Unknown tag {kind!r} in artifact metadata. Refusing to load rather "
        f"than guessing at its meaning."
    )


def _decode_tensor(value, tensors):
    key = value["key"]
    if key not in tensors:
        raise SerializationError(
            f"Metadata points at tensor {key!r}, which is missing from the payload."
        )
    tensor = tensors[key]
    if value["kind"] == "numpy":
        return tensor.numpy().astype(value["dtype"])
    return tensor


def _decode_datetime64(value):
    array = np.asarray(value["values"], dtype="int64").astype(f"datetime64[{value['unit']}]")
    if value["kind"] == "pandas":
        import pandas as pd

        return pd.DatetimeIndex(array, name=value.get("name"))
    if value["kind"] == "polars":
        polars = _polars_or_none()
        if polars is None:
            raise SerializationError("Artifact holds polars data but polars is not installed.")
        return polars.Series(value.get("name") or "", array)
    return array


def _decode_index(value):
    if value["kind"] == "numpy":
        return np.asarray(value["values"], dtype=object).astype(str)
    if value["kind"] == "pandas":
        import pandas as pd

        return pd.Index(value["values"], dtype=value["dtype"], name=value.get("name"))
    polars = _polars_or_none()
    if polars is None:
        raise SerializationError("Artifact holds polars data but polars is not installed.")
    dtype = value["dtype"]
    if dtype not in _POLARS_DTYPES:
        raise SerializationError(f"Unsupported polars dtype {dtype!r} in artifact.")
    return polars.Series(
        value.get("name") or "", value["values"], dtype=getattr(polars, dtype)
    )


def encode_mapping(mapping: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Encode a dict of hyperparameters into (JSON-safe dict, tensors)."""
    tensors: Dict[str, torch.Tensor] = {}
    encoded = {k: encode_value(v, tensors, k) for k, v in mapping.items()}
    return encoded, tensors


def decode_mapping(
    mapping: Dict[str, Any], tensors: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, Any]:
    """Inverse of `encode_mapping`."""
    return {k: decode_value(v, tensors) for k, v in mapping.items()}


# ---------------------------------------------------------------------------
# Tensor payload
# ---------------------------------------------------------------------------


def _remove_duplicate_names(state_dict):
    # Private upstream API. tests/test_serialization.py::test_upstream_canary
    # imports it directly so a breaking change fails in CI, not at a user's load.
    from safetensors.torch import _remove_duplicate_names as upstream

    return upstream(state_dict)


def save_tensors(
    state_dict: Dict[str, torch.Tensor], metadata: Dict[str, str]
) -> bytes:
    """Serialize a state dict to safetensors bytes, handling shared storage.

    `safetensors.save` refuses tied tensors (TCN, FEDformer and TimeLLM all have
    them), so duplicates are dropped and the tie recorded in the header. They are
    re-created at load time by constructing the model before applying weights.
    """
    shared = _remove_duplicate_names(state_dict)
    dropped = {name for names in shared.values() for name in names}
    kept = {
        k: v.detach().cpu().contiguous()
        for k, v in state_dict.items()
        if k not in dropped
    }
    header = dict(metadata)
    header["nf_format"] = FORMAT_VERSION
    header["shared"] = json.dumps(shared)
    return _st_save(kept, metadata=header)


def load_tensors(data: bytes) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """Inverse of `save_tensors`. Returns (tensors, metadata)."""
    if not looks_like_safetensors(data):
        raise SerializationError("Not a safetensors payload.")
    return _st_load(data), read_metadata(data)


def looks_like_safetensors(data: bytes) -> bool:
    """Whether `data` starts with a plausible safetensors header.

    This is how the reader decides which format it is looking at. The decision
    is never delegated to a field *inside* the artifact.
    """
    if len(data) < _SAFETENSORS_HEADER_SIZE:
        return False
    (length,) = struct.unpack("<Q", data[:_SAFETENSORS_HEADER_SIZE])
    if length == 0 or length > _MAX_HEADER_BYTES:
        return False
    header = data[_SAFETENSORS_HEADER_SIZE : _SAFETENSORS_HEADER_SIZE + length]
    if len(header) < length:
        return False
    try:
        return isinstance(json.loads(header), dict)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False


def read_metadata(data: bytes) -> Dict[str, str]:
    """Read the `__metadata__` map out of safetensors bytes."""
    (length,) = struct.unpack("<Q", data[:_SAFETENSORS_HEADER_SIZE])
    header = json.loads(data[_SAFETENSORS_HEADER_SIZE : _SAFETENSORS_HEADER_SIZE + length])
    return header.get("__metadata__", {})


def load_state_dict_exact(
    module: torch.nn.Module,
    tensors: Dict[str, torch.Tensor],
    metadata: Dict[str, str],
) -> None:
    """Apply `tensors` to `module`, allowing only the recorded ties to be missing.

    A truncated or tampered payload must not load as a partially-random model,
    so anything missing beyond the dropped duplicates is an error.
    """
    shared = json.loads(metadata.get("shared", "{}"))
    expected_missing = {name for names in shared.values() for name in names}
    incompatible = module.load_state_dict(tensors, strict=False)
    missing = set(incompatible.missing_keys)
    if missing != expected_missing:
        raise SerializationError(
            f"Checkpoint does not match the model. Missing keys "
            f"{sorted(missing - expected_missing)}, unexpected keys "
            f"{sorted(incompatible.unexpected_keys)}."
        )
