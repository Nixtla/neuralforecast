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

import fsspec
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


_LOCAL_PROTOCOLS = frozenset({"file", "local", "memory"})


def ensure_trusted_path(path, trust_remote: bool) -> None:
    """Refuse a non-local artifact path unless the caller opted in.

    Fetching an artifact over `s3://`, `gcs://` or `http://` is the deployment
    pattern this class of bug is exploited through: whoever can write that object
    -- a leaked CI token, a broad bucket ACL, a staging-to-prod promotion -- picks
    the bytes that get deserialized on the loading host.
    """
    if trust_remote:
        return
    protocol = fsspec.utils.get_protocol(str(path))
    if protocol in _LOCAL_PROTOCOLS:
        return
    raise ValueError(
        f"Refusing to load from the remote path {path!r} ({protocol}://). Anyone "
        f"who can write that location chooses what gets loaded here. Pass "
        f"`trust_remote=True` if you control it, or download it first and inspect "
        f"it."
    )


def registered_classes(kind: str) -> Tuple[type, ...]:
    """Every class currently registered under `kind`, built-in and user."""
    builtin, user = _registries(kind)
    return tuple(dict.fromkeys(list(builtin.values()) + list(user.values())))


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


def encode_value(
    value: Any, tensors: Optional[Dict[str, torch.Tensor]], path: str
) -> Any:
    """Encode one value to JSON-safe data.

    Args:
        value: The value to encode.
        tensors (dict or None): Collects tensors keyed by their pointer, mutated
            in place. Pass None to inline arrays into the JSON instead, for
            metadata that ships without a safetensors sidecar.
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
        state: Dict[str, Any] = {}
        if args is None:
            # Unpickling restores an object without calling `__init__`, so a loss
            # read from a legacy checkpoint has nothing recorded. Recover what the
            # constructor would have been given from the object itself.
            args, state = _recover_loss_args(value, path)
        encoded = {
            TAG: "loss",
            "cls": _name_of(type(value), "loss"),
            "args": encode_value(args, tensors, f"{path}.args"),
        }
        if state:
            encoded["state"] = encode_value(state, tensors, f"{path}.state")
        return encoded
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
        return _encode_array(value.detach().cpu(), tensors, path, "torch")
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "M":
            return _encode_datetime64(value)
        if value.dtype.kind in "USO":
            return _encode_str_array(value, path)
        return _encode_array(value, tensors, path, "numpy")
    if _is_scaler(value):
        return _encode_scaler(value, tensors, path)
    prediction_intervals = _as_prediction_intervals(value)
    if prediction_intervals is not None:
        return prediction_intervals

    pandas_frame = _as_pandas_frame(value, tensors, path)
    if pandas_frame is not None:
        return pandas_frame
    polars_frame = _as_polars_frame(value, tensors, path)
    if polars_frame is not None:
        return polars_frame
    pandas_index = _as_pandas_index(value)
    if pandas_index is not None:
        return pandas_index
    polars_series = _as_polars_series(value, path)
    if polars_series is not None:
        return polars_series
    return None


def _encode_array(value, tensors, path, kind):
    dtype = str(value.dtype)
    if tensors is None:
        array = value.numpy() if isinstance(value, torch.Tensor) else value
        return {
            TAG: "array",
            "kind": kind,
            "dtype": dtype,
            "shape": list(array.shape),
            "values": array.reshape(-1).tolist(),
        }
    key = f"{HPARAM_TENSOR_PREFIX}.{path}"
    if isinstance(value, torch.Tensor):
        tensors[key] = value.contiguous()
        return {TAG: "tensor", "key": key, "kind": "torch"}
    tensors[key] = torch.as_tensor(np.ascontiguousarray(value))
    return {TAG: "tensor", "key": key, "kind": "numpy", "dtype": dtype}


def _encode_str_array(value, path):
    values = value.tolist()
    if not all(v is None or isinstance(v, str) for v in values):
        raise SerializationError(
            f"{path}: object arrays can only be encoded when every element is a "
            f"string or None."
        )
    return {TAG: "index", "kind": "numpy", "dtype": str(value.dtype), "values": values}


# Attributes `__init__` derives rather than stores, which therefore have to be
# re-applied when constructor arguments were recovered instead of recorded.
_LOSS_DERIVED_STATE = ("output_names",)

# Attributes that must match after a recovered rebuild. A mismatch means the
# recovery was lossy and the reconstructed loss would behave differently.
_LOSS_INVARIANTS = ("outputsize_multiplier", "is_distribution_output")


def _recover_loss_args(loss, path):
    """Rebuild a loss's constructor arguments from the object itself.

    Only needed for losses restored from a legacy pickle. This is verified by
    trial reconstruction rather than trusted: if the rebuilt loss would differ
    from the original, saving fails instead of writing a subtly wrong artifact.
    """
    cls = type(loss)
    parameters = list(inspect.signature(cls.__init__).parameters.items())[1:]
    args: Dict[str, Any] = {}
    for name, parameter in parameters:
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if hasattr(loss, name):
            attribute = getattr(loss, name)
            # A Parameter is registered state that `__init__` built; hand it back
            # as plain data. A plain tensor attribute is what the caller passed.
            args[name] = (
                attribute.detach().tolist()
                if isinstance(attribute, torch.nn.Parameter)
                else attribute
            )
        elif parameter.default is not parameter.empty:
            args[name] = parameter.default
        else:
            raise SerializationError(
                f"{path}: cannot recover the `{name}` argument of "
                f"{cls.__name__}, so it cannot be saved. Re-supply the loss at "
                f"load time, e.g. `Model.load(path, loss=...)`."
            )

    num_pieces = _recover_num_pieces(loss)
    if num_pieces is not None:
        args["num_pieces"] = num_pieces

    state = {
        name: getattr(loss, name)
        for name in _LOSS_DERIVED_STATE
        if hasattr(loss, name)
    }
    _verify_recovered_loss(loss, cls, args, state, path)
    return args, state


def _recover_num_pieces(loss):
    """`DistributionLoss` pops `num_pieces` into its `domain_map` partial."""
    domain_map = getattr(loss, "domain_map", None)
    keywords = getattr(domain_map, "keywords", None)
    if isinstance(keywords, dict) and "num_pieces" in keywords:
        return keywords["num_pieces"]
    return None


def _verify_recovered_loss(loss, cls, args, state, path):
    try:
        rebuilt = cls(**args)
    except Exception as e:
        raise SerializationError(
            f"{path}: could not rebuild {cls.__name__} from the arguments "
            f"recovered from it ({e}). Re-supply the loss at load time, e.g. "
            f"`Model.load(path, loss=...)`."
        ) from e
    for name, expected in state.items():
        setattr(rebuilt, name, expected)
    for name in _LOSS_INVARIANTS:
        if getattr(rebuilt, name, None) != getattr(loss, name, None):
            raise SerializationError(
                f"{path}: rebuilding {cls.__name__} from its own attributes "
                f"changes `{name}`, so saving it would produce a different loss. "
                f"Re-supply the loss at load time, e.g. "
                f"`Model.load(path, loss=...)`."
            )


def _as_prediction_intervals(value):
    from neuralforecast.utils import PredictionIntervals

    if type(value) is not PredictionIntervals:
        return None
    return {
        TAG: "prediction_intervals",
        "n_windows": value.n_windows,
        "method": value.method,
        "step_size": value.step_size,
    }


def _scaler_classes():
    from neuralforecast.core import _type2scaler

    return tuple({type(factory()) for factory in _type2scaler.values()})


def _is_scaler(value):
    return isinstance(value, _scaler_classes())


def _scaler_type_name(value):
    """`_type2scaler` is already a closed registry; find the key for this object.

    Some entries share a class and differ only by a constructor argument
    (robust/mad vs robust-iqr), so the probe is compared on those too.
    """
    from neuralforecast.core import _type2scaler

    for name, factory in _type2scaler.items():
        probe = factory()
        if type(probe) is not type(value):
            continue
        if all(
            getattr(probe, attr, None) == getattr(value, attr, None)
            for attr in ("scale", "method", "lower")
        ):
            return name
    return None


def _encode_scaler(value, tensors, path):
    """coreforecast scalers hold their whole fitted state in `stats_`."""
    name = _scaler_type_name(value)
    if name is None:
        raise SerializationError(
            f"{path}: {type(value).__name__} is not one of the supported local "
            f"scalers, so it cannot be saved."
        )
    stats = getattr(value, "stats_", None)
    return {
        TAG: "scaler",
        "type": name,
        "stats_": None
        if stats is None
        else encode_value(stats, tensors, f"{path}.stats_"),
    }


def _encode_datetime64(values, kind="numpy"):
    unit = np.datetime_data(values.dtype)[0]
    return {
        TAG: "datetime64",
        "kind": kind,
        "unit": unit,
        "values": values.astype("int64").tolist(),
    }


def _as_pandas_frame(value, tensors, path):
    import pandas as pd

    if isinstance(value, pd.Series):
        return {
            TAG: "series",
            "kind": "pandas",
            "name": value.name,
            "dtype": str(value.dtype),
            "data": encode_value(value.to_numpy(), tensors, f"{path}.values"),
        }
    if isinstance(value, pd.DataFrame):
        return {
            TAG: "dataframe",
            "kind": "pandas",
            "columns": [
                [str(col), encode_value(value[col], tensors, f"{path}.{col}")]
                for col in value.columns
            ],
        }
    return None


def _as_polars_frame(value, tensors, path):
    polars = _polars_or_none()
    if polars is None or not isinstance(value, polars.DataFrame):
        return None
    return {
        TAG: "dataframe",
        "kind": "polars",
        "columns": [
            [name, encode_value(value[name], tensors, f"{path}.{name}")]
            for name in value.columns
        ],
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
        loss = cls(**decode_value(value["args"], tensors))
        for name, attribute in decode_value(value.get("state", {}), tensors).items():
            if name not in _LOSS_DERIVED_STATE:
                raise SerializationError(
                    f"Refusing to restore unexpected loss attribute {name!r}."
                )
            setattr(loss, name, attribute)
        return loss
    if kind == "torch_cls":
        return _resolve(value["name"], value["kind"])
    if kind == "tensor":
        return _decode_tensor(value, tensors)
    if kind == "datetime64":
        return _decode_datetime64(value)
    if kind == "index":
        return _decode_index(value)
    if kind == "array":
        return _decode_array(value)
    if kind == "scaler":
        return _decode_scaler(value, tensors)
    if kind == "prediction_intervals":
        from neuralforecast.utils import PredictionIntervals

        return PredictionIntervals(
            n_windows=value["n_windows"],
            method=value["method"],
            step_size=value["step_size"],
        )
    if kind == "series":
        return _decode_series(value, tensors)
    if kind == "dataframe":
        return _decode_frame(value, tensors)
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


def _decode_array(value):
    array = np.asarray(value["values"], dtype=value["dtype"]).reshape(value["shape"])
    if value["kind"] == "torch":
        return torch.as_tensor(array)
    return array


def _decode_scaler(value, tensors):
    from neuralforecast.core import _type2scaler

    name = value["type"]
    if name not in _type2scaler:
        raise SerializationError(f"Unknown local scaler type {name!r} in artifact.")
    scaler = _type2scaler[name]()
    stats = value.get("stats_")
    if stats is not None:
        scaler.stats_ = np.ascontiguousarray(decode_value(stats, tensors))
    return scaler


def _decode_series(value, tensors):
    import pandas as pd

    data = decode_value(value["data"], tensors)
    return pd.Series(data, name=value.get("name"), dtype=value["dtype"])


def _decode_frame(value, tensors):
    columns = [(name, decode_value(data, tensors)) for name, data in value["columns"]]
    if value["kind"] == "pandas":
        import pandas as pd

        return pd.DataFrame({name: data for name, data in columns})
    polars = _polars_or_none()
    if polars is None:
        raise SerializationError("Artifact holds polars data but polars is not installed.")
    return polars.DataFrame({name: data for name, data in columns})


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


def encode_mapping(
    mapping: Dict[str, Any], inline: bool = False
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Encode a dict into (JSON-safe dict, tensors).

    Args:
        mapping (dict): Values to encode.
        inline (bool): Write arrays into the JSON instead of a tensor sidecar.
            Used for `configuration.json`, which ships without one.
    """
    tensors: Dict[str, torch.Tensor] = {}
    target = None if inline else tensors
    encoded = {k: encode_value(v, target, k) for k, v in mapping.items()}
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


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

# Constructor arguments per dataset class. Explicit rather than a `vars()` dump,
# so a new attribute is a visible decision instead of a silent round-trip gap.
_DATASET_FIELDS = {
    "TimeSeriesDataset": (
        "temporal",
        "temporal_cols",
        "indptr",
        "y_idx",
        "static",
        "static_cols",
    ),
    "LocalFilesTimeSeriesDataset": (
        "files_ds",
        "temporal_cols",
        "id_col",
        "time_col",
        "target_col",
        "last_times",
        "indices",
        "max_size",
        "min_size",
        "y_idx",
        "static",
        "static_cols",
    ),
}
_DATASET_EXTRA = ("updated",)


def encode_dataset(dataset) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Encode a dataset into (JSON-safe dict, tensors for the safetensors sidecar)."""
    name = _name_of(type(dataset), "dataset")
    tensors: Dict[str, torch.Tensor] = {}
    fields = {
        field: encode_value(getattr(dataset, field), tensors, field)
        for field in _DATASET_FIELDS[name]
    }
    extra = {
        field: encode_value(getattr(dataset, field), tensors, field)
        for field in _DATASET_EXTRA
        if hasattr(dataset, field)
    }
    return {"dataset_class": name, "fields": fields, "extra": extra}, tensors


def decode_dataset(meta: Dict[str, Any], tensors: Dict[str, torch.Tensor]):
    """Inverse of `encode_dataset`."""
    cls = _resolve(meta["dataset_class"], "dataset")
    dataset = cls(**decode_mapping(meta["fields"], tensors))
    for field, value in decode_mapping(meta.get("extra", {}), tensors).items():
        setattr(dataset, field, value)
    return dataset
