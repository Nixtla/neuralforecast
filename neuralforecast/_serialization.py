"""Non-executable serialization for the v2 artifact format.

Tensors go to safetensors, everything else to tagged JSON. Class names resolve
through the closed registries below -- never `importlib`, `eval` or `getattr` on
a module path -- and an unregistered name is a load error, never a fallback.
That registry, not the format, is what makes this safer than pickle.
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

# safetensors has no magic number: a little-endian uint64 header length, then
# that many bytes of JSON.
_SAFETENSORS_HEADER_SIZE = 8
_MAX_HEADER_BYTES = 100_000_000


class SerializationError(Exception):
    """Raised when a value cannot be encoded, or an artifact cannot be decoded."""


# --- Registries ---

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
    """Allow a user-defined loss to be saved and loaded.

    Any `nn.Module` is accepted. One that does not inherit a neuralforecast loss
    base has no recorded init kwargs, so they are recovered from the object and
    checked by trial rebuild at save time.

    Registration is always an explicit in-process call, never driven by data
    read from an artifact -- that would defeat the registry.

    Args:
        cls (type): The loss class.
        name (Optional[str]): Name to register under. Defaults to `cls.__name__`.

    Returns:
        type: `cls`, so this can be used as a decorator.
    """
    return _register(cls, name, _USER_LOSSES, torch.nn.Module, "loss")


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
    """Refuse a non-local path unless opted in: whoever can write it picks the bytes."""
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


# --- Loss constructor-argument capture ---


class _SerializableLoss:
    """Records a loss's init kwargs; `save_hyperparameters` stores only the object."""

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
                # `bind` nests **kwargs under its own name; flatten it.
                captured.update(captured.pop(var_keyword, {}))
            # Copy before `orig` runs: DistributionLoss.__init__ pops num_pieces.
            captured = copy.deepcopy(captured)
            orig(self, *args, **kwargs)
            self._nf_init_kwargs = captured

        __init__._nf_wrapped = True
        cls.__init__ = __init__


# --- Tagged JSON encoding ---

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
        if all(isinstance(k, str) for k in value):
            return {k: encode_value(v, tensors, f"{path}.{k}") for k, v in value.items()}
        # JSON keys are strings; categorical vocabularies hold ints and bools.
        return {
            TAG: "mapping",
            "items": [
                [
                    encode_value(k, tensors, f"{path}.<key>"),
                    encode_value(v, tensors, f"{path}.{k}"),
                ]
                for k, v in value.items()
            ],
        }
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
    if isinstance(value, _SerializableLoss) or type(value) in _USER_LOSSES.values():
        args = getattr(value, "_nf_init_kwargs", None)
        if args is None:
            # Unpickling skips __init__, so a legacy loss recorded nothing.
            args, _ = _recover_loss_args(value, path)
        # `update_quantile` can replace these after __init__.
        state = {
            name: getattr(value, name)
            for name in _LOSS_DERIVED_STATE
            if hasattr(value, name)
        }
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
    if value.dtype.kind == "O" and _looks_like_timestamps(values):
        # A tz-aware `ds` arrives as an object array of pandas Timestamps.
        import pandas as pd

        return _as_pandas_index(pd.DatetimeIndex(value))
    if not all(v is None or isinstance(v, str) for v in values):
        raise SerializationError(
            f"{path}: object arrays can only be encoded when every element is a "
            f"string or None."
        )
    return {TAG: "index", "kind": "numpy", "dtype": str(value.dtype), "values": values}


# Loss attributes that __init__ derives and `update_quantile` can later replace.
_LOSS_DERIVED_STATE = ("output_names", "quantiles")

# A mismatch here means the recovery was lossy.
_LOSS_INVARIANTS = ("outputsize_multiplier", "is_distribution_output")


def _recover_loss_args(loss, path):
    """Rebuild a legacy (unpickled) loss's init kwargs, verified by trial rebuild."""
    cls = type(loss)
    parameters = list(inspect.signature(cls.__init__).parameters.items())[1:]
    args: Dict[str, Any] = {}
    for name, parameter in parameters:
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if hasattr(loss, name):
            attribute = getattr(loss, name)
            # Parameters are state __init__ built; hand them back as plain data.
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

    args.update(_recover_partial_kwargs(loss, parameters, args))

    state = {
        name: getattr(loss, name)
        for name in _LOSS_DERIVED_STATE
        if hasattr(loss, name)
    }
    _verify_recovered_loss(loss, cls, args, state, path)
    return args, state


def _recover_partial_kwargs(loss, parameters, recovered):
    """Read back kwargs `__init__` kept only inside a `functools.partial`.

    `DistributionLoss` pops `num_pieces` and `rho` out of distribution_kwargs and
    binds them to `domain_map` or `scale_decouple`.
    """
    named = {name for name, _ in parameters}
    extra = {}
    for attr in ("domain_map", "scale_decouple"):
        keywords = getattr(getattr(loss, attr, None), "keywords", None)
        if not isinstance(keywords, dict):
            continue
        for key, value in keywords.items():
            if key not in named and key not in recovered:
                extra[key] = value
    return extra


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
    """Find this object's key in `_type2scaler`.

    robust/mad and robust-iqr share a class, so probes compare full state.
    """
    from neuralforecast.core import _type2scaler

    def config(scaler):
        return {k: v for k, v in vars(scaler).items() if k != "stats_"}

    target = config(value)
    for name, factory in _type2scaler.items():
        probe = factory()
        if type(probe) is type(value) and config(probe) == target:
            return name
    return None


def _looks_like_timestamps(values):
    import pandas as pd

    return bool(values) and all(isinstance(v, pd.Timestamp) for v in values)


def _encode_scaler(value, tensors, path):
    """coreforecast scalers hold their fitted state in `stats_`."""
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


def _encode_datetime64(values, kind="numpy", tz=None):
    """Encode datetimes as int64. Tz-aware values are stored in UTC plus a zone."""
    values = np.asarray(values)
    unit = np.datetime_data(values.dtype)[0]
    encoded = {
        TAG: "datetime64",
        "kind": kind,
        "unit": unit,
        "values": values.astype("int64").tolist(),
    }
    if tz is not None:
        encoded["tz"] = tz
    return encoded


def _as_utc_datetimes(value):
    """Split a datetime container into (naive UTC array, zone or None).

    `to_numpy()` on a tz-aware object gives Timestamps numpy cannot read.
    """
    tz = getattr(getattr(value, "dtype", None), "tz", None)
    if tz is None:
        return np.asarray(value), None
    # `.dt` on a Series, direct on an Index.
    accessor = value.dt if hasattr(value, "dt") else value
    naive = accessor.tz_convert("UTC")
    naive = naive.dt if hasattr(naive, "dt") else naive
    return naive.tz_localize(None).to_numpy(), str(tz)


def _is_datetime(value):
    return getattr(getattr(value, "dtype", None), "kind", None) == "M"


def _as_pandas_frame(value, tensors, path):
    import pandas as pd

    if isinstance(value, pd.Series):
        if _is_datetime(value):
            values, tz = _as_utc_datetimes(value)
            data = _encode_datetime64(values, kind="pandas", tz=tz)
        else:
            data = encode_value(value.to_numpy(), tensors, f"{path}.values")
        return {
            TAG: "series",
            "kind": "pandas",
            "name": value.name,
            "dtype": str(value.dtype),
            "data": data,
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
    if _is_datetime(value):
        values, tz = _as_utc_datetimes(value)
        encoded = _encode_datetime64(values, kind="pandas", tz=tz)
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
        tz = getattr(value.dtype, "time_zone", None)
        values = value.dt.replace_time_zone(None).to_numpy() if tz else value.to_numpy()
        return _encode_datetime64(values, kind="polars", tz=tz)
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
    """Inverse of `encode_value`; an unknown tag is an error, not a passthrough."""
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
            if isinstance(attribute, torch.Tensor):
                # Shapes must match before the state dict is applied.
                attribute = torch.nn.Parameter(attribute, requires_grad=False)
            setattr(loss, name, attribute)
        return loss
    if kind == "torch_cls":
        if value["kind"] not in ("optimizer", "lr_scheduler"):
            raise SerializationError(
                f"torch_cls may only name an optimizer or scheduler, not "
                f"{value['kind']!r}."
            )
        return _resolve(value["name"], value["kind"])
    if kind == "tensor":
        return _decode_tensor(value, tensors)
    if kind == "datetime64":
        return _decode_datetime64(value)
    if kind == "index":
        return _decode_index(value)
    if kind == "mapping":
        return {
            decode_value(k, tensors): decode_value(v, tensors)
            for k, v in value["items"]
        }
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
    if isinstance(data, pd.DatetimeIndex):
        return pd.Series(data, name=value.get("name"))
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
    tz = value.get("tz")
    if value["kind"] == "pandas":
        import pandas as pd

        index = pd.DatetimeIndex(array, name=value.get("name"))
        return index.tz_localize("UTC").tz_convert(tz) if tz else index
    if value["kind"] == "polars":
        polars = _polars_or_none()
        if polars is None:
            raise SerializationError("Artifact holds polars data but polars is not installed.")
        series = polars.Series(value.get("name") or "", array)
        return series.dt.replace_time_zone("UTC").dt.convert_time_zone(tz) if tz else series
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
    mapping: Dict[str, Any],
    inline: bool = False,
    droppable: Optional[Any] = None,
):
    """Encode a dict into (JSON-safe dict, tensors[, dropped keys]).

    Args:
        mapping (dict): Values to encode.
        inline (bool): Write arrays into the JSON instead of a tensor sidecar.
            Used for `configuration.json`, which ships without one.
        droppable: Keys whose values may be skipped when they need pickle,
            reported instead of raising. Runtime plumbing such as
            `worker_init_fn` lives in hparams and cannot be removed by the
            caller, but a model argument must never be dropped silently, so the
            caller decides which keys qualify.
    """
    tensors: Dict[str, torch.Tensor] = {}
    target = None if inline else tensors
    encoded: Dict[str, Any] = {}
    dropped = []
    for key, value in mapping.items():
        try:
            encoded[key] = encode_value(value, target, key)
        except SerializationError:
            if droppable is None or key not in droppable:
                raise
            if isinstance(value, dict):
                # Keep the entries that do encode: only the callables inside
                # `dataloader_kwargs` and friends are a problem.
                kept, inner = {}, []
                for inner_key, inner_value in value.items():
                    try:
                        kept[inner_key] = encode_value(
                            inner_value, target, f"{key}.{inner_key}"
                        )
                    except SerializationError:
                        inner.append(f"{key}.{inner_key}")
                encoded[key] = kept
                dropped.extend(inner)
            else:
                dropped.append(key)
    if droppable is None:
        return encoded, tensors
    return encoded, tensors, dropped


def decode_mapping(
    mapping: Dict[str, Any], tensors: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, Any]:
    """Inverse of `encode_mapping`."""
    return {k: decode_value(v, tensors) for k, v in mapping.items()}


# --- Tensor payload ---


def _remove_duplicate_names(state_dict):
    # Private upstream API; test_serialization.py::test_upstream_canary guards it.
    from safetensors.torch import _remove_duplicate_names as upstream

    return upstream(state_dict)


def save_tensors(
    state_dict: Dict[str, torch.Tensor], metadata: Dict[str, str]
) -> bytes:
    """Serialize a state dict to safetensors bytes.

    `safetensors.save` refuses tied tensors, so duplicates are dropped and the
    tie recorded; constructing the model before applying weights re-creates them.
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
    """Whether `data` starts with a safetensors header.

    How the reader picks a format; never a field inside the artifact.
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
    """Apply `tensors`, allowing only recorded ties to be missing.

    A truncated payload must not load as a partially-random model.
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


# --- Datasets ---

# Explicit, so a new attribute is a visible decision, not a silent gap.
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


def decode_dataset(
    meta: Dict[str, Any],
    tensors: Dict[str, torch.Tensor],
    trust_remote: bool = False,
):
    """Inverse of `encode_dataset`."""
    cls = _resolve(meta["dataset_class"], "dataset")
    extra = meta.get("extra", {})
    unexpected = set(extra) - set(_DATASET_EXTRA)
    if unexpected:
        raise SerializationError(
            f"Refusing to restore unexpected dataset attributes: "
            f"{', '.join(sorted(unexpected))}."
        )
    fields = decode_mapping(meta["fields"], tensors)
    # `files_ds` is read with `pd.read_parquet`, which resolves fsspec URLs, so
    # an artifact could otherwise point a later predict() anywhere.
    for parquet_path in fields.get("files_ds") or []:
        ensure_trusted_path(parquet_path, trust_remote)
    dataset = cls(**fields)
    for field, value in decode_mapping(extra, tensors).items():
        setattr(dataset, field, value)
    return dataset
