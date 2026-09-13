"""Search spaces for inference-only forecasting adapters.

These spaces are intentionally separate from ``BaseAuto`` because pretrained
adapters do not train weights or populate Lightning validation metrics in
``fit``. Evaluate candidates with chronological validation outside ``BaseAuto``.
"""

from math import ceil

from ray import tune

from .common._base_auto import BaseAuto

__all__ = ["INFERENCE_TUNING_MODELS", "get_inference_tuning_config"]

INFERENCE_TUNING_MODELS = (
    "Chronos2",
    "Moirai",
    "MoiraiMoE",
    "TimesFM",
    "Toto",
    "Moirai2",
    "ChronosX",
    "BaguanTS",
    "RAG4CTS",
    "TimesFM3",
    "Aurora",
    "ChatTime",
    "TabPFNTS",
)

_REQUIRED_FIXED = {
    "Moirai": ("backend_python",),
    "MoiraiMoE": ("backend_python",),
    "Moirai2": ("backend_python",),
    "ChronosX": (
        "input_size",
        "model_id",
        "backend_python",
        "hidden_dim",
        "num_layers",
    ),
    "BaguanTS": (
        "input_size",
        "source_dir",
        "config_path",
        "model_id",
        "backend_python",
        "futr_exog_list",
    ),
    "RAG4CTS": ("input_size", "source_dir", "futr_exog_list"),
    "Aurora": (
        "source_dir",
        "model_id",
        "tokenizer_path",
        "contexts",
        "stat_exog_list",
    ),
    "ChatTime": ("source_dir", "model_id", "contexts", "stat_exog_list"),
    "TabPFNTS": ("model_id", "futr_exog_list"),
}


def _round_up(value, multiple):
    return multiple * ceil(value / multiple)


def _input_sizes(h, multiple=16, maximum=None):
    if not isinstance(h, int) or isinstance(h, bool) or h < 1:
        raise ValueError("h must be a positive integer.")
    raw = (max(32, 2 * h), max(64, 4 * h), max(128, 8 * h))
    values = sorted({_round_up(value, multiple) for value in raw})
    if maximum is not None:
        values = [value for value in values if value <= maximum]
    if not values:
        raise ValueError("No valid context length remains for this horizon.")
    return values


def _context_only(h, fixed):
    return {"input_size": tune.choice(_input_sizes(h))}


def _moirai(h, fixed):
    return {
        "input_size": tune.choice(_input_sizes(h)),
        "patch_size": tune.choice([8, 16, 32]),
        "num_samples": tune.choice([20, 50, 100]),
    }


def _timesfm(h, fixed):
    return {
        "input_size": tune.choice(_input_sizes(h)),
        "xreg_ridge": tune.loguniform(1e-6, 1e1),
    }


def _toto(h, fixed):
    return {
        "input_size": tune.choice(_input_sizes(h)),
        "num_samples": tune.choice([20, 50, 100]),
    }


def _moirai2(h, fixed):
    return {
        "input_size": tune.choice(_input_sizes(h)),
        "patch_size": 16,
        "num_samples": 100,
    }


def _chronosx(h, fixed):
    return {"num_samples": tune.choice([20, 50, 100])}


def _baguants(h, fixed):
    input_size = fixed["input_size"]
    if input_size <= h:
        raise ValueError("BaguanTS fixed input_size must be greater than h.")
    values = {
        h + 1,
        max(h + 1, input_size // 4),
        max(h + 1, input_size // 2),
        max(h + 1, 3 * input_size // 4),
    }
    contexts = sorted(value for value in values if value <= input_size)
    return {
        "context_size": tune.choice(contexts),
        "neighbors": tune.choice([1, 3, 5, 10]),
        "num_samples": tune.choice([1, 3, 5]),
    }


def _rag4cts(h, fixed):
    input_size = fixed["input_size"]
    queries = [
        value
        for value in (8, 16, 32, 64)
        if input_size >= 2 * value + h
    ]
    if not queries:
        query_size = (input_size - h) // 2
        if query_size < 1:
            raise ValueError(
                "RAG4CTS fixed input_size must leave room for query and bank."
            )
        queries = [query_size]
    return {
        "query_size": tune.choice(queries),
        "neighbors": tune.choice([1, 3, 5, 10]),
        "retrieval_stride": tune.choice([1, 2, 4]),
    }


def _timesfm3(h, fixed):
    return {
        "input_size": tune.choice(_input_sizes(h, maximum=15360)),
        "use_symmetric_averaging": tune.choice([False, True]),
    }


def _aurora(h, fixed):
    return {
        "input_size": tune.choice(_input_sizes(h)),
        "inference_token_len": tune.choice([24, 48, 96]),
        "num_samples": tune.choice([20, 50, 100]),
    }


def _chattime(h, fixed):
    return {
        "input_size": tune.choice(_input_sizes(h)),
        "num_samples": tune.choice([5, 10, 20]),
    }


_BUILDERS = {
    "Chronos2": _context_only,
    "Moirai": _moirai,
    "MoiraiMoE": _moirai,
    "TimesFM": _timesfm,
    "Toto": _toto,
    "Moirai2": _moirai2,
    "ChronosX": _chronosx,
    "BaguanTS": _baguants,
    "RAG4CTS": _rag4cts,
    "TimesFM3": _timesfm3,
    "Aurora": _aurora,
    "ChatTime": _chattime,
    "TabPFNTS": _context_only,
}


def _name(model):
    return model if isinstance(model, str) else model.__name__


def _validate_fixed(name, fixed):
    missing = [
        key
        for key in _REQUIRED_FIXED.get(name, ())
        if key not in fixed or fixed[key] is None
    ]
    if missing:
        raise ValueError(f"{name} tuning requires fixed values for {missing}.")
    if fixed.get("max_steps", 0) != 0:
        raise ValueError("Inference tuning requires max_steps=0.")
    if fixed.get("early_stop_patience_steps", -1) > 0:
        raise ValueError("Inference tuning cannot use early stopping.")
    if name == "Moirai2" and (
        fixed.get("patch_size", 16) != 16 or fixed.get("num_samples", 100) != 100
    ):
        raise ValueError("Moirai2 patch_size=16 and num_samples=100 are fixed.")
    if name == "ChronosX" and not (
        fixed.get("hist_exog_list") or fixed.get("futr_exog_list")
    ):
        raise ValueError("ChronosX requires at least one fixed exogenous list.")
    if name in {"BaguanTS", "RAG4CTS", "TabPFNTS"} and not fixed.get(
        "futr_exog_list"
    ):
        raise ValueError(f"{name} requires nonempty fixed futr_exog_list.")
    if name in {"Aurora", "ChatTime"} and len(
        fixed.get("stat_exog_list") or []
    ) != 1:
        raise ValueError(f"{name} requires one fixed context_id stat_exog column.")


def get_inference_tuning_config(model, h, fixed=None, backend="ray"):
    """Return a model-aware inference search space.

    ``fixed`` holds checkpoint/source/schema values and may also pin any tunable
    key. The returned object is a Ray config dict or an Optuna config callable.
    It is a search-space definition, not a ``BaseAuto`` model.
    """
    name = _name(model)
    if name not in _BUILDERS:
        raise ValueError(
            f"Unknown inference model {name!r}. Expected one of "
            f"{INFERENCE_TUNING_MODELS}."
        )
    fixed = dict(fixed or {})
    _validate_fixed(name, fixed)
    config = {**_BUILDERS[name](h, fixed), **fixed, "max_steps": 0}
    if backend == "ray":
        return config
    if backend == "optuna":
        return BaseAuto._ray_config_to_optuna(config)
    raise ValueError("backend must be 'ray' or 'optuna'.")
