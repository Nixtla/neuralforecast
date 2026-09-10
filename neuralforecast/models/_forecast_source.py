"""Import explicitly supplied, pinned official research checkouts.

No downloading, vendoring, or fallback implementation. These are trusted Python
sources, not a sandbox. Absolute ``layers``/``utils`` imports are scoped so that
SeesawNet and Dualformer can coexist in one process.
"""

import hashlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import types

import torch

from ._research_source import _IMPORT_LOCK

SOURCES = {
    "SeesawNet": ("model/SeesawNet.py", "f0b49c8de2dceb866ffc8eab40ded169e4b2dd5d"),
    "Dualformer": ("models/dualformer.py", "ebd4ccf8bc5634f0c965d0b8d5797d1b926daa19"),
    "SearchCast": ("optuna_ridge.py", "9a12b22525d787c0e0f919b2bd5b26fec5d64d03"),
}


def forecast_source(source_dir, kind):
    """Load a clean pinned checkout, retaining only a private entrypoint module."""
    if source_dir is None:
        raise ValueError(f"{kind} requires source_dir; see docs/short_horizon_models.md.")
    root = Path(source_dir).expanduser().resolve()
    relative, revision = SOURCES[kind]
    filename = root / relative
    if not filename.is_file():
        raise FileNotFoundError(f"Missing official {kind} source: {filename}")
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"{kind} source_dir must be a readable git checkout.") from exc
    if head != revision or dirty:
        raise ValueError(f"{kind} requires a clean checkout at {revision}.")
    name = "_nf_forecast_" + hashlib.sha256(str(filename).encode()).hexdigest()[:16]
    # ponytail: serialized imports; use isolated workers for concurrently loaded
    # untrusted/third-party packages. Model forward passes do not hold this lock.
    with _IMPORT_LOCK:
        if name in sys.modules:
            return sys.modules[name]
        prefixes = ("layers", "utils") if kind != "SearchCast" else ()
        scoped = lambda key: any(key == p or key.startswith(p + ".") for p in prefixes)
        saved = {k: v for k, v in list(sys.modules.items()) if scoped(k)}
        env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        optuna_logging = None
        if kind == "SearchCast":
            try:
                import optuna
            except ImportError as exc:
                raise ImportError("SearchCast requires optuna and its upstream requirements.") from exc
            optuna_logging = optuna.logging.get_verbosity()
        try:
            for key in saved:
                del sys.modules[key]
            for prefix in prefixes:
                module = types.ModuleType(prefix)
                module.__path__ = [str(root / prefix)]
                module.__package__ = prefix
                module.__spec__ = importlib.machinery.ModuleSpec(prefix, loader=None, is_package=True)
                sys.modules[prefix] = module
            spec = importlib.util.spec_from_file_location(name, filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
        finally:
            for key in list(sys.modules):
                if scoped(key):
                    del sys.modules[key]
            sys.modules.update(saved)
            for key, value in env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
            if optuna_logging is not None:
                optuna.logging.set_verbosity(optuna_logging)
        return module
