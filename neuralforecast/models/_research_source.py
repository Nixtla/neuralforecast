"""Load explicitly supplied official source checkouts; never download or vendor them.

The recorded hashes are Git blob IDs (not checkout revisions). Dependencies of
these entrypoints remain the caller's responsibility. Use the pinned checkouts
in docs/exogenous_models_batch2.md, not arbitrary untrusted Python directories.
"""

import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types


SOURCE_FILES = {
    "DAG": ("ts_benchmark/baselines/dag/models/dag_model.py", "ce4d5b0e67e67258fa9a5dfcca7c97d764896de6"),
    "KITE": ("ts_benchmark/baselines/kite/models/KITEModel.py", "d880c5ec5a75d94c8b1c2b1bfa0e821e83035126"),
    "GLAFF": ("plugin/Plugin/model.py", "1fa5e6f2773a431dd0207d82cdee8cc97d199d72"),
    "APT": ("baselines/Normalization/normalization/APT.py", "aecb0ff773d40dedaed3ef776df8ef0750869b60"),
    "APT_orthogonality": ("baselines/Normalization/loss/orthogonality_loss.py", "6fe1dab915023b72371119e0c94d885de66d0440"),
    "APT_balance": ("baselines/Normalization/loss/balanced_loss.py", "af4d295419fa7d7d334b007f81f746639a6b4022"),
    "APT_affine": ("baselines/Normalization/loss/L2affine.py", "c1643d6fc0e6680f8b3890496f83f53430471307"),
    "BaguanTS": ("BaguanTS.py", "d161399ccbd80075bda5e645a30f919e5ec69238"),
    "RAG4CTS": ("RAG4CTS/rag_pipeline.py", "febae8d698d74b0dfbef45449a29f3a5e55ce253"),
}
_IMPORT_LOCK = threading.RLock()


def source_module(source_dir, kind):
    """Import a reviewed entrypoint without executing unrelated package __init__s.

    Relative-import sources get a private namespace. DAG's absolute imports
    require ts_benchmark; an existing different checkout is rejected, not replaced.
    This is a compatibility loader, not a sandbox for untrusted source code.
    """
    if source_dir is None:
        raise ValueError(f"{kind} requires source_dir; see docs/exogenous_models_batch2.md.")
    root = Path(source_dir).expanduser().resolve()
    relative, expected = SOURCE_FILES[kind]
    filename = root / relative
    if not filename.is_file():
        raise FileNotFoundError(f"Missing official {kind} source: {filename}")
    contents = filename.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha1(b"blob " + str(len(contents)).encode() + b"\0" + contents).hexdigest()
    if digest != expected:
        raise ValueError(f"{kind} entrypoint differs from reviewed blob {expected}; use the documented checkout.")
    prefix = "" if kind == "DAG" else "_nf_source_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    parts = Path(relative).with_suffix("").parts
    qualified = ".".join(([prefix] if prefix else []) + list(parts))
    with _IMPORT_LOCK:
        packages = [(prefix, root)] if prefix else []
        for i in range(1, len(parts)):
            name = ".".join(([prefix] if prefix else []) + list(parts[:i]))
            packages.append((name, root.joinpath(*parts[:i])))
        if kind == "DAG":
            # DAG imports masks from this sibling; its __init__ imports all TSL models.
            for suffix in ("time_series_library", "time_series_library.utils"):
                name = "ts_benchmark.baselines." + suffix
                packages.append((name, root / "ts_benchmark" / "baselines" / suffix.replace(".", "/")))
        for name, path in packages:
            existing = sys.modules.get(name)
            if existing is not None:
                locations = [Path(p).resolve() for p in getattr(existing, "__path__", [])]
                if path.resolve() not in locations:
                    raise ImportError(f"Source namespace collision for {name}; use one checkout per process.")
            else:
                module = types.ModuleType(name)
                module.__path__ = [str(path)]
                module.__package__ = name
                module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
                sys.modules[name] = module
        existing = sys.modules.get(qualified)
        if existing is not None:
            if Path(existing.__file__).resolve() != filename.resolve():
                raise ImportError(f"Source module collision for {qualified}.")
            return existing
        spec = importlib.util.spec_from_file_location(qualified, filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(qualified, None)
            raise
        return module


def run_worker(backend_python, config, arrays, timeout):
    """Run the bundled worker in an explicitly selected environment, without pickle."""
    import numpy as np

    if backend_python is None:
        raise ValueError("backend_python must name the separate backend environment's Python.")
    executable = Path(backend_python).expanduser().absolute()  # Preserve venv symlinks.
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError(f"Not an executable Python path: {executable}")
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1:
        raise ValueError("backend_timeout must be a positive integer.")
    # ponytail: one process/model load per batch; use a persistent worker for throughput.
    with tempfile.TemporaryDirectory(prefix="nf-research-") as directory:
        directory = Path(directory)
        np.savez(directory / "inputs.npz", **arrays)
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        worker = Path(__file__).with_name("_research_worker.py")
        try:
            subprocess.run(
                [str(executable), str(worker), str(directory)],
                check=True, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Research backend failed:\n{exc.stderr[-3000:]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Research backend timed out; reduce batch size or raise backend_timeout.") from exc
        with np.load(directory / "outputs.npz", allow_pickle=False) as output:
            return output["prediction"].copy()
