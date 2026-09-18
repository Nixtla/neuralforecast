"""Load the reviewed text-model files without global layers/utils collisions.

Only import paths are rewritten; upstream model calculations are unchanged.
An explicit source checkout is trusted Python code, not a security sandbox.
No source, weights, or input data are downloaded or uploaded by this loader.
"""

import ast
import hashlib
import importlib.machinery
from pathlib import Path
import sys
import types

from ._research_source import _IMPORT_LOCK

# Dependency order and Git blob IDs at the revisions in docs/text_models.md.
_FILES = {
    "SpecTF": {
        "utils/masking.py": "b97bfb1cbd14dba1c39b96319c39940e1aa5e6b6",
        "layers/Embed.py": "ca890c9c82d0e165ff3711e2b5283e92b75abd2a",
        "layers/FreqAttention_Family.py": "a2e66ced75592bbd46864d2f7f2bde71d781fd98",
        "models/SpecTF.py": "1256c3af7a3a3417f94d7a7d7d7379b0b21d093f",
    },
    "TGForecaster": {
        "layers/RevIN.py": "890017090077019d8fa15d19c79cf4d31d63869f",
        "layers/PatchTST_layers.py": "11b5bd68a94c25cfcacf2e1eb0b469ae24342408",
        "layers/TGTSF_torch.py": "3219cc59902f626c4359e215ed18d2ec11c6890a",
        "models/TGTSF_torch.py": "de79c4c591d12251e7534369215c648feb01dd89",
    },
}


def text_source(source_dir: str, kind: str):
    """Load four hash-checked files under one private namespace per checkout."""
    if not source_dir:
        raise ValueError("source_dir is required; see docs/text_models.md.")
    root = Path(source_dir).expanduser().resolve()
    sources = {}
    for relative, expected in _FILES[kind].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Source file escaped its checkout: {relative}")
        content = path.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if digest != expected:
            raise ValueError(f"{kind} source differs from reviewed blob: {relative}")
        sources[relative] = content
    prefix = "_nf_text_" + hashlib.sha256(f"{root}:{kind}".encode()).hexdigest()[:20]
    entry = prefix + "." + next(reversed(sources)).removesuffix(".py").replace("/", ".")
    with _IMPORT_LOCK:
        if entry in sys.modules:
            return sys.modules[entry]
        try:
            for package in (prefix, prefix + ".layers", prefix + ".utils", prefix + ".models"):
                module = types.ModuleType(package)
                module.__package__ = package
                module.__path__ = []  # Never resolve unreviewed checkout modules.
                module.__spec__ = importlib.machinery.ModuleSpec(package, loader=None, is_package=True)
                sys.modules[package] = module
                if "." in package:
                    parent, name = package.rsplit(".", 1)
                    setattr(sys.modules[parent], name, module)
            for relative, content in sources.items():
                filename = str(root / relative)
                tree = ast.parse(content, filename=filename)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        if node.module.split(".")[0] in {"layers", "utils", "models"}:
                            node.module = prefix + "." + node.module
                    elif isinstance(node, ast.Import):
                        if any(a.name.split(".")[0] in {"layers", "utils", "models"} for a in node.names):
                            raise ImportError("Unexpected local import; review the source revision.")
                qualified = prefix + "." + relative.removesuffix(".py").replace("/", ".")
                module = types.ModuleType(qualified)
                module.__file__ = filename
                module.__package__, name = qualified.rsplit(".", 1)
                module.__spec__ = importlib.machinery.ModuleSpec(qualified, loader=None, origin=filename)
                sys.modules[qualified] = module
                setattr(sys.modules[module.__package__], name, module)
                exec(compile(tree, filename, "exec"), module.__dict__)
        except Exception:
            for name in list(sys.modules):
                if name == prefix or name.startswith(prefix + "."):
                    del sys.modules[name]
            raise
        return sys.modules[entry]
