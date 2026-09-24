"""Rewrite a legacy pickle-based model directory in the safe v2 format.

    python -m neuralforecast.migrate <src> [--dst DIR]

Reading the source **executes any code it contains**, so run it only against
artifacts you trust. Remote sources need an explicit flag; migrating one you do
not trust would reintroduce the bug this format change closes.
"""

import argparse
import os
import sys
from typing import List, Optional

import fsspec

from neuralforecast._serialization import ensure_trusted_path

__all__ = ["migrate"]


def migrate(
    src: str,
    dst: Optional[str] = None,
    trust_source: bool = False,
    overwrite: bool = False,
    verbose: bool = True,
    model: Optional[str] = None,
) -> str:
    """Rewrite the legacy artifact at `src` in the v2 format.

    Args:
        src (str): Directory holding the legacy artifacts, or a single `.ckpt`.
        dst (Optional[str]): Where to write the migrated directory. Defaults to
            `<src>_v2`; the source is never modified.
        trust_source (bool): Permit a non-local `src`. Defaults to False.
        overwrite (bool): Permit writing into a non-empty `dst`.
        verbose (bool): Print a summary.
        model (Optional[str]): Model class for a single checkpoint, when the
            filename does not name it.

    Returns:
        str: The path that was written.
    """
    from neuralforecast.core import NeuralForecast

    src = src.rstrip("/")
    dst = (dst or _default_destination(src)).rstrip("/")
    if _same_location(src, dst):
        raise ValueError(
            f"Refusing to migrate {src!r} onto {dst!r}: they resolve to the same "
            f"location. The source is kept intact so a failed migration cannot "
            f"lose the original."
        )

    ensure_trusted_path(src, trust_source)
    fs, _, _ = fsspec.get_fs_token_paths(src)
    if not fs.exists(src):
        raise FileNotFoundError(src)
    if fs.isfile(src):
        return _migrate_checkpoint(src, dst, trust_source, model, verbose)

    files = _listdir(src)
    if "configuration.json" in files:
        _say(verbose, f"{src} is already in the v2 format; nothing to do.")
        return src
    if "configuration.pkl" not in files:
        raise ValueError(f"{src} does not look like a saved NeuralForecast directory.")

    _say(
        verbose,
        f"Reading {src} with pickle. This executes any code contained in those "
        f"files -- only migrate artifacts you trust.",
    )
    forecaster = NeuralForecast.load(src, allow_pickle=True, trust_remote=trust_source)

    has_dataset = getattr(forecaster, "dataset", None) is not None
    forecaster.save(dst, save_dataset=has_dataset, overwrite=overwrite)

    # Prove the result is readable without pickle rather than assuming it.
    NeuralForecast.load(dst, allow_pickle=False, trust_remote=True)

    if verbose:
        print(f"Migrated {src} -> {dst}")
        for fitted in forecaster.models:
            print(f"  model      {fitted.alias} ({type(fitted).__name__})")
        print(f"  dataset    {'included' if has_dataset else 'not stored'}")
        print(f"  files      {', '.join(sorted(_listdir(dst)))}")
        print("  verified   loads with allow_pickle=False")
    return dst


def _default_destination(src: str) -> str:
    stem, extension = os.path.splitext(src)
    return f"{stem}.safetensors" if extension == ".ckpt" else f"{src}_v2"


def _migrate_checkpoint(src, dst, trust_source, model, verbose):
    """Convert a single legacy `.ckpt`, which `NeuralForecast.load` cannot take."""
    from neuralforecast.core import MODEL_FILENAME_DICT

    name = model or os.path.basename(src).rsplit("_", 1)[0].removesuffix(".ckpt")
    cls = MODEL_FILENAME_DICT.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Cannot tell which model {src} holds. A checkpoint saved by "
            f"`NeuralForecast.save` is named `<Model>_<n>.ckpt`; otherwise pass "
            f"`--model <Model>`."
        )

    _say(
        verbose,
        f"Reading {src} with pickle. This executes any code contained in that "
        f"file -- only migrate artifacts you trust.",
    )
    loaded = cls.load(src, allow_pickle=True, trust_remote=trust_source)
    with fsspec.open(dst, "wb") as f:
        f.write(loaded.serialize())
    cls.load(dst, trust_remote=True)

    if verbose:
        print(f"Migrated {src} -> {dst}")
        print(f"  model      {cls.__name__}")
        print("  verified   loads with allow_pickle=False")
    return dst


def _same_location(src: str, dst: str) -> bool:
    """Whether dst resolves onto src, or sits inside it.

    A string compare misses `models` vs `./models` and eats the source.
    """
    if fsspec.utils.get_protocol(src) != fsspec.utils.get_protocol(dst):
        return False
    if fsspec.utils.get_protocol(src) in ("file", "local"):
        src, dst = os.path.realpath(src), os.path.realpath(dst)
    else:
        src, dst = src.rstrip("/"), dst.rstrip("/")
    return dst == src or dst.startswith(f"{src}{os.sep}")


def _listdir(path: str) -> List[str]:
    fs, _, _ = fsspec.get_fs_token_paths(path)
    if not fs.exists(path):
        raise FileNotFoundError(path)
    return [f.split("/")[-1] for f in fs.ls(path) if fs.isfile(f)]


def _say(verbose: bool, message: str) -> None:
    if verbose:
        print(message)


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="python -m neuralforecast.migrate",
        description=(
            "Rewrite a legacy NeuralForecast directory in the safetensors + JSON "
            "format. Reading the source executes code it contains."
        ),
    )
    parser.add_argument("src", help="Legacy directory, or a single .ckpt file.")
    parser.add_argument(
        "--dst", default=None, help="Where to write it. Defaults to <src>_v2."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit writing into a non-empty destination.",
    )
    parser.add_argument(
        "--i-trust-this-source",
        dest="trust_source",
        action="store_true",
        help=(
            "Permit a remote source. Reading it runs code chosen by whoever can "
            "write that location."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model class for a single checkpoint whose filename does not name it.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the summary.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        migrate(
            args.src,
            dst=args.dst,
            trust_source=args.trust_source,
            overwrite=args.overwrite,
            verbose=not args.quiet,
            model=args.model,
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
