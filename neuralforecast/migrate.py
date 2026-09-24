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
) -> str:
    """Rewrite the legacy directory at `src` in the v2 format.

    Args:
        src (str): Directory holding the legacy artifacts.
        dst (Optional[str]): Where to write the migrated directory. Defaults to
            `<src>_v2`; the source is never modified.
        trust_source (bool): Permit a non-local `src`. Defaults to False.
        overwrite (bool): Permit writing into a non-empty `dst`.
        verbose (bool): Print a summary.

    Returns:
        str: The directory that was written.
    """
    from neuralforecast.core import NeuralForecast

    src = src.rstrip("/")
    dst = (dst or f"{src}_v2").rstrip("/")
    if _same_location(src, dst):
        raise ValueError(
            f"Refusing to migrate {src!r} onto {dst!r}: they resolve to the same "
            f"location. The source is kept intact so a failed migration cannot "
            f"lose the original."
        )

    ensure_trusted_path(src, trust_source)
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
        for model in forecaster.models:
            print(f"  model      {model.alias} ({type(model).__name__})")
        print(f"  dataset    {'included' if has_dataset else 'not stored'}")
        print(f"  files      {', '.join(sorted(_listdir(dst)))}")
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
    parser.add_argument("src", help="Directory holding the legacy artifacts.")
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
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
