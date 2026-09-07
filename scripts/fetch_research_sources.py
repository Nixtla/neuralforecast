"""Explicitly fetch reviewed research checkouts. Never overwrite existing paths.

Run: python scripts/fetch_research_sources.py DIRECTORY [DAG KITE ...]
No model weights or dataset files are downloaded. Each source retains its own
license; a source checkout is not a grant to redistribute or commercially use it.
"""

from pathlib import Path
import subprocess
import sys

SOURCES = {
    "DAG": ("decisionintelligence/DAG", "0758990e2c73bb54138ea3e7b11a35cbc5476bcc",
            ["ts_benchmark/baselines/dag", "ts_benchmark/baselines/time_series_library/utils"]),
    "KITE": ("decisionintelligence/KITE", "3140ee824cbd80c5ec7fdf2b54666210519d0b39",
             ["ts_benchmark/baselines/kite"]),
    "GLAFF": ("ForestsKing/GLAFF", "4dedf10e0028b519780645ef5824810f4b1bdb55", ["plugin"]),
    "APT": ("blisky-li/APT", "98a4c9c017666207b02029f842eff92818f0eab8",
            ["baselines/Normalization/normalization", "baselines/Normalization/loss"]),
    "RAG4CTS": ("RAG4CTS-Project/RAG4CTS", "28143b3c3b13ef6be5eba3d28e051fc7bd662cf3", ["RAG4CTS"]),
    "BaguanTS": ("jxgogo/Baguan-TS", "8b55d93eb52c2f2a9d393dfebdd33fc5a7466b8e", ["src", "configs"]),
}


def fetch(destination, names):
    destination = Path(destination).expanduser().absolute()
    for name in names:
        if name not in SOURCES:
            raise ValueError(f"Unknown source {name}; choose from {', '.join(SOURCES)}")
        if (destination / name).exists():
            raise FileExistsError(f"Refusing to overwrite {destination / name}. Use a new directory.")
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        repository, revision, paths = SOURCES[name]
        target = destination / name
        target.mkdir()
        def git(*args):
            subprocess.run(["git", "-C", str(target), *args], check=True, timeout=180)
        git("init")
        git("config", "core.autocrlf", "false")
        git("remote", "add", "origin", f"https://github.com/{repository}.git")
        git("config", "remote.origin.promisor", "true")
        git("config", "remote.origin.partialclonefilter", "blob:none")
        git("sparse-checkout", "init", "--cone")
        git("sparse-checkout", "set", *paths)
        git("fetch", "--filter=blob:none", "--depth=1", "origin", revision)
        git("checkout", "--detach", "FETCH_HEAD")
        print(f"{name}: {revision} -> {target}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    fetch(sys.argv[1], sys.argv[2:] or list(SOURCES))
