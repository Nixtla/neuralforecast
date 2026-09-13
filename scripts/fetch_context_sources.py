"""Fetch pinned, public context-model source files, never datasets or weights.

Usage: python scripts/fetch_context_sources.py DIRECTORY [VoT GPT4MTS ...]
Existing model directories are refused, not updated or deleted.
"""

import argparse
import ast
from pathlib import Path
import subprocess


def main():
    manifest_file = Path(__file__).resolve().parents[1] / "neuralforecast/models/_context_source.py"
    tree = ast.parse(manifest_file.read_text(encoding="utf-8"))
    sources = next(ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "SOURCES")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("models", nargs="*")
    args = parser.parse_args()
    selected = args.models or list(sources)
    if len(selected) != len(set(selected)) or any(name not in sources for name in selected):
        parser.error("Select unique model names from: " + ", ".join(sources))
    root = args.directory.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any((root / name).exists() for name in selected):
        raise FileExistsError("An output model directory exists; choose a new directory or omit that model.")
    for name in selected:
        spec = sources[name]
        target = root / name
        subprocess.run(["git", "init", "--quiet", str(target)], check=True)
        def git(*arguments):
            subprocess.run(["git", "-C", str(target), *arguments], check=True)
        git("remote", "add", "origin", "https://github.com/" + spec["repository"] + ".git")
        git("fetch", "--depth=1", "--filter=blob:none", "origin", spec["revision"])
        git("sparse-checkout", "init", "--no-cone")
        git("sparse-checkout", "set", "--no-cone", *["/" + p for p in spec["files"]],
            "/LICENSE*", "/NOTICE*", "/README.md")
        git("checkout", "--detach", "FETCH_HEAD")
        revision = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if revision != spec["revision"]:
            raise RuntimeError("Checkout revision mismatch for " + name)
        print(name, revision, target)


if __name__ == "__main__":
    main()
