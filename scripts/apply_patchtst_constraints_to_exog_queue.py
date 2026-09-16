"""Sync PatchTST constraints into the reserved exog-source before wti-exog starts.

The running queue (results/exogenous-queue/start.py) verified its reservation
hashes once at execute() entry and runs gasoline-exog from the frozen copy at
results/exogenous-queue/exog-source. Patching ROOT alone therefore never
reaches the reserved wti-exog step.

This script copies the two patched files into the frozen copy, but ONLY after
gasoline-exog has completed, so the running phase-1 trials are not
contaminated. It never touches reservation.json or state.json.

Usage:
    python scripts/apply_patchtst_constraints_to_exog_queue.py --check
    python scripts/apply_patchtst_constraints_to_exog_queue.py --apply  # after gasoline-exog completes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "results/exogenous-queue"
EXOG_SOURCE = QUEUE / "exog-source"
GAS_OUTPUT = ROOT / "results/gasoline-exog"

# ROOT file -> frozen copy, relative to each base.
PATCHED_FILES = [
    "neuralforecast/auto.py",
    "neuralforecast/benchmark_numerics.py",
]

MARKERS = {
    "neuralforecast/auto.py": [
        '"patch_len": tune.choice([4, 8])',
        '"learning_rate": tune.loguniform(1e-4, 1e-3)',
        '"stride": tune.choice([2, 4])',
    ],
    "neuralforecast/benchmark_numerics.py": ['"PatchTST"'],
}

TERMINAL_STATUSES = {"completed", "no_models_above_naive"}


def _status(output: Path) -> str | None:
    path = output / "run_config.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text()).get("status")
    except (json.JSONDecodeError, OSError):
        return None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check() -> int:
    ok = True
    for rel in PATCHED_FILES:
        src = ROOT / rel
        dst = EXOG_SOURCE / rel
        print(f"--- {rel} ---")
        print(f"ROOT   exists={src.is_file()}")
        for marker in MARKERS[rel]:
            found = marker in src.read_text() if src.is_file() else False
            print(f"  ROOT marker {marker!r}: {'OK' if found else 'MISSING'}")
            ok = ok and found
        if dst.is_file():
            print(f"frozen synced={_sha(src) == _sha(dst)}")
        else:
            print("frozen MISSING")
            ok = False
    gas = _status(GAS_OUTPUT)
    print(f"gasoline-exog status: {gas}")
    print(f"wti-exog output exists: {(ROOT / 'results/wti-exog').exists()}")
    print(f"wti-exog smoke exists: {(ROOT / 'results/wti-exog-dynamic-smoke').exists()}")
    return 0 if ok else 1


def apply(force: bool = False) -> int:
    gas = _status(GAS_OUTPUT)
    if not force and gas not in TERMINAL_STATUSES:
        print(
            f"REFUSING: gasoline-exog status is {gas!r}; "
            f"expected one of {sorted(TERMINAL_STATUSES)}. "
            "Re-run with --force only if you intend to contaminate a live run.",
            file=sys.stderr,
        )
        return 2
    if (ROOT / "results/wti-exog").exists():
        print(
            "REFUSING: results/wti-exog already exists; the reserved wti-exog "
            "step may have started. Inspect state before forcing.",
            file=sys.stderr,
        )
        if not force:
            return 2
    for rel in PATCHED_FILES:
        src = ROOT / rel
        dst = EXOG_SOURCE / rel
        text = src.read_text()
        for marker in MARKERS[rel]:
            if marker not in text:
                print(f"ROOT {rel} missing marker {marker!r}", file=sys.stderr)
                return 1
        shutil.copy2(src, dst)
        # Drop stale bytecode so pool workers reimport the patched module.
        for cache in [src.parent / "__pycache__", dst.parent / "__pycache__"]:
            if cache.is_dir():
                for pyc in cache.glob(f"{Path(rel).stem}.*.pyc"):
                    pyc.unlink(missing_ok=True)
        print(f"synced {rel} sha={_sha(dst)[:12]}")
    sidecar = QUEUE / "wti_patch.json"
    sidecar.write_text(
        json.dumps(
            {
                "patch": "patchtst-low-lr-fine-patch",
                "files": {
                    rel: _sha(EXOG_SOURCE / rel) for rel in PATCHED_FILES
                },
                "gasoline_exog_status_at_apply": gas,
                "note": "Does not modify reservation.json; _verify() ran once "
                "at queue start. Applies to the not-yet-started wti-exog step.",
            },
            indent=2,
        )
    )
    print(f"wrote {sidecar}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    raise SystemExit(apply(args.force) if args.apply else check())


if __name__ == "__main__":
    main()
