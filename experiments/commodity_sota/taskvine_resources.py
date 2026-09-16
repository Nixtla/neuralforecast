"""Portable resource identity and VRAM admission; CPU/RAM belong to TaskVine."""

import hashlib
import json
import math
from pathlib import Path
import time

GIB = 2**30
# These parameters change optimization, logging or duration, not model shape.
NON_STRUCTURAL = {
    "learning_rate",
    "random_seed",
    "max_steps",
    "val_check_steps",
    "early_stop_patience_steps",
    "enable_progress_bar",
    "logger",
    "callbacks",
    "devices",
    "accelerator",
    "num_lr_decays",
    "step_size",
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, default=str, indent=2))
    temporary.replace(path)


def structural_value(value):
    if isinstance(value, dict):
        return {str(k): structural_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [structural_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Loss objects have meaningful constructor state; avoid address-based repr.
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "state": {
            k: structural_value(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        },
    }


def category_for(args):
    candidate, config, fold = args[1], args[3], args[4]
    structure = {
        "model": candidate["name"],
        "protocol": candidate.get("protocol"),
        "config": {
            k: structural_value(v) for k, v in config.items() if k not in NON_STRUCTURAL
        },
        "length": 2 ** math.ceil(math.log2(max(1, fold.train_end))),
        "horizon": fold.valid_end - fold.train_end,
    }
    digest = hashlib.sha256(json.dumps(structure, sort_keys=True).encode()).hexdigest()
    return f"{candidate['name']}-{digest[:20]}"


def portable(value):
    if isinstance(value, (str, Path)):
        return not str(value).startswith("/")
    if isinstance(value, dict):
        return all(portable(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(portable(v) for v in value)
    return True


def gpu_admission(snapshot, peers, estimate, exclusive=False, slots=2):
    """Return reservation bytes, or None; stale telemetry never admits work.

    Snapshot usage already includes allocated memory. Add only unused portions
    of reservations when PID accounting exists; otherwise remain conservative.
    """
    if not snapshot or time.time() - snapshot.get("timestamp", 0) > 15:
        return None
    if len(peers) >= slots or any(p["exclusive"] for p in peers):
        return None
    exclusive = exclusive or estimate is None
    if exclusive:
        if peers or snapshot["used"] > 512 * 2**20:
            return None
        return snapshot["total"]
    need = estimate * 1.25
    used = snapshot["used"]
    processes = snapshot.get("processes")
    for peer in peers:
        allocated = (processes or {}).get(str(peer.get("pid")), 0)
        used += max(0, peer["reservation"] - allocated)
    return need if used + need + 2 * GIB <= snapshot["total"] else None
