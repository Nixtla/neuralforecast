"""Delete an experiment group and its explicitly named local outputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time


def _delete_run(run):
    for attempt in range(5):
        try:
            run.delete(delete_artifacts=True)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--also-delete", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allow-completed", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    outputs = [args.output.resolve(), *(path.resolve() for path in args.also_delete)]
    allowed = root / "results"
    if any(allowed not in path.parents or path == allowed for path in outputs):
        raise ValueError("Every deletion target must be a named child of results/")
    config = json.loads((outputs[0] / "run_config.json").read_text())
    if (
        config.get("status") in {"completed", "no_models_above_naive"}
        and not args.allow_completed
    ):
        raise ValueError("Refusing to reset a completed experiment")
    tracking = config["wandb"]
    project = f"{tracking['entity']}/{tracking['project']}"

    import wandb

    api = wandb.Api(timeout=120)
    runs = list(
        api.runs(project, filters={"group": tracking["group"]}, per_page=200)
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "applied": False,
        "project": project,
        "group": tracking["group"],
        "wandb_runs_to_delete": len(runs),
        "wandb_run_ids_to_delete": sorted(run.id for run in runs),
        "local_paths_to_delete": [str(path) for path in outputs if path.exists()],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if not isinstance(v, list)}))
    if not args.apply:
        return

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(_delete_run, run) for run in runs]
        for done, future in enumerate(as_completed(futures), start=1):
            future.result()
            if done % 100 == 0 or done == len(futures):
                print(f"deleted_wandb_runs={done}/{len(futures)}", flush=True)
    for path in outputs:
        if path.exists():
            shutil.rmtree(path)
    manifest["applied"] = True
    manifest["applied_at"] = datetime.now(timezone.utc).isoformat()
    args.manifest.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
