"""Prepare and execute the reserved WTI recovery and exogenous benchmark queue."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import netrc
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "results/exogenous-queue"
LEGACY_SOURCE = QUEUE / "legacy-source"
EXOG_SOURCE = QUEUE / "exog-source"
END = "2026-09-06"
POLICY = {"max_steps": 500, "interval": 10, "patience": 5, "val_size": 16}
PROJECTS = {
    "uni-gasoline-exog": {
        "data": ROOT / "data/gasoline-exog.csv",
        "target": "Oil_EIA_NY_Harbor_Conventional_Gasoline_Spot_Price_Daily_USD_Per_Gallon",
        "start": "2013-08-11",
        "output": ROOT / "results/gasoline-exog",
    },
    "uni-wti-exog": {
        "data": ROOT / "data/wti-exog.csv",
        "target": "Oil_EIA_Cushing_WTI_Spot_Price_Daily_USD_Per_Barrel",
        "start": "2015-03-15",
        "output": ROOT / "results/wti-exog",
    },
}


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def _state(status, **fields):
    _save(
        QUEUE / "state.json",
        {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **fields,
        },
    )


def _environment(source):
    return dict(
        os.environ,
        PYTHONPATH=str(source),
        NF_COLAB_SESSIONS="2",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        WANDB_DISABLE_CODE="true",
        HF_HUB_DISABLE_PROGRESS_BARS="1",
        TOKENIZERS_PARALLELISM="false",
        MPLCONFIGDIR="/tmp/commodity-exog-mpl",
    )


def _command(source, project, settings, exogenous=False):
    command = [
        sys.executable,
        "-u",
        str(source / "experiments/commodity_sota/run.py"),
        "--scheduler",
        "dynamic-pool",
        "--data",
        str(settings["data"]),
        "--date-col",
        "ds",
        "--target",
        settings["target"],
        "--start-date",
        settings["start"],
        "--model-config",
        str(ROOT / "model_config.json"),
        "--horizon",
        "16",
        "--phase2-max-steps",
        str(POLICY["max_steps"]),
        "--phase2-val-check-steps",
        str(POLICY["interval"]),
        "--phase2-patience",
        str(POLICY["patience"]),
        "--phase2-val-size",
        str(POLICY["val_size"]),
        "--phase2-start-ratio",
        "0.7",
        "--wandb-project",
        project,
    ]
    if exogenous:
        command.extend(
            [
                "--end-date",
                END,
                "--auto-hist-exog",
                "--exog-max-abs-corr",
                "0.8",
                "--exog-max-missing-ratio",
                "0.05",
            ]
        )
    return command


def _copy_source(source, destination):
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    destination.mkdir(parents=True)
    shutil.copytree(source / "neuralforecast", destination / "neuralforecast", ignore=ignore)
    shutil.copytree(
        source / "experiments/commodity_sota",
        destination / "experiments/commodity_sota",
        ignore=ignore,
    )


def _hashes(paths):
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def prepare():
    if QUEUE.exists():
        raise FileExistsError(f"Reservation already exists: {QUEUE}")
    for settings in PROJECTS.values():
        if not settings["data"].is_file():
            raise FileNotFoundError(settings["data"])
        if settings["output"].exists():
            raise FileExistsError(settings["output"])

    frozen = ROOT / "results/colab-integration/source"
    _copy_source(frozen, LEGACY_SOURCE)
    shutil.copy2(
        ROOT / "experiments/commodity_sota/pool_worker.py",
        LEGACY_SOURCE / "experiments/commodity_sota/pool_worker.py",
    )
    _copy_source(ROOT, EXOG_SOURCE)

    fingerprints = {}
    for project, settings in PROJECTS.items():
        preflight = settings["output"].with_name(settings["output"].name + "-preflight")
        subprocess.run(
            _command(EXOG_SOURCE, project, settings, exogenous=True)
            + ["--output", str(preflight), "--preflight"],
            env=dict(_environment(EXOG_SOURCE), NF_COLAB_SESSIONS="0"),
            check=True,
        )
        report = json.loads((preflight / "preflight.json").read_text())
        if not report["hist_exog_list"]:
            raise RuntimeError(f"No exogenous features selected for {project}")
        fingerprints[project] = report["fingerprint"]

    source_paths = list(LEGACY_SOURCE.rglob("*.py")) + list(EXOG_SOURCE.rglob("*.py"))
    source_paths += [settings["data"] for settings in PROJECTS.values()]
    source_paths.append(ROOT / "model_config.json")
    reservation = {
        "sequence": [
            "uni-wti",
            "uni-wti-diff",
            "uni-gasoline-exog",
            "uni-wti-exog",
        ],
        "policy": POLICY,
        "end_date": END,
        "fingerprints": fingerprints,
        "hashes": _hashes(source_paths),
    }
    _save(QUEUE / "reservation.json", reservation)
    shutil.copy2(Path(__file__), QUEUE / "start.py")
    _state("queued", sequence=reservation["sequence"])


def _credentials():
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        auth = netrc.netrc().authenticators("api.wandb.ai")
        key = auth[2] if auth else None
    if not key:
        raise RuntimeError("W&B credentials are unavailable")
    return key


def _verify(reservation):
    for name, expected in reservation["hashes"].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Reserved source or data changed: {name}")


def _status(output):
    path = Path(output) / "run_config.json"
    return json.loads(path.read_text()).get("status") if path.is_file() else None


def _run_existing_wti(key):
    output = ROOT / "results/wti"
    if _status(output) in {"completed", "no_models_above_naive"}:
        return
    settings = {
        "data": ROOT / "data/wti.csv",
        "target": "Oil_EIA_Cushing_WTI_Spot_Price_Daily_USD_Per_Barrel",
        "start": "2015-03-15",
    }
    smoke = ROOT / "results/wti-dynamic-smoke/smoke_results.json"
    run_config_path = output / "run_config.json"
    run_config = json.loads(run_config_path.read_text())
    if not run_config.get("candidates"):
        smoke_report = json.loads(smoke.read_text())
        run_config["candidates"] = smoke_report["passed"]
        _save(run_config_path, run_config)
    _state("recovering", project="uni-wti")
    subprocess.run(
        _command(LEGACY_SOURCE, "uni-wti", settings)
        + [
            "--output",
            str(output),
            "--validated-candidates",
            str(smoke),
            "--wandb",
            "--wandb-entity",
            "Beat-Sun",
            "--resume",
        ],
        env=dict(_environment(LEGACY_SOURCE), WANDB_API_KEY=key),
        check=True,
    )
    if _status(output) not in {"completed", "no_models_above_naive"}:
        raise RuntimeError("uni-wti recovery did not complete")


def _run_project(source, project, settings, key, exogenous=False):
    output = Path(settings["output"])
    if _status(output) in {"completed", "no_models_above_naive"}:
        return
    smoke_output = output.with_name(output.name + "-dynamic-smoke")
    smoke_path = smoke_output / "smoke_results.json"
    common = _command(source, project, settings, exogenous=exogenous)
    if not smoke_path.is_file():
        if smoke_output.exists():
            raise RuntimeError(f"Incomplete smoke directory requires review: {smoke_output}")
        _state("smoke", project=project)
        subprocess.run(
            common + ["--output", str(smoke_output), "--smoke-test"],
            env=_environment(source),
            check=True,
        )
    smoke = json.loads(smoke_path.read_text())
    if not smoke.get("passed"):
        raise RuntimeError(f"No smoke-tested candidate passed for {project}")
    arguments = common + [
        "--output",
        str(output),
        "--validated-candidates",
        str(smoke_path),
        "--wandb",
        "--wandb-entity",
        "Beat-Sun",
    ]
    if output.exists():
        arguments.append("--resume")
    _state("experiment", project=project, passed_candidates=len(smoke["passed"]))
    subprocess.run(
        arguments,
        env=dict(_environment(source), WANDB_API_KEY=key),
        check=True,
    )
    if _status(output) not in {"completed", "no_models_above_naive"}:
        raise RuntimeError(f"Experiment did not complete: {project}")


def execute():
    reservation = json.loads((QUEUE / "reservation.json").read_text())
    _verify(reservation)
    subprocess.run(
        [sys.executable, "-c", "import torch; assert torch.cuda.is_available()"],
        check=True,
    )
    key = _credentials()
    _run_existing_wti(key)
    legacy_wti_diff = {
        "data": ROOT / "data/wti.csv",
        "target": "Oil_EIA_Cushing_WTI_Spot_Price_Daily_USD_Per_Barrel",
        "start": "2015-03-15",
        "output": ROOT / "results/wti-diff",
    }
    _run_project(LEGACY_SOURCE, "uni-wti-diff", legacy_wti_diff, key)
    for project, settings in PROJECTS.items():
        _run_project(EXOG_SOURCE, project, settings, key, exogenous=True)
    _state("completed", sequence=reservation["sequence"])


if __name__ == "__main__":
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    try:
        prepare() if args.prepare else execute()
    except BaseException as exc:
        _state("stopped", error=f"{type(exc).__name__}: {exc}")
        raise
