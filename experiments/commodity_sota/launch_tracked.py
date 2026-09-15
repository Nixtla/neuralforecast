"""Launch only the main gasoline experiment with process-scoped W&B auth."""

import argparse
import getpass
import netrc
import os
from pathlib import Path
import subprocess

import requests


PROJECTS = {
    "uni-gasoline": {
        "data": "data/gasoline.csv",
        "target": "Oil_EIA_NY_Harbor_Conventional_Gasoline_Spot_Price_Daily_USD_Per_Gallon",
        "start": "2013-08-11",
        "end": "2026-09-06",
        "exogenous": False,
    },
    "uni-gasoline-diff": {
        "data": "data/gasoline.csv",
        "target": "Oil_EIA_NY_Harbor_Conventional_Gasoline_Spot_Price_Daily_USD_Per_Gallon",
        "start": "2013-08-11",
        "end": "2026-09-06",
        "exogenous": False,
    },
    "uni-gasoline-exog": {
        "data": "data/gasoline-exog.csv",
        "target": "Oil_EIA_NY_Harbor_Conventional_Gasoline_Spot_Price_Daily_USD_Per_Gallon",
        "start": "2013-08-11",
        "end": "2026-09-06",
        "exogenous": True,
    },
    "uni-wti-exog": {
        "data": "data/wti-exog.csv",
        "target": "Oil_EIA_Cushing_WTI_Spot_Price_Daily_USD_Per_Barrel",
        "start": "2015-03-15",
        "end": "2026-09-06",
        "exogenous": True,
    },
}


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="uni-gasoline")
    args = parser.parse_args()
    if args.project not in PROJECTS:
        parser.error(f"Supported projects: {', '.join(PROJECTS)}")
    settings = PROJECTS[args.project]
    experiment = args.project.removeprefix("uni-")
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        try:
            credentials = netrc.netrc().authenticators("api.wandb.ai")
            key = credentials[2] if credentials else None
        except (OSError, netrc.NetrcParseError):
            pass
    if not key:
        key = getpass.getpass("W&B key (hidden): ")
    response = requests.post(
        "https://api.wandb.ai/graphql",
        auth=("api", key),
        timeout=30,
        json={"query": "query { viewer { teams { edges { node { name } } } } }"},
    )
    response.raise_for_status()
    teams = response.json()["data"]["viewer"]["teams"]["edges"]
    if "Beat-Sun" not in {edge["node"]["name"] for edge in teams}:
        raise ValueError("Key does not have access to the configured Beat-Sun team")
    environment = dict(os.environ, WANDB_API_KEY=key)
    command = [
        "systemd-run",
        "--user",
        f"--unit={experiment}-benchmark",
        "--collect",
        f"--property=WorkingDirectory={root}",
        "--property=UMask=0077",
        "--setenv=WANDB_API_KEY",
        "--setenv=WANDB_ENTITY=Beat-Sun",
        "--setenv=WANDB_DISABLE_CODE=true",
        "--setenv=OMP_NUM_THREADS=2",
        "--setenv=PATH=/home/t-lab01/.local/bin:/usr/local/bin:/usr/bin:/bin",
        "--setenv=MKL_NUM_THREADS=2",
        "--setenv=MPLCONFIGDIR=/tmp/gasoline-mpl",
        "--setenv=HF_HUB_DISABLE_PROGRESS_BARS=1",
        "--setenv=TOKENIZERS_PARALLELISM=false",
        str(root / ".venv/bin/python"),
        "experiments/commodity_sota/run.py",
        "--scheduler",
        "dynamic-pool",
        "--data",
        settings["data"],
        "--date-col",
        "ds",
        "--target",
        settings["target"],
        "--start-date",
        settings["start"],
        "--end-date",
        settings["end"],
        "--model-config",
        "model_config.json",
        "--output",
        f"results/{experiment}",
        "--validated-candidates",
        f"results/{experiment}-dynamic-smoke/smoke_results.json",
        "--wandb",
        "--wandb-entity",
        "Beat-Sun",
        "--wandb-project",
        args.project,
    ]
    if settings["exogenous"]:
        command.extend(
            [
                "--auto-hist-exog",
                "--exog-max-abs-corr",
                "0.8",
                "--exog-max-missing-ratio",
                "0.05",
            ]
        )
    subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
