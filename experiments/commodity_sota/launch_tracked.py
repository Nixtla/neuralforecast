"""Launch only the main gasoline experiment with process-scoped W&B auth."""

import getpass
import os
from pathlib import Path
import subprocess

import requests


def main():
    root = Path(__file__).resolve().parents[2]
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
        "--unit=gasoline-benchmark",
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
        "dynamic",
        "--data",
        "data/gasoline.csv",
        "--date-col",
        "ds",
        "--target",
        "Oil_EIA_NY_Harbor_Conventional_Gasoline_Spot_Price_Daily_USD_Per_Gallon",
        "--start-date",
        "2013-08-11",
        "--model-config",
        "model_config.json",
        "--output",
        "results/gasoline",
        "--validated-candidates",
        "results/gasoline-dynamic-smoke/smoke_results.json",
        "--wandb",
        "--wandb-entity",
        "Beat-Sun",
        "--wandb-project",
        "uni-gasoline",
    ]
    subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
