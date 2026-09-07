"""Isolated uni2ts worker. Execute with its environment's Python, not `-m`.

This script deliberately imports no neuralforecast module, since the two
projects' Torch constraints are incompatible. Inputs use JSON and NPZ without
pickle. The private per-call directory is created/removed by the parent.
"""

import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def main(directory: str) -> None:
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    names = {"moirai": "Moirai", "moirai_moe": "MoiraiMoE"}
    prefix = names[config["kind"]]
    package = importlib.import_module("uni2ts.model." + config["kind"])
    torch.manual_seed(config["random_seed"])
    options = {"revision": config["revision"]} if config["revision"] is not None else {}
    module = getattr(package, prefix + "Module").from_pretrained(config["model_id"], **options)
    forecast = getattr(package, prefix + "Forecast")(
        module=module, prediction_length=config["h"], context_length=config["input_size"],
        target_dim=1, feat_dynamic_real_dim=config["futr_size"],
        past_feat_dynamic_real_dim=config["hist_size"], patch_size=config["patch_size"],
        num_samples=config["num_samples"],
    ).to(config["device"]).eval()
    with np.load(directory / "inputs.npz", allow_pickle=False) as payload:
        inputs = {key: torch.from_numpy(payload[key].copy()).to(config["device"]) for key in payload.files}
    with torch.no_grad():
        samples = forecast(**inputs)
        prediction = samples.mean(dim=1).cpu().numpy()
    np.savez(directory / "outputs.npz", prediction=prediction)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python _uni2ts_worker.py PRIVATE_BATCH_DIRECTORY")
    main(sys.argv[1])
