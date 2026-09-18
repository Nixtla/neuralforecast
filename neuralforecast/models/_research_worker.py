"""Private NPZ worker for original ChronosX and Baguan-TS environments.

Execute as a file, not -m: importing neuralforecast would pull in an incompatible
Torch/Chronos version. Only explicitly provided local checkpoints are loaded.
"""

import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from _research_source import source_module


def chronosx_covariates(arrays, history, horizon):
    """Match upstream hf_data_loader: values (-1 if missing), then NaN indicators."""
    past, future = [], []
    if "hist" in arrays:
        past.append(arrays["hist"])
        future.append(np.full((len(arrays["y"]), horizon, arrays["hist"].shape[-1]), np.nan, dtype=np.float32))
    if "futr" in arrays:
        past.append(arrays["futr"][:, :history])
        future.append(arrays["futr"][:, history:])
    past, future = np.concatenate(past, axis=-1), np.concatenate(future, axis=-1)

    def encode(values):
        missing = np.isnan(values)
        return np.concatenate((np.where(missing, -1, values), missing), axis=-1).astype(np.float32)

    return [{"past_feat_dynamic_real": encode(p), "future_feat_dynamic_real": encode(f)}
            for p, f in zip(past, future)]


def _official_chronosx_pipeline():
    worker_dir = Path(__file__).resolve().parent
    original_path = list(sys.path)
    try:
        sys.path[:] = [
            entry
            for entry in sys.path
            if Path(entry or ".").resolve() != worker_dir
        ]
        from chronosx.chronosx import ChronosXPipeline
        return ChronosXPipeline
    finally:
        sys.path[:] = original_path


def predict_chronosx(config, arrays):
    ChronosXPipeline = _official_chronosx_pipeline()
    from safetensors import safe_open

    checkpoint = Path(config["model_id"])
    files = sorted(checkpoint.glob("*.safetensors"))
    if not (checkpoint / "config.json").is_file() or not files:
        raise ValueError("ChronosX requires a local fine-tuned safetensors checkpoint with config.json.")
    keys = set()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as tensors:
            keys.update(tensors.keys())
    if not all(any(key.startswith(prefix) for key in keys) for prefix in ("input_injection_block.", "output_injection_block.")):
        raise ValueError("Checkpoint lacks trained IIB+OIB parameters; plain Chronos weights are not ChronosX.")
    covariates = chronosx_covariates(arrays, config["input_size"], config["h"])
    pipeline = ChronosXPipeline(
        prediction_length=config["h"], num_covariates=covariates[0]["past_feat_dynamic_real"].shape[-1],
        covariate_injection="IIB+OIB", device_map=config["device"],
        hidden_dim=config["hidden_dim"], num_layers=config["num_layers"],
        pretrained_model_name_or_path=str(checkpoint),
    )
    if config["input_size"] > pipeline.tokenizer.config.context_length:
        raise ValueError("input_size exceeds the ChronosX checkpoint's tokenizer context length.")
    if not pipeline.tokenizer.config.use_eos_token:
        raise ValueError("The reviewed ChronosX covariate preparation requires use_eos_token=True.")
    pipeline.chronosx.eval()
    contexts = [torch.from_numpy(y[:, 0].copy()) for y in arrays["y"]]
    with torch.no_grad():
        samples = pipeline.predict(contexts, covariates, num_samples=config["num_samples"])
    return samples.mean(dim=1).cpu().numpy()


def predict_baguan(config, arrays):
    root = Path(config["source_dir"])
    checkpoint_path, config_path = Path(config["model_id"]), Path(config["config_path"])
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("BaguanTS needs an existing tensor checkpoint and matching trusted YAML.")
    # src is an absolute upstream import; this interpreter handles only Baguan.
    sys.path.insert(0, str(root))
    official = source_module(root, "BaguanTS")
    # The original initializer creates an unused, undefined StandardScaler and
    # calls unsafe torch.load. Reconstruct its used attributes without either.
    backend = official.BaguanTS.__new__(official.BaguanTS)
    backend.ckpt_path, backend.config_path, backend.device = str(checkpoint_path), str(config_path), config["device"]
    backend.model = official.ModelFactory.from_config(backend.config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Baguan checkpoint must be a tensor state dictionary.")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("Baguan checkpoint must contain a tensor-only state_dict.")
    state = {key[4:] if key.startswith("net.") else key: value for key, value in state.items()}
    backend.model.load_state_dict(state, strict=True)
    backend.model.to(backend.device).eval()
    predictions = []
    for target, covariates in zip(arrays["y"], arrays["futr"]):
        point, _ = backend.predict(
            covariates[:config["input_size"]].copy(), target[:, 0].copy(),
            covariates[config["input_size"]:].copy(), context_len=config["context_size"],
            K=config["neighbors"], period=1, rag_type="Yscl", rag_window_step=1,
            data_type="TS-tabular", mF=config["num_samples"],
        )
        predictions.append(np.asarray(point).reshape(config["h"]))
    return np.stack(predictions)


def main(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    with np.load(directory / "inputs.npz", allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    random.seed(config["random_seed"])
    functions = {"chronosx": predict_chronosx, "baguants": predict_baguan}
    prediction = np.asarray(functions[config["kind"]](config, arrays))
    if prediction.shape != (len(arrays["y"]), config["h"]) or not np.isfinite(prediction).all():
        raise ValueError("Backend prediction has an invalid shape or non-finite values.")
    np.savez(directory / "outputs.npz", prediction=prediction)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python _research_worker.py PRIVATE_BATCH_DIRECTORY")
    main(sys.argv[1])
