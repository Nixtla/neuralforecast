import numpy as np
import pandas as pd
import pytest
import torch
from ray.tune.search.variant_generator import generate_variants

import neuralforecast.foundation_lora as foundation_lora


def _sample(space, n=5, seed=42):
    random_state = np.random.RandomState(seed)
    return [
        next(generate_variants(space, random_state=random_state))[1]
        for _ in range(n)
    ]


def test_chronos2_lora_search_space_is_sampleable():
    space = foundation_lora.get_foundation_lora_config(
        "Chronos2", h=16, fixed={"backend_device": "cuda"}
    )
    configs = _sample(space)
    assert {config["lora_r"] for config in configs} <= {4, 8, 16}
    assert {config["lora_batch_size"] for config in configs} <= {8, 16, 32}
    assert all("max_steps" not in config for config in configs)


def test_timesfm_lora_uses_transformers_checkpoint_and_drops_xreg():
    space = foundation_lora.get_foundation_lora_config("TimesFM", h=16)
    configs = _sample(space)
    assert {config["model_id"] for config in configs} == {
        "google/timesfm-2.5-200m-transformers"
    }
    assert all("xreg_ridge" not in config for config in configs)


def test_chronos2_lora_uses_official_fit_without_validation(monkeypatch, tmp_path):
    calls = {}

    class Pipeline:
        def fit(self, **kwargs):
            calls.update(kwargs)
            return "finetuned"

    class Chronos2:
        def __init__(self, h, input_size, **kwargs):
            self.h = h
            self.input_size = input_size
            self.__dict__["_backend"] = Pipeline()

        def _get_backend(self):
            return self.__dict__["_backend"]

    monkeypatch.setattr(foundation_lora, "_require_package", lambda name: None)
    config = {
        "h": 16,
        "input_size": 32,
        "learning_rate": 1e-4,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_batch_size": 8,
    }
    train = pd.DataFrame({"y": range(64)})
    adapter = foundation_lora.fit_foundation_lora(
        Chronos2, config, train, h=16, steps=1000, output_dir=tmp_path
    )
    assert adapter.__dict__["_backend"] == "finetuned"
    assert calls["finetune_mode"] == "lora"
    assert calls["validation_inputs"] is None
    assert calls["num_steps"] == 1000
    assert calls["lora_config"]["r"] == 8


def test_timesfm_lora_handle_matches_benchmark_predict_interface():
    class Output:
        mean_predictions = torch.arange(8, dtype=torch.float32).reshape(1, 8)

    class Model:
        def __call__(self, **kwargs):
            assert kwargs["past_values"].shape == (1, 4)
            return Output()

    class Dataset:
        temporal = torch.arange(20, dtype=torch.float32).reshape(-1, 1)
        y_idx = 0

    handle = foundation_lora._TimesFMLoRA(Model(), input_size=4, device="cpu")
    handle.set_test_size(8)
    prediction = handle.predict(Dataset(), test_size=8)
    assert prediction.shape == (8, 1)
    np.testing.assert_array_equal(prediction[:, 0], np.arange(8))


def test_unsupported_foundation_lora_fails_explicitly():
    with pytest.raises(ValueError, match="LoRA is unavailable"):
        foundation_lora.get_foundation_lora_config("TimesFM3", h=16)
