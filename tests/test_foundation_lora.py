import pandas as pd
import pytest

import neuralforecast.foundation_lora as foundation_lora
from neuralforecast.benchmark import sample_ray_configs


def test_chronos2_lora_search_space_is_sampleable():
    space = foundation_lora.get_foundation_lora_config(
        "Chronos2", h=16, fixed={"backend_device": "cuda"}
    )
    configs = sample_ray_configs(space, n=5, seed=42)
    assert {config["lora_r"] for config in configs} <= {4, 8, 16}
    assert {config["lora_batch_size"] for config in configs} <= {8, 16, 32}
    assert all("max_steps" not in config for config in configs)


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

    monkeypatch.setattr(foundation_lora, "_require_peft", lambda: None)
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


def test_unsupported_foundation_lora_fails_explicitly():
    with pytest.raises(ValueError, match="LoRA is unavailable"):
        foundation_lora.get_foundation_lora_config("TimesFM3", h=16)
