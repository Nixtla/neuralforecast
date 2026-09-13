import numpy as np
import pandas as pd
import pytest
import torch
from ray.tune.search.variant_generator import generate_variants

import neuralforecast.foundation_lora as foundation_lora


def _sample(space, n=5, seed=42):
    random_state = np.random.RandomState(seed)
    return [
        next(generate_variants(space, random_state=random_state))[1] for _ in range(n)
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


def test_external_training_restores_global_precision_on_failure():
    if not hasattr(torch.backends, "fp32_precision"):
        pytest.skip("new precision API unavailable")
    original = torch.backends.fp32_precision
    with pytest.raises(ValueError):
        with foundation_lora._preserve_precision():
            torch.backends.fp32_precision = "tf32"
            raise ValueError("training failed")
    assert torch.backends.fp32_precision == original
    # Lightning's legacy getter must still be usable after HF training.
    assert torch.get_float32_matmul_precision() in {"highest", "high", "medium"}


@pytest.mark.parametrize("backend", ["Chronos2", "TimesFM"])
def test_lora_cumulative_training_with_tiny_local_models(
    backend, monkeypatch, tmp_path
):
    """Exercise real optional trainers/PEFT without downloading model weights."""
    pytest.importorskip("peft")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    if backend == "Chronos2":
        pytest.importorskip("chronos")
        from chronos.chronos2 import Chronos2Model, Chronos2Pipeline
        from chronos.chronos2.config import Chronos2CoreConfig

        class Chronos2:
            def __init__(self, h, input_size, **kwargs):
                self.input_size = input_size
                config = Chronos2CoreConfig(
                    d_model=16,
                    d_kv=8,
                    d_ff=32,
                    num_layers=1,
                    num_heads=2,
                    chronos_config=dict(
                        context_length=16,
                        input_patch_size=4,
                        input_patch_stride=4,
                        output_patch_size=4,
                        quantiles=[i / 10 for i in range(1, 10)],
                        max_output_patches=1,
                        use_reg_token=True,
                    ),
                )
                self.pipeline = Chronos2Pipeline(Chronos2Model(config))

            def _get_backend(self):
                return self.pipeline

        cls = Chronos2
    else:
        from transformers import TimesFm2_5Config, TimesFm2_5ModelForPrediction

        def local_model(*args, **kwargs):
            return TimesFm2_5ModelForPrediction(
                TimesFm2_5Config(
                    patch_length=4,
                    context_length=16,
                    horizon_length=4,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=8,
                    output_quantile_len=4,
                    use_continuous_quantile_head=False,
                )
            )

        monkeypatch.setattr(
            TimesFm2_5ModelForPrediction, "from_pretrained", local_model
        )
        cls = type("TimesFM", (), {})
    config = dict(
        h=4,
        input_size=16,
        backend_device="cpu",
        learning_rate=1e-3,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_batch_size=2,
    )
    frame = pd.DataFrame({"y": 3 + np.sin(np.arange(68) / 7)})
    options = dict(
        model_cls=cls,
        config=config,
        train=frame.iloc[:64],
        h=4,
        validation=frame.iloc[64:],
        stopping={"interval": 1, "patience": 20},
        schedule_steps=4,
    )
    try:
        first = foundation_lora.fit_foundation_lora(
            **options, steps=2, output_dir=tmp_path / "first"
        )
        second = foundation_lora.fit_foundation_lora(
            **options,
            steps=4,
            checkpoint=first.resume_checkpoint,
            output_dir=tmp_path / "second",
        )
        assert first.early_stopping_info["actual_steps"] == 2
        assert second.early_stopping_info["actual_steps"] == 4
        assert (
            second.early_stopping_info["best_validation_loss"]
            <= first.early_stopping_info["best_validation_loss"]
        )
        if backend == "TimesFM":
            state = torch.load(second.resume_checkpoint, weights_only=False)
            assert state["global_step"] == 4
            assert state["optimizer"]["state"]
        else:
            from pathlib import Path

            assert (Path(second.resume_checkpoint) / "optimizer.pt").is_file()
    finally:
        torch.set_num_threads(old_threads)
