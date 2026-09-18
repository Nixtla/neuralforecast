"""Regression tests for native probabilistic pretrained context adapters."""

import numpy as np
import pandas as pd
import pytest
import torch

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE, MQLoss
from neuralforecast.models import Aurora, TabPFNTS


class _TensorBatch(dict):
    def to(self, device):
        return _TensorBatch({key: value.to(device) for key, value in self.items()})


class _Tokenizer:
    def __call__(self, text, return_tensors):
        assert text and return_tensors == "pt"
        return _TensorBatch(
            input_ids=torch.ones(1, 2, dtype=torch.long),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
        )


class _AuroraBackend:
    def generate(self, inputs, max_output_length, num_samples, **kwargs):
        assert inputs.shape[0] == 1
        assert kwargs["text_input_ids"].shape == (1, 2)
        samples = torch.arange(num_samples, dtype=inputs.dtype, device=inputs.device)
        return samples.reshape(1, num_samples, 1).expand(1, num_samples, max_output_length)


class _TabPFNPipeline:
    def predict_df(self, context_df, future_df, quantiles):
        assert "target" in context_df and "target" not in future_df
        target = future_df["covariate_0"].to_numpy() + 10.0
        result = pd.DataFrame({"target": target})
        for quantile in quantiles:
            result[quantile] = target + (quantile - 0.5) * 2.0
        return result


def _series_frame(length=8):
    return pd.DataFrame(
        {
            "unique_id": "a",
            "ds": pd.date_range("2026-01-01", periods=length, freq="D"),
            "y": np.arange(length, dtype=np.float32),
        }
    )


@pytest.mark.parametrize("probabilistic", [False, True])
def test_aurora_quantiles_use_generated_samples(tmp_path, probabilistic):
    checkpoint = tmp_path / "aurora"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"test")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()

    model = Aurora(
        h=2,
        input_size=4,
        source_dir=str(tmp_path),
        model_id=str(checkpoint),
        tokenizer_path=str(tokenizer),
        contexts=["demand context"],
        stat_exog_list=["context_id"],
        num_samples=5,
        accelerator="cpu",
        loss=MQLoss(quantiles=[0.1, 0.5, 0.9]) if probabilistic else MAE(),
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.models[0].__dict__["_backend"] = (_AuroraBackend(), _Tokenizer())
    static = pd.DataFrame({"unique_id": ["a"], "context_id": [0.0]})
    nf.fit(df=_series_frame(), static_df=static)

    forecast = nf.predict()
    columns = [column for column in forecast.columns if column.startswith("Aurora")]
    assert columns == [f"Aurora{name}" for name in nf.models[0].loss.output_names]
    np.testing.assert_allclose(
        forecast[columns].to_numpy(),
        np.tile([0.4, 2.0, 3.6] if probabilistic else [2.0], (2, 1)),
        atol=1e-6,
    )


@pytest.mark.parametrize("probabilistic", [False, True])
def test_tabpfnts_quantiles_use_official_pipeline_columns(tmp_path, probabilistic):
    checkpoint = tmp_path / "tabpfn.ckpt"
    checkpoint.write_bytes(b"test")
    model = TabPFNTS(
        h=2,
        input_size=4,
        model_id=str(checkpoint),
        futr_exog_list=["schedule"],
        accelerator="cpu",
        loss=MQLoss(quantiles=[0.1, 0.5, 0.9]) if probabilistic else MAE(),
    )
    frame = _series_frame()
    frame["schedule"] = np.arange(len(frame), dtype=np.float32)
    nf = NeuralForecast(models=[model], freq="D")
    nf.models[0].__dict__["_backend"] = _TabPFNPipeline()
    nf.fit(df=frame)
    future = nf.make_future_dataframe()
    future["schedule"] = [8.0, 9.0]

    forecast = nf.predict(futr_df=future)
    columns = [column for column in forecast.columns if column.startswith("TabPFNTS")]
    assert columns == [f"TabPFNTS{name}" for name in nf.models[0].loss.output_names]
    np.testing.assert_allclose(
        forecast[columns].to_numpy(),
        [[17.2, 18.0, 18.8], [18.2, 19.0, 19.8]] if probabilistic else [[18.0], [19.0]],
        atol=1e-6,
    )
