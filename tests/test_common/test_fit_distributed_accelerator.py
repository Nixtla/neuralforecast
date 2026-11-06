"""Regression test: distributed fit on a machine without CUDA.

`trainer_kwargs["accelerator"]` is only set when CUDA is available, so a
Spark cluster of CPU-only workers used to fail with ``KeyError: 'accelerator'``.
"""
import sys
import types

from neuralforecast.common._base_model import BaseModel, DistributedConfig


def test_fit_distributed_without_accelerator_kwarg(monkeypatch):
    captured = {}

    class FakeTorchDistributor:
        def __init__(self, num_processes, local_mode, use_gpu):
            captured["use_gpu"] = use_gpu

        def run(self, *args, **kwargs):
            return "trained"

    fake_distributor = types.ModuleType("pyspark.ml.torch.distributor")
    fake_distributor.TorchDistributor = FakeTorchDistributor
    for name in ("pyspark", "pyspark.ml", "pyspark.ml.torch"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "pyspark.ml.torch.distributor", fake_distributor)

    # no CUDA -> `accelerator` was never put in trainer_kwargs
    model = types.SimpleNamespace(trainer_kwargs={}, hparams={})
    config = DistributedConfig(partitions_path="unused", num_nodes=2, devices=1)

    result = BaseModel._fit_distributed(
        model, config, datamodule=None, val_size=0, test_size=0
    )

    assert result == "trained"
    assert captured["use_gpu"] is False
