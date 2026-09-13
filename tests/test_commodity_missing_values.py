"""Missing-value policy for the commodity benchmark."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def runner(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "experiments/commodity_sota/run.py"
    spec = importlib.util.spec_from_file_location("commodity_run_missing_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_missing_values_use_only_prior_observation(runner):
    frame = pd.DataFrame({"y": [1.0, np.nan, np.nan, 4.0]})
    assert runner._fill(frame)["y"].tolist() == [1.0, 1.0, 1.0, 4.0]


def test_leading_missing_value_is_rejected(runner):
    with pytest.raises(ValueError, match="begins with missing"):
        runner._fill(pd.DataFrame({"y": [np.nan, 2.0]}))
