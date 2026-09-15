"""Historical-exogenous selection policy for commodity experiments."""

import importlib.util
from datetime import date
from decimal import Decimal
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def runner(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "experiments/commodity_sota/run.py"
    spec = importlib.util.spec_from_file_location("commodity_run_exog_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def exporter(monkeypatch):
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/commodity_sota/export_postgres.py"
    )
    spec = importlib.util.spec_from_file_location("commodity_export_exog_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def frame(rows=20):
    y = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "unique_id": "oil",
            "ds": pd.date_range("2020-01-05", periods=rows, freq="W-SUN"),
            "y": y,
            "positive": y,
            "negative": -y,
            "future_only": np.r_[y[:10], -y[10:]],
            "weak": np.sin(y),
        }
    )


def test_selection_uses_absolute_correlation_and_only_initial_training(runner):
    values = frame()
    selected, report, columns = runner._select_historical_exogenous(
        values, training_rows=10, max_abs_correlation=0.8
    )

    assert columns == ["weak"]
    assert selected.columns.tolist() == [
        "unique_id",
        "ds",
        "y",
        "weak",
    ]
    indexed = report.set_index("feature")
    assert indexed.loc["negative", "correlation"] == pytest.approx(-1)
    assert indexed.loc["future_only", "reason"] == (
        "absolute_correlation_at_or_above_threshold"
    )


def test_selection_excludes_the_exact_absolute_correlation_threshold(runner):
    values = frame()
    threshold = abs(values.iloc[:10]["positive"].corr(values.iloc[:10]["y"]))

    _, report, columns = runner._select_historical_exogenous(
        values, training_rows=10, max_abs_correlation=threshold
    )

    assert "positive" not in columns
    assert "weak" in columns
    assert report.set_index("feature").loc["positive", "reason"] == (
        "absolute_correlation_at_or_above_threshold"
    )


def test_missing_threshold_is_exclusive_and_forward_fill_is_causal(runner):
    values = frame(rows=40)
    values["exactly_five_percent"] = values["y"]
    values.loc[10, "exactly_five_percent"] = np.nan
    values.loc[30, "exactly_five_percent"] = np.nan
    values["below_five_percent"] = values["weak"]
    values.loc[10, "below_five_percent"] = np.nan

    selected, report, columns = runner._select_historical_exogenous(
        values, training_rows=10, max_missing_ratio=0.05
    )
    indexed = report.set_index("feature")

    assert "exactly_five_percent" not in columns
    assert indexed.loc["exactly_five_percent", "missing_ratio"] == pytest.approx(0.05)
    assert indexed.loc["exactly_five_percent", "reason"] == (
        "missing_ratio_at_or_above_threshold"
    )
    assert selected.loc[10, "below_five_percent"] == selected.loc[
        9, "below_five_percent"
    ]


def test_selection_rejects_leading_missing_and_constant_columns(runner):
    values = frame(rows=40)
    values["leading"] = values["y"]
    values.loc[0, "leading"] = np.nan
    values["constant"] = 1.0

    _, report, columns = runner._select_historical_exogenous(values, training_rows=10)
    reasons = report.set_index("feature")["reason"]

    assert "leading" not in columns
    assert reasons["leading"] == "missing_at_period_start"
    assert reasons["constant"] == "correlation_unavailable"


def test_weekly_frame_keeps_exogenous_only_when_requested(runner, tmp_path):
    path = tmp_path / "wide.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=21, freq="D"),
            "target": np.arange(21),
            "x": np.arange(21) * 2,
        }
    ).to_csv(path, index=False)

    target_only = runner._weekly_frame(path, "date", "target")
    wide = runner._weekly_frame(
        path,
        "date",
        "target",
        start_date="2020-01-05",
        end_date="2020-01-12",
        include_exogenous=True,
    )

    assert target_only.columns.tolist() == ["unique_id", "ds", "y"]
    assert wide.columns.tolist() == ["unique_id", "ds", "y", "x"]
    assert wide["ds"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-01-05",
        "2020-01-12",
    ]


def test_candidate_configs_receive_fixed_historical_exogenous(runner):
    configs = [{"input_size": 16}, {"input_size": 32}]

    result = runner._with_historical_exog(configs, ["a", "b"])

    assert all(config["hist_exog_list"] == ["a", "b"] for config in result)
    assert all(config["futr_exog_list"] is None for config in result)


def test_wide_export_preserves_headers_missing_values_and_date_bounds(
    exporter, monkeypatch, tmp_path
):
    target = exporter.DATASETS["wti"]["target"]
    rows = [
        (date(2020, 1, 5), Decimal("10"), None),
        (date(2020, 1, 12), Decimal("11"), Decimal("2")),
    ]
    monkeypatch.setattr(
        exporter,
        "_snapshot",
        lambda *args: ([(target, "target_internal"), ("feature", "x")], rows),
    )
    output = tmp_path / "wide.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_postgres.py",
            "--dataset",
            "wti",
            "--start-date",
            "2020-01-05",
            "--end-date",
            "2020-01-12",
            "--include-exogenous",
            "--output",
            str(output),
        ],
    )

    exporter.main()

    exported = pd.read_csv(output)
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert exported.columns.tolist() == ["ds", target, "feature"]
    assert exported["feature"].isna().sum() == 1
    assert manifest["include_exogenous"] is True
    assert manifest["missing_ratio"]["feature"] == 0.5

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        exporter.main()
