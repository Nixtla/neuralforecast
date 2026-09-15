import importlib.util
import json
from pathlib import Path


def load_backfill():
    path = (
        Path(__file__).parents[1]
        / "experiments/commodity_sota/backfill_wandb_forecasts.py"
    )
    spec = importlib.util.spec_from_file_location("wandb_forecast_backfill_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_forecasts_keeps_latest_successful_rung(tmp_path):
    folder = tmp_path / "checkpoints_phase1" / "GRU" / "0" / "2"
    folder.mkdir(parents=True)
    for budget, ok in ((100, True), (250, True), (500, False)):
        (folder / f"result-{budget}.json").write_text(
            json.dumps(
                {
                    "ok": ok,
                    "candidate": "GRU",
                    "config_id": 0,
                    "fold": 2,
                    "budget": budget,
                    "actual": [1, 2],
                    "prediction": [2, 3],
                    "forecast_origin": 0,
                }
            )
        )
    selected = load_backfill().collect_forecasts(tmp_path, ("phase1",))
    assert selected[("phase1", "GRU", 0, 2)]["budget"] == 250


def test_collect_forecasts_skips_run_when_latest_rung_is_already_tracked(tmp_path):
    folder = tmp_path / "checkpoints_phase2" / "GRU" / "0" / "2"
    folder.mkdir(parents=True)
    base = {
        "ok": True,
        "candidate": "GRU",
        "config_id": 0,
        "fold": 2,
        "actual": [1],
        "prediction": [2],
        "forecast_origin": 0,
    }
    (folder / "result-100.json").write_text(json.dumps({**base, "budget": 100}))
    (folder / "result-500.json").write_text(
        json.dumps({**base, "budget": 500, "forecast_tracking_version": 1})
    )
    assert not load_backfill().collect_forecasts(tmp_path, ("phase2",))
