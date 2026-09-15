import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def load_migration():
    path = (
        Path(__file__).parents[1]
        / "experiments/commodity_sota/migrate_phase2_window.py"
    )
    spec = importlib.util.spec_from_file_location("phase2_window_migration_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_inventory_separates_only_early_phase2_runs():
    migration = load_migration()
    runs = [
        SimpleNamespace(name="phase1/GRU/config-0/fold-0", id="p1"),
        SimpleNamespace(name="phase2/GRU/config-0/fold-323", id="early"),
        SimpleNamespace(name="phase2/GRU/config-0/fold-324", id="keep"),
        SimpleNamespace(name="summary/benchmark/config-0/fold--1", id="summary"),
    ]
    api = SimpleNamespace(runs=lambda *args, **kwargs: runs)
    config = {"wandb": {"entity": "team", "project": "p", "group": "g"}}

    path, early, retained, phase1, summaries, unexpected = migration._inventory(
        api, config, 324
    )

    assert path == "team/p"
    assert [run.id for run in early] == ["early"]
    assert [run.id for run in retained] == ["keep"]
    assert [run.id for run in phase1] == ["p1"]
    assert [run.id for run in summaries] == ["summary"]
    assert unexpected == []


def test_local_inventory_requires_every_retained_result(tmp_path):
    migration = load_migration()
    root = tmp_path / "checkpoints_phase2" / "GRU" / "0"
    for fold in range(3):
        folder = root / str(fold)
        folder.mkdir(parents=True)
        (folder / "result-500.json").write_text("{}")

    early, retained = migration._local_inventory(
        tmp_path, {"selected": ["GRU"]}, first_fold=1, last_fold=2
    )

    assert [path.name for path in early] == ["0"]
    assert [path.name for path in retained] == ["1", "2"]
