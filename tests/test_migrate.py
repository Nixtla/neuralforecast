"""The `migrate` tool: rewriting a legacy directory in the safe format."""

import json
import os
import pickle
import shutil

import pytest
import torch

from neuralforecast import NeuralForecast
from neuralforecast.migrate import main, migrate
from neuralforecast.models import NLinear
from neuralforecast.utils import AirPassengersPanel

V1_FIXTURES = "tests/backward_comp/data"
FIXTURE_NAMES = ["NLinear", "DeepAR", "TSMixer"]


@pytest.fixture
def legacy(tmp_path, request):
    name = getattr(request, "param", "NLinear")
    destination = tmp_path / name
    shutil.copytree(f"{V1_FIXTURES}/{name}", destination)
    return str(destination)


@pytest.fixture
def legacy_with_dataset(tmp_path):
    """A legacy directory that stores its dataset, which cannot be read safely."""
    panel = (
        AirPassengersPanel[["unique_id", "ds", "y"]]
        .groupby("unique_id")
        .tail(40)
        .reset_index(drop=True)
    )
    nf = NeuralForecast(
        models=[
            NLinear(
                h=4,
                input_size=8,
                max_steps=1,
                enable_progress_bar=False,
                logger=False,
                # macOS CI runs this file and its MPS pool is tiny; the suite's
                # convention is to pin tests that build models to the CPU.
                accelerator="cpu",
                devices=1,
            )
        ],
        freq="ME",
    )
    nf.fit(panel)

    directory = tmp_path / "legacy_ds"
    directory.mkdir()
    model = nf.models[0]
    hparams = {k: v for k, v in dict(model.hparams).items() if k != "callbacks"}
    torch.save(
        {"hyper_parameters": hparams, "state_dict": model.state_dict()},
        directory / "NLinear_0.ckpt",
    )
    with open(directory / "alias_to_model.pkl", "wb") as f:
        pickle.dump({"NLinear": "nlinear"}, f)
    with open(directory / "dataset.pkl", "wb") as f:
        pickle.dump(nf.dataset, f)
    with open(directory / "configuration.pkl", "wb") as f:
        pickle.dump(
            {
                "h": nf.h,
                "freq": nf.freq,
                "_fitted": nf._fitted,
                "local_scaler_type": None,
                "scalers_": {},
                "static_scalers_": {},
                "categorical_vocab_": {},
                "id_col": nf.id_col,
                "time_col": nf.time_col,
                "target_col": nf.target_col,
                "prediction_intervals": None,
                "_cs_df": None,
                "uids": nf.uids,
                "last_dates": nf.last_dates,
                "ds": nf.ds,
            },
            f,
        )
    return str(directory), nf



def _on_cpu(nf):
    """Pin a forecaster loaded from a fixture to the CPU.

    The fixtures were fitted before these tests pinned the accelerator, so their
    stored trainer kwargs still auto-select one. macOS CI has almost no MPS
    memory available by the time this file runs.
    """
    for model in nf.models:
        model.trainer_kwargs["accelerator"] = "cpu"
        model.trainer_kwargs["devices"] = 1
    return nf


@pytest.mark.parametrize("legacy", FIXTURE_NAMES, indirect=True)
def test_migrated_directory_loads_without_pickle(legacy):
    destination = migrate(legacy, verbose=False)
    assert sorted(os.listdir(destination)) == [
        f"{os.path.basename(legacy)}_0.safetensors",
        "configuration.json",
    ]
    assert NeuralForecast.load(destination, allow_pickle=False) is not None


@pytest.mark.parametrize("legacy", FIXTURE_NAMES, indirect=True)
def test_migration_preserves_predictions(legacy):
    # The fixtures were fitted on the full panel; DeepAR also needs `trend` as a
    # future exogenous feature, so predict the same way test_backward_comp does.
    horizon = 12
    panel = AirPassengersPanel.copy()
    train_df = panel[panel.ds < panel["ds"].values[-horizon]]
    test_df = panel[panel.ds >= panel["ds"].values[-horizon]]

    with pytest.warns(UserWarning):
        before = _on_cpu(NeuralForecast.load(legacy, allow_pickle=True)).predict(
            df=train_df, futr_df=test_df
        )

    destination = migrate(legacy, verbose=False)
    after = _on_cpu(NeuralForecast.load(destination, allow_pickle=False)).predict(
        df=train_df, futr_df=test_df
    )
    assert before.equals(after)


def test_migration_recovers_a_loss_from_an_unpickled_object(legacy):
    """Unpickling skips `__init__`, so the loss carries no recorded arguments."""
    with pytest.warns(UserWarning):
        original = NeuralForecast.load(legacy, allow_pickle=True).models[0].loss
    assert not hasattr(original, "_nf_init_kwargs")

    destination = migrate(legacy, verbose=False)
    restored = NeuralForecast.load(destination, allow_pickle=False).models[0].loss
    assert type(restored) is type(original)
    assert restored.output_names == original.output_names
    assert restored.outputsize_multiplier == original.outputsize_multiplier


def test_source_is_left_untouched(legacy):
    before = sorted(os.listdir(legacy))
    migrate(legacy, verbose=False)
    assert sorted(os.listdir(legacy)) == before


def test_default_destination_is_a_sibling(legacy):
    assert migrate(legacy, verbose=False) == f"{legacy}_v2"


def test_explicit_destination(legacy, tmp_path):
    destination = str(tmp_path / "elsewhere")
    assert migrate(legacy, dst=destination, verbose=False) == destination
    assert NeuralForecast.load(destination, allow_pickle=False) is not None


@pytest.mark.parametrize(
    "alias", ["{src}", "{src}/", "./{name}", "{src}/../{name}", "{src}/inner"]
)
def test_refuses_a_destination_that_resolves_onto_the_source(legacy, alias, monkeypatch):
    """A raw string compare missed `models` vs `./models` and ate the source."""
    monkeypatch.chdir(os.path.dirname(legacy))
    name = os.path.basename(legacy)
    before = sorted(os.listdir(legacy))

    with pytest.raises(ValueError, match="same location"):
        migrate(legacy, dst=alias.format(src=legacy, name=name), verbose=False)
    assert sorted(os.listdir(legacy)) == before


def test_refuses_a_remote_source_without_opt_in():
    with pytest.raises(ValueError, match="Refusing to load from the remote path"):
        migrate("s3://bucket/models", dst="/tmp/out", verbose=False)


def test_already_migrated_directory_is_a_no_op(legacy, capsys):
    destination = migrate(legacy, verbose=False)
    assert migrate(destination, verbose=True) == destination
    assert "already in the v2 format" in capsys.readouterr().out


def test_rejects_a_directory_that_is_not_a_saved_model(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="does not look like a saved"):
        migrate(str(empty), verbose=False)


def test_existing_destination_needs_overwrite(legacy, tmp_path):
    destination = str(tmp_path / "taken")
    os.makedirs(destination)
    with open(f"{destination}/stale.json", "w") as f:
        json.dump({}, f)

    with pytest.raises(Exception, match="not empty"):
        migrate(legacy, dst=destination, verbose=False)
    assert migrate(legacy, dst=destination, overwrite=True, verbose=False)


def test_dataset_is_carried_over(legacy_with_dataset):
    source, nf = legacy_with_dataset
    destination = migrate(source, verbose=False)

    loaded = NeuralForecast.load(destination, allow_pickle=False)
    assert loaded.dataset is not None
    assert torch.equal(loaded.dataset.temporal, nf.dataset.temporal)
    assert nf.predict().equals(loaded.predict())


def test_dataset_directory_is_unreadable_before_migration(legacy_with_dataset):
    """The case migration exists for: `dataset.pkl` has no safe reader."""
    source, _ = legacy_with_dataset
    with pytest.raises(ValueError, match="Refusing to load the legacy dataset"):
        NeuralForecast.load(source, allow_pickle=False)

    destination = migrate(source, verbose=False)
    assert NeuralForecast.load(destination, allow_pickle=False) is not None


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def test_cli_migrates_and_reports(legacy, capsys):
    assert main([legacy]) == 0
    out = capsys.readouterr().out
    assert "executes any code" in out
    assert f"Migrated {legacy} -> {legacy}_v2" in out
    assert "verified   loads with allow_pickle=False" in out


def test_cli_quiet(legacy, capsys):
    assert main([legacy, "--quiet"]) == 0
    assert capsys.readouterr().out == ""


def test_cli_reports_failure(tmp_path, capsys):
    assert main([str(tmp_path / "missing")]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_refuses_remote_without_the_flag(capsys):
    assert main(["s3://bucket/models"]) == 1
    assert "Refusing to load from the remote path" in capsys.readouterr().err
