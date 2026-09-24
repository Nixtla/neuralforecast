"""Directory-level save/load: the v2 layout and the legacy read paths."""

import json
import os
import pickle

import numpy as np
import pytest

from neuralforecast import NeuralForecast
from neuralforecast._serialization import TAG
from neuralforecast.models import NLinear
from neuralforecast.utils import AirPassengersPanel, PredictionIntervals

V1_FIXTURES = "tests/backward_comp/data"


@pytest.fixture(scope="module")
def panel():
    return (
        AirPassengersPanel[["unique_id", "ds", "y"]]
        .groupby("unique_id")
        .tail(60)
        .reset_index(drop=True)
    )


def _fit(panel, **kwargs):
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
        **kwargs,
    )
    nf.fit(panel)
    return nf


@pytest.fixture(scope="module")
def saved(panel, tmp_path_factory):
    nf = _fit(panel, local_scaler_type="standard")
    path = str(tmp_path_factory.mktemp("v2"))
    nf.save(path, overwrite=True)
    return nf, path


# --------------------------------------------------------------------------
# v2 layout
# --------------------------------------------------------------------------


def test_v2_layout_has_no_pickles(saved):
    _, path = saved
    files = sorted(os.listdir(path))
    assert files == [
        "NLinear_0.safetensors",
        "configuration.json",
        "configuration.safetensors",
        "dataset.json",
        "dataset.safetensors",
    ]


def test_alias_to_model_is_folded_into_the_configuration(saved):
    _, path = saved
    with open(f"{path}/configuration.json") as f:
        document = json.load(f)
    assert document["configuration"]["alias_to_model"] == {"NLinear": "nlinear"}


def test_v2_roundtrip_predicts_identically(saved):
    nf, path = saved
    assert nf.predict().equals(NeuralForecast.load(path).predict())


def test_v2_roundtrip_restores_scalers(saved):
    nf, path = saved
    loaded = NeuralForecast.load(path)
    np.testing.assert_array_equal(
        loaded.scalers_["y"].stats_, nf.scalers_["y"].stats_
    )
    assert type(loaded.scalers_["y"]) is type(nf.scalers_["y"])


def test_v2_roundtrip_restores_index_attributes(saved):
    nf, path = saved
    loaded = NeuralForecast.load(path)
    assert loaded.uids.equals(nf.uids)
    assert loaded.last_dates.equals(nf.last_dates)
    np.testing.assert_array_equal(loaded.ds, nf.ds)


def test_v2_roundtrip_with_prediction_intervals(panel, tmp_path):
    nf = _fit_with_intervals(panel)
    path = str(tmp_path / "pi")
    nf.save(path, overwrite=True)

    loaded = NeuralForecast.load(path)
    assert loaded.prediction_intervals.n_windows == nf.prediction_intervals.n_windows
    assert loaded.prediction_intervals.method == nf.prediction_intervals.method
    assert loaded._cs_df.equals(nf._cs_df)
    assert nf.predict(level=[80]).equals(loaded.predict(level=[80]))


def _fit_with_intervals(panel):
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
    nf.fit(panel, prediction_intervals=PredictionIntervals(n_windows=2), val_size=4)
    return nf


def test_v2_roundtrip_without_dataset(panel, tmp_path):
    nf = _fit(panel)
    path = str(tmp_path / "nods")
    nf.save(path, save_dataset=False, overwrite=True)
    assert sorted(os.listdir(path)) == ["NLinear_0.safetensors", "configuration.json"]


    loaded = NeuralForecast.load(path)
    assert nf.predict(df=panel).equals(loaded.predict(df=panel))


def test_v2_never_calls_pickle_load(saved, monkeypatch):
    _, path = saved

    def forbidden(*args, **kwargs):
        raise AssertionError("pickle.load must not be reached on the v2 path")

    monkeypatch.setattr(pickle, "load", forbidden)
    NeuralForecast.load(path)


def test_v2_wins_over_stray_legacy_files(saved, tmp_path):
    """A leftover .pkl next to a v2 directory must not downgrade the reader."""
    nf, path = saved
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    for name in os.listdir(path):
        (mixed / name).write_bytes(open(f"{path}/{name}", "rb").read())
    with open(mixed / "configuration.pkl", "wb") as f:
        pickle.dump({"freq": "D", "_fitted": False}, f)
    with open(mixed / "alias_to_model.pkl", "wb") as f:
        pickle.dump({"NLinear": "nlinear"}, f)

    loaded = NeuralForecast.load(str(mixed))
    assert loaded.freq == nf.freq, "must have read configuration.json, not the .pkl"


def test_unknown_model_name_is_refused(saved, tmp_path):
    nf, path = saved
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    for name in os.listdir(path):
        (tampered / name).write_bytes(open(f"{path}/{name}", "rb").read())
    with open(tampered / "configuration.json") as f:
        document = json.load(f)
    document["configuration"]["alias_to_model"] = {"NLinear": "os.system"}
    with open(tampered / "configuration.json", "w") as f:
        json.dump(document, f)

    with pytest.raises(ValueError, match="not a known model"):
        NeuralForecast.load(str(tampered))


# --------------------------------------------------------------------------
# Remote paths
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["s3://bucket/models", "gs://b/m", "http://h/m"])
def test_remote_paths_are_refused_without_opt_in(path):
    with pytest.raises(ValueError, match="Refusing to load from the remote path"):
        NeuralForecast.load(path)
    with pytest.raises(ValueError, match="Refusing to load from the remote path"):
        NLinear.load(path)


def test_remote_gate_runs_before_any_fetch(monkeypatch):
    import fsspec

    def forbidden(*args, **kwargs):
        raise AssertionError("must not touch the remote before the gate")

    monkeypatch.setattr(fsspec, "get_fs_token_paths", forbidden)
    monkeypatch.setattr(fsspec, "open", forbidden)
    with pytest.raises(ValueError, match="Refusing to load"):
        NeuralForecast.load("s3://bucket/models")


def test_local_paths_pass_the_gate(saved):
    _, path = saved
    assert NeuralForecast.load(path) is not None
    assert NeuralForecast.load(f"file://{path}") is not None


# --------------------------------------------------------------------------
# Legacy v1 directories
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["NLinear", "DeepAR", "TSMixer"])
def test_v1_directory_loads_with_explicit_consent(name):
    with pytest.warns(UserWarning, match="executes any code"):
        nf = NeuralForecast.load(f"{V1_FIXTURES}/{name}", allow_pickle=True)
    assert len(nf.models) == 1


def test_v1_dataset_pkl_is_refused_outright(tmp_path):
    directory = tmp_path / "legacy"
    directory.mkdir()
    for name in os.listdir(f"{V1_FIXTURES}/NLinear"):
        (directory / name).write_bytes(
            open(f"{V1_FIXTURES}/NLinear/{name}", "rb").read()
        )
    with open(directory / "dataset.pkl", "wb") as f:
        pickle.dump({"not": "a real dataset"}, f)

    with pytest.raises(ValueError) as excinfo:
        NeuralForecast.load(str(directory), allow_pickle=False)
    message = str(excinfo.value)
    assert "save_dataset=False" in message and "predict()" in message


def test_v1_dataset_pkl_is_refused_before_pickle_runs(tmp_path, monkeypatch):
    directory = tmp_path / "legacy"
    directory.mkdir()
    for name in os.listdir(f"{V1_FIXTURES}/NLinear"):
        (directory / name).write_bytes(
            open(f"{V1_FIXTURES}/NLinear/{name}", "rb").read()
        )
    with open(directory / "dataset.pkl", "wb") as f:
        pickle.dump({"not": "a real dataset"}, f)

    monkeypatch.setattr(
        pickle, "load", lambda *a, **k: pytest.fail("pickle.load was reached")
    )
    with pytest.raises(ValueError):
        NeuralForecast.load(str(directory), allow_pickle=False)


def test_restricted_sidecar_reader_rejects_the_dangerous_globals():
    """`_load_from_bytes` is an unrestricted load; allowlisting it undoes this.

    Deleting these assertions is a security decision, not a bug fix.
    """
    from neuralforecast.core import (
        _V1_SIDECAR_DENIED,
        _RestrictedUnpickler,
        _v1_sidecar_allowlist,
    )

    denied = {
        "torch.storage._load_from_bytes",
        "pandas._libs.internals._unpickle_block",
        "pandas._libs.arrays.__pyx_unpickle_NDArrayBacked",
    }
    assert denied <= set(_V1_SIDECAR_DENIED)
    assert not denied & set(_v1_sidecar_allowlist())

    unpickler = _RestrictedUnpickler.__new__(_RestrictedUnpickler)
    for full in denied:
        module, name = full.rsplit(".", 1)
        with pytest.raises(pickle.UnpicklingError, match="explicitly denied"):
            unpickler.find_class(module, name)


def test_restricted_sidecar_reader_refuses_anything_unlisted():
    from neuralforecast.core import _RestrictedUnpickler

    unpickler = _RestrictedUnpickler.__new__(_RestrictedUnpickler)
    for module, name in [("os", "system"), ("builtins", "eval"), ("subprocess", "Popen")]:
        with pytest.raises(pickle.UnpicklingError, match="not on the allowlist"):
            unpickler.find_class(module, name)


def test_restricted_sidecar_reader_returns_registered_classes():
    from coreforecast.scalers import LocalStandardScaler

    from neuralforecast.core import _RestrictedUnpickler

    unpickler = _RestrictedUnpickler.__new__(_RestrictedUnpickler)
    assert (
        unpickler.find_class("coreforecast.scalers", "LocalStandardScaler")
        is LocalStandardScaler
    )


# --------------------------------------------------------------------------
# Save is atomic, and encodes before it deletes (PR #1625 review)
# --------------------------------------------------------------------------


def test_encoding_failure_leaves_the_previous_save_intact(saved, monkeypatch):
    """`save` used to rm the directory before encoding the configuration."""
    from neuralforecast import core

    nf, path = saved
    before = sorted(os.listdir(path))

    def boom(*args, **kwargs):
        raise RuntimeError("encode failed")

    monkeypatch.setattr(core, "encode_mapping", boom)
    with pytest.raises(RuntimeError, match="encode failed"):
        nf.save(path, overwrite=True)

    assert sorted(os.listdir(path)) == before
    assert NeuralForecast.load(path) is not None


def test_timezone_aware_index_roundtrips(panel, tmp_path):
    localized = panel.assign(ds=panel["ds"].dt.tz_localize("UTC"))
    nf = _fit(localized)
    path = str(tmp_path / "tz")
    nf.save(path, overwrite=True)

    loaded = NeuralForecast.load(path)
    assert loaded.last_dates.equals(nf.last_dates)
    assert nf.predict().equals(loaded.predict())


def test_robust_iqr_scaler_is_not_saved_as_mad(panel, tmp_path):
    """Both robust variants share a class, so the wrong one reloaded silently."""
    nf = _fit(panel, local_scaler_type="robust-iqr")
    path = str(tmp_path / "iqr")
    nf.save(path, overwrite=True)

    loaded = NeuralForecast.load(path)
    assert loaded.local_scaler_type == "robust-iqr"
    assert loaded.scalers_["y"]._scaler_type == nf.scalers_["y"]._scaler_type
    assert nf.predict().equals(loaded.predict())


def test_remote_parquet_paths_in_a_dataset_are_refused(tmp_path):
    """`files_ds` is handed to pd.read_parquet, which resolves fsspec URLs."""
    from neuralforecast._serialization import SerializationError, decode_dataset

    meta = {
        "dataset_class": "LocalFilesTimeSeriesDataset",
        "fields": {"files_ds": ["s3://attacker/x.parquet"]},
        "extra": {},
    }
    with pytest.raises(ValueError, match="Refusing to load from the remote path"):
        decode_dataset(meta, {})

    meta["extra"] = {"__getitem__": 1}
    with pytest.raises(SerializationError, match="unexpected dataset attributes"):
        decode_dataset(meta, {})


# --------------------------------------------------------------------------
# Arrays live in a sidecar, not inline JSON (PR #1625 review)
# --------------------------------------------------------------------------


def test_large_arrays_do_not_land_in_the_configuration_json(saved):
    """`ds` has one row per training row; JSON stamps are ~2.7x the tensors."""
    _, path = saved
    with open(f"{path}/configuration.json") as f:
        document = json.load(f)

    stored = document["configuration"]
    assert stored["ds"][TAG] == "datetime64"
    assert "values" not in stored["ds"], "the int64s belong in the sidecar"
    assert stored["ds"]["data"][TAG] == "tensor"
    assert os.path.exists(f"{path}/configuration.safetensors")


def test_configuration_json_stays_small(panel, tmp_path):
    nf = _fit(panel, local_scaler_type="standard")
    path = str(tmp_path / "small")
    nf.save(path, overwrite=True)

    with open(f"{path}/configuration.json") as f:
        text = f.read()
    assert len(text) < 8_000, "configuration.json should hold schema, not data"
    assert nf.predict().equals(NeuralForecast.load(path).predict())


def test_a_sidecar_free_configuration_still_loads(saved, tmp_path):
    """Artifacts written before the sidecar inline their arrays."""
    from neuralforecast._serialization import decode_mapping, encode_mapping

    nf, _ = saved
    config = {"ds": nf.ds, "last_dates": nf.last_dates, "uids": nf.uids}
    inline, _ = encode_mapping(config, inline=True)

    restored = decode_mapping(json.loads(json.dumps(inline)))
    assert (restored["ds"] == nf.ds).all()
    assert restored["last_dates"].equals(nf.last_dates)
