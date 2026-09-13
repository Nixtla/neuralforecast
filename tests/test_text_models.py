"""Actual official-source tests. No backend doubles or language-model downloads.

Set NF_TEXT_SOURCE_ROOT to a directory containing the SpecTF and TGForecaster
checkouts prepared by scripts/fetch_research_sources.py. Source-dependent tests
skip in the normal suite when it is unset; the dedicated CI always sets it.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from neuralforecast import NeuralForecast, models
from neuralforecast.core import MODEL_FILENAME_DICT
from neuralforecast.models import SpecTF, TGForecaster
from neuralforecast.models._text_source import _FILES, text_source


@pytest.fixture
def sources():
    root = os.environ.get("NF_TEXT_SOURCE_ROOT")
    if not root:
        pytest.skip("Set NF_TEXT_SOURCE_ROOT to run actual official-source tests.")
    return Path(root)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(cls, sources, **kwargs):
    options = dict(h=8, input_size=16, source_dir=str(sources / cls.__name__),
                   max_steps=2, val_check_steps=2, windows_batch_size=4,
                   inference_windows_batch_size=4, logger=False,
                   enable_progress_bar=False, enable_model_summary=False,
                   accelerator="cpu", devices=1, random_seed=13)
    if cls is SpecTF:
        options.update(hist_exog_list=[f"emb{i}" for i in range(4)],
                       mm_emb_size=8, mm_hidden_size=16, dropout=0.0, text_dropout=0.0)
    else:
        options.update(futr_exog_list=[f"text{i}" for i in range(16)],
                       text_dim=8, n_heads=2, encoder_layers=1, patch_len=4,
                       dropout=0.0)
    options.update(kwargs)
    return cls(**options)


def windows(cls, length=16, horizon=8):
    generator = torch.Generator().manual_seed(19)
    return dict(
        insample_y=torch.randn(2, length, 1, generator=generator),
        insample_mask=torch.ones(2, length, 1),
        hist_exog=torch.randn(2, length, 4, generator=generator) if cls is SpecTF else None,
        futr_exog=torch.randn(2, length + horizon, 16, generator=generator) if cls is TGForecaster else None,
        stat_exog=None,
    )


@pytest.mark.parametrize("cls", [SpecTF, TGForecaster])
def test_export_and_checkpoint_registry(cls):
    assert cls.__name__ in models.__all__
    assert getattr(models, cls.__name__) is MODEL_FILENAME_DICT[cls.__name__.lower()]


@pytest.mark.parametrize("cls", [SpecTF, TGForecaster])
def test_real_source_gradient_conditioning_and_label_exclusion(cls, sources):
    model = make_model(cls, sources).eval()
    batch = windows(cls)
    key = "hist_exog" if cls is SpecTF else "futr_exog"
    batch[key].requires_grad_()
    output = model(batch)
    assert model.model.__class__.__module__.startswith("_nf_text_")
    assert output.shape == (2, 8, 1) and torch.isfinite(output).all()
    output.square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert batch[key].grad is not None and batch[key].grad.abs().sum() > 0
    if cls is TGForecaster:
        # Both news and description must affect the forecast, including values
        # between patch starts (no silent every-fourth-row subsampling).
        gradient = batch[key].grad[:, 16:]
        assert gradient[..., :8].abs().sum() > 0
        assert gradient[..., 8:].abs().sum() > 0
        assert gradient[:, 1::4].abs().sum() > 0
    changed = dict(batch, **{key: batch[key].detach() * -2})
    assert not torch.allclose(output, model(changed))
    batch["outsample_y"] = torch.full((2, 8, 1), float("nan"))
    batch["future_target"] = torch.full((2, 8, 1), 1e12)
    torch.testing.assert_close(output, model(batch))


@pytest.mark.parametrize("cls", [SpecTF, TGForecaster])
def test_constant_history_and_state_dict_roundtrip(cls, sources):
    first = make_model(cls, sources).eval()
    batch = windows(cls)
    batch["insample_y"].fill_(3.0)
    output = first(batch)
    assert torch.isfinite(output).all()
    second = make_model(cls, sources).eval()
    second.load_state_dict(first.state_dict(), strict=True)
    torch.testing.assert_close(output, second(batch))


@pytest.mark.parametrize("cls", [SpecTF, TGForecaster])
def test_missing_nonfinite_and_wrong_shape_fail(cls, sources):
    model = make_model(cls, sources).eval()
    batch = windows(cls)
    batch["insample_mask"][:, 0] = 0
    with pytest.raises(ValueError, match="complete history"):
        model(batch)
    key = "hist_exog" if cls is SpecTF else "futr_exog"
    batch = windows(cls)
    batch[key][0, -1, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(batch)
    batch = windows(cls)
    batch[key] = batch[key][:, :-1]
    with pytest.raises(ValueError, match="shape"):
        model(batch)


@pytest.mark.parametrize("cls", [SpecTF, TGForecaster])
@pytest.mark.parametrize("options", [
    {"scaler_type": "standard"}, {"start_padding_enabled": True},
    {"training_data_availability_threshold": 0.5}, {"h": 0},
    {"stat_exog_list": ["static"]},
])
def test_reject_unsupported_configurations(cls, sources, options):
    with pytest.raises(Exception):
        make_model(cls, sources, **options)


def test_model_specific_schema_validation(sources):
    for options in ({"hist_exog_list": []}, {"mm_emb_size": 3}, {"input_size": 10000},
                    {"futr_exog_list": ["future"]}):
        with pytest.raises(Exception):
            make_model(SpecTF, sources, **options)
    for options in ({"futr_exog_list": ["wrong"]}, {"h": 7}, {"input_size": 15},
                    {"n_heads": 3}, {"hist_exog_list": ["past"]}):
        with pytest.raises(Exception):
            make_model(TGForecaster, sources, **options)


def test_tg_forecaster_patch_pooling(sources):
    model = make_model(TGForecaster, sources).eval()
    batch = windows(TGForecaster)
    captured = []
    handle = model.model.register_forward_pre_hook(lambda _, args: captured.append(args))
    model(batch)
    handle.remove()
    _, news, description, news_mask = captured[0]
    expected = batch["futr_exog"][:, 16:].reshape(2, 2, 4, 16).mean(2)
    torch.testing.assert_close(news.squeeze(2), expected[..., :8])
    torch.testing.assert_close(description.squeeze(2), expected[..., 8:])
    assert not news_mask.any()


def test_source_imports_do_not_replace_global_packages(sources):
    before = {key: sys.modules.get(key) for key in ("layers", "utils", "models")}
    path = list(sys.path)
    meta = list(sys.meta_path)
    first = make_model(SpecTF, sources)
    second = make_model(TGForecaster, sources)
    assert first.model.__class__.__module__ != second.model.__class__.__module__
    assert {key: sys.modules.get(key) for key in before} == before
    assert sys.path == path and sys.meta_path == meta


def test_dependency_tamper_is_rejected_even_after_cached_load(sources, tmp_path):
    for relative in _FILES["TGForecaster"]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources / "TGForecaster" / relative, destination)
    text_source(str(tmp_path), "TGForecaster")
    dependency = tmp_path / "layers/PatchTST_layers.py"
    dependency.write_bytes(dependency.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="reviewed blob"):
        text_source(str(tmp_path), "TGForecaster")


def test_neuralforecast_fit_predict_and_cold_reload(sources, tmp_path):
    generator = np.random.default_rng(31)
    n = 56
    frame = pd.DataFrame({
        "unique_id": np.repeat(["a", "b"], n),
        "ds": np.tile(pd.date_range("2024-01-01", periods=n), 2),
        "y": np.tile(np.sin(np.arange(n) / 5), 2),
    })
    for name in [f"emb{i}" for i in range(4)] + [f"text{i}" for i in range(16)]:
        frame[name] = generator.normal(size=len(frame))
    nf = NeuralForecast(models=[make_model(cls, sources) for cls in (SpecTF, TGForecaster)], freq="D")
    nf.fit(df=frame, val_size=8)
    for model in nf.models:
        assert model.train_trajectories and all(np.isfinite(v) for _, v in model.train_trajectories)
    future = nf.make_future_dataframe()
    for i in range(16):
        future[f"text{i}"] = generator.normal(size=len(future))
    output = nf.predict(futr_df=future)
    names = ["SpecTF", "TGForecaster"]
    assert len(output) == 16 and np.isfinite(output[names]).all().all()
    checkpoint = tmp_path / "checkpoint"
    nf.save(path=str(checkpoint), overwrite=True, save_dataset=True)
    future.to_csv(tmp_path / "future.csv", index=False)
    # A fresh interpreter proves persistence does not depend on cached imports.
    program = (
        "import sys, numpy as np, pandas as pd, torch; "
        "from neuralforecast import NeuralForecast; torch.set_num_threads(1); "
        "root=sys.argv[1]; nf=NeuralForecast.load(path=root+'/checkpoint'); "
        "future=pd.read_csv(root+'/future.csv',parse_dates=['ds']); "
        "p=nf.predict(futr_df=future); "
        "np.save(root+'/reloaded.npy',p[['SpecTF','TGForecaster']].to_numpy())"
    )
    subprocess.run([sys.executable, "-c", program, str(tmp_path)], check=True,
                   capture_output=True, text=True, timeout=90)
    np.testing.assert_allclose(output[names], np.load(tmp_path / "reloaded.npy"), rtol=1e-5, atol=1e-5)
    assert nf.models[0].hparams["source_dir"] == str((sources / "SpecTF").resolve())
