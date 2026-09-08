"""Official-source integration checks; see each test for any controlled component.

Set NF_CONTEXT_SOURCE_ROOT to pinned checkouts from fetch_context_sources.py.
No published checkpoint or model-accuracy claim is made by tiny synthetic tests.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import Aurora, ChatTime, GPT4MTS, LangTime, TabPFNTS, UniTime, VoT
from neuralforecast.models._context_source import SOURCES, official_module

TRAINABLE = [VoT, GPT4MTS, UniTime, LangTime]


@pytest.fixture(scope="module")
def sources():
    root = os.environ.get("NF_CONTEXT_SOURCE_ROOT")
    if not root:
        pytest.skip("Set NF_CONTEXT_SOURCE_ROOT for actual-source integration tests.")
    path = Path(root)
    assert all((path / name).is_dir() for name in SOURCES), "Missing reviewed checkouts"
    torch.set_num_threads(1)
    return path


@pytest.fixture(scope="module")
def gpt2_path(tmp_path_factory):
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    path = tmp_path_factory.mktemp("tiny-gpt2")
    vocab = {c: i for i, c in enumerate(bytes_to_unicode().values())}
    vocab["<|endoftext|>"] = 256
    (path / "vocab.json").write_text(json.dumps(vocab))
    (path / "merges.txt").write_text("#version: 0.2\n")
    GPT2Tokenizer(str(path / "vocab.json"), str(path / "merges.txt")).save_pretrained(path)
    torch.manual_seed(42)
    GPT2LMHeadModel(GPT2Config(vocab_size=257, n_positions=256, n_embd=32,
        n_head=4, n_layer=1, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).save_pretrained(path)
    return str(path)


def make_model(cls, sources, gpt2_path, **overrides):
    args = dict(h=4, input_size=16, source_dir=str(sources / cls.__name__), max_steps=2,
                windows_batch_size=2, logger=False, enable_progress_bar=False)
    if cls is VoT:
        args.update(hist_exog_list=[f"text_{i}" for i in range(8)], d_model=16,
                    n_heads=4, d_ff=32, e_layers=1, patch_len=4, stride=4, dropout=0)
    elif cls is GPT4MTS:
        args.update(hist_exog_list=[f"text_{i}" for i in range(32)], n_heads=4,
                    gpt_layers=1, patch_len=4, stride=4)
    else:
        args.update(backbone_path=gpt2_path, contexts=["external up", "external down"],
                    stat_exog_list=["context_id"], patch_len=4, dropout=0)
        if cls is UniTime:
            args.update(gpt_layers=1, max_tokens=64)
        else:
            args.update(d_model=16, n_heads=4, d_ff=32, e_layers=1)
    args.update(overrides)
    return cls(**args)


def window(model, count=2):
    torch.manual_seed(9)
    return dict(insample_y=torch.randn(count, model.input_size, 1),
                insample_mask=torch.ones(count, model.input_size, 1),
                hist_exog=torch.randn(count, model.input_size, model.hist_exog_size)
                if model.hist_exog_size else None,
                futr_exog=torch.randn(count, model.input_size + model.h, model.futr_exog_size)
                if model.futr_exog_size else None,
                stat_exog=(torch.arange(count) % 2).float().reshape(count, 1)
                if model.stat_exog_size else None)


def frames(model):
    frame = pd.DataFrame({"unique_id": np.repeat(["a", "b"], 32),
                          "ds": list(pd.date_range("2024-01-01", periods=32)) * 2,
                          "y": np.sin(np.arange(64) / 3)})
    for j, name in enumerate(model.hist_exog_list + model.futr_exog_list):
        frame[name] = np.cos(np.arange(64) / (j + 1) + j)
    static = pd.DataFrame({"unique_id": ["a", "b"], "context_id": [0., 1.]}) if model.stat_exog_size else None
    return frame, static


@pytest.mark.parametrize("cls", TRAINABLE)
def test_actual_forward_backward_condition_and_label_exclusion(cls, sources, gpt2_path):
    model = make_model(cls, sources, gpt2_path)
    batch = window(model)
    output = model(batch)
    assert output.shape == (2, 4, 1) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    model.eval()
    reference = model(batch)
    changed = dict(batch)
    if model.hist_exog_size:
        changed["hist_exog"] = batch["hist_exog"] + 1
    else:
        changed["stat_exog"] = 1 - batch["stat_exog"]
    assert not torch.allclose(reference, model(changed), atol=1e-7, rtol=1e-7)
    changed = {**batch, "outsample_y": torch.full((2, 4, 1), 1e9)}
    torch.testing.assert_close(reference, model(changed))
    # The first forecast is independent of unrelated/later windows in the batch.
    changed = {key: value.clone() if value is not None else None for key, value in batch.items()}
    changed["insample_y"][1] += 99
    if model.hist_exog_size:
        changed["hist_exog"][1] -= 99
    torch.testing.assert_close(reference[:1], model(changed)[:1])


@pytest.mark.parametrize("cls", TRAINABLE)
def test_actual_nf_fit_predict_reload_fresh_process(cls, sources, gpt2_path, tmp_path):
    model = make_model(cls, sources, gpt2_path)
    frame, static = frames(model)
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=frame, static_df=static)
    expected = nf.predict()[cls.__name__].to_numpy()
    checkpoint = tmp_path / "nf"
    nf.save(path=str(checkpoint), save_dataset=True)
    script = """import sys, numpy as np, torch
from neuralforecast import NeuralForecast
torch.set_num_threads(1)
nf=NeuralForecast.load(path=sys.argv[1])
np.save(sys.argv[2],nf.predict()[sys.argv[3]].to_numpy())
"""
    result = subprocess.run([sys.executable, "-c", script, str(checkpoint),
        str(tmp_path / "actual.npy"), cls.__name__], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    np.testing.assert_allclose(expected, np.load(tmp_path / "actual.npy"), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("cls", TRAINABLE)
def test_complete_history_and_input_schema(cls, sources, gpt2_path):
    model = make_model(cls, sources, gpt2_path)
    batch = window(model)
    batch["insample_mask"][0, 0] = 0
    with pytest.raises(ValueError, match="complete"):
        model(batch)
    with pytest.raises(ValueError, match="identity"):
        make_model(cls, sources, gpt2_path, scaler_type="standard")
    if model.stat_exog_size:
        for invalid in [float("nan"), -1., 0.5, 2.]:
            batch = window(model)
            batch["stat_exog"][0] = invalid
            with pytest.raises(ValueError, match="context_id"):
                model(batch)


@pytest.mark.parametrize("kind", list(SOURCES))
def test_modified_transitive_source_is_rejected(kind, sources, tmp_path):
    spec = SOURCES[kind]
    for relative in spec["files"]:
        dest = tmp_path / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources / kind / relative, dest)
    victim = tmp_path / next(iter(spec["files"]))
    victim.write_bytes(victim.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="Changed"):
        official_module(tmp_path, kind)


def test_gpt4mts_singleton_batch_singleton_patch(sources, gpt2_path):
    model = make_model(GPT4MTS, sources, gpt2_path, input_size=4, h=1)
    assert model(window(model, 1)).shape == (1, 1, 1)


def test_gpt4mts_local_language_checkpoint(sources, gpt2_path):
    model = make_model(GPT4MTS, sources, gpt2_path, backbone_path=gpt2_path, freeze_backbone=True)
    assert torch.isfinite(model(window(model, 1))).all()
    assert all(not p.requires_grad for name, p in model.model.gpt2.named_parameters()
               if "ln" not in name and "wpe" not in name)


def test_actual_aurora_checkpoint_and_text_condition(sources, tmp_path, monkeypatch):
    """Real tiny BERT/ViT/Aurora, trained four synthetic steps, not published weights."""
    from transformers import BertConfig, ViTConfig, ViTImageProcessor, BertTokenizerFast
    module = official_module(sources / "Aurora", "Aurora")
    connectors = sys.modules[module.__name__.rsplit(".", 1)[0] + ".modality_connector"]
    bert, vit = tmp_path / "bert", tmp_path / "vit"
    bert.mkdir(); vit.mkdir()
    BertConfig(vocab_size=12, hidden_size=16, intermediate_size=32,
               num_attention_heads=4, num_hidden_layers=1).to_json_file(bert / "config.json")
    ViTConfig(image_size=16, patch_size=8, hidden_size=16, intermediate_size=32,
              num_attention_heads=4, num_hidden_layers=1).to_json_file(vit / "config.json")
    ViTImageProcessor(size={"height": 16, "width": 16}).save_pretrained(vit)
    for cls, path in [(connectors.TextEncoder, bert), (connectors.VisionEncoder, vit),
                      (connectors.UnifiedImageProcessor, vit)]:
        monkeypatch.setattr(cls, "config_path", str(path))
    (bert / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ndemand\nup\ndown\nsupply\ntoday\nforecast\ntrend\n")
    tokenizer = BertTokenizerFast(vocab_file=str(bert / "vocab.txt"))
    tokenizer.save_pretrained(bert)
    torch.manual_seed(11)
    cfg = module.AuroraConfig(token_len=4, hidden_size=16, intermediate_size=32,
        num_enc_layers=1, num_dec_layers=1, num_attention_heads=4, num_sampling_steps=2,
        flow_loss_depth=1, num_prototypes=8, num_distill=2, dropout_rate=0)
    net = module.AuroraForPrediction(cfg)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    text = tokenizer(["demand up", "demand down"], return_tensors="pt", padding=True)
    history = torch.randn(2, 16)
    # Upstream initializes the flow output head to zero. Use its original loss,
    # not a fabricated prediction layer, before checking conditional forecasts.
    for _ in range(4):
        optimizer.zero_grad()
        loss = net(input_ids=history, labels=torch.randn(2, 4), inference_token_len=4,
                   **{"text_" + k: v for k, v in text.items()}).loss
        assert torch.isfinite(loss)
        loss.backward(); optimizer.step()
    checkpoint = tmp_path / "aurora"
    net.eval().save_pretrained(checkpoint)
    model = Aurora(h=4, input_size=16, source_dir=str(sources / "Aurora"),
        model_id=str(checkpoint), tokenizer_path=str(bert), contexts=["demand up", "demand down"],
        stat_exog_list=["context_id"], num_samples=2, inference_token_len=4)
    model._get_backend()  # Lazy construction consumes RNG; compare sampling after load.
    batch = window(model)
    torch.manual_seed(7); original = model(batch)
    torch.manual_seed(7); changed = model({**batch, "stat_exog": 1 - batch["stat_exog"]})
    assert original.shape == (2, 4, 1) and torch.isfinite(original).all()
    assert not torch.allclose(original, changed, atol=1e-7, rtol=1e-7)
    torch.manual_seed(7)
    torch.testing.assert_close(original, model({**batch, "outsample_y": torch.full((2, 4, 1), 1e9)}))
    # Reference persistence, not external-weight serialization.
    frame, static = frames(model)
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=frame, static_df=static)
    path = tmp_path / "nf"
    nf.save(path=str(path), save_dataset=True)
    loaded = NeuralForecast.load(path=str(path))
    loaded.models[0]._get_backend()
    torch.manual_seed(13); expected = nf.predict()["Aurora"].to_numpy()
    torch.manual_seed(13); actual = loaded.predict()["Aurora"].to_numpy()
    np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-6)


def test_actual_chattime_api_with_controlled_generation(sources, tmp_path, monkeypatch):
    """Actual Llama/tokenizer and ChatTime API; only generated responses are controlled."""
    import sentencepiece as spm
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("demand up demand down Forecast ### Response 0.1234 external " * 20) + "\n")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path / "tokenizer"),
        vocab_size=32, model_type="bpe", bos_id=1, eos_id=2, unk_id=0, minloglevel=2)
    tokenizer = LlamaTokenizer(vocab_file=str(tmp_path / "tokenizer.model"))
    checkpoint = tmp_path / "chattime"
    tokenizer.save_pretrained(checkpoint)
    LlamaForCausalLM(LlamaConfig(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4)).save_pretrained(checkpoint)
    module = official_module(sources / "ChatTime", "ChatTime")
    seen = []
    def controlled_generation(**kwargs):
        assert isinstance(kwargs["model"], LlamaForCausalLM)
        def predict(prompt):
            seen.append(prompt)
            value = "0.2" if "demand up" in prompt else "-0.2"
            return [{"generated_text": prompt + ("###" + value + "### ") * 4}
                    for _ in range(kwargs["num_return_sequences"])]
        return predict
    monkeypatch.setattr(module, "pipeline", controlled_generation)
    model = ChatTime(h=4, input_size=16, source_dir=str(sources / "ChatTime"),
        model_id=str(checkpoint), contexts=["demand up", "demand down"], stat_exog_list=["context_id"],
        num_samples=2)
    batch = window(model)
    original = model(batch)
    assert original.shape == (2, 4, 1) and torch.isfinite(original).all()
    assert all("### Response:\n" in p for p in seen)
    changed = model({**batch, "stat_exog": 1 - batch["stat_exog"]})
    assert not torch.allclose(original, changed)
    torch.testing.assert_close(original, model({**batch, "outsample_y": torch.full((2, 4, 1), 1e9)}))
    frame, static = frames(model)
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=frame, static_df=static)
    nf.save(path=str(tmp_path / "nf"), save_dataset=True)
    loaded = NeuralForecast.load(path=str(tmp_path / "nf"))
    np.testing.assert_allclose(nf.predict()["ChatTime"], loaded.predict()["ChatTime"])
    monkeypatch.setattr(module, "pipeline", lambda **kw: lambda prompt: [{"generated_text": prompt + "not numbers"}])
    with pytest.raises(ValueError):
        model(batch)


def test_tabpfnts_official_pipeline_local_only(tmp_path, monkeypatch):
    """Official preprocessing/worker/pipeline; gated TabPFN regressor methods are doubles."""
    if not os.environ.get("NF_TEST_TABPFN"):
        pytest.skip("NF_TEST_TABPFN=1 in dedicated optional-package CI")
    import tabpfn
    import tabpfn_client
    import tabpfn.model_loading
    import tabpfn_time_series
    seen = []
    def forbidden(*args, **kwargs):
        raise AssertionError("Network/cloud path must not be used")
    monkeypatch.setattr(tabpfn_client, "init", forbidden)
    monkeypatch.setattr(tabpfn.model_loading, "download_model", forbidden)
    def init(self, **kwargs):
        assert Path(kwargs["model_path"]).is_file()
        assert kwargs["device"] == "cpu"
    def fit(self, X, y, **kwargs):
        self._nf_y = np.asarray(y)
        seen.append((X.copy(), np.asarray(y).copy()))
        return self
    def predict(self, X, quantiles, **kwargs):
        assert "target" not in X.columns
        assert "covariate_0" in X.columns
        values = X["covariate_0"].to_numpy() + self._nf_y.mean()
        return {"median": values, "mean": values, "quantiles": [values for _ in quantiles]}
    monkeypatch.setattr(tabpfn.TabPFNRegressor, "__init__", init)
    monkeypatch.setattr(tabpfn.TabPFNRegressor, "fit", fit)
    monkeypatch.setattr(tabpfn.TabPFNRegressor, "predict", predict)
    checkpoint = tmp_path / "tabpfn-v3-regressor-v3_default.ckpt"
    checkpoint.write_bytes(b"explicit checkpoint path; regressor is controlled in this test")
    model = TabPFNTS(h=4, input_size=16, model_id=str(checkpoint), futr_exog_list=["schedule"])
    batch = window(model)
    original = model(batch)
    assert isinstance(model._get_backend(), tabpfn_time_series.TabPFNTSPipeline)
    assert original.shape == (2, 4, 1) and len(seen) == 2
    assert seen[0][0].shape[0] == 16
    changed = batch["futr_exog"].clone(); changed[:, 16:] += 2
    torch.testing.assert_close(model({**batch, "futr_exog": changed}), original + 2)
    torch.testing.assert_close(model({**batch, "outsample_y": torch.full((2, 4, 1), 1e9)}), original)
    # Per-window fit must not share any other window's labels.
    np.testing.assert_allclose(seen[0][1], batch["insample_y"][0, :, 0].numpy())
    frame, _ = frames(model)
    nf = NeuralForecast(models=[model], freq="D"); nf.fit(df=frame)
    future = nf.make_future_dataframe(); future["schedule"] = 0.25
    expected = nf.predict(futr_df=future)["TabPFNTS"].to_numpy()
    nf.save(path=str(tmp_path / "nf"), save_dataset=True)
    loaded = NeuralForecast.load(path=str(tmp_path / "nf"))
    np.testing.assert_allclose(expected, loaded.predict(futr_df=future)["TabPFNTS"])


@pytest.mark.parametrize("cls", [Aurora, ChatTime, TabPFNTS])
def test_frozen_adapters_refuse_implicit_checkpoints(cls, tmp_path):
    kwargs = dict(h=4, input_size=16, model_id="online/model")
    if cls is not TabPFNTS:
        kwargs.update(source_dir=str(tmp_path), contexts=["description"], stat_exog_list=["context_id"])
        if cls is Aurora:
            kwargs["tokenizer_path"] = str(tmp_path)
    else:
        kwargs["futr_exog_list"] = ["schedule"]
    with pytest.raises((ValueError, FileNotFoundError)):
        cls(**kwargs)


def test_optional_packages_not_imported_in_clean_nf_process():
    script = """import sys
import neuralforecast
assert not any(p in sys.modules for p in ('tabpfn', 'tabpfn_client', 'tabpfn_time_series'))
from neuralforecast.models import VoT, GPT4MTS, UniTime, LangTime, Aurora, ChatTime, TabPFNTS
from neuralforecast.core import MODEL_FILENAME_DICT
assert all(MODEL_FILENAME_DICT[c.__name__.lower()] is c for c in (VoT,GPT4MTS,UniTime,LangTime,Aurora,ChatTime,TabPFNTS))
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
