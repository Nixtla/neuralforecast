"""Regression checks for every exported Auto search-space family."""

import importlib
import inspect
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
import pytest
import torch
from ray import tune
from ray.tune.search.sample import Domain

from neuralforecast import auto, auto_external
from neuralforecast.common._base_auto import BaseAuto
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models.gpt4mts import GPT4MTS
from tests.test_auto_external import _autos

NATIVE = [
    name for name in auto.__all__
    if name.startswith("Auto") and hasattr(getattr(auto, name), "default_config")
]
MULTIVARIATE = [name for name in NATIVE if "n_series" in getattr(auto, name).default_config]
CONTEXT_NOOPS = ["AutoRNN", "AutoLSTM", "AutoGRU", "AutoTCN", "AutoDilatedRNN"]
ADDED_ARCHITECTURE = {
    "AutoRNN": "encoder_dropout encoder_bias decoder_layers windows_batch_size scaler_type",
    "AutoLSTM": "encoder_dropout encoder_bias decoder_layers windows_batch_size scaler_type",
    "AutoGRU": "encoder_dropout encoder_bias decoder_layers windows_batch_size scaler_type",
    "AutoTCN": "kernel_size dilations encoder_activation decoder_layers windows_batch_size",
    "AutoDeepAR": "decoder_hidden_layers decoder_hidden_size",
    "AutoDilatedRNN": "decoder_layers windows_batch_size scaler_type",
    "AutoxLSTM": "encoder_bias decoder_layers decoder_dropout decoder_activation",
    "AutoNBEATS": "n_blocks mlp_units activation shared_weights n_harmonics n_basis basis",
    "AutoNBEATSx": "n_blocks mlp_units activation shared_weights n_harmonics n_polynomials dropout_prob_theta",
    "AutoNHITS": "n_blocks mlp_units activation pooling_mode interpolation_mode dropout_prob_theta",
    "AutoDeepNPTS": "batch_norm",
    "AutoKAN": "n_hidden_layers scale_noise scale_base scale_spline grid_range",
    "AutoTFT": "dropout attn_dropout n_rnn_layers rnn_type grn_activation",
    "AutoVanillaTransformer": "dropout encoder_layers decoder_layers conv_hidden_size activation decoder_input_size_multiplier",
    "AutoInformer": "dropout encoder_layers decoder_layers conv_hidden_size activation factor distil",
    "AutoAutoformer": "dropout encoder_layers decoder_layers conv_hidden_size activation factor MovingAvg_window",
    "AutoFEDformer": "dropout encoder_layers decoder_layers conv_hidden_size activation modes mode_select MovingAvg_window",
    "AutoPatchTST": "encoder_layers linear_hidden_size dropout head_dropout attn_dropout stride activation res_attention batch_normalization learn_pos_embed",
    "AutoiTransformer": "e_layers d_ff dropout use_norm windows_batch_size",
    "AutoTimeXer": "e_layers d_ff dropout use_norm patch_len windows_batch_size",
    "AutoTimesNet": "encoder_layers dropout top_k num_kernels",
    "AutoStemGNN": "dropout_rate leaky_rate windows_batch_size",
    "AutoTSMixer": "revin windows_batch_size",
    "AutoTSMixerx": "revin windows_batch_size",
    "AutoMLPMultivariate": "windows_batch_size",
    "AutoSOFTS": "e_layers d_ff dropout use_norm windows_batch_size",
    "AutoSOFTSSharp": "e_layers d_ff dropout use_norm windows_batch_size",
    "AutoTimeMixer": "dropout e_layers moving_avg channel_independence down_sampling_window down_sampling_method use_norm windows_batch_size",
    "AutoRMoK": "dropout revin_affine windows_batch_size",
    "AutoXLinear": "temporal_ff channel_ff temporal_dropout channel_dropout embed_dropout head_dropout windows_batch_size",
}


def _native(name, h, backend="ray", config=None):
    kwargs = {"h": h, "backend": backend}
    if name in MULTIVARIATE:
        kwargs["n_series"] = 3
    if config is not None:
        kwargs["config"] = config
    return getattr(auto, name)(**kwargs)


def _sample(config, seed):
    rng = np.random.RandomState(seed)
    return {
        key: value.sample(random_state=rng) if isinstance(value, Domain) else value
        for key, value in config.items()
    }


def _batch(model):
    n_series = model.n_series if model.MULTIVARIATE else 1
    length = max(model.input_size + model.h + 8, 32)
    temporal = torch.cat(
        (torch.randn(n_series, 1, length), torch.ones(n_series, 1, length)), dim=1
    )
    return {"temporal": temporal, "temporal_cols": pd.Index(["y", "available_mask"]), "y_idx": 0}


def test_all_auto_declarations_are_accounted_for():
    names = [name for name in auto.__all__ if name.startswith("Auto")]
    assert len(NATIVE) == 35
    assert len(auto_external.__all__) == 15
    assert set(names) == set(NATIVE) | set(auto_external.__all__) | {"AutoHINT"}


@pytest.mark.parametrize("name", NATIVE)
def test_native_defaults_and_missing_architecture_controls(name):
    model = _native(name, 12)
    config = model.config
    parameters = inspect.signature(model.cls_model).parameters
    assert set(config) <= set(parameters)
    assert isinstance(config["num_lr_decays"], Domain)
    for key in ADDED_ARCHITECTURE.get(name, "").split():
        assert isinstance(config[key], Domain), (name, key)
    for seed in range(20):
        sample = _sample(config, seed)
        assert type(sample["max_steps"]) is int
        assert sample["max_steps"] >= 500
        assert not any(isinstance(value, Domain) for value in sample.values())
    if name in MULTIVARIATE:
        assert getattr(auto, name).default_config["batch_size"] is None
        assert config["batch_size"] == config["n_series"] == 3


@pytest.mark.parametrize("name", NATIVE)
@pytest.mark.parametrize("h", [1, 2, 7, 12])
def test_native_spaces_convert_to_optuna_without_expanding_integer_bounds(name, h):
    ray_model = _native(name, h)
    optuna_model = _native(name, h, backend="optuna")
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
    trial = study.ask()
    sample = optuna_model.config(trial)
    assert type(sample["max_steps"]) is int
    for key, domain in ray_model.config.items():
        if isinstance(domain, tune.search.sample.Integer):
            assert domain.lower <= sample[key] < domain.upper
            assert trial.distributions[key].high == domain.upper - 1
        elif isinstance(domain, tune.search.sample.Categorical):
            assert sample[key] in domain.categories
        elif key not in ("loss", "valid_loss") and not isinstance(domain, Domain):
            assert sample[key] == domain


def test_integer_conversion_uses_exclusive_ray_upper_bound():
    study = optuna.create_study()
    trial = study.ask()
    convert = BaseAuto._ray_config_to_optuna({"layers": tune.randint(1, 4)})
    convert(trial)
    assert trial.distributions["layers"].high == 3


@pytest.mark.parametrize("name", CONTEXT_NOOPS)
def test_deprecated_context_is_none_and_has_no_output_effect(name):
    wrapper = _native(name, 2)
    assert wrapper.config["context_size"] is None
    assert wrapper.config["inference_input_size"] is None
    assert "inference_input_size_multiplier" not in wrapper.default_config
    outputs = []
    for context in (None, 5, 50):
        model = wrapper.cls_model(h=2, input_size=16, context_size=context, random_seed=7)
        model.eval()
        torch.manual_seed(21)
        window = {
            "insample_y": torch.randn(2, 16, 1),
            "insample_mask": torch.ones(2, 16, 1),
            "hist_exog": None, "futr_exog": None, "stat_exog": None,
        }
        outputs.append(model(window).detach())
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)


@pytest.mark.parametrize("name", MULTIVARIATE)
def test_multivariate_none_batch_supports_default_config_copy(name):
    wrapper = _native(name, 12, config=deepcopy(getattr(auto, name).default_config))
    config = _sample(wrapper.config, 0)
    # Isolate series-batch semantics from patch/window constraints.
    config.update(input_size=32, max_steps=1, windows_batch_size=2)
    model = wrapper.cls_model(**deepcopy(config))
    assert model.batch_size == model.n_series == 3


@pytest.mark.parametrize("name", ["AutoNBEATS", "AutoNBEATSx"])
def test_one_step_basis_search_is_disabled(name):
    config = _native(name, 1).config
    assert config["stack_types"] == ["identity"] * 3
    assert config["n_harmonics"] is None
    assert config["n_basis" if name == "AutoNBEATS" else "n_polynomials"] is None


@pytest.mark.parametrize("h,seed", [(1, 0), (2, 1), (12, 2)])
@pytest.mark.parametrize("name", NATIVE)
def test_native_sampled_configuration_forward_backward(name, h, seed):
    wrapper = _native(name, h)
    config = _sample(wrapper.config, seed)
    # Bound test cost; leave all sampled architecture/input choices intact.
    config.update(max_steps=2, windows_batch_size=2, logger=False, enable_progress_bar=False)
    if name == "AutoxLSTM":
        pytest.importorskip("xlstm")
        pytest.importorskip("mlstm_kernels")
    model = wrapper.cls_model(**deepcopy(config))
    loss = model.training_step(_batch(model), 0)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
    model.eval()
    model.val_size = h + 1
    with torch.no_grad():
        assert torch.isfinite(model.validation_step(_batch(model), 0))


@pytest.mark.parametrize("wrapper", _autos(), ids=lambda model: type(model).__name__)
def test_every_external_space_keeps_schema_and_converts(wrapper):
    config = wrapper.config
    assert isinstance(config["num_lr_decays"], Domain)
    converter = BaseAuto._ray_config_to_optuna(config)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=11))
    sample = converter(study.ask())
    for name in ("h", "source_dir", "backbone_path", "hist_exog_list", "futr_exog_list", "stat_exog_list"):
        if name in config:
            assert sample[name] == config[name]
    assert sample["random_seed"] == 1
    assert sample["num_lr_decays"] in (0, 1, 3)


@pytest.mark.parametrize("name", ["AutoCrossLinear", "AutoTimerXL"])
def test_native_external_ports_forward_backward(name):
    wrapper = getattr(auto_external, name)(h=2)
    config = _sample(wrapper.config, 11)
    config.update(max_steps=2, windows_batch_size=2, logger=False, enable_progress_bar=False)
    model = wrapper.cls_model(**deepcopy(config))
    loss = model.training_step(_batch(model), 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.parameters())


def test_gpt4mts_search_heads_follow_checkpoint_mode():
    kwargs = {"h": 2, "source_dir": "/unused", "hist_exog_list": [f"text{i}" for i in range(8)]}
    scratch = auto_external.AutoGPT4MTS(**kwargs).config
    pretrained = auto_external.AutoGPT4MTS(**kwargs, backbone_path="/weights").config
    assert isinstance(scratch["n_heads"], Domain)
    assert pretrained["n_heads"] is None
    assert pretrained["gpt_layers"] == 2


def test_gpt4mts_none_heads_resolves_real_checkpoint_config(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.GPT2Config(n_embd=12, n_head=3, n_layer=2, n_positions=256)
    cfg.save_pretrained(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"configuration-only test")
    received = {}

    class Capture(torch.nn.Module):
        def __init__(self, args, device):
            super().__init__()
            received.update(args.backbone_config)
            self.gpt2 = SimpleNamespace(config=cfg)

    module = importlib.import_module("neuralforecast.models.gpt4mts")
    monkeypatch.setattr(module, "official_module", lambda *args: SimpleNamespace(GPT4MTS=Capture))
    GPT4MTS(h=2, input_size=32, source_dir=str(tmp_path), backbone_path=str(tmp_path),
            hist_exog_list=[f"text{i}" for i in range(12)], n_heads=None)
    assert received["n_head"] == 3
    assert received["n_embd"] == 12
    with pytest.raises(ValueError, match="n_heads"):
        GPT4MTS(h=2, input_size=32, source_dir=str(tmp_path),
                hist_exog_list=[f"text{i}" for i in range(12)], n_heads=None)


def test_custom_spaces_and_hint_remain_user_controlled():
    config = {"input_size": 16, "encoder_hidden_size": 17, "max_steps": 2}
    wrapper = _native("AutoGRU", 2, config=deepcopy(config))
    assert set(wrapper.config) == set(config) | {"h", "loss", "valid_loss"}
    assert wrapper.config["encoder_hidden_size"] == 17
    hint_config = {"input_size": 16, "hidden_size": 16, "num_layers": 2,
                   "max_steps": 2, "reconciliation": tune.choice(["BottomUp", "MinTraceOLS"])}
    hint = auto.AutoHINT(h=2, cls_model=auto.MLP, loss=MAE(), valid_loss=MAE(),
                         S=np.array([[1., 1.], [1., 0.], [0., 1.]]), config=hint_config)
    assert hint.config["reconciliation"].categories == ["BottomUp", "MinTraceOLS"]
    assert "num_lr_decays" not in hint.config


@pytest.mark.parametrize("ratio", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("input_size", [12, 13, 32])
def test_fedformer_fourier_decoder_matches_configured_context(ratio, input_size):
    wrapper = _native("AutoFEDformer", 12)
    config = deepcopy(_sample(wrapper.config, 0))
    config.update(input_size=input_size, decoder_input_size_multiplier=ratio,
                  modes=64, mode_select="low", windows_batch_size=2, max_steps=2)
    model = wrapper.cls_model(**config)
    expected_modes = min(config["modes"], (model.label_len + model.h) // 2)
    attention = model.decoder.layers[0].self_attention.inner_correlation
    assert len(attention.index) == expected_modes
    loss = model.training_step(_batch(model), 0)
    assert torch.isfinite(loss)
    loss.backward()


def test_ray_trial_preserves_explicit_none_batch_sentinels(monkeypatch):
    wrapper = _native("AutoTimeXer", 2)
    received = {}
    def capture(**kwargs):
        received.update(kwargs["config"])
    monkeypatch.setattr(wrapper, "_fit_model", capture)
    config = _sample(wrapper.config, 0)
    config.update(batch_size=None, windows_batch_size=None)
    wrapper._train_tune(config, wrapper.cls_model, dataset=None, val_size=2, test_size=0)
    assert received["batch_size"] is None
    assert received["windows_batch_size"] is None
