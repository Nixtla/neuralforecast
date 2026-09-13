import pytest

from neuralforecast.auto_external import (
    AutoAPT,
    AutoCrossLinear,
    AutoDAG,
    AutoDualformer,
    AutoGLAFF,
    AutoGPT4MTS,
    AutoKITE,
    AutoLangTime,
    AutoSeesawNet,
    AutoSpecTF,
    AutoTGForecaster,
    AutoTinyTimeMixer,
    AutoTimerXL,
    AutoUniTime,
    AutoVoT,
)
from neuralforecast.inference_tuning import (
    INFERENCE_TUNING_MODELS,
    get_inference_tuning_config,
)


class _FirstTrial:
    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_int(self, name, low, high, **kwargs):
        return low

    def suggest_uniform(self, name, low, high):
        return low

    def suggest_loguniform(self, name, low, high):
        return low

    def suggest_float(self, name, low, high, **kwargs):
        return low


def _autos():
    text_hist = [f"text_{i}" for i in range(8)]
    tg_future = [f"news_{i}" for i in range(4)] + [
        f"description_{i}" for i in range(4)
    ]
    return [
        AutoCrossLinear(h=12),
        AutoTimerXL(h=12),
        AutoTinyTimeMixer(h=12),
        AutoDAG(h=12, source_dir="/src/dag", futr_exog_list=["x"]),
        AutoKITE(h=12, source_dir="/src/kite", hist_exog_list=["x"]),
        AutoGLAFF(
            h=12,
            source_dir="/src/glaff",
            futr_exog_list=[f"calendar_{i}" for i in range(6)],
        ),
        AutoAPT(
            h=12,
            source_dir="/src/apt",
            futr_exog_list=["time_of_day", "day_of_week"],
        ),
        AutoVoT(h=12, source_dir="/src/vot", hist_exog_list=text_hist),
        AutoGPT4MTS(
            h=12,
            source_dir="/src/gpt4mts",
            hist_exog_list=text_hist,
        ),
        AutoUniTime(
            h=12,
            source_dir="/src/unitime",
            backbone_path="/weights/gpt2",
            contexts=["energy"],
            stat_exog_list=["context_id"],
        ),
        AutoLangTime(
            h=12,
            source_dir="/src/langtime",
            backbone_path="/weights/gpt2",
            contexts=["energy"],
            stat_exog_list=["context_id"],
        ),
        AutoSpecTF(
            h=12,
            source_dir="/src/spectf",
            hist_exog_list=text_hist,
        ),
        AutoTGForecaster(
            h=12,
            source_dir="/src/tg",
            futr_exog_list=tg_future,
        ),
        AutoSeesawNet(h=12, source_dir="/src/seesaw"),
        AutoDualformer(h=12, source_dir="/src/dualformer"),
    ]


def _inference_fixed():
    return {
        "Chronos2": {},
        "Moirai": {"backend_python": "/env/bin/python"},
        "MoiraiMoE": {"backend_python": "/env/bin/python"},
        "TimesFM": {},
        "Toto": {},
        "Moirai2": {"backend_python": "/env/bin/python"},
        "ChronosX": {
            "input_size": 96,
            "model_id": "/weights/chronosx",
            "backend_python": "/env/bin/python",
            "hidden_dim": 256,
            "num_layers": 1,
            "hist_exog_list": ["inventory"],
        },
        "BaguanTS": {
            "input_size": 96,
            "source_dir": "/src/baguants",
            "config_path": "/weights/baguants.yaml",
            "model_id": "/weights/baguants.pt",
            "backend_python": "/env/bin/python",
            "futr_exog_list": ["inventory"],
        },
        "RAG4CTS": {
            "input_size": 96,
            "source_dir": "/src/rag4cts",
            "futr_exog_list": ["inventory"],
        },
        "TimesFM3": {},
        "Aurora": {
            "source_dir": "/src/aurora",
            "model_id": "/weights/aurora",
            "tokenizer_path": "/weights/tokenizer",
            "contexts": ["energy"],
            "stat_exog_list": ["context_id"],
        },
        "ChatTime": {
            "source_dir": "/src/chattime",
            "model_id": "/weights/chattime",
            "contexts": ["energy"],
            "stat_exog_list": ["context_id"],
        },
        "TabPFNTS": {
            "model_id": "/weights/tabpfn.ckpt",
            "futr_exog_list": ["inventory"],
        },
    }


def test_all_external_auto_wrappers_build_search_configs():
    autos = _autos()
    assert len(autos) == 15
    for model in autos:
        assert model.config["h"] == 12
        assert model.config["random_seed"] == 1
        assert model.config["early_stop_patience_steps"] == 5
        assert "input_size" in model.config


def test_external_auto_optuna_config_keeps_fixed_values():
    model = AutoDualformer(
        h=12,
        source_dir="/src/dualformer",
        hist_exog_list=["inventory"],
        backend="optuna",
    )
    config = model.config(_FirstTrial())
    assert config["source_dir"] == "/src/dualformer"
    assert config["hist_exog_list"] == ["inventory"]
    assert config["random_seed"] == 1


def test_external_auto_schema_guards_fail_before_hpo():
    with pytest.raises(ValueError, match="exactly six"):
        AutoGLAFF(h=12, source_dir="/src/glaff", futr_exog_list=["month"])
    with pytest.raises(ValueError, match="exactly one"):
        AutoKITE(
            h=12,
            source_dir="/src/kite",
            hist_exog_list=["past"],
            futr_exog_list=["future"],
        )
    with pytest.raises(ValueError, match="paired"):
        AutoTGForecaster(
            h=12,
            source_dir="/src/tg",
            futr_exog_list=["a", "b", "c"],
        )


@pytest.mark.parametrize("name", INFERENCE_TUNING_MODELS)
def test_every_inference_model_builds_a_separate_space(name):
    config = get_inference_tuning_config(
        name,
        h=12,
        fixed=_inference_fixed()[name],
    )
    assert config["max_steps"] == 0
    assert "learning_rate" not in config


def test_inference_models_have_separate_spaces():
    assert len(INFERENCE_TUNING_MODELS) == 13
    config = get_inference_tuning_config("Chronos2", h=12)
    assert config["max_steps"] == 0
    assert "learning_rate" not in config
    assert "input_size" in config


def test_inference_spaces_preserve_fixed_checkpoint_contracts():
    moirai2 = get_inference_tuning_config(
        "Moirai2",
        h=12,
        fixed={"backend_python": "/env/bin/python"},
    )
    assert moirai2["patch_size"] == 16
    assert moirai2["num_samples"] == 100

    with pytest.raises(ValueError, match="fixed values"):
        get_inference_tuning_config("ChronosX", h=12, fixed={})


def test_rag4cts_space_respects_strictly_historical_bank():
    config = get_inference_tuning_config(
        "RAG4CTS",
        h=12,
        fixed={
            "input_size": 96,
            "source_dir": "/src/rag4cts",
            "futr_exog_list": ["x"],
        },
    )
    for query_size in config["query_size"].categories:
        assert 96 >= 2 * query_size + 12


def test_inference_optuna_space_is_callable():
    config = get_inference_tuning_config(
        "TimesFM3",
        h=12,
        backend="optuna",
    )(_FirstTrial())
    assert config["max_steps"] == 0
    assert config["input_size"] >= 12
