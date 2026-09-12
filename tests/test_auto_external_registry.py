from neuralforecast import auto, models
from neuralforecast.auto_external import AutoDualformer
from neuralforecast.core import MODEL_FILENAME_DICT


def test_external_auto_names_load_as_underlying_models():
    names = (
        "CrossLinear",
        "TimerXL",
        "TinyTimeMixer",
        "DAG",
        "KITE",
        "GLAFF",
        "APT",
        "VoT",
        "GPT4MTS",
        "UniTime",
        "LangTime",
        "SpecTF",
        "TGForecaster",
        "SeesawNet",
        "Dualformer",
    )
    for name in names:
        assert MODEL_FILENAME_DICT[f"auto{name.lower()}"] is getattr(models, name)


def test_external_auto_models_are_available_from_standard_auto_module():
    assert auto.AutoDualformer is AutoDualformer
    assert "AutoDualformer" in auto.__all__
