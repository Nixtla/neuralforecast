from neuralforecast import models
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
