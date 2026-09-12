import logging
from importlib.metadata import version

__version__ = version("neuralforecast")
__all__ = ['NeuralForecast']

# Suppress PyTorch Lightning's "💡 Tip:" promos for LitLogger / cloud uploads,
# emitted via rank_zero_info on the `pytorch_lightning.utilities.rank_zero`
# logger every time a Trainer is constructed.
class _DropLightningTips(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("\U0001f4a1 Tip:")

logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(
    _DropLightningTips()
)

from .common._base_model import DistributedConfig  # noqa: F401, E402
from .core import NeuralForecast  # noqa: E402

# Register optional adapters after core is initialized, without importing their
# external packages. This keeps NeuralForecast.save/load's filename lookup valid.
from . import models as _models  # noqa: E402
from .core import MODEL_FILENAME_DICT as _model_filename_dict  # noqa: E402

for _model_name in (
    "VoT", "GPT4MTS", "UniTime", "LangTime", "Aurora", "ChatTime", "TabPFNTS",
    "CrossLinear", "TimerXL", "TinyTimeMixer", "Chronos2",
    "Moirai", "MoiraiMoE", "TimesFM", "Toto",
    "DAG", "KITE", "GLAFF", "APT", "Moirai2", "ChronosX", "BaguanTS", "RAG4CTS",
    "SpecTF", "TGForecaster",
    "TimesFM3", "SeesawNet", "Dualformer", "SearchCast",
):
    _model_filename_dict[_model_name.lower()] = getattr(_models, _model_name)

for _model_name in (
    "CrossLinear", "TimerXL", "TinyTimeMixer", "DAG", "KITE", "GLAFF", "APT",
    "VoT", "GPT4MTS", "UniTime", "LangTime", "SpecTF", "TGForecaster",
    "SeesawNet", "Dualformer",
):
    _model_filename_dict[f"auto{_model_name.lower()}"] = getattr(_models, _model_name)

# Keep the standard `from neuralforecast.auto import Auto...` API for the fork's
# external Auto wrappers while their implementation remains isolated by concern.
from . import auto as _auto  # noqa: E402
from . import auto_external as _auto_external  # noqa: E402

for _auto_name in _auto_external.__all__:
    setattr(_auto, _auto_name, getattr(_auto_external, _auto_name))
    if _auto_name not in _auto.__all__:
        _auto.__all__.append(_auto_name)

del _models, _model_filename_dict, _model_name, _auto, _auto_external, _auto_name
