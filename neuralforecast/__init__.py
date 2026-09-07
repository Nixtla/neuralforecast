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
    "CrossLinear", "TimerXL", "TinyTimeMixer", "Chronos2",
    "Moirai", "MoiraiMoE", "TimesFM", "Toto",
):
    _model_filename_dict[_model_name.lower()] = getattr(_models, _model_name)
del _models, _model_filename_dict, _model_name
