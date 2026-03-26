from importlib.metadata import version

__version__ = version("neuralforecast")
__all__ = ['NeuralForecast']
from .common._base_model import DistributedConfig  # noqa: F401
from .core import NeuralForecast
