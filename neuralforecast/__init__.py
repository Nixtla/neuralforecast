__version__ = "1.7.4"
__all__ = ['NeuralForecast']
from .core import NeuralForecast
from .common._base_model import DistributedConfig  # noqa: F401
