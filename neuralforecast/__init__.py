__version__ = "2.0.0"
__all__ = ['NeuralForecast']
from .core import NeuralForecast
from .common._base_model import DistributedConfig  # noqa: F401
