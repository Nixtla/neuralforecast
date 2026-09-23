"""Backward-compatible imports for text-conditioned forecasting adapters."""

from .spectf import SpecTF
from .tgforecaster import TGForecaster

__all__ = ["SpecTF", "TGForecaster"]
