"""Moirai-MoE pretrained forecasting adapter."""

from .moirai import Moirai

__all__ = ["MoiraiMoE"]


class MoiraiMoE(Moirai):
    """Moirai-MoE-1.0 using the same isolated covariate interface as Moirai.

    This loads MoiraiMoEModule/MoiraiMoEForecast, not dense Moirai weights.
    """

    DEFAULT_MODEL_ID = "Salesforce/moirai-moe-1.0-R-small"
    BACKEND_KIND = "moirai_moe"
