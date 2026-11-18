from enum import Enum


class TimeSeriesDatasetEnum(str, Enum):
    Datetime = "ds"
    UniqueId = "unique_id"
    Target = "y"


class ExplainerEnum(str, Enum):
    IntegratedGradients = "IntegratedGradients"
    ShapleyValueSampling = "ShapleyValueSampling"
    InputXGradient = "InputXGradient"


ADDITIVE_EXPLAINERS = frozenset(
    {ExplainerEnum.IntegratedGradients, ExplainerEnum.ShapleyValueSampling}
)
ALL_EXPLAINERS = tuple(ExplainerEnum)
