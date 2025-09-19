from enum import Enum


class TimeSeriesDatasetEnum(str, Enum):
    Datetime = "ds"
    UniqueId = "unique_id"
    Target = "y"


class ExplainerEnum(str, Enum):
    IntegratedGradients = "IntegratedGradients"
    ShapleyValueSampling = "ShapleyValueSampling"
    InputXGradient = "InputXGradient"

    AdditiveExplainers = [IntegratedGradients, ShapleyValueSampling]
    AllExplainers = [IntegratedGradients, ShapleyValueSampling, InputXGradient]
