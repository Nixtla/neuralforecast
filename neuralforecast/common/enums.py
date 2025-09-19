from enum import Enum


class TimeSeriesDatasetEnum(str, Enum):
    Datetime = "ds"
    UniqueId = "unique_id"
    Target = "y"


class ExplainerEnum(str, Enum):
    IntegratedGradients = "IntegratedGradients"
    ShapleyValueSampling = "ShapleyValueSampling"
    Lime = "Lime"
    KernelShap = "KernelShap"
    InputXGradient = "InputXGradient"

    AddictiveExplainers = [IntegratedGradients, ShapleyValueSampling]
    AllExplainers = [IntegratedGradients, ShapleyValueSampling, Lime, KernelShap, InputXGradient]
