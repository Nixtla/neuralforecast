from enum import Enum

class TimeSeriesDatasetEnum(str, Enum):
    Datetime = "ds"
    UniqueId = "unique_id"
    Target = "y"