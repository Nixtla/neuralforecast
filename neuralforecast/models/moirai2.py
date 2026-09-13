"""Moirai 2.0 pretrained forecasting adapter."""

from .moirai import Moirai

__all__ = ["Moirai2"]


class Moirai2(Moirai):
    """Official Moirai 2.0 quantile model, exposing its median (not quantile mean).

    Uses the same isolated uni2ts environment and numerical past/future fields
    as Moirai. Its official architecture fixes patch size and quantile levels;
    inherited patch_size=16/num_samples=100 are compatibility metadata only.
    Nondefault values are rejected instead of pretending to alter this backend.
    """

    DEFAULT_MODEL_ID = "Salesforce/moirai-2.0-R-small"
    BACKEND_KIND = "moirai2"

    def __init__(self, h, input_size, **kwargs):
        if (
            kwargs.get("patch_size", 16) != 16
            or kwargs.get("num_samples", 100) != 100
        ):
            raise ValueError(
                "Moirai2 uses the official fixed patches/quantiles; "
                "do not change patch_size or num_samples."
            )
        super().__init__(h=h, input_size=input_size, **kwargs)
