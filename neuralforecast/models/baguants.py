"""Baguan-TS pretrained research forecasting adapter."""

from pathlib import Path

from ._isolated_research_forecast import _IsolatedResearchForecast

__all__ = ["BaguanTS"]


class BaguanTS(_IsolatedResearchForecast):
    """Official Baguan-TS TS-tabular pipeline with window-local retrieval.

    Args:
        h: Forecast horizon, strictly smaller than input_size.
        input_size: Historical context length.
        source_dir: Checkout of jxgogo/Baguan-TS, in the backend environment.
        config_path: Trusted official model YAML matching the checkpoint.
        model_id: Explicit local Baguan checkpoint (tensor-only state_dict).
        context_size: Retrieved sequence width, h < context_size <= input_size.
        neighbors: Maximum number of retrieved historical contexts.
        num_samples: Official mF/n_repeat count, not forecast quantile count.
        **kwargs: NF options, nonempty futr_exog_list and separate backend_python.

    The worker calls the unchanged official predict method. Its small safe
    initializer avoids the source's unused undefined StandardScaler reference
    and loads checkpoint tensors with weights_only=True, never unsafe fallback.
    No trained Baguan checkpoint is bundled or implicitly substituted.
    """

    BACKEND_KIND = "baguants"
    EXOGENOUS_HIST = False

    def __init__(
        self,
        h,
        input_size,
        source_dir=None,
        config_path=None,
        context_size=None,
        neighbors=5,
        num_samples=1,
        **kwargs,
    ):
        context_size = (
            max(h + 1, input_size // 2)
            if context_size is None
            else context_size
        )
        if (
            not isinstance(context_size, int)
            or isinstance(context_size, bool)
            or not h < context_size <= input_size
        ):
            raise ValueError(
                "BaguanTS requires h < context_size <= input_size."
            )
        if (
            not isinstance(neighbors, int)
            or isinstance(neighbors, bool)
            or neighbors < 1
        ):
            raise ValueError("neighbors must be a positive integer.")
        if source_dir is None or config_path is None:
            raise ValueError(
                "BaguanTS requires source_dir and a trusted official config_path."
            )
        super().__init__(
            h=h,
            input_size=input_size,
            num_samples=num_samples,
            **kwargs,
        )
        if not self.futr_exog_size:
            raise ValueError("BaguanTS requires nonempty futr_exog_list.")
        self.source_dir, self.config_path = str(source_dir), str(config_path)
        self.context_size, self.neighbors = context_size, neighbors

    def _configuration(self):
        return dict(
            super()._configuration(),
            source_dir=str(Path(self.source_dir).expanduser().absolute()),
            config_path=str(Path(self.config_path).expanduser().absolute()),
            context_size=self.context_size,
            neighbors=self.neighbors,
        )
