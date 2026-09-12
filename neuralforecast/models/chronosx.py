"""ChronosX pretrained research forecasting adapter."""

from ._isolated_research_forecast import _IsolatedResearchForecast

__all__ = ["ChronosX"]


class ChronosX(_IsolatedResearchForecast):
    """Official ChronosX IIB+OIB inference from a *fine-tuned* local checkpoint.

    Args:
        h: Forecast horizon.
        input_size: Complete context, within the checkpoint's context limit.
        model_id: Local ChronosX checkpoint directory containing safetensors.
            Base amazon/chronos-t5 weights are not a trained covariate model.
        hidden_dim: Injection-block width used during original fine-tuning.
        num_layers: Injection-block depth used during original fine-tuning.
        **kwargs: PretrainedExogenousModel options and a separate backend_python.

    Channels follow the official HF data loader: values then missing indicators.
    Historical-only features have -1 and indicator=1 in the future. Known-future
    values are passed unchanged, with indicator=0. Feature order is hist then
    futr and must match the order used to train the checkpoint. NF fit does not
    fine-tune the model; use the original ChronosX training pipeline for that.
    """

    BACKEND_KIND = "chronosx"

    def __init__(
        self,
        h,
        input_size,
        hidden_dim=256,
        num_layers=1,
        **kwargs,
    ):
        if any(
            not isinstance(v, int) or isinstance(v, bool) or v < 1
            for v in (hidden_dim, num_layers)
        ):
            raise ValueError(
                "hidden_dim and num_layers must be positive integers."
            )
        super().__init__(h=h, input_size=input_size, **kwargs)
        if not self.hist_exog_size + self.futr_exog_size:
            raise ValueError(
                "ChronosX requires at least one numerical exogenous feature."
            )
        self.hidden_dim, self.num_layers = hidden_dim, num_layers

    def _configuration(self):
        return dict(
            super()._configuration(),
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
