"""TabPFN-TS pretrained forecasting adapter."""

from ..losses.pytorch import MQLoss
from ._context_utils import _local_path, _options
from ._exogenous import PretrainedExogenousModel

__all__ = ["TabPFNTS"]


class TabPFNTS(PretrainedExogenousModel):
    """Official TabPFN-TS pipeline in LOCAL mode with known-future covariates.

    model_id is an explicit local compatible TabPFN regressor checkpoint file.
    Only the running index is engineered; provide true calendar values through
    futr_exog_list because NF windows do not carry the original datetime index.
    With loss=MQLoss(...), return the official regressor's requested quantiles;
    MAE/MSE retains the original median point forecast.
    """

    EXOGENOUS_HIST = False

    NATIVE_QUANTILES = True

    def __init__(self, h, input_size, model_id, **kwargs):
        super().__init__(
            h=h,
            input_size=input_size,
            model_id=_local_path(model_id, directory=False),
            **_options(kwargs),
        )
        if not self.futr_exog_size:
            raise ValueError(
                "TabPFNTS requires known-future numerical covariates."
            )

    def _load_backend(self):
        import os

        os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
        from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline
        from tabpfn_time_series.features import RunningIndexFeature

        return TabPFNTSPipeline(
            max_context_length=self.input_size,
            temporal_features=[RunningIndexFeature()],
            tabpfn_mode=TabPFNMode.LOCAL,
            tabpfn_output_selection="median",
            tabpfn_model_config={
                "model_path": self.model_id,
                "device": self.backend_device,
            },
        )

    def forward(self, windows_batch):
        import numpy as np
        import pandas as pd

        y, mask, _, futr = self._inputs(windows_batch)
        self._complete_history(mask)
        pipeline = self._get_backend()
        quantile_mode = isinstance(self.loss, MQLoss)
        quantiles = (
            self.loss.quantiles.detach().cpu().tolist()
            if quantile_mode
            else [0.5]
        )
        dates = pd.date_range(
            "2000-01-01",
            periods=self.input_size + self.h,
            freq="s",
        )
        output = []
        for i in range(len(y)):
            frame = pd.DataFrame(
                futr[i].detach().cpu().numpy(),
                columns=[
                    f"covariate_{j}"
                    for j in range(self.futr_exog_size)
                ],
            )
            frame["timestamp"] = dates
            history = frame.iloc[: self.input_size].copy()
            history["target"] = y[i, :, 0].detach().cpu().numpy()
            future = frame.iloc[self.input_size :].copy()
            result = pipeline.predict_df(
                context_df=history,
                future_df=future,
                quantiles=quantiles,
            )
            if quantile_mode:
                if (
                    not result.columns.is_unique
                    or any(q not in result.columns for q in quantiles)
                ):
                    raise ValueError(
                        "TabPFN-TS did not return each requested "
                        "quantile column exactly once."
                    )
                output.append(
                    result.loc[:, quantiles].to_numpy()
                )
            else:
                output.append(result["target"].to_numpy())
        if quantile_mode:
            return self._quantile_output(np.stack(output), y)
        return self._point_output(np.stack(output), y)
