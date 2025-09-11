import numpy as np
import pytest

from neuralforecast import NeuralForecast
from neuralforecast.common.enums import TimeSeriesDatasetEnum
from neuralforecast.common._base_model import DISTRIBUTION_LOSSES, MULTIQUANTILE_LOSSES
from neuralforecast.losses.pytorch import (
    DistributionLoss,
    PMM,
    GMM,
    NBMM,
    MQLoss,
    HuberMQLoss,
    SMAPE,
)
from tests.dummy.dummy_models import DummyRecurrent


class TestDummyRecurrent:
    """Test suite for recurrent dummy models to validate horizon predictions functionality."""

    def test_larger_horizon(self, longer_horizon_test):
        model = DummyRecurrent(
            h=longer_horizon_test.h,
            input_size=longer_horizon_test.input_size,
            futr_exog_list=longer_horizon_test.calendar_cols,
        )

        nf = NeuralForecast(
            models=[model],
            freq="ME",
        )
        # dummy fit
        nf.fit(df=longer_horizon_test.train_df)

        # standard forecast
        forecasts = nf.predict(futr_df=longer_horizon_test.test_df)
        np.testing.assert_almost_equal(
            forecasts[
                forecasts[TimeSeriesDatasetEnum.UniqueId]
                == longer_horizon_test.series1_id
            ]["DummyRecurrent"].values,
            np.array([405.0] * 4),
        )
        np.testing.assert_almost_equal(
            forecasts[
                forecasts[TimeSeriesDatasetEnum.UniqueId]
                == longer_horizon_test.series2_id
            ]["DummyRecurrent"].values,
            np.array([705.0] * 4),
        )

        # longer horizon forecast
        forecasts = nf.predict(
            futr_df=longer_horizon_test.test_df, h=longer_horizon_test.longer_h
        )
        np.testing.assert_almost_equal(
            forecasts[
                forecasts[TimeSeriesDatasetEnum.UniqueId]
                == longer_horizon_test.series1_id
            ]["DummyRecurrent"].values,
            np.array([405.0] * 10),
        )
        np.testing.assert_almost_equal(
            forecasts[
                forecasts[TimeSeriesDatasetEnum.UniqueId]
                == longer_horizon_test.series2_id
            ]["DummyRecurrent"].values,
            np.array([705.0] * 10),
        )

    @pytest.mark.parametrize(
        ("loss", "quantile", "raised"),
        [
            (DistributionLoss("StudentT"), 0.52, True),
            (PMM(), 0.52, True),
            (GMM(), 0.52, True),
            (NBMM(), 0.52, True),
            (MQLoss(), 0.52, True),
            (HuberMQLoss(), 0.52, True),
            (SMAPE(), 0.52, False),
            (DistributionLoss("StudentT"), 0.5, False),
            (PMM(), 0.5, False),
            (GMM(), 0.5, False),
            (NBMM(), 0.5, False),
            (MQLoss(), 0.5, False),
            (HuberMQLoss(), 0.5, False),
            (SMAPE(), 0.5, False),
        ],
    )
    def test_maybe_get_quantile_idx(self, longer_horizon_test, loss, quantile, raised):
        model = DummyRecurrent(
            h=longer_horizon_test.h,
            input_size=longer_horizon_test.input_size,
            futr_exog_list=longer_horizon_test.calendar_cols,
            loss=loss,
        )

        if raised:
            with pytest.raises(
                ValueError, match="Model was not trained with a median quantile."
            ):
                model._maybe_get_quantile_idx(quantile)
        else:
            if isinstance(loss, DISTRIBUTION_LOSSES + MULTIQUANTILE_LOSSES):
                if isinstance(loss, DISTRIBUTION_LOSSES):
                    assert 1 == model._maybe_get_quantile_idx(quantile)
                else:
                    assert 0 == model._maybe_get_quantile_idx(quantile)
            else:
                model._maybe_get_quantile_idx(quantile) is None
