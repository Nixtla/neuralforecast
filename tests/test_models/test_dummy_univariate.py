import numpy as np
import pandas as pd
import pytest

from neuralforecast import NeuralForecast
from neuralforecast.common._base_model import DISTRIBUTION_LOSSES, MULTIQUANTILE_LOSSES
from neuralforecast.common.enums import TimeSeriesDatasetEnum
from neuralforecast.losses.pytorch import (
    DistributionLoss,
    PMM,
    GMM,
    HuberIQLoss,
    IQLoss,
    NBMM,
    MQLoss,
    HuberMQLoss,
    SMAPE,
)
from neuralforecast.utils import PredictionIntervals
from tests.dummy.dummy_models import DummyUnivariate


class TestDummyUnivariate:
    """Test suite for univariate dummy models to validate horizon predictions functionality."""

    def test_larger_horizon(self, longer_horizon_test):
        model = DummyUnivariate(
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
            ]["DummyUnivariate"].values,
            np.array([463.0, 407.0, 362.0, 405.0]),
        )
        np.testing.assert_almost_equal(
            forecasts[
                forecasts[TimeSeriesDatasetEnum.UniqueId]
                == longer_horizon_test.series2_id
            ]["DummyUnivariate"].values,
            np.array([763.0, 707.0, 662.0, 705.0]),
        )

        # longer horizon forecast
        forecasts = nf.predict(
            futr_df=longer_horizon_test.test_df, h=longer_horizon_test.longer_h
        )

        np.testing.assert_almost_equal(
            forecasts[
                forecasts[TimeSeriesDatasetEnum.UniqueId]
                == longer_horizon_test.series1_id
            ]["DummyUnivariate"].values,
            np.array(
                [463.0, 407.0, 362.0, 405.0, 463.0, 407.0, 362.0, 405.0, 463.0, 407.0]
            ),
        )
        np.testing.assert_almost_equal(
            forecasts[
                forecasts[TimeSeriesDatasetEnum.UniqueId]
                == longer_horizon_test.series2_id
            ]["DummyUnivariate"].values,
            np.array(
                [763.0, 707.0, 662.0, 705.0, 763.0, 707.0, 662.0, 705.0, 763.0, 707.0]
            ),
        )

        # cross-validation using a different h parameter
        with pytest.raises(
            ValueError,
            match="The specified horizon h={} is larger than the horizon of the fitted models: {}. Set refit=True in this setting.".format(
                longer_horizon_test.cross_val_h, longer_horizon_test.h
            ),
        ):
            nf.cross_validation(
                df=longer_horizon_test.train_df,
                n_windows=2,
                h=longer_horizon_test.cross_val_h,
            )

        cross_val_df = nf.cross_validation(
            df=longer_horizon_test.train_df,
            n_windows=2,
            h=longer_horizon_test.cross_val_h,
            refit=True,
        )
        assert cross_val_df[TimeSeriesDatasetEnum.Target].mean() > 0.0
        assert cross_val_df["DummyUnivariate"].mean() > 0.0

    def test_conformal_prediction(self, longer_horizon_test):
        model = DummyUnivariate(
            h=longer_horizon_test.h,
            input_size=longer_horizon_test.input_size,
            futr_exog_list=longer_horizon_test.calendar_cols,
        )
        nf = NeuralForecast(
            models=[model],
            freq="ME",
        )
        # dummy fit
        nf.fit(
            df=longer_horizon_test.train_df, prediction_intervals=PredictionIntervals()
        )
        error_msg = (
            "The specified horizon h=10 is larger than the horizon of the fitted models: 4. "
            "Forecast with prediction intervals requires a refit with the specified horizon."
        )

        # forecast with longer horizon not supported
        with pytest.raises(ValueError, match=error_msg):
            nf.predict(
                futr_df=longer_horizon_test.test_df, 
                h=longer_horizon_test.longer_h,
                level=[80]
            )

        forecasts = nf.predict(futr_df=longer_horizon_test.test_df, h=longer_horizon_test.h, level=[80])
        assert "DummyUnivariate-lo-80" in forecasts.columns
        assert "DummyUnivariate-hi-80" in forecasts.columns
                

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
        model = DummyUnivariate(
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
                    assert model._maybe_get_quantile_idx(quantile) == 1
                else:
                    assert model._maybe_get_quantile_idx(quantile) == 0
            else:
                model._maybe_get_quantile_idx(quantile) is None

    @pytest.mark.parametrize("loss_type", [IQLoss, HuberIQLoss])
    def test_iqloss(self, longer_horizon_test, loss_type):
        model = DummyUnivariate(
            h=longer_horizon_test.h,
            input_size=longer_horizon_test.input_size,
            futr_exog_list=longer_horizon_test.calendar_cols,
            loss=loss_type(),
        )

        nf = NeuralForecast(
            models=[model],
            freq="ME",
        )
        # dummy fit
        nf.fit(df=longer_horizon_test.train_df)

        # longer horizon forecast
        forecasts = nf.predict(
            futr_df=longer_horizon_test.test_df, h=longer_horizon_test.longer_h
        )
        group_cnt = forecasts.groupby(TimeSeriesDatasetEnum.UniqueId)[
            "DummyUnivariate_ql0.5"
        ].count()
        expected = pd.Series(
            data=[10, 10],
            index=[longer_horizon_test.series1_id, longer_horizon_test.series2_id],
            name="DummyUnivariate_ql0.5",
        )
        expected.index.name = TimeSeriesDatasetEnum.UniqueId
        pd.testing.assert_series_equal(group_cnt, expected)
