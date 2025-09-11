import numpy as np

from neuralforecast import NeuralForecast
from neuralforecast.common.enums import TimeSeriesDatasetEnum
from tests.dummy.dummy_models import DummyMultivariate


class TestDummyMultivariate:
    """Test suite for univariate dummy models to validate horizon predictions functionality."""

    def test_larger_horizon(self, longer_horizon_test):
        model = DummyMultivariate(
            h=longer_horizon_test.train_h,
            input_size=longer_horizon_test.input_size,
            n_series=longer_horizon_test.n_series,
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
            ]["DummyMultivariate"].values,
            np.array([463.0, 407.0, 362.0, 405.0]),
        )
        np.testing.assert_almost_equal(
            forecasts[
                forecasts[TimeSeriesDatasetEnum.UniqueId]
                == longer_horizon_test.series2_id
            ]["DummyMultivariate"].values,
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
            ]["DummyMultivariate"].values,
            np.array(
                [463.0, 407.0, 362.0, 405.0, 463.0, 407.0, 362.0, 405.0, 463.0, 407.0]
            ),
        )
        np.testing.assert_almost_equal(
            forecasts[
                forecasts[TimeSeriesDatasetEnum.UniqueId]
                == longer_horizon_test.series2_id
            ]["DummyMultivariate"].values,
            np.array(
                [763.0, 707.0, 662.0, 705.0, 763.0, 707.0, 662.0, 705.0, 763.0, 707.0]
            ),
        )
