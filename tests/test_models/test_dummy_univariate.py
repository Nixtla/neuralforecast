import numpy as np

from neuralforecast import NeuralForecast
from neuralforecast.common.enums import TimeSeriesDatasetEnum

from tests.dummy.dummy_models import DummyUnivariate
from tests.helpers.data import air_passengers


class TestDummyUnivariate:
    """Test suite for univariate dummy models to validate horizon predictions functionality."""

    def test_larger_horizon(self):
        train_df, test_df, calendar_cols, _ = air_passengers(
            h=12, augment_calendar=True
        )
        test_df[TimeSeriesDatasetEnum.Target] = np.nan

        h = 4
        longer_h = 10
        input_size = 14

        model = DummyUnivariate(
            h=h,
            input_size=input_size,
            futr_exog_list=calendar_cols,
        )

        nf = NeuralForecast(
            models=[model],
            freq="ME",
        )
        # dummy fit
        nf.fit(df=train_df)

        # standard forecast
        forecasts = nf.predict(futr_df=test_df)
        np.testing.assert_almost_equal(
            forecasts[forecasts[TimeSeriesDatasetEnum.UniqueId] == "Airline1"][
                "DummyUnivariate"
            ].values,
            np.array(
                [463.0, 407.0, 362.0, 405.0]
            ),
        )
        np.testing.assert_almost_equal(
            forecasts[forecasts[TimeSeriesDatasetEnum.UniqueId] == "Airline2"][
                "DummyUnivariate"
            ].values,
            np.array(
                [763.0, 707.0, 662.0, 705.0]
            ),
        )

        # longer horizon forecast
        forecasts = nf.predict(futr_df=test_df, h=longer_h)

        np.testing.assert_almost_equal(
            forecasts[forecasts[TimeSeriesDatasetEnum.UniqueId] == "Airline1"][
                "DummyUnivariate"
            ].values,
            np.array(
                [463.0, 407.0, 362.0, 405.0, 463.0, 407.0, 362.0, 405.0, 463.0, 407.0]
            ),
        )
        np.testing.assert_almost_equal(
            forecasts[forecasts[TimeSeriesDatasetEnum.UniqueId] == "Airline2"][
                "DummyUnivariate"
            ].values,
            np.array(
                [763.0, 707.0, 662.0, 705.0, 763.0, 707.0, 662.0, 705.0, 763.0, 707.0]
            ),
        )
