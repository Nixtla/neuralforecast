import numpy as np

from neuralforecast import NeuralForecast
from neuralforecast.common.enums import TimeSeriesDatasetEnum

from tests.dummy.dummy_models import DummyRecurrent
from tests.helpers.data import air_passengers


class TestDummyRecurrent:
    """Test suite for recurrent dummy models to validate horizon predictions functionality."""

    def test_larger_horizon(self):
        train_df, test_df, calendar_cols, _ = air_passengers(
            h=12, augment_calendar=True
        )
        test_df[TimeSeriesDatasetEnum.Target] = np.nan

        h = 4
        longer_h = 10
        input_size = 14

        model = DummyRecurrent(
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
                "DummyRecurrent"
            ].values,
            np.array([405.0] * 4),
        )
        np.testing.assert_almost_equal(
            forecasts[forecasts[TimeSeriesDatasetEnum.UniqueId] == "Airline2"][
                "DummyRecurrent"
            ].values,
            np.array([705.0] * 4),
        )

        # longer horizon forecast
        forecasts = nf.predict(futr_df=test_df, h=longer_h)
        np.testing.assert_almost_equal(
            forecasts[forecasts[TimeSeriesDatasetEnum.UniqueId] == "Airline1"][
                "DummyRecurrent"
            ].values,
            np.array([405.0] * 10),
        )
        np.testing.assert_almost_equal(
            forecasts[forecasts[TimeSeriesDatasetEnum.UniqueId] == "Airline2"][
                "DummyRecurrent"
            ].values,
            np.array([705.0] * 10),
        )
