import numpy as np
import pytest

from neuralforecast import NeuralForecast
from neuralforecast.common.enums import TimeSeriesDatasetEnum
from neuralforecast.utils import PredictionIntervals
from tests.dummy.dummy_models import DummyRecurrent


class TestDummyRecurrent:
    """Test suite for recurrent dummy models to validate horizon predictions functionality."""

    def test_larger_horizon(self, longer_horizon_test):
        model = DummyRecurrent(
            h=longer_horizon_test.h,
            input_size=longer_horizon_test.input_size,
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
        assert cross_val_df["DummyRecurrent"].mean() > 0.0

    def test_conformal_prediction(self, longer_horizon_test):
        model = DummyRecurrent(
            h=longer_horizon_test.h,
            input_size=longer_horizon_test.input_size,
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
            "Forecast with prediction intervals is not supported."
        )

        # forecast with longer horizon not supported
        with pytest.raises(ValueError, match=error_msg):
            nf.predict(
                futr_df=longer_horizon_test.test_df,
                h=longer_horizon_test.longer_h,
                level=[80],
            )

        forecasts = nf.predict(
            futr_df=longer_horizon_test.test_df, h=longer_horizon_test.h, level=[80]
        )
        assert "DummyRecurrent-lo-80" in forecasts.columns
        assert "DummyRecurrent-hi-80" in forecasts.columns

    def test_futr_exog(self, longer_horizon_test):
        model = DummyRecurrent(
            h=longer_horizon_test.h,
            input_size=longer_horizon_test.input_size,
            futr_exog_list=longer_horizon_test.futr_exog_list,
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
            np.array([405.0 * 1, 405.0 * 1, 405.0 * 1, 405.0 * 2]),
        )
        np.testing.assert_almost_equal(
            forecasts[
                forecasts[TimeSeriesDatasetEnum.UniqueId]
                == longer_horizon_test.series2_id
            ]["DummyRecurrent"].values,
            np.array([705.0 * 1, 705.0 * 1, 705.0 * 1, 705.0 * 2]),
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
            np.array(
                [
                    405.0 * 1,
                    405.0 * 1,
                    405.0 * 1,
                    405.0 * 2,
                    (405.0 * 2) * 2,
                    (405.0 * 2 * 2) * 2,
                    (405.0 * 2 * 2 * 2) * 3,
                    (405.0 * 2 * 2 * 2 * 3) * 3,
                    (405.0 * 2 * 2 * 2 * 3 * 3) * 3,
                    (405.0 * 2 * 2 * 2 * 3 * 3 * 3) * 4,
                ]
            ),
        )
        np.testing.assert_almost_equal(
            forecasts[
                forecasts[TimeSeriesDatasetEnum.UniqueId]
                == longer_horizon_test.series2_id
            ]["DummyRecurrent"].values,
            np.array(
                [
                    705.0 * 1,
                    705.0 * 1,
                    705.0 * 1,
                    705.0 * 2,
                    (705.0 * 2) * 2,
                    (705.0 * 2 * 2) * 2,
                    (705.0 * 2 * 2 * 2) * 3,
                    (705.0 * 2 * 2 * 2 * 3) * 3,
                    (705.0 * 2 * 2 * 2 * 3 * 3) * 3,
                    (705.0 * 2 * 2 * 2 * 3 * 3 * 3) * 4,
                ]
            ),
        )
