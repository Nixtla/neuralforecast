import numpy as np
import pytest

from neuralforecast import NeuralForecast
from neuralforecast.common.enums import TimeSeriesDatasetEnum
from neuralforecast.utils import PredictionIntervals
from tests.dummy.dummy_models import DummyMultivariate


class TestDummyMultivariate:
    """Test suite for univariate dummy models to validate horizon predictions functionality."""

    def test_larger_horizon(self, longer_horizon_test):
        model = DummyMultivariate(
            h=longer_horizon_test.h,
            input_size=longer_horizon_test.input_size,
            n_series=longer_horizon_test.n_series,
        )

        nf = NeuralForecast(
            models=[model],
            freq="ME",
        )
        # dummy fit
        nf.fit(df=longer_horizon_test.train_df)

        # standard forecast, also test consistency of predict_horizon upon prediction
        assert nf.models[0].predict_horizon == longer_horizon_test.h
        forecasts = nf.predict(futr_df=longer_horizon_test.test_df)
        assert nf.models[0].predict_horizon == longer_horizon_test.h

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
        assert cross_val_df["DummyMultivariate"].mean() > 0.0


    def test_single_series(self, longer_horizon_test):
        assert longer_horizon_test.single_train_df[TimeSeriesDatasetEnum.UniqueId].nunique() == 1
        assert longer_horizon_test.single_test_df[TimeSeriesDatasetEnum.UniqueId].nunique() == 1
        model = DummyMultivariate(
            h=longer_horizon_test.h,
            input_size=longer_horizon_test.input_size,
            n_series=longer_horizon_test.single_n_series,
        )

        nf = NeuralForecast(
            models=[model],
            freq="ME",
        )
        # dummy fit
        nf.fit(df=longer_horizon_test.single_train_df)
        # standard forecast
        forecasts = nf.predict(futr_df=longer_horizon_test.single_test_df)
        np.testing.assert_almost_equal(
            forecasts["DummyMultivariate"].values,
            np.array([463.0, 407.0, 362.0, 405.0]),
        )

        # longer horizon forecast
        forecasts = nf.predict(
            futr_df=longer_horizon_test.single_test_df, h=longer_horizon_test.longer_h
        )
        np.testing.assert_almost_equal(
            forecasts["DummyMultivariate"].values,
            np.array(
                [463.0, 407.0, 362.0, 405.0, 463.0, 407.0, 362.0, 405.0, 463.0, 407.0]
            ),
        )

    def test_conformal_prediction(self, longer_horizon_test):
        model = DummyMultivariate(
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

        # forecast, cross_validation with longer horizon not supported
        with pytest.raises(ValueError, match=error_msg):
            nf.predict(
                futr_df=longer_horizon_test.test_df,
                h=longer_horizon_test.longer_h,
                level=[80],
            )
        with pytest.raises(ValueError, match=error_msg):
            nf.cross_validation(
                df=longer_horizon_test.train_df,
                n_windows=2,
                h=longer_horizon_test.longer_h,
            )

        forecasts = nf.predict(
            futr_df=longer_horizon_test.test_df, h=longer_horizon_test.h, level=[80]
        )
        assert "DummyMultivariate-lo-80" in forecasts.columns
        assert "DummyMultivariate-hi-80" in forecasts.columns


    def test_futr_exog(self, longer_horizon_test):
        model = DummyMultivariate(
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
            ]["DummyMultivariate"].values,
            np.array([463.0 * 1, 407.0 * 1, 362.0 * 1, 405.0 * 2]),
        )
        np.testing.assert_almost_equal(
            forecasts[
                forecasts[TimeSeriesDatasetEnum.UniqueId]
                == longer_horizon_test.series2_id
            ]["DummyMultivariate"].values,
            np.array([763.0 * 1, 707.0 * 1, 662.0 * 1, 705.0 * 2]),
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
                [
                    463.0 * 1,
                    407.0 * 1,
                    362.0 * 1,
                    405.0 * 2,
                    (463.0 * 1) * 2,
                    (407.0 * 1) * 2,
                    (362.0 * 1) * 3,
                    (405.0 * 2) * 3,
                    (463.0 * 1 * 2) * 3,
                    (407.0 * 1 * 2) * 4,
                ]
            ),
        )
        np.testing.assert_almost_equal(
            forecasts[
                forecasts[TimeSeriesDatasetEnum.UniqueId]
                == longer_horizon_test.series2_id
            ]["DummyMultivariate"].values,
            np.array(
                [
                    763.0 * 1,
                    707.0 * 1,
                    662.0 * 1,
                    705.0 * 2,
                    (763.0 * 1) * 2,
                    (707.0 * 1) * 2,
                    (662.0 * 1) * 3,
                    (705.0 * 2) * 3,
                    (763.0 * 1 * 2) * 3,
                    (707.0 * 1 * 2) * 4,
                ]
            ),
        )
