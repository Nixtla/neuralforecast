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
    MAE,
    NBMM,
    MQLoss,
    HuberMQLoss,
    SMAPE,
)
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

        # longer horizon forecast, also test consistency of predict_horizon upon prediction
        assert nf.models[0].predict_horizon == longer_horizon_test.h
        forecasts = nf.predict(
            futr_df=longer_horizon_test.test_df, h=longer_horizon_test.longer_h
        )
        assert nf.models[0].predict_horizon == longer_horizon_test.h
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
        assert "DummyRecurrent-lo-80" in forecasts.columns
        assert "DummyRecurrent-hi-80" in forecasts.columns

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

    @pytest.mark.parametrize(
        "loss_type,target_col",
        [
            (MAE(), "DummyRecurrent"),
            (DistributionLoss(distribution="Normal"), "DummyRecurrent"),
            (IQLoss(), "DummyRecurrent_ql0.5"),
            (MQLoss(), "DummyRecurrent-median"),
            (HuberIQLoss(), "DummyRecurrent_ql0.5"),
        ],
    )
    def test_various_loss_types(self, longer_horizon_test, loss_type, target_col):
        model = DummyRecurrent(
            h=longer_horizon_test.h,
            input_size=longer_horizon_test.input_size,
            loss=loss_type,
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
            target_col
        ].count()
        expected = pd.Series(
            data=[longer_horizon_test.longer_h] * 2,
            index=[longer_horizon_test.series1_id, longer_horizon_test.series2_id],
            name=target_col,
        )
        expected.index.name = TimeSeriesDatasetEnum.UniqueId
        pd.testing.assert_series_equal(group_cnt, expected)

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
