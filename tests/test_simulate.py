"""Tests for neuralforecast simulation."""

import numpy as np
import pandas as pd
import pytest


def _get_model_cols(df):
    """Get model columns (everything except unique_id, ds, sample_id)."""
    return [c for c in df.columns if c not in ("unique_id", "ds", "sample_id")]


class TestSimulation:
    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------
    @pytest.fixture
    def fitted_nf_distribution(self):
        """Small NHITS model with DistributionLoss(Normal) on AirPassengers."""
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import DistributionLoss
        from neuralforecast.models import NHITS
        from neuralforecast.utils import AirPassengersDF

        model = NHITS(
            h=12,
            input_size=24,
            max_steps=5,
            loss=DistributionLoss(distribution="Normal"),
            accelerator="cpu",
        )
        nf = NeuralForecast(models=[model], freq="MS")
        nf.fit(df=AirPassengersDF)
        return nf

    @pytest.fixture
    def fitted_nf_mqloss(self):
        """Small NHITS model with MQLoss on AirPassengers."""
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MQLoss
        from neuralforecast.models import NHITS
        from neuralforecast.utils import AirPassengersDF

        model = NHITS(
            h=12,
            input_size=24,
            max_steps=5,
            loss=MQLoss(level=[90]),
            accelerator="cpu",
        )
        nf = NeuralForecast(models=[model], freq="MS")
        nf.fit(df=AirPassengersDF)
        return nf

    @pytest.fixture
    def fitted_nf_gmm(self):
        """Small NHITS model with GMM loss on AirPassengers."""
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import GMM
        from neuralforecast.models import NHITS
        from neuralforecast.utils import AirPassengersDF

        model = NHITS(
            h=12,
            input_size=24,
            max_steps=5,
            loss=GMM(n_components=2),
            accelerator="cpu",
        )
        nf = NeuralForecast(models=[model], freq="MS")
        nf.fit(df=AirPassengersDF)
        return nf

    @pytest.fixture
    def fitted_nf_recurrent(self):
        """Small RNN model with DistributionLoss(Normal) on AirPassengers."""
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import DistributionLoss
        from neuralforecast.models import RNN
        from neuralforecast.utils import AirPassengersDF

        model = RNN(
            h=6,
            input_size=12,
            max_steps=5,
            loss=DistributionLoss(distribution="Normal"),
            recurrent=True,
            accelerator="cpu",
        )
        nf = NeuralForecast(models=[model], freq="MS")
        nf.fit(df=AirPassengersDF)
        return nf

    @pytest.fixture
    def fitted_nf_iqloss(self):
        """Small NHITS model with IQLoss on AirPassengers."""
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import IQLoss
        from neuralforecast.models import NHITS
        from neuralforecast.utils import AirPassengersDF

        model = NHITS(
            h=12,
            input_size=24,
            max_steps=5,
            loss=IQLoss(),
            accelerator="cpu",
        )
        nf = NeuralForecast(models=[model], freq="MS")
        nf.fit(df=AirPassengersDF)
        return nf

    @pytest.fixture
    def fitted_nf_mae_conformal(self):
        """Small NHITS model with MAE + conformal prediction intervals."""
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MAE
        from neuralforecast.models import NHITS
        from neuralforecast.utils import AirPassengersDF, PredictionIntervals

        model = NHITS(
            h=12,
            input_size=24,
            max_steps=5,
            loss=MAE(),
            accelerator="cpu",
        )
        nf = NeuralForecast(models=[model], freq="MS")
        nf.fit(
            df=AirPassengersDF,
            prediction_intervals=PredictionIntervals(
                n_windows=2, method="conformal_error"
            ),
        )
        return nf

    @pytest.fixture
    def fitted_nf_multivariate(self):
        """TSMixer multivariate model on AirPassengersPanel."""
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import DistributionLoss
        from neuralforecast.models import TSMixer
        from neuralforecast.utils import AirPassengersPanel

        model = TSMixer(
            h=12,
            input_size=24,
            n_series=2,
            max_steps=5,
            loss=DistributionLoss(distribution="Normal"),
            accelerator="cpu",
        )
        nf = NeuralForecast(models=[model], freq="MS")
        nf.fit(df=AirPassengersPanel)
        return nf

    @pytest.fixture
    def fitted_nf_multi_model(self):
        """Multiple models with different loss types fitted together."""
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import DistributionLoss, MQLoss
        from neuralforecast.models import NHITS
        from neuralforecast.utils import AirPassengersDF

        models = [
            NHITS(
                h=12,
                input_size=24,
                max_steps=5,
                loss=DistributionLoss(distribution="Normal"),
                alias="NHITS_Normal",
                accelerator="cpu",
            ),
            NHITS(
                h=12,
                input_size=24,
                max_steps=5,
                loss=MQLoss(level=[90]),
                alias="NHITS_MQ",
                accelerator="cpu",
            ),
        ]
        nf = NeuralForecast(models=models, freq="MS")
        nf.fit(df=AirPassengersDF)
        return nf

    @pytest.fixture
    def fitted_nf_multi_series(self):
        """DistributionLoss model on a multi-series dataset (2 unique_ids)."""
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import DistributionLoss
        from neuralforecast.models import NHITS
        from neuralforecast.utils import AirPassengersPanel

        model = NHITS(
            h=12,
            input_size=24,
            max_steps=5,
            loss=DistributionLoss(distribution="Normal"),
            accelerator="cpu",
        )
        nf = NeuralForecast(models=[model], freq="MS")
        nf.fit(df=AirPassengersPanel)
        return nf

    # ------------------------------------------------------------------
    # Existing tests
    # ------------------------------------------------------------------
    def test_simulate_not_fitted(self):
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import DistributionLoss
        from neuralforecast.models import NHITS

        model = NHITS(
            h=12, input_size=24, max_steps=1,
            loss=DistributionLoss(distribution="Normal"),
            accelerator="cpu",
        )
        nf = NeuralForecast(models=[model], freq="MS")
        with pytest.raises(Exception, match="must fit"):
            nf.simulate(n_paths=5)

    def test_output_format(self, fitted_nf_distribution):
        result = fitted_nf_distribution.simulate(n_paths=10, seed=0)
        assert isinstance(result, pd.DataFrame)
        assert "unique_id" in result.columns
        assert "ds" in result.columns
        assert "sample_id" in result.columns
        model_cols = _get_model_cols(result)
        assert len(model_cols) >= 1

    def test_output_shape(self, fitted_nf_distribution):
        n_paths = 10
        h = 12
        result = fitted_nf_distribution.simulate(n_paths=n_paths, seed=0)
        n_series = result["unique_id"].nunique()
        assert len(result) == n_series * n_paths * h
        assert result["sample_id"].nunique() == n_paths

    def test_seed_reproducibility(self, fitted_nf_distribution):
        r1 = fitted_nf_distribution.simulate(n_paths=5, seed=42)
        r2 = fitted_nf_distribution.simulate(n_paths=5, seed=42)
        model_cols = _get_model_cols(r1)
        for col in model_cols:
            np.testing.assert_array_equal(r1[col].values, r2[col].values)

    def test_different_seeds(self, fitted_nf_distribution):
        r1 = fitted_nf_distribution.simulate(n_paths=5, seed=1)
        r2 = fitted_nf_distribution.simulate(n_paths=5, seed=2)
        model_cols = _get_model_cols(r1)
        for col in model_cols:
            assert not np.allclose(r1[col].values, r2[col].values)

    def test_mqloss_smoke(self, fitted_nf_mqloss):
        """Gaussian Copula works with MQLoss (not just DistributionLoss)."""
        result = fitted_nf_mqloss.simulate(n_paths=10, seed=0)
        assert isinstance(result, pd.DataFrame)
        assert result["sample_id"].nunique() == 10
        model_cols = _get_model_cols(result)
        assert len(model_cols) >= 1

    def test_gmm_smoke(self, fitted_nf_gmm):
        result = fitted_nf_gmm.simulate(n_paths=10, seed=0)
        assert isinstance(result, pd.DataFrame)
        assert result["sample_id"].nunique() == 10

    def test_recurrent_smoke(self, fitted_nf_recurrent):
        result = fitted_nf_recurrent.simulate(n_paths=5, seed=0)
        assert isinstance(result, pd.DataFrame)
        assert result["sample_id"].nunique() == 5
        n_series = result["unique_id"].nunique()
        assert len(result) == n_series * 5 * 6  # h=6

    def test_recurrent_path_divergence(self, fitted_nf_recurrent):
        """Paths should show variation across samples."""
        result = fitted_nf_recurrent.simulate(n_paths=50, seed=0)
        model_col = _get_model_cols(result)[0]
        uid = result["unique_id"].iloc[0]
        series_df = result[result["unique_id"] == uid]
        by_sample = series_df.pivot_table(
            index="sample_id", columns="ds", values=model_col,
        )
        stds = by_sample.std(axis=0).values
        assert stds[-1] >= stds[0] * 0.5

    def test_mean_plausibility(self, fitted_nf_distribution):
        result = fitted_nf_distribution.simulate(n_paths=100, seed=0)
        model_cols = _get_model_cols(result)
        for col in model_cols:
            assert not np.any(np.isnan(result[col].values))
            assert not np.any(np.isinf(result[col].values))

    def test_ds_matches_predict(self, fitted_nf_distribution):
        """The ds values should match those from predict()."""
        sim_df = fitted_nf_distribution.simulate(n_paths=3, seed=0)
        pred_df = fitted_nf_distribution.predict()
        sim_sample0 = sim_df[sim_df["sample_id"] == 0].sort_values(
            ["unique_id", "ds"]
        )
        pred_sorted = pred_df.sort_values(["unique_id", "ds"])
        np.testing.assert_array_equal(
            sim_sample0["ds"].values, pred_sorted["ds"].values
        )
        np.testing.assert_array_equal(
            sim_sample0["unique_id"].values, pred_sorted["unique_id"].values
        )

    def test_custom_quantile_grid(self, fitted_nf_distribution):
        """Custom quantile grid should work."""
        result = fitted_nf_distribution.simulate(
            n_paths=5, seed=0, quantiles=[0.1, 0.5, 0.9],
        )
        assert isinstance(result, pd.DataFrame)
        assert result["sample_id"].nunique() == 5

    # ------------------------------------------------------------------
    # Multivariate model (TSMixer)
    # ------------------------------------------------------------------
    def test_multivariate_smoke(self, fitted_nf_multivariate):
        """TSMixer multivariate model produces valid simulation output."""
        result = fitted_nf_multivariate.simulate(n_paths=5, seed=0)
        assert isinstance(result, pd.DataFrame)
        assert "sample_id" in result.columns
        assert result["sample_id"].nunique() == 5
        model_cols = _get_model_cols(result)
        assert len(model_cols) >= 1

    def test_multivariate_shape(self, fitted_nf_multivariate):
        """Multivariate output should have rows for all series × paths × H."""
        n_paths = 5
        h = 12
        result = fitted_nf_multivariate.simulate(n_paths=n_paths, seed=0)
        n_uids = result["unique_id"].nunique()
        assert n_uids == 2  # AirPassengersPanel has 2 series
        assert len(result) == n_uids * n_paths * h

    def test_multivariate_no_nan(self, fitted_nf_multivariate):
        """Multivariate simulations should contain no NaN or Inf."""
        result = fitted_nf_multivariate.simulate(n_paths=10, seed=0)
        model_cols = _get_model_cols(result)
        for col in model_cols:
            assert not np.any(np.isnan(result[col].values))
            assert not np.any(np.isinf(result[col].values))

    def test_multivariate_ds_matches_predict(self, fitted_nf_multivariate):
        """Multivariate ds/unique_id values should match predict()."""
        sim_df = fitted_nf_multivariate.simulate(n_paths=3, seed=0)
        pred_df = fitted_nf_multivariate.predict()
        sim_sample0 = sim_df[sim_df["sample_id"] == 0].sort_values(
            ["unique_id", "ds"]
        )
        pred_sorted = pred_df.sort_values(["unique_id", "ds"])
        np.testing.assert_array_equal(
            sim_sample0["ds"].values, pred_sorted["ds"].values
        )
        np.testing.assert_array_equal(
            sim_sample0["unique_id"].values, pred_sorted["unique_id"].values
        )

    # ------------------------------------------------------------------
    # Schaake shuffle method
    # ------------------------------------------------------------------
    def test_schaake_shuffle_smoke(self, fitted_nf_distribution):
        """Schaake shuffle method produces valid output."""
        result = fitted_nf_distribution.simulate(
            n_paths=10, seed=0, method="schaake_shuffle"
        )
        assert isinstance(result, pd.DataFrame)
        assert result["sample_id"].nunique() == 10
        model_cols = _get_model_cols(result)
        assert len(model_cols) >= 1

    def test_schaake_shuffle_shape(self, fitted_nf_distribution):
        """Schaake shuffle has same shape as gaussian_copula."""
        n_paths = 10
        r_copula = fitted_nf_distribution.simulate(
            n_paths=n_paths, seed=0, method="gaussian_copula"
        )
        r_schaake = fitted_nf_distribution.simulate(
            n_paths=n_paths, seed=0, method="schaake_shuffle"
        )
        assert r_copula.shape == r_schaake.shape
        assert list(r_copula.columns) == list(r_schaake.columns)

    def test_schaake_shuffle_differs_from_copula(self, fitted_nf_distribution):
        """Schaake shuffle and gaussian_copula should produce different paths."""
        r_copula = fitted_nf_distribution.simulate(
            n_paths=20, seed=42, method="gaussian_copula"
        )
        r_schaake = fitted_nf_distribution.simulate(
            n_paths=20, seed=42, method="schaake_shuffle"
        )
        model_col = _get_model_cols(r_copula)[0]
        assert not np.allclose(r_copula[model_col].values, r_schaake[model_col].values)

    def test_schaake_shuffle_reproducible(self, fitted_nf_distribution):
        """Schaake shuffle with same seed produces identical results."""
        r1 = fitted_nf_distribution.simulate(
            n_paths=5, seed=42, method="schaake_shuffle"
        )
        r2 = fitted_nf_distribution.simulate(
            n_paths=5, seed=42, method="schaake_shuffle"
        )
        model_cols = _get_model_cols(r1)
        for col in model_cols:
            np.testing.assert_array_equal(r1[col].values, r2[col].values)

    def test_schaake_shuffle_no_nan(self, fitted_nf_distribution):
        """Schaake shuffle should produce no NaN or Inf."""
        result = fitted_nf_distribution.simulate(
            n_paths=50, seed=0, method="schaake_shuffle"
        )
        model_cols = _get_model_cols(result)
        for col in model_cols:
            assert not np.any(np.isnan(result[col].values))
            assert not np.any(np.isinf(result[col].values))

    # ------------------------------------------------------------------
    # IQLoss
    # ------------------------------------------------------------------
    def test_iqloss_smoke(self, fitted_nf_iqloss):
        """IQLoss model produces valid simulation output."""
        result = fitted_nf_iqloss.simulate(n_paths=10, seed=0)
        assert isinstance(result, pd.DataFrame)
        assert result["sample_id"].nunique() == 10
        model_cols = _get_model_cols(result)
        assert len(model_cols) >= 1

    def test_iqloss_no_nan(self, fitted_nf_iqloss):
        """IQLoss simulations should contain no NaN or Inf."""
        result = fitted_nf_iqloss.simulate(n_paths=10, seed=0)
        model_cols = _get_model_cols(result)
        for col in model_cols:
            assert not np.any(np.isnan(result[col].values))
            assert not np.any(np.isinf(result[col].values))

    # ------------------------------------------------------------------
    # MAE + conformal prediction
    # ------------------------------------------------------------------
    def test_mae_conformal_smoke(self, fitted_nf_mae_conformal):
        """MAE model with conformal intervals produces valid simulation."""
        result = fitted_nf_mae_conformal.simulate(n_paths=10, seed=0)
        assert isinstance(result, pd.DataFrame)
        assert result["sample_id"].nunique() == 10
        model_cols = _get_model_cols(result)
        assert len(model_cols) >= 1

    def test_mae_conformal_no_nan(self, fitted_nf_mae_conformal):
        """MAE + conformal simulations should contain no NaN or Inf."""
        result = fitted_nf_mae_conformal.simulate(n_paths=10, seed=0)
        model_cols = _get_model_cols(result)
        for col in model_cols:
            assert not np.any(np.isnan(result[col].values))
            assert not np.any(np.isinf(result[col].values))

    def test_mae_without_conformal_raises(self):
        """MAE model WITHOUT conformal intervals should raise on simulate."""
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MAE
        from neuralforecast.models import NHITS
        from neuralforecast.utils import AirPassengersDF

        model = NHITS(h=12, input_size=24, max_steps=5, loss=MAE(), accelerator="cpu")
        nf = NeuralForecast(models=[model], freq="MS")
        nf.fit(df=AirPassengersDF)
        with pytest.raises(ValueError, match="prediction_intervals"):
            nf.simulate(n_paths=5)

    # ------------------------------------------------------------------
    # Invalid method
    # ------------------------------------------------------------------
    def test_invalid_method_raises(self, fitted_nf_distribution):
        """Unknown method string should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown simulation method"):
            fitted_nf_distribution.simulate(n_paths=5, method="bogus")

    # ------------------------------------------------------------------
    # Multiple models simultaneously
    # ------------------------------------------------------------------
    def test_multi_model_output(self, fitted_nf_multi_model):
        """Simulate with multiple models produces columns for each model."""
        result = fitted_nf_multi_model.simulate(n_paths=5, seed=0)
        model_cols = _get_model_cols(result)
        assert "NHITS_Normal" in model_cols
        assert "NHITS_MQ" in model_cols
        assert result["sample_id"].nunique() == 5

    def test_multi_model_no_nan(self, fitted_nf_multi_model):
        """Multi-model simulations should contain no NaN or Inf."""
        result = fitted_nf_multi_model.simulate(n_paths=10, seed=0)
        model_cols = _get_model_cols(result)
        for col in model_cols:
            assert not np.any(np.isnan(result[col].values))
            assert not np.any(np.isinf(result[col].values))

    # ------------------------------------------------------------------
    # Multi-series (>1 unique_id, univariate model)
    # ------------------------------------------------------------------
    def test_multi_series_shape(self, fitted_nf_multi_series):
        """Univariate model on multi-series dataset has correct shape."""
        n_paths = 5
        h = 12
        result = fitted_nf_multi_series.simulate(n_paths=n_paths, seed=0)
        n_uids = result["unique_id"].nunique()
        assert n_uids == 2
        assert len(result) == n_uids * n_paths * h

    def test_multi_series_per_uid_paths(self, fitted_nf_multi_series):
        """Each unique_id should have exactly n_paths * H rows."""
        n_paths = 5
        h = 12
        result = fitted_nf_multi_series.simulate(n_paths=n_paths, seed=0)
        for uid in result["unique_id"].unique():
            uid_df = result[result["unique_id"] == uid]
            assert len(uid_df) == n_paths * h
            assert uid_df["sample_id"].nunique() == n_paths

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------
    def test_single_path(self, fitted_nf_distribution):
        """n_paths=1 should work and produce exactly 1 sample."""
        result = fitted_nf_distribution.simulate(n_paths=1, seed=0)
        assert isinstance(result, pd.DataFrame)
        assert result["sample_id"].nunique() == 1
        n_series = result["unique_id"].nunique()
        assert len(result) == n_series * 12  # 1 path × H

    def test_schaake_shuffle_with_mqloss(self, fitted_nf_mqloss):
        """Schaake shuffle works with MQLoss (not just DistributionLoss)."""
        result = fitted_nf_mqloss.simulate(
            n_paths=10, seed=0, method="schaake_shuffle"
        )
        assert isinstance(result, pd.DataFrame)
        assert result["sample_id"].nunique() == 10

    def test_schaake_shuffle_mae_conformal(self, fitted_nf_mae_conformal):
        """Schaake shuffle works for MAE + conformal path."""
        result = fitted_nf_mae_conformal.simulate(
            n_paths=10, seed=0, method="schaake_shuffle"
        )
        assert isinstance(result, pd.DataFrame)
        assert result["sample_id"].nunique() == 10
