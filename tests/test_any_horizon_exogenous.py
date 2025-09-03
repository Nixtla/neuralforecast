"""
Test cases for exogenous variable handling in any horizon predictions.

These tests ensure that future, historical, and static exogenous variables
are correctly handled when predicting horizons longer than the trained horizon.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import DummyRNN, DummyMultivariate, DummyUnivariate
from neuralforecast.losses.pytorch import MAE
from neuralforecast.utils import generate_series


@pytest.fixture
def setup_exog_data():
    """Create test data with various exogenous variables."""
    # Generate base series
    n_series = 2
    n_timesteps = 60
    dates = pd.date_range('2020-01-01', periods=n_timesteps, freq='M')
    
    df_list = []
    for i in range(n_series):
        # Base trend and seasonal patterns
        trend = np.arange(n_timesteps) * 0.1
        seasonal = 5 * np.sin(2 * np.pi * np.arange(n_timesteps) / 12)
        noise = np.random.normal(0, 2, n_timesteps)
        y = 100 + trend + seasonal + noise
        
        # Various exogenous variables
        df_list.append(pd.DataFrame({
            'unique_id': f'series_{i}',
            'ds': dates,
            'y': y,
            # Future exogenous (can be provided for future periods)
            'temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(n_timesteps) / 12) + np.random.normal(0, 1, n_timesteps),
            'promotion': np.random.choice([0, 1], n_timesteps, p=[0.8, 0.2]),
            # Historical exogenous (derived from past values)
            'y_lag1': np.concatenate([[np.nan], y[:-1]]),
            'y_rolling_mean': pd.Series(y).rolling(3, min_periods=1).mean().values,
            # Mixed variables that could be both
            'economic_index': 1000 + np.arange(n_timesteps) * 2 + np.random.normal(0, 5, n_timesteps),
        }))
    
    df = pd.concat(df_list, ignore_index=True)
    
    # Static exogenous
    static_df = pd.DataFrame({
        'unique_id': [f'series_{i}' for i in range(n_series)],
        'region': ['north', 'south'],
        'category': [1, 2],
        'market_size': [1000, 1500],
    })
    
    return df, static_df


@pytest.fixture
def setup_future_exog():
    """Create future exogenous data for testing."""
    def create_future_data(df, horizon):
        future_dfs = []
        for uid in df['unique_id'].unique():
            series_df = df[df['unique_id'] == uid].copy()
            last_date = series_df['ds'].max()
            
            # Generate future dates
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1), 
                periods=horizon, 
                freq='M'
            )
            
            # Continue patterns for future exogenous
            future_temp = 20 + 10 * np.sin(2 * np.pi * np.arange(len(series_df), len(series_df) + horizon) / 12)
            future_promo = np.random.choice([0, 1], horizon, p=[0.8, 0.2])
            future_econ = series_df['economic_index'].iloc[-1] + np.arange(1, horizon + 1) * 2
            
            future_dfs.append(pd.DataFrame({
                'unique_id': uid,
                'ds': future_dates,
                'temperature': future_temp,
                'promotion': future_promo,
                'economic_index': future_econ,
            }))
        
        return pd.concat(future_dfs, ignore_index=True)
    
    return create_future_data


class TestExogenousAnyHorizon:
    """Test exogenous variable handling with any horizon predictions."""
    
    def test_future_exog_longer_horizon(self, setup_exog_data, setup_future_exog):
        """Test future exogenous variables work with longer horizons."""
        df, static_df = setup_exog_data
        create_future = setup_future_exog
        
        train_h = 6
        predict_h = 18
        
        # Model with future exogenous
        model = DummyRNN(
            h=train_h,
            input_size=12, 
            max_steps=1,
            futr_exog_list=['temperature', 'promotion']
        )
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df)
        
        # Create future exogenous data
        future_df = create_future(df, predict_h)
        
        # Make predictions with longer horizon
        predictions = nf.predict(h=predict_h, futr_df=future_df)
        
        # Validate results
        assert len(predictions) == 2 * predict_h
        assert 'DummyRNN' in predictions.columns
        assert not predictions['DummyRNN'].isna().any()
    
    def test_static_exog_longer_horizon(self, setup_exog_data):
        """Test static exogenous variables work with longer horizons."""
        df, static_df = setup_exog_data
        
        train_h = 4
        predict_h = 12
        
        # Model with static exogenous
        model = DummyRNN(
            h=train_h,
            input_size=12,
            max_steps=1, 
            stat_exog_list=['region', 'category', 'market_size']
        )
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df, static_df=static_df)
        
        # Predict longer horizon - static exog should be handled automatically
        predictions = nf.predict(h=predict_h)
        
        assert len(predictions) == 2 * predict_h
        assert 'DummyRNN' in predictions.columns
        assert not predictions['DummyRNN'].isna().any()
    
    def test_historical_exog_error_longer_horizon(self, setup_exog_data):
        """Test that historical exog raises error for longer horizons."""
        df, _ = setup_exog_data
        
        train_h = 6
        predict_h = 12
        
        # Model with historical exogenous
        model = DummyRNN(
            h=train_h,
            input_size=12,
            max_steps=1,
            hist_exog_list=['y_lag1', 'y_rolling_mean']
        )
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df)
        
        # Should raise error for historical exog with longer horizon
        with pytest.raises(NotImplementedError, match="historic exogenous features"):
            nf.predict(h=predict_h)
    
    def test_mixed_exog_types_longer_horizon(self, setup_exog_data, setup_future_exog):
        """Test combination of different exogenous types with longer horizons."""
        df, static_df = setup_exog_data
        create_future = setup_future_exog
        
        train_h = 6
        predict_h = 18
        
        # Model with future and static exogenous (no historical)
        model = DummyRNN(
            h=train_h,
            input_size=12,
            max_steps=1,
            futr_exog_list=['temperature', 'economic_index'],
            stat_exog_list=['region', 'market_size']
        )
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df, static_df=static_df)
        
        # Create future data
        future_df = create_future(df, predict_h)
        
        # Predict with mixed exogenous
        predictions = nf.predict(h=predict_h, futr_df=future_df)
        
        assert len(predictions) == 2 * predict_h
        assert 'DummyRNN' in predictions.columns
        assert not predictions['DummyRNN'].isna().any()
    
    def test_multivariate_exog_longer_horizon(self, setup_exog_data, setup_future_exog):
        """Test multivariate models with exogenous variables and longer horizons."""
        df, static_df = setup_exog_data
        create_future = setup_future_exog
        
        train_h = 4
        predict_h = 12
        
        # Multivariate model with exogenous
        model = DummyMultivariate(
            h=train_h,
            n_series=2,
            input_size=12,
            max_steps=1,
            futr_exog_list=['temperature'],
            stat_exog_list=['category']
        )
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df, static_df=static_df)
        
        future_df = create_future(df, predict_h)
        predictions = nf.predict(h=predict_h, futr_df=future_df)
        
        assert len(predictions) == 2 * predict_h
        assert 'DummyMultivariate' in predictions.columns
        assert not predictions['DummyMultivariate'].isna().any()
    
    def test_direct_model_exog_longer_horizon(self, setup_exog_data, setup_future_exog):
        """Test direct (non-recurrent) models with exogenous and longer horizons."""
        df, static_df = setup_exog_data
        create_future = setup_future_exog
        
        train_h = 6
        predict_h = 18
        
        # Direct model with future and static exog
        model = DummyUnivariate(
            h=train_h,
            input_size=12,
            max_steps=1,
            futr_exog_list=['temperature', 'promotion'],
            stat_exog_list=['region']
        )
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df, static_df=static_df)
        
        future_df = create_future(df, predict_h)
        predictions = nf.predict(h=predict_h, futr_df=future_df)
        
        assert len(predictions) == 2 * predict_h
        assert 'DummyUnivariate' in predictions.columns
        assert not predictions['DummyUnivariate'].isna().any()
    
    def test_missing_future_exog_error(self, setup_exog_data):
        """Test error when required future exogenous is missing."""
        df, _ = setup_exog_data
        
        train_h = 6
        predict_h = 12
        
        # Model requiring future exog
        model = DummyRNN(
            h=train_h,
            input_size=12,
            max_steps=1,
            futr_exog_list=['temperature', 'promotion']
        )
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df)
        
        # Try to predict without providing future exog
        with pytest.raises(ValueError, match="Models require the following future exogenous features"):
            nf.predict(h=predict_h)
    
    def test_incomplete_future_exog_error(self, setup_exog_data, setup_future_exog):
        """Test error when future exogenous data is incomplete."""
        df, _ = setup_exog_data
        create_future = setup_future_exog
        
        train_h = 6
        predict_h = 12
        
        model = DummyRNN(
            h=train_h,
            input_size=12,
            max_steps=1,
            futr_exog_list=['temperature', 'promotion']
        )
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df)
        
        # Create incomplete future data (missing 'promotion')
        future_df = create_future(df, predict_h)
        incomplete_future_df = future_df.drop(columns=['promotion'])
        
        with pytest.raises(ValueError, match="missing from `futr_df`"):
            nf.predict(h=predict_h, futr_df=incomplete_future_df)
    
    def test_exog_scaling_consistency(self, setup_exog_data, setup_future_exog):
        """Test that exogenous variables are scaled consistently for longer horizons."""
        df, static_df = setup_exog_data
        create_future = setup_future_exog
        
        train_h = 6
        predict_h = 18
        
        # Test with scaling
        model = DummyRNN(
            h=train_h,
            input_size=12,
            max_steps=1,
            futr_exog_list=['temperature', 'economic_index'],
            stat_exog_list=['market_size'],
            scaler_type='standard'
        )
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df, static_df=static_df)
        
        future_df = create_future(df, predict_h)
        
        # Should work with scaling
        predictions = nf.predict(h=predict_h, futr_df=future_df)
        
        assert len(predictions) == 2 * predict_h
        assert not predictions['DummyRNN'].isna().any()
        assert not predictions['DummyRNN'].isin([np.inf, -np.inf]).any()
    
    def test_cross_validation_with_exog_longer_horizon(self, setup_exog_data, setup_future_exog):
        """Test cross-validation with exogenous variables and longer horizons."""
        df, static_df = setup_exog_data
        
        train_h = 4
        predict_h = 8
        
        # Model with future and static exog
        model = DummyRNN(
            h=train_h,
            input_size=12,
            max_steps=1,
            futr_exog_list=['temperature'],
            stat_exog_list=['region']
        )
        nf = NeuralForecast(models=[model], freq='M')
        
        # Cross-validation with refit=True (required for longer horizons)
        cv_results = nf.cross_validation(
            df, 
            static_df=static_df,
            h=predict_h,
            n_windows=2,
            refit=True
        )
        
        # Check results
        assert 'DummyRNN' in cv_results.columns
        assert not cv_results['DummyRNN'].isna().any()
        assert len(cv_results['unique_id'].unique()) == 2
    
    def test_exog_variable_order_independence(self, setup_exog_data, setup_future_exog):
        """Test that order of exogenous variables doesn't affect results."""
        df, static_df = setup_exog_data
        create_future = setup_future_exog
        
        train_h = 6
        predict_h = 12
        
        # Model 1: specific order
        model1 = DummyRNN(
            h=train_h,
            input_size=12,
            max_steps=1,
            futr_exog_list=['temperature', 'promotion', 'economic_index'],
            stat_exog_list=['region', 'category'],
            random_seed=42
        )
        nf1 = NeuralForecast(models=[model1], freq='M')
        nf1.fit(df, static_df=static_df)
        
        # Model 2: different order
        model2 = DummyRNN(
            h=train_h,
            input_size=12,
            max_steps=1,
            futr_exog_list=['promotion', 'economic_index', 'temperature'],
            stat_exog_list=['category', 'region'],
            random_seed=42
        )
        nf2 = NeuralForecast(models=[model2], freq='M')
        nf2.fit(df, static_df=static_df)
        
        future_df = create_future(df, predict_h)
        
        pred1 = nf1.predict(h=predict_h, futr_df=future_df)
        pred2 = nf2.predict(h=predict_h, futr_df=future_df)
        
        # Results should be very similar (allowing for small numerical differences)
        np.testing.assert_allclose(
            pred1['DummyRNN'].values,
            pred2['DummyRNN'].values,
            rtol=1e-3
        )


class TestExogenousEdgeCases:
    """Test edge cases for exogenous variable handling."""
    
    def test_empty_exog_lists(self, setup_exog_data):
        """Test models with empty exogenous lists work for longer horizons."""
        df, _ = setup_exog_data
        
        train_h = 6
        predict_h = 15
        
        # Model with no exogenous variables
        model = DummyRNN(
            h=train_h,
            input_size=12,
            max_steps=1,
            futr_exog_list=[],
            hist_exog_list=[],
            stat_exog_list=[]
        )
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df)
        
        predictions = nf.predict(h=predict_h)
        
        assert len(predictions) == 2 * predict_h
        assert not predictions['DummyRNN'].isna().any()
    
    def test_single_exog_variable(self, setup_exog_data, setup_future_exog):
        """Test with single exogenous variable of each type."""
        df, static_df = setup_exog_data
        create_future = setup_future_exog
        
        train_h = 6
        predict_h = 12
        
        # Test each type individually
        test_cases = [
            {'futr_exog_list': ['temperature'], 'need_future': True},
            {'stat_exog_list': ['region'], 'need_future': False},
        ]
        
        for case in test_cases:
            model = DummyRNN(h=train_h, input_size=12, max_steps=1, **case)
            nf = NeuralForecast(models=[model], freq='M')
            
            if 'stat_exog_list' in case:
                nf.fit(df, static_df=static_df)
            else:
                nf.fit(df)
            
            if case.get('need_future', False):
                future_df = create_future(df, predict_h)
                predictions = nf.predict(h=predict_h, futr_df=future_df)
            else:
                predictions = nf.predict(h=predict_h)
            
            assert len(predictions) == 2 * predict_h
            assert not predictions['DummyRNN'].isna().any()
    
    def test_large_number_exog_variables(self, setup_exog_data, setup_future_exog):
        """Test with many exogenous variables."""
        df, static_df = setup_exog_data
        create_future = setup_future_exog
        
        # Add more exogenous variables
        for i in range(10):
            df[f'extra_futr_{i}'] = np.random.normal(0, 1, len(df))
            static_df[f'extra_stat_{i}'] = [i, i+1]
        
        train_h = 6
        predict_h = 12
        
        futr_vars = [f'extra_futr_{i}' for i in range(10)] + ['temperature']
        stat_vars = [f'extra_stat_{i}' for i in range(10)] + ['region']
        
        model = DummyRNN(
            h=train_h,
            input_size=12,
            max_steps=1,
            futr_exog_list=futr_vars,
            stat_exog_list=stat_vars
        )
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df, static_df=static_df)
        
        # Create extended future data
        future_df = create_future(df, predict_h)
        for i in range(10):
            future_df[f'extra_futr_{i}'] = np.random.normal(0, 1, len(future_df))
        
        predictions = nf.predict(h=predict_h, futr_df=future_df)
        
        assert len(predictions) == 2 * predict_h
        assert not predictions['DummyRNN'].isna().any()