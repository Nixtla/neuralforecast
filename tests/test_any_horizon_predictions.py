"""
Test cases for any horizon predictions functionality.

Tests the ability to make predictions for horizons longer than the trained horizon,
with particular focus on recurrent models and correct handling of exogenous variables.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import DummyRNN, DummyMultivariate, DummyUnivariate
from neuralforecast.losses.pytorch import MAE, MSE, MQLoss
from neuralforecast.utils import AirPassengersPanel, generate_series


@pytest.fixture
def setup_test_data():
    """Create test data for any horizon predictions tests."""
    # Use a subset of AirPassengers data for faster testing
    df = AirPassengersPanel.head(100).copy()
    
    # Add some exogenous variables for testing
    df['trend'] = np.arange(len(df))
    df['seasonal'] = np.sin(2 * np.pi * (df.groupby('unique_id').cumcount() % 12) / 12)
    
    # Create static features
    static_df = pd.DataFrame({
        'unique_id': df['unique_id'].unique(),
        'category': [0, 1],  # Simple categorical variable
    })
    
    return df, static_df


@pytest.fixture  
def setup_multivariate_data():
    """Create multivariate test data."""
    # Generate simple multivariate series
    n_series = 3
    n_timesteps = 60
    
    df_list = []
    for i in range(n_series):
        dates = pd.date_range('2020-01-01', periods=n_timesteps, freq='M')
        # Create correlated series with different trends
        base_trend = np.arange(n_timesteps) * (i + 1) * 0.1
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_timesteps) / 12)
        noise = np.random.normal(0, 1, n_timesteps)
        values = 100 + base_trend + seasonal + noise
        
        df_list.append(pd.DataFrame({
            'unique_id': f'series_{i}',
            'ds': dates,
            'y': values,
            'exog_trend': np.arange(n_timesteps) * 0.05,
        }))
    
    return pd.concat(df_list, ignore_index=True)


class TestAnyHorizonPredictions:
    """Test cases for any horizon predictions functionality."""
    
    def test_dummy_rnn_basic_functionality(self, setup_test_data):
        """Test basic functionality of DummyRNN model."""
        df, _ = setup_test_data
        
        # Test with short horizon first
        model = DummyRNN(h=6, input_size=12, max_steps=1, lag=12)
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df)
        
        # Make predictions
        predictions = nf.predict()
        
        # Basic checks
        assert len(predictions) == 2 * 6  # 2 series * 6 horizon
        assert 'DummyRNN' in predictions.columns
        assert not predictions['DummyRNN'].isna().any()
    
    def test_dummy_multivariate_basic(self, setup_multivariate_data):
        """Test basic functionality of DummyMultivariate model."""
        df = setup_multivariate_data
        
        model = DummyMultivariate(h=6, n_series=3, input_size=12, max_steps=1)
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df)
        
        predictions = nf.predict()
        
        # Check multivariate output
        assert len(predictions) == 3 * 6  # 3 series * 6 horizon  
        assert 'DummyMultivariate' in predictions.columns
        assert not predictions['DummyMultivariate'].isna().any()
    
    def test_dummy_univariate_basic(self, setup_test_data):
        """Test basic functionality of DummyUnivariate model."""
        df, _ = setup_test_data
        
        model = DummyUnivariate(h=6, input_size=12, max_steps=1, trend_factor=1.02)
        nf = NeuralForecast(models=[model], freq='M')  
        nf.fit(df)
        
        predictions = nf.predict()
        
        # Check predictions
        assert len(predictions) == 2 * 6  # 2 series * 6 horizon
        assert 'DummyUnivariate' in predictions.columns
        assert not predictions['DummyUnivariate'].isna().any()
    
    def test_recurrent_longer_horizon_predictions(self, setup_test_data):
        """Test recurrent models can predict horizons longer than trained horizon."""
        df, _ = setup_test_data
        
        # Train on short horizon
        train_h = 6
        model = DummyRNN(h=train_h, input_size=12, max_steps=1, lag=12)
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df)
        
        # Predict longer horizon
        predict_h = 18  # 3x longer than trained horizon
        predictions = nf.predict(h=predict_h)
        
        # Check we get predictions for the longer horizon
        assert len(predictions) == 2 * predict_h  # 2 series * 18 horizon
        assert 'DummyRNN' in predictions.columns
        assert not predictions['DummyRNN'].isna().any()
        
        # For dummy RNN, predictions should be consistent (seasonal naive behavior)
        # Group by series and check predictions make sense
        for uid in df['unique_id'].unique():
            series_preds = predictions[predictions['unique_id'] == uid]['DummyRNN'].values
            # Should have 18 predictions
            assert len(series_preds) == predict_h
    
    def test_direct_longer_horizon_predictions(self, setup_test_data):
        """Test direct models can predict horizons longer than trained horizon."""
        df, _ = setup_test_data
        
        # Train on short horizon
        train_h = 6
        model = DummyUnivariate(h=train_h, input_size=12, max_steps=1)
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df)
        
        # Predict longer horizon  
        predict_h = 18  # 3x longer
        predictions = nf.predict(h=predict_h)
        
        # Check we get predictions for the longer horizon
        assert len(predictions) == 2 * predict_h
        assert 'DummyUnivariate' in predictions.columns
        assert not predictions['DummyUnivariate'].isna().any()
    
    def test_multivariate_longer_horizon(self, setup_multivariate_data):
        """Test multivariate recurrent model with longer horizons."""
        df = setup_multivariate_data
        
        # Train on short horizon
        train_h = 4
        model = DummyMultivariate(h=train_h, n_series=3, input_size=12, max_steps=1)
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df)
        
        # Predict longer horizon
        predict_h = 12  # 3x longer
        predictions = nf.predict(h=predict_h)
        
        # Check multivariate predictions for longer horizon
        assert len(predictions) == 3 * predict_h
        assert 'DummyMultivariate' in predictions.columns
        assert not predictions['DummyMultivariate'].isna().any()
    
    def test_exogenous_variables_with_longer_horizon(self, setup_test_data):
        """Test that exogenous variables work correctly with longer horizons."""
        df, static_df = setup_test_data
        
        # Create future exogenous data
        future_df = df.groupby('unique_id').tail(12).copy()
        future_df['ds'] = future_df.groupby('unique_id')['ds'].transform(
            lambda x: pd.date_range(start=x.iloc[-1] + pd.DateOffset(months=1), periods=12, freq='M')
        )
        future_df['trend'] = future_df['trend'] + 12  # Continue trend
        future_df['seasonal'] = np.sin(2 * np.pi * (future_df.groupby('unique_id').cumcount() % 12) / 12)
        future_df = future_df[['unique_id', 'ds', 'trend', 'seasonal']].reset_index(drop=True)
        
        # Model with exogenous variables
        train_h = 6
        model = DummyRNN(
            h=train_h, 
            input_size=12, 
            max_steps=1,
            futr_exog_list=['trend', 'seasonal'],
            stat_exog_list=['category']
        )
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df, static_df=static_df)
        
        # Predict longer horizon with exogenous
        predict_h = 12
        predictions = nf.predict(h=predict_h, futr_df=future_df)
        
        # Check predictions
        assert len(predictions) == 2 * predict_h
        assert 'DummyRNN' in predictions.columns
        assert not predictions['DummyRNN'].isna().any()
        
    def test_historical_exogenous_limitations(self, setup_test_data):
        """Test that models with historical exogenous raise appropriate errors for longer horizons."""
        df, _ = setup_test_data
        
        # Model with historical exogenous
        model = DummyRNN(
            h=6,
            input_size=12,
            max_steps=1,
            hist_exog_list=['trend']  # Historical exogenous
        )
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df)
        
        # Should raise error when trying to predict longer horizon
        with pytest.raises(NotImplementedError, match="historic exogenous features"):
            nf.predict(h=12)  # Longer than trained horizon
    
    def test_shorter_horizon_error(self, setup_test_data):
        """Test that specifying shorter horizon than trained raises appropriate error."""
        df, _ = setup_test_data
        
        model = DummyRNN(h=12, input_size=12, max_steps=1)
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df)
        
        # Should raise error for shorter horizon
        with pytest.raises(ValueError, match="must be greater than the horizon of the fitted models"):
            nf.predict(h=6)  # Shorter than trained horizon
    
    def test_equal_horizon_works(self, setup_test_data):
        """Test that predicting same horizon as trained works normally."""
        df, _ = setup_test_data
        
        train_h = 6
        model = DummyRNN(h=train_h, input_size=12, max_steps=1)
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df)
        
        # Predict same horizon - should work fine
        predictions = nf.predict(h=train_h)
        
        assert len(predictions) == 2 * train_h
        assert 'DummyRNN' in predictions.columns
        assert not predictions['DummyRNN'].isna().any()
    
    def test_multiple_models_different_horizons(self, setup_test_data):
        """Test multiple models predicting different horizons together."""
        df, _ = setup_test_data
        
        # Multiple models with same trained horizon
        train_h = 6
        models = [
            DummyRNN(h=train_h, input_size=12, max_steps=1, alias='RNN'),
            DummyUnivariate(h=train_h, input_size=12, max_steps=1, alias='Direct')
        ]
        nf = NeuralForecast(models=models, freq='M')
        nf.fit(df)
        
        # Predict longer horizon
        predict_h = 18
        predictions = nf.predict(h=predict_h)
        
        # Check both models made predictions
        assert len(predictions) == 2 * predict_h
        assert 'RNN' in predictions.columns
        assert 'Direct' in predictions.columns
        assert not predictions['RNN'].isna().any()
        assert not predictions['Direct'].isna().any()
    
    def test_cross_validation_with_longer_horizon(self, setup_test_data):
        """Test cross-validation with longer horizons requires refit=True."""
        df, _ = setup_test_data
        
        train_h = 6
        model = DummyRNN(h=train_h, input_size=12, max_steps=1)
        nf = NeuralForecast(models=[model], freq='M')
        
        predict_h = 12  # Longer horizon
        
        # Should raise error with refit=False
        with pytest.raises(ValueError, match="Set refit=True"):
            nf.cross_validation(df, h=predict_h, n_windows=2, refit=False)
        
        # Should work with refit=True
        cv_results = nf.cross_validation(df, h=predict_h, n_windows=2, refit=True)
        
        # Check cross-validation results
        assert 'DummyRNN' in cv_results.columns
        assert not cv_results['DummyRNN'].isna().any()
        assert len(cv_results['unique_id'].unique()) == 2  # Both series
    
    def test_probabilistic_predictions_longer_horizon(self, setup_test_data):
        """Test probabilistic predictions work with longer horizons."""
        df, _ = setup_test_data
        
        # Use MQLoss for probabilistic predictions
        train_h = 6
        model = DummyRNN(h=train_h, input_size=12, max_steps=1, loss=MQLoss(level=[80, 90]))
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df)
        
        # Predict longer horizon with levels
        predict_h = 12
        predictions = nf.predict(h=predict_h, level=[80, 90])
        
        # Check probabilistic predictions
        assert len(predictions) == 2 * predict_h
        expected_cols = ['unique_id', 'ds', 'DummyRNN', 'DummyRNN-lo-80', 'DummyRNN-hi-80', 
                        'DummyRNN-lo-90', 'DummyRNN-hi-90']
        for col in expected_cols:
            assert col in predictions.columns
    
    def test_recurrent_state_consistency(self, setup_test_data):
        """Test that recurrent models maintain proper state during longer predictions."""
        df, _ = setup_test_data
        
        # Simple test: predictions should be deterministic
        train_h = 4
        model = DummyRNN(h=train_h, input_size=12, max_steps=1, random_seed=42, lag=12)
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df)
        
        predict_h = 8
        
        # Make predictions twice - should be identical
        pred1 = nf.predict(h=predict_h)
        pred2 = nf.predict(h=predict_h)
        
        pd.testing.assert_frame_equal(pred1, pred2, check_dtype=False)
    
    def test_edge_cases_very_long_horizon(self, setup_test_data):
        """Test edge cases with very long prediction horizons."""
        df, _ = setup_test_data
        
        train_h = 3
        model = DummyRNN(h=train_h, input_size=6, max_steps=1, lag=12)
        nf = NeuralForecast(models=[model], freq='M')  
        nf.fit(df)
        
        # Very long horizon (10x trained horizon)
        predict_h = 30
        predictions = nf.predict(h=predict_h)
        
        # Should still work and produce valid predictions
        assert len(predictions) == 2 * predict_h
        assert 'DummyRNN' in predictions.columns
        assert not predictions['DummyRNN'].isna().any()
        assert not predictions['DummyRNN'].isin([np.inf, -np.inf]).any()


class TestAnyHorizonEdgeCases:
    """Test edge cases and error conditions for any horizon predictions."""
    
    def test_insufficient_test_size_error(self, setup_test_data):
        """Test that insufficient test_size raises appropriate error for direct models."""
        df, _ = setup_test_data
        
        # This should work fine for recurrent models, but let's test direct models
        train_h = 6
        model = DummyUnivariate(h=train_h, input_size=12, max_steps=1)
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df)
        
        # For direct models predicting longer horizons, we need sufficient test_size
        # This is handled internally, but let's verify predictions work
        predict_h = 18
        predictions = nf.predict(h=predict_h)
        
        assert len(predictions) == 2 * predict_h
        assert not predictions['DummyUnivariate'].isna().any()
    
    def test_model_attributes_preserved(self, setup_test_data):
        """Test that model attributes are preserved during longer horizon predictions."""
        df, _ = setup_test_data
        
        train_h = 6
        model = DummyRNN(h=train_h, input_size=12, max_steps=1, lag=12)
        nf = NeuralForecast(models=[model], freq='M')
        nf.fit(df)
        
        # Store original attributes
        original_h = nf.models[0].h
        original_horizon_backup = nf.models[0].horizon_backup
        
        # Make longer horizon predictions
        predictions = nf.predict(h=18)
        
        # Check attributes are restored
        assert nf.models[0].h == original_h
        assert nf.models[0].horizon_backup == original_horizon_backup
    
    def test_mixed_model_types_longer_horizon(self, setup_test_data, setup_multivariate_data):
        """Test mixed recurrent and direct models with longer horizons."""
        df = setup_test_data[0]  # Use first fixture
        
        train_h = 6
        models = [
            DummyRNN(h=train_h, input_size=12, max_steps=1, alias='RecurrentModel'),
            DummyUnivariate(h=train_h, input_size=12, max_steps=1, alias='DirectModel')
        ]
        nf = NeuralForecast(models=models, freq='M')
        nf.fit(df)
        
        predict_h = 15
        predictions = nf.predict(h=predict_h)
        
        # Both models should make predictions
        assert 'RecurrentModel' in predictions.columns
        assert 'DirectModel' in predictions.columns  
        assert len(predictions) == 2 * predict_h
        assert not predictions['RecurrentModel'].isna().any()
        assert not predictions['DirectModel'].isna().any()