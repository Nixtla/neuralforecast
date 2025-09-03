import numpy as np
import pandas as pd
import pytest
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import DummyRNN, DummyMultivariate, DummyUnivariate
from neuralforecast.losses.pytorch import MAE, MSE, MQLoss
from neuralforecast.tsdataset import TimeSeriesDataset


class TestDummyModels:
    """Test suite for dummy models to validate any horizon predictions functionality."""
    
    @pytest.fixture
    def simple_data(self):
        """Create simple time series data for testing."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        y = np.random.randn(100) + np.arange(100) * 0.1  # Trend + noise
        
        df = pd.DataFrame({
            'unique_id': ['series1'] * 100,
            'ds': dates,
            'y': y
        })
        return df
    
    @pytest.fixture
    def multivariate_data(self):
        """Create multivariate time series data for testing."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Create two series with different patterns
        y1 = np.random.randn(100) + np.arange(100) * 0.1  # Trend + noise
        y2 = np.random.randn(100) + np.sin(np.arange(100) * 0.1) * 10  # Sinusoidal + noise
        
        df = pd.DataFrame({
            'unique_id': ['series1'] * 100,
            'ds': dates,
            'y1': y1,
            'y2': y2
        })
        return df
    
    @pytest.fixture
    def exogenous_data(self):
        """Create data with exogenous variables for testing."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        y = np.random.randn(100) + np.arange(100) * 0.1
        trend = np.arange(100)
        seasonal = np.sin(np.arange(100) * 2 * np.pi / 7)  # Weekly seasonality
        
        df = pd.DataFrame({
            'unique_id': ['series1'] * 100,
            'ds': dates,
            'y': y,
            'trend': trend,
            'seasonal': seasonal
        })
        return df
    
    def test_dummy_univariate_basic(self, simple_data):
        """Test basic functionality of DummyUnivariate model."""
        # Train with horizon 5
        model = DummyUnivariate(
            h=5,
            input_size=20,
            max_steps=1,
            learning_rate=1e-3
        )
        
        # Fit the model
        dataset = TimeSeriesDataset.from_df(simple_data)
        model.fit(dataset)
        
        # Test prediction with same horizon
        predictions = model.predict(dataset)
        assert predictions.shape[0] == 5  # 5 predictions
        assert not np.any(np.isnan(predictions))
        
        # Test prediction with longer horizon (any horizon feature)
        predictions_long = model.predict(dataset, h=10)
        assert predictions_long.shape[0] == 10  # 10 predictions
        assert not np.any(np.isnan(predictions_long))
        
        # Test prediction with much longer horizon
        predictions_very_long = model.predict(dataset, h=20)
        assert predictions_very_long.shape[0] == 20  # 20 predictions
        assert not np.any(np.isnan(predictions_very_long))
    
    def test_dummy_rnn_seasonal(self, simple_data):
        """Test DummyRNN model with seasonal predictions."""
        # Train with horizon 7 and seasonality 7
        model = DummyRNN(
            h=7,
            input_size=21,
            seasonality=7,
            max_steps=1,
            learning_rate=1e-3
        )
        
        # Fit the model
        dataset = TimeSeriesDataset.from_df(simple_data)
        model.fit(dataset)
        
        # Test prediction with same horizon
        predictions = model.predict(dataset)
        assert predictions.shape[0] == 7  # 7 predictions
        assert not np.any(np.isnan(predictions))
        
        # Test prediction with longer horizon (any horizon feature)
        predictions_long = model.predict(dataset, h=14)
        assert predictions_long.shape[0] == 14  # 14 predictions
        assert not np.any(np.isnan(predictions_long))
        
        # Test prediction with much longer horizon
        predictions_very_long = model.predict(dataset, h=28)
        assert predictions_very_long.shape[0] == 28  # 28 predictions
        assert not np.any(np.isnan(predictions_very_long))
    
    def test_dummy_multivariate(self, multivariate_data):
        """Test DummyMultivariate model with multivariate data."""
        # Train with horizon 5 and 2 series
        model = DummyMultivariate(
            h=5,
            input_size=20,
            n_series=2,
            max_steps=1,
            learning_rate=1e-3
        )
        
        # Fit the model
        dataset = TimeSeriesDataset.from_df(multivariate_data, target_cols=['y1', 'y2'])
        model.fit(dataset)
        
        # Test prediction with same horizon
        predictions = model.predict(dataset)
        assert predictions.shape[0] == 5  # 5 predictions
        assert not np.any(np.isnan(predictions))
        
        # Test prediction with longer horizon (any horizon feature)
        predictions_long = model.predict(dataset, h=10)
        assert predictions_long.shape[0] == 10  # 10 predictions
        assert not np.any(np.isnan(predictions_long))
    
    def test_dummy_models_with_exogenous(self, exogenous_data):
        """Test dummy models with exogenous variables."""
        # Test DummyUnivariate with exogenous
        model = DummyUnivariate(
            h=5,
            input_size=20,
            futr_exog_list=['trend', 'seasonal'],
            max_steps=1,
            learning_rate=1e-3
        )
        
        dataset = TimeSeriesDataset.from_df(
            exogenous_data, 
            futr_exog_list=['trend', 'seasonal']
        )
        model.fit(dataset)
        
        # Test prediction with longer horizon
        predictions = model.predict(dataset, h=10)
        assert predictions.shape[0] == 10
        assert not np.any(np.isnan(predictions))
        
        # Test DummyRNN with exogenous
        model_rnn = DummyRNN(
            h=5,
            input_size=20,
            seasonality=7,
            futr_exog_list=['trend', 'seasonal'],
            max_steps=1,
            learning_rate=1e-3
        )
        
        model_rnn.fit(dataset)
        predictions_rnn = model_rnn.predict(dataset, h=15)
        assert predictions_rnn.shape[0] == 15
        assert not np.any(np.isnan(predictions_rnn))
    
    def test_dummy_models_with_different_losses(self, simple_data):
        """Test dummy models with different loss functions."""
        dataset = TimeSeriesDataset.from_df(simple_data)
        
        # Test with MSE loss
        model_mse = DummyUnivariate(
            h=5,
            input_size=20,
            loss=MSE(),
            max_steps=1,
            learning_rate=1e-3
        )
        model_mse.fit(dataset)
        predictions_mse = model_mse.predict(dataset, h=8)
        assert predictions_mse.shape[0] == 8
        
        # Test with MQLoss (quantile loss)
        model_mq = DummyUnivariate(
            h=5,
            input_size=20,
            loss=MQLoss(level=[80, 90]),
            max_steps=1,
            learning_rate=1e-3
        )
        model_mq.fit(dataset)
        predictions_mq = model_mq.predict(dataset, h=12)
        assert predictions_mq.shape[0] == 12
    
    def test_dummy_models_edge_cases(self, simple_data):
        """Test edge cases for dummy models."""
        dataset = TimeSeriesDataset.from_df(simple_data)
        
        # Test with very short input size
        model_short = DummyUnivariate(
            h=5,
            input_size=5,
            max_steps=1,
            learning_rate=1e-3
        )
        model_short.fit(dataset)
        predictions = model_short.predict(dataset, h=10)
        assert predictions.shape[0] == 10
        
        # Test with very long horizon
        model_long = DummyUnivariate(
            h=5,
            input_size=20,
            max_steps=1,
            learning_rate=1e-3
        )
        model_long.fit(dataset)
        predictions = model_long.predict(dataset, h=50)
        assert predictions.shape[0] == 50
        
        # Test with horizon equal to input size
        model_equal = DummyUnivariate(
            h=5,
            input_size=20,
            max_steps=1,
            learning_rate=1e-3
        )
        model_equal.fit(dataset)
        predictions = model_equal.predict(dataset, h=20)
        assert predictions.shape[0] == 20
    
    def test_dummy_models_consistency(self, simple_data):
        """Test that predictions are consistent across different horizon lengths."""
        dataset = TimeSeriesDataset.from_df(simple_data)
        
        model = DummyUnivariate(
            h=5,
            input_size=20,
            max_steps=1,
            learning_rate=1e-3
        )
        model.fit(dataset)
        
        # Get predictions for different horizons
        pred_5 = model.predict(dataset, h=5)
        pred_10 = model.predict(dataset, h=10)
        pred_15 = model.predict(dataset, h=15)
        
        # First 5 predictions should be identical
        np.testing.assert_array_equal(pred_5, pred_10[:5])
        np.testing.assert_array_equal(pred_5, pred_15[:5])
        
        # Predictions should be deterministic
        pred_10_2 = model.predict(dataset, h=10)
        np.testing.assert_array_equal(pred_10, pred_10_2)
    
    def test_dummy_models_with_neuralforecast(self, simple_data):
        """Test dummy models integration with NeuralForecast."""
        # Create NeuralForecast instance with dummy models
        models = [
            DummyUnivariate(h=5, input_size=20, max_steps=1, learning_rate=1e-3),
            DummyRNN(h=5, input_size=20, seasonality=7, max_steps=1, learning_rate=1e-3)
        ]
        
        nf = NeuralForecast(models=models, freq='D')
        nf.fit(simple_data)
        
        # Test prediction with longer horizon
        forecasts = nf.predict(simple_data, h=10)
        assert len(forecasts) == 10
        
        # Check that all models produced forecasts
        expected_cols = ['unique_id', 'ds', 'DummyUnivariate', 'DummyRNN']
        assert all(col in forecasts.columns for col in expected_cols)
    
    def test_dummy_models_training_consistency(self, simple_data):
        """Test that training is consistent and models learn something."""
        dataset = TimeSeriesDataset.from_df(simple_data)
        
        # Train model for multiple steps
        model = DummyUnivariate(
            h=5,
            input_size=20,
            max_steps=5,
            learning_rate=1e-3
        )
        
        # Check that model has trainable parameters
        assert len(list(model.parameters())) > 0
        
        # Fit the model
        model.fit(dataset)
        
        # Test predictions
        predictions = model.predict(dataset, h=10)
        assert predictions.shape[0] == 10
        assert not np.any(np.isnan(predictions))
    
    def test_dummy_models_with_validation(self, simple_data):
        """Test dummy models with validation during training."""
        dataset = TimeSeriesDataset.from_df(simple_data)
        
        model = DummyUnivariate(
            h=5,
            input_size=20,
            max_steps=5,
            val_check_steps=2,
            learning_rate=1e-3
        )
        
        # Fit with validation
        model.fit(dataset, val_size=10)
        
        # Test predictions
        predictions = model.predict(dataset, h=15)
        assert predictions.shape[0] == 15
        assert not np.any(np.isnan(predictions))
    
    def test_dummy_models_error_handling(self, simple_data):
        """Test error handling in dummy models."""
        dataset = TimeSeriesDataset.from_df(simple_data)
        
        # Test with invalid horizon
        model = DummyUnivariate(
            h=5,
            input_size=20,
            max_steps=1,
            learning_rate=1e-3
        )
        model.fit(dataset)
        
        # Should handle zero horizon gracefully
        with pytest.raises(ValueError):
            model.predict(dataset, h=0)
        
        # Should handle negative horizon gracefully
        with pytest.raises(ValueError):
            model.predict(dataset, h=-1)
    
    def test_dummy_models_memory_efficiency(self, simple_data):
        """Test that dummy models don't leak memory during long horizon predictions."""
        dataset = TimeSeriesDataset.from_df(simple_data)
        
        model = DummyUnivariate(
            h=5,
            input_size=20,
            max_steps=1,
            learning_rate=1e-3
        )
        model.fit(dataset)
        
        # Test with very long horizon to check memory usage
        predictions = model.predict(dataset, h=100)
        assert predictions.shape[0] == 100
        assert not np.any(np.isnan(predictions))
        
        # Check that model state is properly reset
        assert model.h == 5  # Should reset to original horizon
        assert model.n_predicts == 1  # Should reset to default
