import numpy as np
import pandas as pd
import pytest
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import RNN, LSTM, GRU, TCN, DeepAR, MLP
from neuralforecast.losses.pytorch import MAE, MSE, MQLoss
from neuralforecast.tsdataset import TimeSeriesDataset


class TestAnyHorizonPredictions:
    """Test suite for any horizon predictions functionality."""
    
    @pytest.fixture
    def simple_data(self):
        """Create simple time series data for testing."""
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        np.random.seed(42)
        y = np.random.randn(200) + np.arange(200) * 0.1  # Trend + noise
        
        df = pd.DataFrame({
            'unique_id': ['series1'] * 200,
            'ds': dates,
            'y': y
        })
        return df
    
    @pytest.fixture
    def multivariate_data(self):
        """Create multivariate time series data for testing."""
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        # Create two series with different patterns
        y1 = np.random.randn(200) + np.arange(200) * 0.1  # Trend + noise
        y2 = np.random.randn(200) + np.sin(np.arange(200) * 0.1) * 10  # Sinusoidal + noise
        
        df = pd.DataFrame({
            'unique_id': ['series1'] * 200,
            'ds': dates,
            'y1': y1,
            'y2': y2
        })
        return df
    
    @pytest.fixture
    def exogenous_data(self):
        """Create data with exogenous variables for testing."""
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        y = np.random.randn(200) + np.arange(200) * 0.1
        trend = np.arange(200)
        seasonal = np.sin(np.arange(200) * 2 * np.pi / 7)  # Weekly seasonality
        
        df = pd.DataFrame({
            'unique_id': ['series1'] * 200,
            'ds': dates,
            'y': y,
            'trend': trend,
            'seasonal': seasonal
        })
        return df
    
    def test_rnn_any_horizon_basic(self, simple_data):
        """Test RNN model with any horizon predictions."""
        # Train with horizon 12
        model = RNN(
            h=12,
            input_size=24,
            encoder_hidden_size=32,
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        
        # Fit the model
        dataset = TimeSeriesDataset.from_df(simple_data)
        model.fit(dataset)
        
        # Test prediction with same horizon
        predictions = model.predict(dataset)
        assert predictions.shape[0] == 12  # 12 predictions
        
        # Test prediction with longer horizon (any horizon feature)
        predictions_long = model.predict(dataset, h=24)
        assert predictions_long.shape[0] == 24  # 24 predictions
        
        # Test prediction with much longer horizon
        predictions_very_long = model.predict(dataset, h=48)
        assert predictions_very_long.shape[0] == 48  # 48 predictions
        
        # Test prediction with horizon that's not a multiple of original
        predictions_odd = model.predict(dataset, h=30)
        assert predictions_odd.shape[0] == 30  # 30 predictions
    
    def test_lstm_any_horizon_basic(self, simple_data):
        """Test LSTM model with any horizon predictions."""
        # Train with horizon 7
        model = LSTM(
            h=7,
            input_size=21,
            encoder_hidden_size=32,
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        
        # Fit the model
        dataset = TimeSeriesDataset.from_df(simple_data)
        model.fit(dataset)
        
        # Test prediction with same horizon
        predictions = model.predict(dataset)
        assert predictions.shape[0] == 7  # 7 predictions
        
        # Test prediction with longer horizon (any horizon feature)
        predictions_long = model.predict(dataset, h=21)
        assert predictions_long.shape[0] == 21  # 21 predictions
        
        # Test prediction with much longer horizon
        predictions_very_long = model.predict(dataset, h=35)
        assert predictions_very_long.shape[0] == 35  # 35 predictions
    
    def test_gru_any_horizon_basic(self, simple_data):
        """Test GRU model with any horizon predictions."""
        # Train with horizon 10
        model = GRU(
            h=10,
            input_size=30,
            encoder_hidden_size=32,
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        
        # Fit the model
        dataset = TimeSeriesDataset.from_df(simple_data)
        model.fit(dataset)
        
        # Test prediction with same horizon
        predictions = model.predict(dataset)
        assert predictions.shape[0] == 10  # 10 predictions
        
        # Test prediction with longer horizon (any horizon feature)
        predictions_long = model.predict(dataset, h=25)
        assert predictions_long.shape[0] == 25  # 25 predictions
        
        # Test prediction with much longer horizon
        predictions_very_long = model.predict(dataset, h=50)
        assert predictions_long.shape[0] == 25  # 25 predictions
    
    def test_tcn_any_horizon_basic(self, simple_data):
        """Test TCN model with any horizon predictions."""
        # Train with horizon 8
        model = TCN(
            h=8,
            input_size=24,
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        
        # Fit the model
        dataset = TimeSeriesDataset.from_df(simple_data)
        model.fit(dataset)
        
        # Test prediction with same horizon
        predictions = model.predict(dataset)
        assert predictions.shape[0] == 8  # 8 predictions
        
        # Test prediction with longer horizon (any horizon feature)
        predictions_long = model.predict(dataset, h=20)
        assert predictions_long.shape[0] == 20  # 20 predictions
        
        # Test prediction with much longer horizon
        predictions_very_long = model.predict(dataset, h=32)
        assert predictions_very_long.shape[0] == 32  # 32 predictions
    
    def test_deepar_any_horizon_basic(self, simple_data):
        """Test DeepAR model with any horizon predictions."""
        # Train with horizon 14
        model = DeepAR(
            h=14,
            input_size=28,
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        
        # Fit the model
        dataset = TimeSeriesDataset.from_df(simple_data)
        model.fit(dataset)
        
        # Test prediction with same horizon
        predictions = model.predict(dataset)
        assert predictions.shape[0] == 14  # 14 predictions
        
        # Test prediction with longer horizon (any horizon feature)
        predictions_long = model.predict(dataset, h=28)
        assert predictions_long.shape[0] == 28  # 28 predictions
        
        # Test prediction with much longer horizon
        predictions_very_long = model.predict(dataset, h=42)
        assert predictions_very_long.shape[0] == 42  # 42 predictions
    
    def test_mlp_any_horizon_basic(self, simple_data):
        """Test MLP model with any horizon predictions (non-recurrent)."""
        # Train with horizon 6
        model = MLP(
            h=6,
            input_size=18,
            max_steps=2,
            learning_rate=1e-3
        )
        
        # Fit the model
        dataset = TimeSeriesDataset.from_df(simple_data)
        model.fit(dataset)
        
        # Test prediction with same horizon
        predictions = model.predict(dataset)
        assert predictions.shape[0] == 6  # 6 predictions
        
        # Test prediction with longer horizon (any horizon feature)
        predictions_long = model.predict(dataset, h=12)
        assert predictions_long.shape[0] == 12  # 12 predictions
        
        # Test prediction with much longer horizon
        predictions_very_long = model.predict(dataset, h=18)
        assert predictions_very_long.shape[0] == 18  # 18 predictions
    
    def test_any_horizon_with_exogenous(self, exogenous_data):
        """Test any horizon predictions with exogenous variables."""
        # Test RNN with exogenous
        model = RNN(
            h=8,
            input_size=24,
            encoder_hidden_size=32,
            futr_exog_list=['trend', 'seasonal'],
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        
        dataset = TimeSeriesDataset.from_df(
            exogenous_data, 
            futr_exog_list=['trend', 'seasonal']
        )
        model.fit(dataset)
        
        # Test prediction with longer horizon
        predictions = model.predict(dataset, h=20)
        assert predictions.shape[0] == 20
        
        # Test prediction with much longer horizon
        predictions_long = model.predict(dataset, h=40)
        assert predictions_long.shape[0] == 40
    
    def test_any_horizon_with_different_losses(self, simple_data):
        """Test any horizon predictions with different loss functions."""
        dataset = TimeSeriesDataset.from_df(simple_data)
        
        # Test with MSE loss
        model_mse = RNN(
            h=8,
            input_size=24,
            encoder_hidden_size=32,
            loss=MSE(),
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        model_mse.fit(dataset)
        predictions_mse = model_mse.predict(dataset, h=16)
        assert predictions_mse.shape[0] == 16
        
        # Test with MQLoss (quantile loss)
        model_mq = RNN(
            h=8,
            input_size=24,
            encoder_hidden_size=32,
            loss=MQLoss(level=[80, 90]),
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        model_mq.fit(dataset)
        predictions_mq = model_mq.predict(dataset, h=24)
        assert predictions_mq.shape[0] == 24
    
    def test_any_horizon_edge_cases(self, simple_data):
        """Test edge cases for any horizon predictions."""
        dataset = TimeSeriesDataset.from_df(simple_data)
        
        # Test with very short input size
        model_short = RNN(
            h=5,
            input_size=10,
            encoder_hidden_size=32,
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        model_short.fit(dataset)
        predictions = model_short.predict(dataset, h=20)
        assert predictions.shape[0] == 20
        
        # Test with very long horizon
        model_long = RNN(
            h=5,
            input_size=20,
            encoder_hidden_size=32,
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        model_long.fit(dataset)
        predictions = model_long.predict(dataset, h=100)
        assert predictions.shape[0] == 100
        
        # Test with horizon equal to input size
        model_equal = RNN(
            h=5,
            input_size=20,
            encoder_hidden_size=32,
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        model_equal.fit(dataset)
        predictions = model_equal.predict(dataset, h=20)
        assert predictions.shape[0] == 20
    
    def test_any_horizon_consistency(self, simple_data):
        """Test that predictions are consistent across different horizon lengths."""
        dataset = TimeSeriesDataset.from_df(simple_data)
        
        model = RNN(
            h=8,
            input_size=24,
            encoder_hidden_size=32,
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        model.fit(dataset)
        
        # Get predictions for different horizons
        pred_8 = model.predict(dataset, h=8)
        pred_16 = model.predict(dataset, h=16)
        pred_24 = model.predict(dataset, h=24)
        
        # First 8 predictions should be identical
        np.testing.assert_array_equal(pred_8, pred_16[:8])
        np.testing.assert_array_equal(pred_8, pred_24[:8])
        
        # Predictions should be deterministic
        pred_16_2 = model.predict(dataset, h=16)
        np.testing.assert_array_equal(pred_16, pred_16_2)
    
    def test_any_horizon_with_neuralforecast(self, simple_data):
        """Test any horizon predictions integration with NeuralForecast."""
        # Create NeuralForecast instance with recurrent models
        models = [
            RNN(h=8, input_size=24, encoder_hidden_size=32, max_steps=2, learning_rate=1e-3, recurrent=True),
            LSTM(h=8, input_size=24, encoder_hidden_size=32, max_steps=2, learning_rate=1e-3, recurrent=True)
        ]
        
        nf = NeuralForecast(models=models, freq='D')
        nf.fit(simple_data)
        
        # Test prediction with longer horizon
        forecasts = nf.predict(simple_data, h=20)
        assert len(forecasts) == 20
        
        # Check that all models produced forecasts
        expected_cols = ['unique_id', 'ds', 'RNN', 'LSTM']
        assert all(col in forecasts.columns for col in expected_cols)
    
    def test_any_horizon_training_consistency(self, simple_data):
        """Test that training is consistent and models learn something."""
        dataset = TimeSeriesDataset.from_df(simple_data)
        
        # Train model for multiple steps
        model = RNN(
            h=8,
            input_size=24,
            encoder_hidden_size=32,
            max_steps=5,
            learning_rate=1e-3,
            recurrent=True
        )
        
        # Check that model has trainable parameters
        assert len(list(model.parameters())) > 0
        
        # Fit the model
        model.fit(dataset)
        
        # Test predictions
        predictions = model.predict(dataset, h=20)
        assert predictions.shape[0] == 20
        assert not np.any(np.isnan(predictions))
    
    def test_any_horizon_with_validation(self, simple_data):
        """Test any horizon predictions with validation during training."""
        dataset = TimeSeriesDataset.from_df(simple_data)
        
        model = RNN(
            h=8,
            input_size=24,
            encoder_hidden_size=32,
            max_steps=5,
            val_check_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        
        # Fit with validation
        model.fit(dataset, val_size=20)
        
        # Test predictions
        predictions = model.predict(dataset, h=25)
        assert predictions.shape[0] == 25
        assert not np.any(np.isnan(predictions))
    
    def test_any_horizon_error_handling(self, simple_data):
        """Test error handling in any horizon predictions."""
        dataset = TimeSeriesDataset.from_df(simple_data)
        
        # Test with invalid horizon
        model = RNN(
            h=8,
            input_size=24,
            encoder_hidden_size=32,
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        model.fit(dataset)
        
        # Should handle zero horizon gracefully
        with pytest.raises(ValueError):
            model.predict(dataset, h=0)
        
        # Should handle negative horizon gracefully
        with pytest.raises(ValueError):
            model.predict(dataset, h=-1)
    
    def test_any_horizon_memory_efficiency(self, simple_data):
        """Test that models don't leak memory during long horizon predictions."""
        dataset = TimeSeriesDataset.from_df(simple_data)
        
        model = RNN(
            h=8,
            input_size=24,
            encoder_hidden_size=32,
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        model.fit(dataset)
        
        # Test with very long horizon to check memory usage
        predictions = model.predict(dataset, h=100)
        assert predictions.shape[0] == 100
        assert not np.any(np.isnan(predictions))
        
        # Check that model state is properly reset
        assert model.h == 8  # Should reset to original horizon
        assert model.n_predicts == 1  # Should reset to default
    
    def test_any_horizon_with_multiple_series(self, simple_data):
        """Test any horizon predictions with multiple time series."""
        # Create multiple series
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        series_data = []
        for i in range(3):
            y = np.random.randn(200) + np.arange(200) * 0.1 + i * 10
            series_data.append(pd.DataFrame({
                'unique_id': [f'series{i+1}'] * 200,
                'ds': dates,
                'y': y
            }))
        
        multi_df = pd.concat(series_data, ignore_index=True)
        
        # Train model
        model = RNN(
            h=10,
            input_size=30,
            encoder_hidden_size=32,
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        
        dataset = TimeSeriesDataset.from_df(multi_df)
        model.fit(dataset)
        
        # Test predictions with longer horizon
        predictions = model.predict(dataset, h=25)
        assert predictions.shape[0] == 25
        
        # Test with much longer horizon
        predictions_long = model.predict(dataset, h=50)
        assert predictions_long.shape[0] == 50
    
    def test_any_horizon_with_step_size(self, simple_data):
        """Test any horizon predictions with different step sizes."""
        dataset = TimeSeriesDataset.from_df(simple_data)
        
        model = RNN(
            h=8,
            input_size=24,
            encoder_hidden_size=32,
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        model.fit(dataset)
        
        # Test with different step sizes
        for step_size in [1, 2, 4]:
            predictions = model.predict(dataset, h=20, step_size=step_size)
            assert predictions.shape[0] == 20
    
    def test_any_horizon_with_quantiles(self, simple_data):
        """Test any horizon predictions with quantile outputs."""
        dataset = TimeSeriesDataset.from_df(simple_data)
        
        model = RNN(
            h=8,
            input_size=24,
            encoder_hidden_size=32,
            loss=MQLoss(level=[80, 90]),
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        model.fit(dataset)
        
        # Test predictions with longer horizon
        predictions = model.predict(dataset, h=20)
        assert predictions.shape[0] == 20
        
        # Test with much longer horizon
        predictions_long = model.predict(dataset, h=40)
        assert predictions_long.shape[0] == 40
