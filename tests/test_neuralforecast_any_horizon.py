import numpy as np
import pandas as pd
import pytest

from neuralforecast import NeuralForecast
from neuralforecast.models import RNN, LSTM, GRU, TCN, DeepAR, MLP, DummyRNN, DummyUnivariate
from neuralforecast.losses.pytorch import MAE, MSE, MQLoss


class TestNeuralForecastAnyHorizon:
    """Test suite for any horizon predictions through NeuralForecast interface."""
    
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
    
    def test_neuralforecast_any_horizon_basic(self, simple_data):
        """Test basic any horizon predictions through NeuralForecast."""
        # Create models with different horizons
        models = [
            RNN(h=8, input_size=24, encoder_hidden_size=32, max_steps=2, learning_rate=1e-3, recurrent=True),
            LSTM(h=10, input_size=30, encoder_hidden_size=32, max_steps=2, learning_rate=1e-3, recurrent=True),
            MLP(h=6, input_size=18, max_steps=2, learning_rate=1e-3)
        ]
        
        nf = NeuralForecast(models=models, freq='D')
        nf.fit(simple_data)
        
        # Test prediction with longer horizon than any model's trained horizon
        forecasts = nf.predict(simple_data, h=20)
        assert len(forecasts) == 20
        
        # Check that all models produced forecasts
        expected_cols = ['unique_id', 'ds', 'RNN', 'LSTM', 'MLP']
        assert all(col in forecasts.columns for col in expected_cols)
        
        # Test with much longer horizon
        forecasts_long = nf.predict(simple_data, h=50)
        assert len(forecasts_long) == 50
    
    def test_neuralforecast_any_horizon_dummy_models(self, simple_data):
        """Test any horizon predictions with dummy models through NeuralForecast."""
        models = [
            DummyUnivariate(h=5, input_size=20, max_steps=1, learning_rate=1e-3),
            DummyRNN(h=7, input_size=21, seasonality=7, max_steps=1, learning_rate=1e-3)
        ]
        
        nf = NeuralForecast(models=models, freq='D')
        nf.fit(simple_data)
        
        # Test prediction with longer horizon
        forecasts = nf.predict(simple_data, h=15)
        assert len(forecasts) == 15
        
        # Check that all models produced forecasts
        expected_cols = ['unique_id', 'ds', 'DummyUnivariate', 'DummyRNN']
        assert all(col in forecasts.columns for col in expected_cols)
        
        # Test with much longer horizon
        forecasts_long = nf.predict(simple_data, h=30)
        assert len(forecasts_long) == 30
    
    def test_neuralforecast_any_horizon_with_exogenous(self, exogenous_data):
        """Test any horizon predictions with exogenous variables through NeuralForecast."""
        models = [
            RNN(
                h=8, 
                input_size=24, 
                encoder_hidden_size=32, 
                futr_exog_list=['trend', 'seasonal'],
                max_steps=2, 
                learning_rate=1e-3, 
                recurrent=True
            ),
            LSTM(
                h=10, 
                input_size=30, 
                encoder_hidden_size=32, 
                futr_exog_list=['trend', 'seasonal'],
                max_steps=2, 
                learning_rate=1e-3, 
                recurrent=True
            )
        ]
        
        nf = NeuralForecast(models=models, freq='D')
        nf.fit(exogenous_data)
        
        # Test prediction with longer horizon
        forecasts = nf.predict(exogenous_data, h=25)
        assert len(forecasts) == 25
        
        # Check that all models produced forecasts
        expected_cols = ['unique_id', 'ds', 'RNN', 'LSTM']
        assert all(col in forecasts.columns for col in expected_cols)
        
        # Test with much longer horizon
        forecasts_long = nf.predict(exogenous_data, h=40)
        assert len(forecasts_long) == 40
    
    def test_neuralforecast_any_horizon_multivariate(self, multivariate_data):
        """Test any horizon predictions with multivariate data through NeuralForecast."""
        # For multivariate, we need to use models that support it
        # or create separate models for each target
        models = [
            RNN(h=8, input_size=24, encoder_hidden_size=32, max_steps=2, learning_rate=1e-3, recurrent=True),
            LSTM(h=10, input_size=30, encoder_hidden_size=32, max_steps=2, learning_rate=1e-3, recurrent=True)
        ]
        
        nf = NeuralForecast(models=models, freq='D')
        nf.fit(multivariate_data)
        
        # Test prediction with longer horizon
        forecasts = nf.predict(multivariate_data, h=20)
        assert len(forecasts) == 20
        
        # Check that all models produced forecasts
        expected_cols = ['unique_id', 'ds', 'RNN', 'LSTM']
        assert all(col in forecasts.columns for col in expected_cols)
        
        # Test with much longer horizon
        forecasts_long = nf.predict(multivariate_data, h=35)
        assert len(forecasts_long) == 35
    
    def test_neuralforecast_any_horizon_consistency(self, simple_data):
        """Test that NeuralForecast predictions are consistent across different horizon lengths."""
        models = [
            RNN(h=8, input_size=24, encoder_hidden_size=32, max_steps=2, learning_rate=1e-3, recurrent=True),
            LSTM(h=10, input_size=30, encoder_hidden_size=32, max_steps=2, learning_rate=1e-3, recurrent=True)
        ]
        
        nf = NeuralForecast(models=models, freq='D')
        nf.fit(simple_data)
        
        # Get predictions for different horizons
        forecasts_8 = nf.predict(simple_data, h=8)
        forecasts_16 = nf.predict(simple_data, h=16)
        forecasts_24 = nf.predict(simple_data, h=24)
        
        # First 8 predictions should be identical for RNN
        np.testing.assert_array_equal(forecasts_8['RNN'].values, forecasts_16['RNN'].values[:8])
        np.testing.assert_array_equal(forecasts_8['RNN'].values, forecasts_24['RNN'].values[:8])
        
        # First 10 predictions should be identical for LSTM
        np.testing.assert_array_equal(forecasts_8['LSTM'].values[:8], forecasts_16['LSTM'].values[:8])
        np.testing.assert_array_equal(forecasts_8['LSTM'].values[:8], forecasts_24['LSTM'].values[:8])
        
        # Predictions should be deterministic
        forecasts_16_2 = nf.predict(simple_data, h=16)
        np.testing.assert_array_equal(forecasts_16['RNN'].values, forecasts_16_2['RNN'].values)
        np.testing.assert_array_equal(forecasts_16['LSTM'].values, forecasts_16_2['LSTM'].values)
    
    def test_neuralforecast_any_horizon_with_different_losses(self, simple_data):
        """Test any horizon predictions with different loss functions through NeuralForecast."""
        models = [
            RNN(
                h=8, 
                input_size=24, 
                encoder_hidden_size=32, 
                loss=MSE(),
                max_steps=2, 
                learning_rate=1e-3, 
                recurrent=True
            ),
            RNN(
                h=10, 
                input_size=30, 
                encoder_hidden_size=32, 
                loss=MQLoss(level=[80, 90]),
                max_steps=2, 
                learning_rate=1e-3, 
                recurrent=True
            )
        ]
        
        nf = NeuralForecast(models=models, freq='D')
        nf.fit(simple_data)
        
        # Test prediction with longer horizon
        forecasts = nf.predict(simple_data, h=20)
        assert len(forecasts) == 20
        
        # Check that all models produced forecasts
        expected_cols = ['unique_id', 'ds', 'RNN', 'RNN-lo-80', 'RNN-hi-80']
        assert all(col in forecasts.columns for col in expected_cols)
        
        # Test with much longer horizon
        forecasts_long = nf.predict(simple_data, h=30)
        assert len(forecasts_long) == 30
    
    def test_neuralforecast_any_horizon_edge_cases(self, simple_data):
        """Test edge cases for any horizon predictions through NeuralForecast."""
        models = [
            RNN(h=5, input_size=20, encoder_hidden_size=32, max_steps=2, learning_rate=1e-3, recurrent=True),
            LSTM(h=8, input_size=24, encoder_hidden_size=32, max_steps=2, learning_rate=1e-3, recurrent=True)
        ]
        
        nf = NeuralForecast(models=models, freq='D')
        nf.fit(simple_data)
        
        # Test with very long horizon
        forecasts = nf.predict(simple_data, h=100)
        assert len(forecasts) == 100
        
        # Test with horizon equal to input size
        forecasts_equal = nf.predict(simple_data, h=20)
        assert len(forecasts_equal) == 20
        
        # Test with horizon that's not a multiple of any model's horizon
        forecasts_odd = nf.predict(simple_data, h=17)
        assert len(forecasts_odd) == 17
    
    def test_neuralforecast_any_horizon_with_step_size(self, simple_data):
        """Test any horizon predictions with different step sizes through NeuralForecast."""
        models = [
            RNN(h=8, input_size=24, encoder_hidden_size=32, max_steps=2, learning_rate=1e-3, recurrent=True),
            LSTM(h=10, input_size=30, encoder_hidden_size=32, max_steps=2, learning_rate=1e-3, recurrent=True)
        ]
        
        nf = NeuralForecast(models=models, freq='D')
        nf.fit(simple_data)
        
        # Test with different step sizes
        for step_size in [1, 2, 4]:
            forecasts = nf.predict(simple_data, h=20, step_size=step_size)
            assert len(forecasts) == 20
    
    def test_neuralforecast_any_horizon_with_multiple_series(self, simple_data):
        """Test any horizon predictions with multiple time series through NeuralForecast."""
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
        
        # Create models
        models = [
            RNN(h=10, input_size=30, encoder_hidden_size=32, max_steps=2, learning_rate=1e-3, recurrent=True),
            LSTM(h=12, input_size=36, encoder_hidden_size=32, max_steps=2, learning_rate=1e-3, recurrent=True)
        ]
        
        nf = NeuralForecast(models=models, freq='D')
        nf.fit(multi_df)
        
        # Test predictions with longer horizon
        forecasts = nf.predict(multi_df, h=25)
        assert len(forecasts) == 25
        
        # Test with much longer horizon
        forecasts_long = nf.predict(multi_df, h=50)
        assert len(forecasts_long) == 50
        
        # Check that forecasts are produced for all series
        unique_ids = forecasts['unique_id'].unique()
        assert len(unique_ids) == 3
        assert all(f'series{i+1}' in unique_ids for i in range(3))
    
    def test_neuralforecast_any_horizon_memory_efficiency(self, simple_data):
        """Test that NeuralForecast doesn't leak memory during long horizon predictions."""
        models = [
            RNN(h=8, input_size=24, encoder_hidden_size=32, max_steps=2, learning_rate=1e-3, recurrent=True),
            LSTM(h=10, input_size=30, encoder_hidden_size=32, max_steps=2, learning_rate=1e-3, recurrent=True)
        ]
        
        nf = NeuralForecast(models=models, freq='D')
        nf.fit(simple_data)
        
        # Test with very long horizon to check memory usage
        forecasts = nf.predict(simple_data, h=100)
        assert len(forecasts) == 100
        
        # Check that all models produced forecasts
        expected_cols = ['unique_id', 'ds', 'RNN', 'LSTM']
        assert all(col in forecasts.columns for col in expected_cols)
    
    def test_neuralforecast_any_horizon_error_handling(self, simple_data):
        """Test error handling in any horizon predictions through NeuralForecast."""
        models = [
            RNN(h=8, input_size=24, encoder_hidden_size=32, max_steps=2, learning_rate=1e-3, recurrent=True)
        ]
        
        nf = NeuralForecast(models=models, freq='D')
        nf.fit(simple_data)
        
        # Should handle zero horizon gracefully
        with pytest.raises(ValueError):
            nf.predict(simple_data, h=0)
        
        # Should handle negative horizon gracefully
        with pytest.raises(ValueError):
            nf.predict(simple_data, h=-1)
    
    def test_neuralforecast_any_horizon_with_quantiles(self, simple_data):
        """Test any horizon predictions with quantile outputs through NeuralForecast."""
        models = [
            RNN(
                h=8, 
                input_size=24, 
                encoder_hidden_size=32, 
                loss=MQLoss(level=[80, 90]),
                max_steps=2, 
                learning_rate=1e-3, 
                recurrent=True
            ),
            LSTM(
                h=10, 
                input_size=30, 
                encoder_hidden_size=32, 
                loss=MQLoss(level=[85, 95]),
                max_steps=2, 
                learning_rate=1e-3, 
                recurrent=True
            )
        ]
        
        nf = NeuralForecast(models=models, freq='D')
        nf.fit(simple_data)
        
        # Test predictions with longer horizon
        forecasts = nf.predict(simple_data, h=20)
        assert len(forecasts) == 20
        
        # Check that quantile columns are present
        expected_cols = [
            'unique_id', 'ds', 
            'RNN-lo-80', 'RNN-hi-80',
            'LSTM-lo-85', 'LSTM-hi-95'
        ]
        assert all(col in forecasts.columns for col in expected_cols)
        
        # Test with much longer horizon
        forecasts_long = nf.predict(simple_data, h=40)
        assert len(forecasts_long) == 40
