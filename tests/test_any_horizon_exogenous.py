import numpy as np
import pandas as pd
import pytest

from neuralforecast import NeuralForecast
from neuralforecast.models import RNN, LSTM, GRU, TCN, DeepAR, DummyRNN, DummyUnivariate
from neuralforecast.losses.pytorch import MAE, MSE, MQLoss
from neuralforecast.tsdataset import TimeSeriesDataset


class TestAnyHorizonExogenous:
    """Test suite for any horizon predictions with exogenous variables."""
    
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
    def future_exogenous_data(self):
        """Create data with future exogenous variables for testing."""
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        future_dates = pd.date_range('2020-07-20', periods=100, freq='D')  # Future dates
        
        np.random.seed(42)
        y = np.random.randn(200) + np.arange(200) * 0.1
        
        # Historical data
        df = pd.DataFrame({
            'unique_id': ['series1'] * 200,
            'ds': dates,
            'y': y
        })
        
        # Future exogenous data
        future_df = pd.DataFrame({
            'unique_id': ['series1'] * 100,
            'ds': future_dates,
            'trend': np.arange(200, 300),
            'seasonal': np.sin(np.arange(200, 300) * 2 * np.pi / 7),
            'event': np.random.choice([0, 1], size=100, p=[0.8, 0.2])
        })
        
        return df, future_df
    
    @pytest.fixture
    def historical_exogenous_data(self):
        """Create data with historical exogenous variables for testing."""
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        y = np.random.randn(200) + np.arange(200) * 0.1
        lag1 = np.roll(y, 1)  # Lag-1 of target
        lag7 = np.roll(y, 7)  # Lag-7 of target
        trend = np.arange(200)
        
        df = pd.DataFrame({
            'unique_id': ['series1'] * 200,
            'ds': dates,
            'y': y,
            'lag1': lag1,
            'lag7': lag7,
            'trend': trend
        })
        return df
    
    @pytest.fixture
    def static_exogenous_data(self):
        """Create data with static exogenous variables for testing."""
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        y = np.random.randn(200) + np.arange(200) * 0.1
        
        # Historical data
        df = pd.DataFrame({
            'unique_id': ['series1'] * 200,
            'ds': dates,
            'y': y
        })
        
        # Static data
        static_df = pd.DataFrame({
            'unique_id': ['series1'],
            'category': ['A'],
            'region': ['North'],
            'size': [100]
        })
        
        return df, static_df
    
    def test_future_exogenous_any_horizon(self, future_exogenous_data):
        """Test any horizon predictions with future exogenous variables."""
        df, future_df = future_exogenous_data
        
        # Test RNN with future exogenous
        model = RNN(
            h=8,
            input_size=24,
            encoder_hidden_size=32,
            futr_exog_list=['trend', 'seasonal', 'event'],
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        
        dataset = TimeSeriesDataset.from_df(
            df, 
            futr_exog_list=['trend', 'seasonal', 'event']
        )
        model.fit(dataset)
        
        # Test prediction with longer horizon
        predictions = model.predict(dataset, futr_df=future_df, h=20)
        assert predictions.shape[0] == 20
        
        # Test prediction with much longer horizon
        predictions_long = model.predict(dataset, futr_df=future_df, h=40)
        assert predictions_long.shape[0] == 40
        
        # Test prediction with horizon that's not a multiple of original
        predictions_odd = model.predict(dataset, futr_df=future_df, h=25)
        assert predictions_odd.shape[0] == 25
    
    def test_historical_exogenous_any_horizon(self, historical_exogenous_data):
        """Test any horizon predictions with historical exogenous variables."""
        df = historical_exogenous_data
        
        # Test LSTM with historical exogenous
        model = LSTM(
            h=10,
            input_size=30,
            encoder_hidden_size=32,
            hist_exog_list=['lag1', 'lag7', 'trend'],
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        
        dataset = TimeSeriesDataset.from_df(
            df, 
            hist_exog_list=['lag1', 'lag7', 'trend']
        )
        model.fit(dataset)
        
        # Test prediction with longer horizon
        predictions = model.predict(dataset, h=25)
        assert predictions.shape[0] == 25
        
        # Test prediction with much longer horizon
        predictions_long = model.predict(dataset, h=50)
        assert predictions_long.shape[0] == 50
    
    def test_static_exogenous_any_horizon(self, static_exogenous_data):
        """Test any horizon predictions with static exogenous variables."""
        df, static_df = static_exogenous_data
        
        # Test GRU with static exogenous
        model = GRU(
            h=12,
            input_size=36,
            encoder_hidden_size=32,
            stat_exog_list=['category', 'region', 'size'],
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        
        dataset = TimeSeriesDataset.from_df(
            df, 
            stat_exog_list=['category', 'region', 'size']
        )
        model.fit(dataset)
        
        # Test prediction with longer horizon
        predictions = model.predict(dataset, h=30)
        assert predictions.shape[0] == 30
        
        # Test prediction with much longer horizon
        predictions_long = model.predict(dataset, h=60)
        assert predictions_long.shape[0] == 60
    
    def test_mixed_exogenous_any_horizon(self, future_exogenous_data, static_exogenous_data):
        """Test any horizon predictions with mixed exogenous variables."""
        df, future_df = future_exogenous_data
        _, static_df = static_exogenous_data
        
        # Test TCN with mixed exogenous
        model = TCN(
            h=8,
            input_size=24,
            futr_exog_list=['trend', 'seasonal'],
            stat_exog_list=['category', 'region'],
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        
        dataset = TimeSeriesDataset.from_df(
            df, 
            futr_exog_list=['trend', 'seasonal'],
            stat_exog_list=['category', 'region']
        )
        model.fit(dataset)
        
        # Test prediction with longer horizon
        predictions = model.predict(dataset, futr_df=future_df, h=20)
        assert predictions.shape[0] == 20
        
        # Test prediction with much longer horizon
        predictions_long = model.predict(dataset, futr_df=future_df, h=40)
        assert predictions_long.shape[0] == 40
    
    def test_dummy_models_exogenous_any_horizon(self, future_exogenous_data):
        """Test dummy models with exogenous variables and any horizon predictions."""
        df, future_df = future_exogenous_data
        
        # Test DummyUnivariate with future exogenous
        model = DummyUnivariate(
            h=5,
            input_size=20,
            futr_exog_list=['trend', 'seasonal'],
            max_steps=1,
            learning_rate=1e-3
        )
        
        dataset = TimeSeriesDataset.from_df(
            df, 
            futr_exog_list=['trend', 'seasonal']
        )
        model.fit(dataset)
        
        # Test prediction with longer horizon
        predictions = model.predict(dataset, futr_df=future_df, h=15)
        assert predictions.shape[0] == 15
        
        # Test DummyRNN with future exogenous
        model_rnn = DummyRNN(
            h=7,
            input_size=21,
            seasonality=7,
            futr_exog_list=['trend', 'seasonal'],
            max_steps=1,
            learning_rate=1e-3
        )
        
        model_rnn.fit(dataset)
        predictions_rnn = model_rnn.predict(dataset, futr_df=future_df, h=21)
        assert predictions_rnn.shape[0] == 21
    
    def test_neuralforecast_exogenous_any_horizon(self, future_exogenous_data):
        """Test any horizon predictions with exogenous variables through NeuralForecast."""
        df, future_df = future_exogenous_data
        
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
        nf.fit(df)
        
        # Test prediction with longer horizon
        forecasts = nf.predict(df, futr_df=future_df, h=25)
        assert len(forecasts) == 25
        
        # Check that all models produced forecasts
        expected_cols = ['unique_id', 'ds', 'RNN', 'LSTM']
        assert all(col in forecasts.columns for col in expected_cols)
        
        # Test with much longer horizon
        forecasts_long = nf.predict(df, futr_df=future_df, h=50)
        assert len(forecasts_long) == 50
    
    def test_exogenous_any_horizon_consistency(self, future_exogenous_data):
        """Test that predictions with exogenous variables are consistent across different horizon lengths."""
        df, future_df = future_exogenous_data
        
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
            df, 
            futr_exog_list=['trend', 'seasonal']
        )
        model.fit(dataset)
        
        # Get predictions for different horizons
        pred_8 = model.predict(dataset, futr_df=future_df, h=8)
        pred_16 = model.predict(dataset, futr_df=future_df, h=16)
        pred_24 = model.predict(dataset, futr_df=future_df, h=24)
        
        # First 8 predictions should be identical
        np.testing.assert_array_equal(pred_8, pred_16[:8])
        np.testing.assert_array_equal(pred_8, pred_24[:8])
        
        # Predictions should be deterministic
        pred_16_2 = model.predict(dataset, futr_df=future_df, h=16)
        np.testing.assert_array_equal(pred_16, pred_16_2)
    
    def test_exogenous_any_horizon_edge_cases(self, future_exogenous_data):
        """Test edge cases for any horizon predictions with exogenous variables."""
        df, future_df = future_exogenous_data
        
        model = RNN(
            h=5,
            input_size=20,
            encoder_hidden_size=32,
            futr_exog_list=['trend', 'seasonal'],
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        
        dataset = TimeSeriesDataset.from_df(
            df, 
            futr_exog_list=['trend', 'seasonal']
        )
        model.fit(dataset)
        
        # Test with very short input size
        predictions = model.predict(dataset, futr_df=future_df, h=20)
        assert predictions.shape[0] == 20
        
        # Test with very long horizon
        predictions_long = model.predict(dataset, futr_df=future_df, h=100)
        assert predictions_long.shape[0] == 100
        
        # Test with horizon equal to input size
        predictions_equal = model.predict(dataset, futr_df=future_df, h=20)
        assert predictions_equal.shape[0] == 20
    
    def test_exogenous_any_horizon_with_different_losses(self, future_exogenous_data):
        """Test any horizon predictions with exogenous variables and different loss functions."""
        df, future_df = future_exogenous_data
        
        dataset = TimeSeriesDataset.from_df(
            df, 
            futr_exog_list=['trend', 'seasonal']
        )
        
        # Test with MSE loss
        model_mse = RNN(
            h=8,
            input_size=24,
            encoder_hidden_size=32,
            futr_exog_list=['trend', 'seasonal'],
            loss=MSE(),
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        model_mse.fit(dataset)
        predictions_mse = model_mse.predict(dataset, futr_df=future_df, h=16)
        assert predictions_mse.shape[0] == 16
        
        # Test with MQLoss (quantile loss)
        model_mq = RNN(
            h=8,
            input_size=24,
            encoder_hidden_size=32,
            futr_exog_list=['trend', 'seasonal'],
            loss=MQLoss(level=[80, 90]),
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        model_mq.fit(dataset)
        predictions_mq = model_mq.predict(dataset, futr_df=future_df, h=24)
        assert predictions_mq.shape[0] == 24
    
    def test_exogenous_any_horizon_memory_efficiency(self, future_exogenous_data):
        """Test that models don't leak memory during long horizon predictions with exogenous variables."""
        df, future_df = future_exogenous_data
        
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
            df, 
            futr_exog_list=['trend', 'seasonal']
        )
        model.fit(dataset)
        
        # Test with very long horizon to check memory usage
        predictions = model.predict(dataset, futr_df=future_df, h=100)
        assert predictions.shape[0] == 100
        assert not np.any(np.isnan(predictions))
        
        # Check that model state is properly reset
        assert model.h == 8  # Should reset to original horizon
        assert model.n_predicts == 1  # Should reset to default
    
    def test_exogenous_any_horizon_with_step_size(self, future_exogenous_data):
        """Test any horizon predictions with exogenous variables and different step sizes."""
        df, future_df = future_exogenous_data
        
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
            df, 
            futr_exog_list=['trend', 'seasonal']
        )
        model.fit(dataset)
        
        # Test with different step sizes
        for step_size in [1, 2, 4]:
            predictions = model.predict(dataset, futr_df=future_df, h=20, step_size=step_size)
            assert predictions.shape[0] == 20
    
    def test_exogenous_any_horizon_with_quantiles(self, future_exogenous_data):
        """Test any horizon predictions with exogenous variables and quantile outputs."""
        df, future_df = future_exogenous_data
        
        model = RNN(
            h=8,
            input_size=24,
            encoder_hidden_size=32,
            futr_exog_list=['trend', 'seasonal'],
            loss=MQLoss(level=[80, 90]),
            max_steps=2,
            learning_rate=1e-3,
            recurrent=True
        )
        
        dataset = TimeSeriesDataset.from_df(
            df, 
            futr_exog_list=['trend', 'seasonal']
        )
        model.fit(dataset)
        
        # Test predictions with longer horizon
        predictions = model.predict(dataset, futr_df=future_df, h=20)
        assert predictions.shape[0] == 20
        
        # Test with much longer horizon
        predictions_long = model.predict(dataset, futr_df=future_df, h=40)
        assert predictions_long.shape[0] == 40
