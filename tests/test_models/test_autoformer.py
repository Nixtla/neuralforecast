import numpy as np
import pytest
import torch

from neuralforecast.auto import AutoAutoformer, Autoformer
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model
from neuralforecast.losses.pytorch import MAE, MAPE, MSE
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengersDF as Y_df

from .test_helpers import check_args


def test_autoformer_basic(suppress_warnings):
    """Test basic Autoformer model functionality using check_model."""
    _ = suppress_warnings  # Fixture used to suppress warnings during test
    check_model(Autoformer, ["airpassengers"])


@pytest.fixture
def setup_autoformer_data():
    """Setup train/test dataset specifically for Autoformer testing."""
    Y_train_df = Y_df[Y_df.ds < Y_df['ds'].values[-12]]  # 132 train
    Y_test_df = Y_df[Y_df.ds >= Y_df['ds'].values[-12]]   # 12 test
    
    dataset, *_ = TimeSeriesDataset.from_df(df=Y_train_df)
    return dataset, Y_train_df, Y_test_df


@pytest.fixture
def basic_autoformer_config():
    """Basic Autoformer configuration for testing."""
    return {
        'h': 12,
        'input_size': 24,
        'hidden_size': 16,
        'encoder_layers': 1,
        'decoder_layers': 1,
        'max_steps': 2,
        'val_check_steps': 1,
        'windows_batch_size': None
    }


@pytest.fixture
def minimal_autoformer_config():
    """Minimal Autoformer configuration for fast testing."""
    return {
        'h': 6,
        'input_size': 12,
        'hidden_size': 8,
        'encoder_layers': 1,
        'decoder_layers': 1,
        'max_steps': 1,
        'val_check_steps': 1
    }


@pytest.fixture
def large_autoformer_config():
    """Larger Autoformer configuration for comprehensive testing."""
    return {
        'h': 12,
        'input_size': 48,
        'hidden_size': 32,
        'encoder_layers': 2,
        'decoder_layers': 2,
        'n_head': 4,
        'max_steps': 5,
        'val_check_steps': 2,
        'windows_batch_size': None
    }


@pytest.fixture
def autoformer_with_exog_config():
    """Autoformer configuration with exogenous variables."""
    return {
        'h': 12,
        'input_size': 24,
        'futr_exog_list': ['exog1', 'exog2'],
        'hidden_size': 16,
        'encoder_layers': 1,
        'decoder_layers': 1,
        'max_steps': 2,
        'val_check_steps': 1,
        'windows_batch_size': None
    }


@pytest.fixture
def basic_autoformer(basic_autoformer_config):
    """Create a basic Autoformer model for testing."""
    return Autoformer(**basic_autoformer_config)


@pytest.fixture
def minimal_autoformer(minimal_autoformer_config):
    """Create a minimal Autoformer model for fast testing."""
    return Autoformer(**minimal_autoformer_config)


@pytest.fixture
def autoformer_data_with_exog():
    """Setup dataset with exogenous variables for testing."""
    Y_df_exog = Y_df.copy()
    Y_df_exog['exog1'] = np.sin(np.arange(len(Y_df_exog)) * 2 * np.pi / 12)
    Y_df_exog['exog2'] = np.cos(np.arange(len(Y_df_exog)) * 2 * np.pi / 12)
    
    Y_train_df = Y_df_exog[Y_df_exog.ds < Y_df_exog['ds'].values[-12]]
    dataset, *_ = TimeSeriesDataset.from_df(
        df=Y_train_df, 
        futr_exog_list=['exog1', 'exog2']
    )
    return dataset, Y_train_df


def test_autoformer_training_and_prediction(setup_autoformer_data, basic_autoformer):
    """Test Autoformer model training and prediction with small dataset."""
    dataset, Y_train_df, Y_test_df = setup_autoformer_data
    model = basic_autoformer
    
    model.fit(dataset=dataset)
    y_hat = model.predict(dataset=dataset)
    
    # Check output shape
    assert y_hat.shape == (12,), f"Expected shape (12,), got {y_hat.shape}"
    
    # Check that predictions are finite
    assert np.all(np.isfinite(y_hat)), "Predictions contain non-finite values"
    
    # Test prediction consistency
    y_hat2 = model.predict(dataset=dataset)
    np.testing.assert_array_equal(y_hat, y_hat2, "Predictions should be consistent")


def test_autoformer_with_validation(setup_autoformer_data, basic_autoformer):
    """Test Autoformer model with validation split."""
    dataset, Y_train_df, Y_test_df = setup_autoformer_data
    model = basic_autoformer
    
    # Fit with validation
    model.fit(dataset=dataset, val_size=12)
    y_hat_w_val = model.predict(dataset=dataset)
    
    # Check output shape and finite values
    assert y_hat_w_val.shape == (12,), f"Expected shape (12,), got {y_hat_w_val.shape}"
    assert np.all(np.isfinite(y_hat_w_val)), "Predictions with validation contain non-finite values"


def test_autoformer_no_leakage(setup_autoformer_data, basic_autoformer):
    """Test that there's no data leakage when using test_size."""
    dataset, *_ = setup_autoformer_data
    model = basic_autoformer
    
    model.fit(dataset=dataset, test_size=12)
    y_hat_test = model.predict(dataset=dataset, step_size=1)
    
    # Test prediction consistency
    y_hat_test2 = model.predict(dataset=dataset, step_size=1)
    np.testing.assert_array_equal(y_hat_test, y_hat_test2, "Test predictions should be consistent")


def test_autoformer_parameters(minimal_autoformer_config):
    """Test various Autoformer model parameters and configurations."""
    # Test different activation functions
    for activation in ["relu", "gelu"]:
        config = minimal_autoformer_config.copy()
        config['activation'] = activation
        model = Autoformer(**config)
        assert model is not None
    
    # Test different loss functions
    for loss in [MAE(), MSE(), MAPE()]:
        config = minimal_autoformer_config.copy()
        config['loss'] = loss
        model = Autoformer(**config)
        assert model is not None
    
    # Test different layer configurations
    config = minimal_autoformer_config.copy()
    config.update({
        'hidden_size': 16,
        'encoder_layers': 2,
        'decoder_layers': 2,
        'n_head': 2
    })
    model = Autoformer(**config)
    assert model is not None


def test_autoformer_input_validation(minimal_autoformer_config):
    """Test input validation and error handling."""
    # Test invalid decoder_input_size_multiplier
    config = minimal_autoformer_config.copy()
    config['decoder_input_size_multiplier'] = 1.5  # Invalid: > 1
    with pytest.raises(Exception, match="Check decoder_input_size_multiplier"):
        Autoformer(**config)
    
    config = minimal_autoformer_config.copy()
    config['decoder_input_size_multiplier'] = 0.0  # Invalid: <= 0
    with pytest.raises(Exception, match="Check decoder_input_size_multiplier"):
        Autoformer(**config)
    
    # Test invalid activation
    config = minimal_autoformer_config.copy()
    config['activation'] = "invalid_activation"
    with pytest.raises(Exception, match="Check activation"):
        Autoformer(**config)


def test_autoformer_with_exogenous_variables(autoformer_data_with_exog, autoformer_with_exog_config):
    """Test Autoformer model with future exogenous variables."""
    dataset, Y_train_df = autoformer_data_with_exog
    model = Autoformer(**autoformer_with_exog_config)
    
    model.fit(dataset=dataset)
    y_hat = model.predict(dataset=dataset)
    
    assert y_hat.shape == (12,), f"Expected shape (12,), got {y_hat.shape}"
    assert np.all(np.isfinite(y_hat)), "Predictions with exogenous variables contain non-finite values"


def test_autoformer_model_attributes(minimal_autoformer):
    """Test that Autoformer has correct model attributes."""
    model = minimal_autoformer
    
    # Test class attributes
    assert model.EXOGENOUS_FUTR == True
    assert model.EXOGENOUS_HIST == False
    assert model.EXOGENOUS_STAT == False
    assert model.MULTIVARIATE == False
    assert model.RECURRENT == False


def test_autoformer_forward_pass(minimal_autoformer):
    """Test the forward pass of Autoformer model."""
    model = minimal_autoformer
    
    # Create dummy input
    batch_size = 2
    windows_batch = {
        'insample_y': torch.randn(batch_size, 12, 1),
        'futr_exog': torch.randn(batch_size, 18, 0)  # input_size + h, no exog features
    }
    
    model.eval()
    with torch.no_grad():
        output = model.forward(windows_batch)
    
    assert output.shape == (batch_size, 6, 1), f"Expected shape ({batch_size}, 6, 1), got {output.shape}"
    assert torch.all(torch.isfinite(output)), "Forward pass output contains non-finite values"


def test_autoformer_different_configurations(minimal_autoformer_config):
    """Test Autoformer with different architectural configurations."""
    configs = [
        {'hidden_size': 8, 'n_head': 1, 'encoder_layers': 1, 'decoder_layers': 1},
        {'hidden_size': 16, 'n_head': 2, 'encoder_layers': 2, 'decoder_layers': 1},
        {'hidden_size': 32, 'n_head': 4, 'encoder_layers': 1, 'decoder_layers': 2},
    ]
    
    for config_override in configs:
        config = minimal_autoformer_config.copy()
        config.update(config_override)
        model = Autoformer(**config)
        assert model is not None
        
        # Test forward pass
        windows_batch = {
            'insample_y': torch.randn(1, 12, 1),
            'futr_exog': torch.randn(1, 18, 0)
        }
        
        model.eval()
        with torch.no_grad():
            output = model.forward(windows_batch)
        
        assert output.shape == (1, 6, 1), f"Config {config_override} failed forward pass"


def test_autoformer_moving_average_window(minimal_autoformer_config):
    """Test Autoformer with different MovingAvg_window sizes."""
    for window_size in [5, 15, 25, 35]:
        config = minimal_autoformer_config.copy()
        config.update({
            'input_size': max(24, window_size + 1),  # Ensure input_size > window_size
            'MovingAvg_window': window_size
        })
        model = Autoformer(**config)
        assert model is not None


def test_autoautoformer(setup_dataset):
    """Test AutoAutoformer hyperparameter optimization."""
    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoAutoformer, exclude_args=['cls_model'])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoAutoformer.get_default_config(h=12, backend='optuna')
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update({
            'max_steps': 1, 
            'val_check_steps': 1, 
            'input_size': 12, 
            'hidden_size': 8,
            'encoder_layers': 1,
            'decoder_layers': 1
        })
        return config

    model = AutoAutoformer(h=12, config=my_config_new, backend='optuna', num_samples=1, cpus=1)
    assert model.config(MockTrial())['h'] == 12
    model.fit(dataset=setup_dataset)

    # Unit test for situation: Ray with updated default config
    my_config = AutoAutoformer.get_default_config(h=12, backend='ray')
    my_config.update({
        'max_steps': 1,
        'val_check_steps': 1,
        'input_size': 12,
        'hidden_size': 8,
        'encoder_layers': 1,
        'decoder_layers': 1
    })
    model = AutoAutoformer(h=12, config=my_config, backend='ray', num_samples=1, cpus=1)
    model.fit(dataset=setup_dataset)


def test_autoformer_edge_cases(minimal_autoformer_config):
    """Test edge cases and boundary conditions."""
    # Test minimum viable configuration
    config = minimal_autoformer_config.copy()
    config.update({
        'h': 1,
        'input_size': 2,
        'hidden_size': 4
    })
    model = Autoformer(**config)
    assert model is not None
    
    # Test with very small factor
    config = minimal_autoformer_config.copy()
    config['factor'] = 1
    model = Autoformer(**config)
    assert model is not None
    
    # Test with high dropout
    config = minimal_autoformer_config.copy()
    config['dropout'] = 0.9
    model = Autoformer(**config)
    assert model is not None


def test_autoformer_scaler_types(minimal_autoformer_config):
    """Test Autoformer with different scaler types."""
    scaler_types = ['identity', 'standard', 'robust', 'minmax']
    
    for scaler_type in scaler_types:
        config = minimal_autoformer_config.copy()
        config['scaler_type'] = scaler_type
        model = Autoformer(**config)
        assert model is not None


def test_basic_autoformer_fixture_functionality(basic_autoformer, setup_autoformer_data):
    """Test that the basic_autoformer fixture works correctly."""
    dataset, _, _ = setup_autoformer_data
    model = basic_autoformer
    
    # Test that the model can be trained and produce predictions
    model.fit(dataset=dataset)
    predictions = model.predict(dataset=dataset)
    
    assert predictions is not None
    assert len(predictions) == 12  # h=12 from basic config
    assert np.all(np.isfinite(predictions))


def test_large_autoformer_configuration(large_autoformer_config, setup_autoformer_data):
    """Test the large Autoformer configuration."""
    dataset, _, _ = setup_autoformer_data
    model = Autoformer(**large_autoformer_config)
    
    # Test model creation and basic functionality
    assert model.h == 12
    assert model.input_size == 48
    assert model is not None
    
    # Test forward pass with appropriate input size
    batch_size = 1
    windows_batch = {
        'insample_y': torch.randn(batch_size, 48, 1),  # input_size from large config
        'futr_exog': torch.randn(batch_size, 60, 0)    # input_size + h
    }
    
    model.eval()
    with torch.no_grad():
        output = model.forward(windows_batch)
    
    assert output.shape == (batch_size, 12, 1)


def test_config_fixture_independence(minimal_autoformer_config, basic_autoformer_config):
    """Test that config fixtures are independent and don't interfere with each other."""
    # Modify one config
    minimal_autoformer_config['h'] = 999
    
    # Check that the other config is not affected
    assert basic_autoformer_config['h'] == 12
    
    # Reset for other tests
    minimal_autoformer_config['h'] = 6