import numpy as np
import pandas as pd
import pytest
import torch

from neuralforecast import NeuralForecast
from neuralforecast.common._scalers import TemporalNorm
from neuralforecast.losses.pytorch import GMM, HuberLoss
from neuralforecast.models import NHITS, RNN
from neuralforecast.utils import AirPassengersDF as Y_df


# Unit test for masked predict filtering
# Declare synthetic batch to normalize
@pytest.mark.parametrize("scaler_type", [None, 'identity', 'standard', 'robust', 'minmax', 'minmax1', 'invariant', 'revin'])
def test_temporal_norm(scaler_type):
    x1 = 10**0 * np.arange(36)[:, None]
    x2 = 10**1 * np.arange(36)[:, None]

    np_x = np.concatenate([x1, x2], axis=1)
    np_x = np.repeat(np_x[None, :,:], repeats=2, axis=0)
    np_x[0,:,:] = np_x[0,:,:] + 100

    np_mask = np.ones(np_x.shape)
    np_mask[:, -12:, :] = 0

    # Validate scalers
    x = 1.0*torch.tensor(np_x).unsqueeze(-1)
    mask = torch.tensor(np_mask).unsqueeze(-1)
    scaler = TemporalNorm(scaler_type=scaler_type, dim=1, num_features=np_x.shape[-1])
    x_scaled = scaler.transform(x=x, mask=mask)
    x_recovered = scaler.inverse_transform(x_scaled)
    assert torch.allclose(x, x_recovered, atol=1e-3), f'Recovered data is not the same as original with {scaler_type}'

@pytest.fixture
def base_model_params():
    """Common parameters for all models in revin tests."""
    return {
        'h': 12,
        'input_size': 24,
        'max_steps': 1,
        'early_stop_patience_steps': 10,
        'val_check_steps': 50,
        'scaler_type': 'revin',
        'learning_rate': 1e-3
    }


class TestRevinScaler:
    """Test suite for revin scaler functionality across different models."""

    def _run_cross_validation_test(self, model):
        """Helper method to run cross validation test for any model."""
        nf = NeuralForecast(models=[model], freq='MS')
        Y_hat_df = nf.cross_validation(df=Y_df, val_size=12, n_windows=1)

        assert Y_hat_df is not None
        # The model name is used as the prediction column
        model_name = model.__class__.__name__
        assert model_name in Y_hat_df.columns, f"Prediction column {model_name} should be in output"
        assert 'y' in Y_hat_df.columns, "True values column 'y' should be in output"
        assert Y_hat_df.shape[0] > 0, "Should have predictions"
        return Y_hat_df

    def test_nhits_with_huber_loss_basic(self, base_model_params):
        """Test NHITS model with Huber loss (no exogenous variables)."""
        model = NHITS(
            loss=HuberLoss(),
            **base_model_params
        )
        self._run_cross_validation_test(model)

    def test_nhits_with_huber_loss_different_configs(self, base_model_params):
        """Test NHITS model with Huber loss and different configurations."""
        # Test with different input sizes
        configs = [
            {'input_size': 12},
            {'input_size': 36},
            {'learning_rate': 5e-4}
        ]

        for config in configs:
            params = base_model_params.copy()
            params.update(config)
            model = NHITS(
                loss=HuberLoss(),
                **params
            )
            self._run_cross_validation_test(model)

    def test_nhits_with_gmm_loss(self, base_model_params):
        """Test NHITS model with GMM loss."""
        model = NHITS(
            loss=GMM(n_components=10, level=[90]),
            **base_model_params
        )
        self._run_cross_validation_test(model)

    def test_rnn_with_huber_loss(self, base_model_params):
        """Test RNN model with Huber loss."""
        model = RNN(
            loss=HuberLoss(),
            **base_model_params
        )
        self._run_cross_validation_test(model)