import numpy as np
import pandas as pd
import pytest
import torch

from neuralforecast import NeuralForecast
from neuralforecast.common._scalers import TemporalNorm
from neuralforecast.models import NHITS
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
