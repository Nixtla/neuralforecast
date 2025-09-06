# Set MPS fallback environment variables before any imports
import os
import numpy as np

# Critical: Set these environment variables BEFORE PyTorch is imported
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Removes MPS memory ceiling
os.environ["DEVICE"] = "cpu"  # Optional: if your code uses this

import logging
import warnings

import pytest

from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengersDF as Y_df
from neuralforecast.utils import AirPassengersPanel


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configure the environment for all neural forecast tests."""
    logging.warning("Running on macOS CI - forcing MPS fallback settings")
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
    yield


@pytest.fixture
def suppress_warnings():
    """Suppress warnings for individual tests."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture
def setup_dataset():
    """Setup train/test dataset for model testing."""
    Y_train_df = Y_df[Y_df.ds <= "1959-12-31"]  # 132 train
    _ = Y_df[Y_df.ds > "1959-12-31"]  # 12 test
    dataset, *_ = TimeSeriesDataset.from_df(Y_train_df)
    return dataset


@pytest.fixture
def full_dataset_split():
    """Provide both train and test dataframes if needed."""
    Y_train_df = Y_df[Y_df.ds <= "1959-12-31"]  # 132 train
    Y_test_df = Y_df[Y_df.ds > "1959-12-31"]  # 12 test
    return Y_train_df, Y_test_df


@pytest.fixture
def setup_airplane_data():
    AirPassengersPanel_train = AirPassengersPanel[
        AirPassengersPanel["ds"] < AirPassengersPanel["ds"].values[-12]
    ].reset_index(drop=True)
    AirPassengersPanel_test = AirPassengersPanel[
        AirPassengersPanel["ds"] >= AirPassengersPanel["ds"].values[-12]
    ].reset_index(drop=True)
    AirPassengersPanel_test["y"] = np.nan
    AirPassengersPanel_test["y_[lag12]"] = np.nan
    return AirPassengersPanel_train, AirPassengersPanel_test