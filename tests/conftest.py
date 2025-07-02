import logging
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # set before neuralforecast import
import warnings

import pytest

from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengersDF as Y_df


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configure the environment for all neural forecast tests."""
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
    Y_test_df = Y_df[Y_df.ds > "1959-12-31"]  # 12 test
    dataset, *_ = TimeSeriesDataset.from_df(Y_train_df)
    return dataset


@pytest.fixture
def full_dataset_split():
    """Provide both train and test dataframes if needed."""
    Y_train_df = Y_df[Y_df.ds <= "1959-12-31"]  # 132 train
    Y_test_df = Y_df[Y_df.ds > "1959-12-31"]  # 12 test
    return Y_train_df, Y_test_df
