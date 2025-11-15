import numpy as np
import pytest

from neuralforecast.models import UR2CUTE
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengersDF as Y_df


def test_ur2cute_fit_predict(suppress_warnings):
    dataset, *_ = TimeSeriesDataset.from_df(Y_df)
    model = UR2CUTE(
        h=6,
        input_size=12,
        max_steps=1,
        val_check_steps=1,
        batch_size=8,
    )
    model.fit(dataset=dataset, test_size=6)
    preds = model.predict(dataset=dataset, step_size=1)
    assert preds.shape[-1] == 1
    assert np.isfinite(preds).all()
    assert (preds >= 0).all()


@pytest.fixture
def airpassengers_dataset():
    dataset, *_ = TimeSeriesDataset.from_df(Y_df)
    return dataset


def test_ur2cute_auto_threshold(airpassengers_dataset):
    model = UR2CUTE(
        h=12,
        input_size=24,
        classification_threshold="auto",
        max_steps=1,
        val_check_steps=1,
        batch_size=8,
    )
    model.fit(dataset=airpassengers_dataset, test_size=12)

    assert model.classification_threshold_ is not None
    assert 0.0 <= model.classification_threshold_ <= 1.0
