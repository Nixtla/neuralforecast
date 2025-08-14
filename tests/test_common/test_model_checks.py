import pytest

from neuralforecast.common._model_checks import check_model
from neuralforecast.core import MODEL_FILENAME_DICT


@pytest.mark.parametrize("model", [v for k,v in MODEL_FILENAME_DICT.items() if 'auto' not in k])
def test_model_checks(model):
    check_model(model, checks=["losses"])
