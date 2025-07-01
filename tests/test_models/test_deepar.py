import logging
import warnings

from neuralforecast.common._model_checks import check_model
from neuralforecast.models import DeepAR


def test_deepar():
    # Unit tests for models
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    check_model(DeepAR, ["airpassengers"])
