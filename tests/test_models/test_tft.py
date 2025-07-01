import logging
import os
import warnings

from neuralforecast.common._model_checks import check_model
from neuralforecast.models import TFT

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def test_tft_model():
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_model(TFT, ["airpassengers"])
