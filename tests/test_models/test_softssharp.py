from neuralforecast.common._model_checks import check_model
from neuralforecast.models import SOFTSSharp


def test_softssharp_model(suppress_warnings):
    check_model(SOFTSSharp, ["airpassengers"])
