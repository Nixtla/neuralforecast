import pytest

import neuralforecast.common._model_checks as model_checks
from neuralforecast.common._model_checks import check_loss_functions, check_model
from neuralforecast.core import MODEL_FILENAME_DICT
from neuralforecast.models import MLP


# TimeLLM loads an LLM, so it is not tested (see test_timellm.py)
@pytest.mark.parametrize("model", [v for k,v in MODEL_FILENAME_DICT.items() if 'auto' not in k and k != 'timellm'])
def test_model_checks(model):
    check_model(model, checks=["losses"])


def test_check_loss_functions_fails_on_any_error(monkeypatch):
    def fail(model_class, config):
        raise ValueError("model failed")

    monkeypatch.setattr(model_checks, "_run_model_tests", fail)
    with pytest.raises(Exception, match="MLP: checking MAE failed"):
        check_loss_functions(MLP)
