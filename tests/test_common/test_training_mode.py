import pytest

from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, MLP
from neuralforecast.utils import AirPassengersPanel

DF = AirPassengersPanel[["unique_id", "ds", "y"]]


def _mlp():
    return MLP(
        h=12,
        input_size=24,
        max_steps=2,
        accelerator="cpu",
        enable_progress_bar=False,
        logger=False,
    )


def _recurrent_lstm():
    return LSTM(
        h=12,
        input_size=24,
        max_steps=2,
        recurrent=True,
        accelerator="cpu",
        enable_progress_bar=False,
        logger=False,
    )


def _raise(*args, **kwargs):
    raise RuntimeError("interrupted")


def _fit(model):
    nf = NeuralForecast([model], freq="ME")
    nf.fit(DF)
    return nf


def _record_training_steps(model):
    seen = []
    step = model.training_step

    def training_step(*args, **kwargs):
        seen.append(model.training)
        return step(*args, **kwargs)

    model.training_step = training_step
    return seen


@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
@pytest.mark.parametrize(
    "make_model, predict_kwargs",
    [
        (_mlp, {}),
        (_mlp, {"h": 24}),
        (_recurrent_lstm, {}),
    ],
    ids=["direct", "h_above_model_h", "recurrent"],
)
def test_failed_predict_keeps_training_mode(make_model, predict_kwargs, training):
    nf = _fit(make_model())
    model = nf.models[0]
    model.train(training)

    forward = model.forward
    model.forward = _raise
    with pytest.raises(RuntimeError, match="interrupted"):
        nf.predict(**predict_kwargs)
    model.forward = forward

    assert model.training is training


def test_failed_predict_does_not_leave_next_fit_in_eval_mode():
    nf = _fit(_mlp())
    model = nf.models[0]

    forward = model.forward
    model.forward = _raise
    with pytest.raises(RuntimeError, match="interrupted"):
        nf.predict()
    model.forward = forward

    seen = _record_training_steps(model)
    nf.fit(DF, use_init_models=False)
    assert seen and all(seen)
