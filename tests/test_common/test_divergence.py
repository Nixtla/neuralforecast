"""A diverged model must report an infinite loss, never a finite one.

Regression coverage for issue #1601: `_weighted_mean` collapsed non-finite
losses to 0.0, so a diverged trial reported the lowest possible loss and won
the Auto search.
"""

import logging
import warnings

import pytest
import torch

from neuralforecast.losses.pytorch import MAE
from neuralforecast.models.mlp import MLP
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengersDF as Y_df

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


@pytest.fixture
def dataset():
    return TimeSeriesDataset.from_df(Y_df[Y_df.ds <= "1959-12-31"])[0]


def _standalone_model(monkeypatch, **kwargs):
    """An MLP whose logged metrics are captured, usable without a Trainer."""
    model = MLP(h=12, input_size=12, max_steps=1, loss=MAE(), **kwargs)
    model.logged_metrics = {}
    monkeypatch.setattr(
        model, "log", lambda name, value, *a, **kw: model.logged_metrics.update({name: value})
    )
    monkeypatch.setattr(model, "all_gather", lambda tensor: tensor)
    return model


def _diverging(bad_value):
    """A forward that emits `bad_value` while keeping the autograd graph."""
    original_forward = MLP.forward

    def forward(self, windows_batch):
        # `* 0 + bad_value` rather than `full_like`, so the output still has a
        # grad_fn and Lightning can run backward on it.
        return original_forward(self, windows_batch) * 0 + bad_value

    return forward


@pytest.mark.parametrize(
    "bad_value", [float("inf"), float("-inf"), float("nan")], ids=["inf", "-inf", "nan"]
)
def test_non_finite_val_loss_is_reported_as_inf(bad_value, monkeypatch):
    """`ptl/val_loss` is what Ray and Optuna rank on, so it must be +inf.

    nan does not order reliably (`nan < x` is always False), which is how a
    broken trial survives every ASHA rung.
    """
    model = _standalone_model(monkeypatch)
    model.val_size = 12
    # One validation batch over 8 windows, with a non-finite weighted loss sum.
    model.validation_step_outputs = [torch.tensor([bad_value, 8.0])]

    with pytest.warns(UserWarning, match="Validation loss is not finite"):
        model.on_validation_epoch_end()

    assert model.logged_metrics["ptl/val_loss"] == float("inf")


def test_finite_val_loss_is_untouched(monkeypatch):
    model = _standalone_model(monkeypatch)
    model.val_size = 12
    model.validation_step_outputs = [
        torch.tensor([12.0, 4.0]),
        torch.tensor([4.0, 4.0]),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.on_validation_epoch_end()

    assert model.logged_metrics["ptl/val_loss"] == 2.0
    assert model.valid_trajectories[-1][1] == 2.0


def test_validation_does_not_raise_on_diverged_model(dataset, monkeypatch):
    """Validation reports, it does not abort: erroring the trial would deprive
    the hyperparameter search of a ranking for this configuration."""
    monkeypatch.setattr(MLP, "forward", _diverging(float("inf")))
    model = MLP(
        h=12,
        input_size=12,
        max_steps=1,
        val_check_steps=1,
        loss=MAE(),
        enable_progress_bar=False,
        logger=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(dataset=dataset, val_size=12)

    val_losses = [loss for _, loss in model.valid_trajectories]
    assert val_losses, "validation should have run"
    assert all(loss == float("inf") for loss in val_losses), val_losses


def test_train_loss_still_raises_on_nan(dataset, monkeypatch):
    """A NaN training loss means the optimizer step is meaningless. The guard
    already existed in `training_step`; before this fix it was unreachable
    because `_weighted_mean` zeroed the NaN first."""
    monkeypatch.setattr(MLP, "forward", _diverging(float("nan")))
    model = MLP(
        h=12,
        input_size=12,
        max_steps=1,
        loss=MAE(),
        enable_progress_bar=False,
        logger=False,
    )

    with pytest.raises(Exception, match="Loss is NaN"):
        model.fit(dataset=dataset, val_size=0)


def test_infinite_train_loss_warns_but_does_not_raise(dataset, monkeypatch):
    """Under mixed precision an overflowed loss is expected and GradScaler
    recovers by skipping the step, so an infinite train loss only warns."""
    monkeypatch.setattr(MLP, "forward", _diverging(float("inf")))
    model = MLP(
        h=12,
        input_size=12,
        max_steps=1,
        loss=MAE(),
        enable_progress_bar=False,
        logger=False,
    )

    with pytest.warns(UserWarning, match="Training loss is infinite"):
        model.fit(dataset=dataset, val_size=0)


def test_diverged_trial_does_not_win_the_search(dataset, monkeypatch):
    """The reported symptom: a diverged trial scored 0.0, the lowest possible
    loss, so `best_config` stored the broken configuration."""
    import optuna

    from neuralforecast.auto import AutoMLP

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    diverging_hidden_size = 8
    original_forward = MLP.forward

    def forward(self, windows_batch):
        output = original_forward(self, windows_batch)
        if self.hparams["hidden_size"] == diverging_hidden_size:
            return output * 0 + float("inf")
        return output

    monkeypatch.setattr(MLP, "forward", forward)

    def config(trial):
        return {
            "hidden_size": trial.suggest_categorical(
                "hidden_size", [diverging_hidden_size, 16]
            ),
            "input_size": 12,
            "max_steps": 1,
            "val_check_steps": 1,
            "enable_progress_bar": False,
            "logger": False,
        }

    auto = AutoMLP(
        h=12,
        config=config,
        backend="optuna",
        num_samples=2,
        search_alg=optuna.samplers.GridSampler(
            {"hidden_size": [diverging_hidden_size, 16]}
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auto.fit(dataset=dataset, val_size=12)

    losses = {
        trial.params["hidden_size"]: trial.user_attrs["METRICS"]["loss"]
        for trial in auto.results.trials
    }
    assert losses[diverging_hidden_size] == float("inf"), (
        f"the diverged trial must rank worst, got {losses[diverging_hidden_size]}"
    )
    assert torch.isfinite(torch.tensor(losses[16]))

    best_trial = auto.results.best_trial
    assert best_trial.user_attrs["ALL_PARAMS"]["hidden_size"] != diverging_hidden_size
    assert best_trial.user_attrs["METRICS"]["loss"] != float("inf")
