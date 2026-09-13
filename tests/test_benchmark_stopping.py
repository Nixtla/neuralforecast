import torch
import pytest
from neuralforecast.benchmark_stopping import ValidationStopper


def test_restores_best_checkpoint_after_patience(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    stopper = ValidationStopper(tmp_path, interval=10, patience=5)
    for step, loss in zip(range(10, 71, 10), [3, 1, 2, 2, 2, 2, 2]):
        model.weight.data.fill_(step)
        stopped = stopper.update(model, step, loss)
    assert stopped
    stopper.restore(model)
    assert model.weight.item() == 20
    assert stopper.summary() == {
        "best_step": 20,
        "actual_steps": 70,
        "best_validation_loss": 1,
        "stop_reason": "early_stopping",
    }


def test_ties_do_not_reset_patience_and_nonfinite_fails(tmp_path):
    model = torch.nn.Linear(1, 1)
    stopper = ValidationStopper(tmp_path, patience=1)
    assert not stopper.update(model, 10, 1)
    assert stopper.update(model, 20, 1)
    with pytest.raises(ValueError, match="Non-finite"):
        stopper.update(model, 30, float("nan"))


def test_lora_checkpoint_only_restores_trainable_parameters(tmp_path):
    model = torch.nn.Linear(1, 1)
    model.weight.requires_grad_(False)
    stopper = ValidationStopper(tmp_path, adapters_only=True)
    model.bias.data.fill_(2)
    stopper.update(model, 10, 1)
    model.bias.data.fill_(9)
    model.weight.data.fill_(7)
    stopper.restore(model)
    assert model.bias.item() == 2
    assert model.weight.item() == 7


def test_patience_and_best_weights_are_portable(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    first = ValidationStopper(tmp_path / "first", patience=3)
    model.weight.data.fill_(2)
    first.update(model, 10, 1.0)
    first.update(model, 20, 2.0)
    second = ValidationStopper(tmp_path / "second", patience=3)
    second.load_state_dict(first.state_dict())
    assert not second.update(model, 30, 2.0)
    assert second.update(model, 40, 2.0)
    model.weight.data.fill_(10)
    second.restore(model)
    assert model.weight.item() == 2
