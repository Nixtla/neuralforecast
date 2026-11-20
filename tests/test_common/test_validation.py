import pytest
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping

from neuralforecast.models import MLP
from neuralforecast.tsdataset import TimeSeriesDataset


@pytest.mark.parametrize(
    ("options", "expected_steps"),
    [
        ({}, []),
        ({"limit_val_batches": 1}, [0, 1, 2]),
        ({"callbacks": Callback()}, []),
        (
            {"callbacks": [EarlyStopping(monitor="ptl/val_loss", strict=False)]},
            [0, 1, 2],
        ),
    ],
)
def test_validation_schedule(setup_airplane_data, monkeypatch, options, expected_steps):
    train_df, _ = setup_airplane_data
    dataset = TimeSeriesDataset.from_df(train_df)[0]
    model = MLP(
        h=2,
        input_size=4,
        max_steps=2,
        val_check_steps=1,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        **options,
    )
    calls = []
    validation_step = MLP.validation_step

    def record_step(self, batch, batch_idx):
        calls.append(self.global_step)
        return validation_step(self, batch, batch_idx)

    monkeypatch.setattr(MLP, "validation_step", record_step)
    model.fit(dataset, val_size=0)
    assert calls == expected_steps

    calls.clear()
    model.fit(dataset, val_size=2)
    assert calls == [0, 1, 2]
    assert [step for step, _ in model.valid_trajectories] == [1, 2]
