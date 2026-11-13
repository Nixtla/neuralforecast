import pytest
import torch

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import GMM, NBMM, PMM, DistributionLoss
from neuralforecast.models import MLP
from neuralforecast.utils import AirPassengersPanel

DF = AirPassengersPanel[["unique_id", "ds", "y"]]


@pytest.mark.parametrize(
    "loss",
    [
        DistributionLoss("Normal", level=[80]),
        PMM(level=[80]),
        GMM(level=[80]),
        NBMM(level=[80]),
    ],
    ids=lambda loss: type(loss).__name__,
)
def test_update_quantile_none_restores_trained_quantiles(loss):
    quantiles = loss.quantiles.detach().clone()
    output_names = list(loss.output_names)

    loss.update_quantile(q=[0.1, 0.9])
    assert loss.output_names != output_names

    loss.update_quantile(q=None)
    assert torch.equal(loss.quantiles, quantiles)
    assert loss.output_names == output_names


def test_predict_without_quantiles_keeps_trained_levels(tmp_path):
    model = MLP(
        h=12,
        input_size=24,
        loss=DistributionLoss("Normal", level=[80]),
        max_steps=2,
        enable_progress_bar=False,
        logger=False,
    )
    nf = NeuralForecast(models=[model], freq="ME")
    nf.fit(DF)
    before = nf.predict().columns.tolist()
    assert "MLP-lo-80" in before

    nf.predict(quantiles=[0.1, 0.9])
    assert nf.predict().columns.tolist() == before

    nf.predict(quantiles=[0.1, 0.9])
    nf.save(str(tmp_path / "ckpt"), overwrite=True)
    loaded = NeuralForecast.load(str(tmp_path / "ckpt"))
    assert loaded.predict().columns.tolist() == before
