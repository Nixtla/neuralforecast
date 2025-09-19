import os
import pytest
import sys

from neuralforecast import NeuralForecast
from neuralforecast.models import (
    DeepAR,
    NLinear,
    TSMixer,
)
from neuralforecast.utils import AirPassengersPanel


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason=(
        "RuntimeError: MPS backend out of memory (MPS allocated: 8.00 MB, other allocations: 16.00 KB). Tried to allocate 256 bytes on shared pool."
        "Test failed in MacOS as shown by https://github.com/Nixtla/neuralforecast/actions/runs/17668835879/job/50215797674?pr=1368"
    )
)
@pytest.mark.parametrize(
    "model,kwargs",
    [
        (
            DeepAR,
            {"h": 5, "input_size": 12, "max_steps": 2, "futr_exog_list": ["trend"]},
        ),
        (NLinear, {"h": 5, "input_size": 12, "max_steps": 2}),
        (TSMixer, {"h": 5, "input_size": 12, "n_series": 2, "max_steps": 2}),
    ],
)
def test_backward_comptability(model, kwargs, save_model=False):
    # Fixture generated with save_model=True, on codespace env (Ubuntu)
    save_path = "./tests/backward_comp/data/{}".format(model.__name__)
    horizon = 12
    panel = AirPassengersPanel.copy()
    train_df = panel[panel.ds < panel["ds"].values[-horizon]]  # 132 train
    test_df = panel[panel.ds >= panel["ds"].values[-horizon]]

    if save_model:
        nf = NeuralForecast(
            models=[model(**kwargs)],
            freq="ME",
        )

        nf.fit(df=train_df)
        nf.predict(futr_df=test_df)
        os.makedirs(save_path, exist_ok=True)
        nf.save(path=save_path, model_index=None, overwrite=True, save_dataset=False)
    else:
        # backwarc compatibility test
        fcst = NeuralForecast.load(path=save_path)
        # standard forecast
        fcst.predict(df=train_df, futr_df=test_df)
        # prediction with longer horizon
        fcst.predict(df=train_df, futr_df=test_df, h=horizon)
        fcst.cross_validation(df=train_df, n_windows=2)
