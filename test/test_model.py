# mypy: ignore-errors
# #%% Test all models against all losses
import pytest
import itertools
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.utils import generate_series
#%% Unit test for model - loss - valid loss combinations   
seed = 0
test_size = 14
FREQ = "D"

# 1 series, no exogenous
N_SERIES_1 = 1
df = generate_series(n_series=N_SERIES_1, seed=seed, freq=FREQ, equal_ends=True)
max_ds = df.ds.max() - pd.Timedelta(test_size, FREQ)
Y_TRAIN_DF_1 = df[df.ds < max_ds]
Y_TEST_DF_1 = df[df.ds >= max_ds]

# 5 series, no exogenous
N_SERIES_2 = 5
df = generate_series(n_series=N_SERIES_2, seed=seed, freq=FREQ, equal_ends=True)
max_ds = df.ds.max() - pd.Timedelta(test_size, FREQ)
Y_TRAIN_DF_2 = df[df.ds < max_ds]
Y_TEST_DF_2 = df[df.ds >= max_ds]

# 1 series, with static and temporal exogenous
N_SERIES_3 = 1
df, STATIC_3 = generate_series(n_series=N_SERIES_3, n_static_features=2, 
                     n_temporal_features=2, seed=seed, freq=FREQ, equal_ends=True)
max_ds = df.ds.max() - pd.Timedelta(test_size, FREQ)
Y_TRAIN_DF_3 = df[df.ds < max_ds]
Y_TEST_DF_3 = df[df.ds >= max_ds]

# 5 series, with static and temporal exogenous
N_SERIES_4 = 5
df, STATIC_4 = generate_series(n_series=N_SERIES_4, n_static_features=2, 
                     n_temporal_features=2, seed=seed, freq=FREQ, equal_ends=True)
max_ds = df.ds.max() - pd.Timedelta(test_size, FREQ)
Y_TRAIN_DF_4 = df[df.ds < max_ds]
Y_TEST_DF_4 = df[df.ds >= max_ds]

# Generic test for a given config for a model
def _run_model_tests(model_class, config):
    if model_class.RECURRENT:
        config["inference_input_size"] = config["input_size"]

    # DF_1
    if model_class.MULTIVARIATE:
        config["n_series"] = N_SERIES_1
    if isinstance(config["loss"], relMSE):
        config["loss"].y_train = Y_TRAIN_DF_1["y"].values   
    if isinstance(config["valid_loss"], relMSE):
        config["valid_loss"].y_train = Y_TRAIN_DF_1["y"].values   

    model = model_class(**config)
    fcst = NeuralForecast(models=[model], freq=FREQ)
    fcst.fit(df=Y_TRAIN_DF_1, val_size=24)
    forecasts = fcst.predict(futr_df=Y_TEST_DF_1)
    # DF_2
    if model_class.MULTIVARIATE:
        config["n_series"] = N_SERIES_2
    if isinstance(config["loss"], relMSE):
        config["loss"].y_train = Y_TRAIN_DF_2["y"].values   
    if isinstance(config["valid_loss"], relMSE):
        config["valid_loss"].y_train = Y_TRAIN_DF_2["y"].values
    model = model_class(**config)
    fcst = NeuralForecast(models=[model], freq=FREQ)
    fcst.fit(df=Y_TRAIN_DF_2, val_size=24)
    forecasts = fcst.predict(futr_df=Y_TEST_DF_2)

    if model.EXOGENOUS_STAT and model.EXOGENOUS_FUTR:
        # DF_3
        if model_class.MULTIVARIATE:
            config["n_series"] = N_SERIES_3
        if isinstance(config["loss"], relMSE):
            config["loss"].y_train = Y_TRAIN_DF_3["y"].values   
        if isinstance(config["valid_loss"], relMSE):
            config["valid_loss"].y_train = Y_TRAIN_DF_3["y"].values
        model = model_class(**config)
        fcst = NeuralForecast(models=[model], freq=FREQ)
        fcst.fit(df=Y_TRAIN_DF_3, static_df=STATIC_3, val_size=24)
        forecasts = fcst.predict(futr_df=Y_TEST_DF_3)
        # DF_4
        if model_class.MULTIVARIATE:
            config["n_series"] = N_SERIES_4
        if isinstance(config["loss"], relMSE):
            config["loss"].y_train = Y_TRAIN_DF_4["y"].values   
        if isinstance(config["valid_loss"], relMSE):
            config["valid_loss"].y_train = Y_TRAIN_DF_4["y"].values 
        model = model_class(**config)
        fcst = NeuralForecast(models=[model], freq=FREQ)
        fcst.fit(df=Y_TRAIN_DF_4, static_df=STATIC_4, val_size=24)
        forecasts = fcst.predict(futr_df=Y_TEST_DF_4)       # noqa F841

#%% Test model - loss - valid_loss combinations
import neuralforecast.models as nfm
from neuralforecast.losses.pytorch import MAE, MSE, RMSE, MAPE, SMAPE, MASE, QuantileLoss, MQLoss, IQLoss, DistributionLoss, PMM, GMM, NBMM, HuberLoss, TukeyLoss, HuberQLoss, HuberMQLoss, relMSE

models = [nfm.NHITS, nfm.DeepAR, nfm.TSMixer]

losses = [MAE(), MSE(), RMSE(), MAPE(), SMAPE(), MASE(seasonality=7), QuantileLoss(q=0.5), MQLoss(), 
           IQLoss(), DistributionLoss("Normal"), DistributionLoss("StudentT"), DistributionLoss("Poisson"), 
           DistributionLoss("NegativeBinomial"), DistributionLoss("Tweedie", rho=1.5), DistributionLoss("ISQF"), 
           PMM(), PMM(weighted=True), GMM(), GMM(weighted=True), NBMM(), NBMM(weighted=True), HuberLoss(), 
           TukeyLoss(), HuberQLoss(q=0.5), HuberMQLoss()]
valid_losses = [None]
combinations = list(itertools.product(*[models, losses, valid_losses]))


@pytest.mark.parametrize("model_class, loss, valid_loss",
                         combinations
                         )
def test_losses(model_class, loss, valid_loss):
    config = {'max_steps': 2,
              'h': 7,
              'input_size': 28,
              'loss': loss,
              'valid_loss': valid_loss}
    
    _run_model_tests(model_class, config)    

#%% Test return params on distribution losses on all model class types
models = [
             nfm.DeepAR, nfm.NHITS, nfm.TSMixer
           ]
losses = [
          DistributionLoss("Normal", return_params=True), DistributionLoss("StudentT", return_params=True), DistributionLoss("Poisson", return_params=True), 
           DistributionLoss("NegativeBinomial", return_params=True), 
           PMM(return_params=True), PMM(weighted=True, return_params=True), GMM(return_params=True), GMM(weighted=True, return_params=True), NBMM(return_params=True), NBMM(weighted=True, return_params=True)
           ]
valid_losses = [None]
combinations = list(itertools.product(*[models, losses, valid_losses]))

@pytest.mark.parametrize("model_class, loss, valid_loss",
                         combinations
                         )
def test_dlosses_return_params(model_class, loss, valid_loss):
    config = {'max_steps': 2,
              'h': 7,
              'input_size': 28,
              'loss': loss,
              'valid_loss': valid_loss}
    
    _run_model_tests(model_class, config)      
#%% Test horizon weights
import torch

models = [
             nfm.DeepAR, nfm.NHITS, nfm.TSMixer
           ]
losses = [MAE(horizon_weight=torch.randint(0, 1, (7, ))),
          DistributionLoss("Normal", horizon_weight=torch.randint(0, 1, (7, ))),
          DistributionLoss("Normal", horizon_weight=torch.randint(0, 1, (7, )), return_params=True)]
valid_losses = [None]
combinations = list(itertools.product(*[models, losses, valid_losses]))

@pytest.mark.parametrize("model_class, loss, valid_loss",
                         combinations
                         )
def test_losses_horizon_weight(model_class, loss, valid_loss):
    config = {'max_steps': 2,
              'h': 7,
              'input_size': 28,
              'loss': loss,
              'valid_loss': valid_loss}
    
    _run_model_tests(model_class, config)         