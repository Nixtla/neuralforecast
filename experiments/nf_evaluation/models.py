from neuralforecast.auto import *
from neuralforecast.losses.pytorch import HuberLoss, DistributionLoss


def get_model(model_name, horizon, num_samples):
    """Returns the model class given the model name.
    """
    model_dict = {
        'lstm': AutoLSTM(config=None, horizon=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'rnn': AutoRNN(config=None, horizon=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'gru': AutoGRU(config=None, horizon=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'dilatedrnn': AutoDilatedRNN(config=None, horizon=horizon, loss=HuberLoss(), num_samples=num_samples),
        'deepar': AutoDeepAR(config=None, horizon=horizon,
                             loss=DistributionLoss(distribution='StudentT', level=[80, 90], return_params=False),
                             num_samples=num_samples, alias='model'),
        'tcn': AutoTCN(config=None, horizon=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'mlp': AutoMLP(config=None, horizon=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'nbeats': AutoNBEATS(config=None, horizon=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'nhits': AutoNHITS(config=None, horizon=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'dlinear': AutoDLinear(config=None, horizon=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'tft': AutoTFT(config=None, horizon=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'vanillatransformer': AutoVanillaTransformer(config=None, horizon=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'informer': AutoInformer(config=None, horizon=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'autoformer': AutoAutoformer(config=None, horizon=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'fedformer': AutoFEDformer(config=None, horizon=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'timesnet': AutoTimesNet(config=None, horizon=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'patchtst': AutoPatchTST(config=None, horizon=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model')
    }

    return model_dict[model_name]