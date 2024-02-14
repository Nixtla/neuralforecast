from neuralforecast.auto import *
from neuralforecast.losses.pytorch import HuberLoss, DistributionLoss


def get_model(model_name, horizon, num_samples):
    """Returns the model class given the model name.
    """
    model_dict = {
        'AutoLSTM': AutoLSTM(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples),
        'AutoRNN': AutoRNN(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples),
        'AutoGRU': AutoGRU(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'AutoDilatedRNN': AutoDilatedRNN(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples),
        'AutoDeepAR': AutoDeepAR(config=None, h=horizon,
                             loss=DistributionLoss(distribution='StudentT', level=[80, 90], return_params=False),
                             num_samples=num_samples, alias='model'),
        'AutoTCN': AutoTCN(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'AutoMLP': AutoMLP(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'AutoNBEATS': AutoNBEATS(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'AutoNHITS': AutoNHITS(config=None, h=horizon, loss=HuberLoss(), backend='optuna', num_samples=num_samples, alias='model'),
        'AutoDLinear': AutoDLinear(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'AutoTFT': AutoTFT(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'AutoVanillaTransformer': AutoVanillaTransformer(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'AutoInformer': AutoInformer(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'AutoAutoformer': AutoAutoformer(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'AutoFEDformer': AutoFEDformer(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'AutoTimesNet': AutoTimesNet(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model'),
        'AutoPatchTST': AutoPatchTST(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples, alias='model')
    }

    return model_dict[model_name]