from neuralforecast.auto import *
from neuralforecast.losses.pytorch import HuberLoss, DistributionLoss


def get_model(model_name, horizon, num_samples):
    """Returns the model class given the model name.
    """
    model_dict = {
        'AutoLSTM': AutoLSTM(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples),
        'AutoRNN': AutoRNN(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples),
        'AutoGRU': AutoGRU(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples),
        'AutoDilatedRNN': AutoDilatedRNN(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples),
        'AutoDeepAR': AutoDeepAR(config=None, h=horizon,
                             loss=DistributionLoss(distribution='StudentT', level=[80, 90], return_params=False),
                             num_samples=num_samples),
        'AutoTCN': AutoTCN(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples),
        'AutoMLP': AutoMLP(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples),
        'AutoNBEATS': AutoNBEATS(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples),
        'AutoNHITS': AutoNHITS(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples),
        'AutoDLinear': AutoDLinear(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples),
        'AutoTFT': AutoTFT(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples),
        'AutoVanillaTransformer': AutoVanillaTransformer(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples),
        'AutoInformer': AutoInformer(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples),
        'AutoAutoformer': AutoAutoformer(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples),
        'AutoFEDformer': AutoFEDformer(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples),
        'AutoTimesNet': AutoTimesNet(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples),
        'AutoPatchTST': AutoPatchTST(config=None, h=horizon, loss=HuberLoss(), num_samples=num_samples)
    }

    return model_dict[model_name]