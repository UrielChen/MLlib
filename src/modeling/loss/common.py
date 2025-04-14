import torch.nn as nn
import torch.nn.functional as F
from .build import LOSS_FUNC_REGISTRY


@LOSS_FUNC_REGISTRY.register()
def mean_square_error():
    return nn.MSELoss()


@LOSS_FUNC_REGISTRY.register()
def l1_loss():
    return nn.L1Loss()


@LOSS_FUNC_REGISTRY.register()
def smooth_l1_loss():
    return nn.SmoothL1Loss()


@LOSS_FUNC_REGISTRY.register()
def cross_entropy_loss():
    return nn.CrossEntropyLoss()
    # return F.binary_cross_entropy


@LOSS_FUNC_REGISTRY.register()
def nll_loss():
    return nn.NLLLoss()
