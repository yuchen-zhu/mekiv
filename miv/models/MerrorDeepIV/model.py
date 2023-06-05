import numpy as np
from scipy.spatial.distance import cdist
import torch

from miv.data.data_class import TrainDataSet, TestDataSet
from miv.models.LVM.model_old import LatentVariableModel

from torch import nn

class MerrorDeepIVVModel(LatentVariableModel):
    def __init__(self, response_net: nn.Module, iv_net_hidden: nn.Module, iv_net_obs: nn.Module, data_name: str):
        super(MerrorDeepIVVModel, self).__init__(response_net, iv_net_hidden, iv_net_obs, data_name)
