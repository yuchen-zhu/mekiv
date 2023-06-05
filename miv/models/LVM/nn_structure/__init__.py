from typing import Tuple, Optional

import torch
from torch import nn

from .nn_structure_for_demand_low_dim import build_net_for_demand_low_dim
from .nn_structure_for_linear_and_sigmoid import build_net_for_linear_and_sigmoid
from miv.util import dotdict
# from .nn_structure_for_demand_image import build_net_for_demand_image
# from .nn_structure_for_dsprite import build_net_for_dsprite

import logging

logger = logging.getLogger()


def build_extractor(data_name: str, **args) -> dotdict:
    if data_name == "demand":
        logger.info("build demand LVM model without image")
        return build_net_for_demand_low_dim(**args)
    elif data_name == "sigmoid":
        logger.info("build sigmoid LVM model")
        return build_net_for_linear_and_sigmoid(**args)
    elif data_name == "linear":
        logger.info("build linear LVM model")
        return build_net_for_linear_and_sigmoid(**args)
    elif data_name == "linear_cp":
        logger.info("build linear LVM model")
        return build_net_for_linear_and_sigmoid(**args)
    elif data_name == "demand_image":
        raise ValueError(f"data name {data_name} is not implemented")
    elif data_name == "dsprite":
        raise ValueError(f"data name {data_name} is not implemented")
    else:
        raise ValueError(f"data name {data_name} is not implemented")
