from typing import Optional

import torch
from torch import nn
from torch.distributions import Normal, OneHotCategorical
import numpy as np

from miv.data.data_class import TestDataSetTorch, TestDataSet
from miv.data import preprocess

class LatentVariableModel:
    def __init__(self, response_net: nn.Module, iv_net_hidden: nn.Module, iv_net_obs: nn.Module, data_name: str):
        self.iv_net_hidden = iv_net_hidden
        self.iv_net_obs = iv_net_obs
        self.response_net = response_net
        self.data_name = data_name


    @classmethod
    def sample_from_density(cls, n_sample: int, response_net: nn.Module, norm_hidden: Normal,
                            norm_obs: Optional[Normal]=None, cat_obs: Optional[OneHotCategorical]=None,
                            covariate: Optional[torch.Tensor]=None):
        pred_list = []
        for i in range(n_sample):
            sample_obs = None
            if norm_obs:
                assert cat_obs is not None
                cat_sample = cat_obs.sample().unsqueeze(1)  # size = [B, 1, n_component]
                norm_sample = norm_obs.sample()  # size = [B, output_dim, n_component]
                sample_obs = torch.sum(cat_sample * norm_sample, dim=2)  # size = [B, output_dim]
            sample_hidden = norm_hidden.sample()  # size = [B, X_hidden]
            sample = torch.cat([sample_hidden, sample_obs], dim=-1) if sample_obs else sample_hidden
            if covariate is not None:
                pred = response_net(sample, covariate)
            else:
                pred = response_net(sample)
            pred_list.append(pred)
        return torch.cat(pred_list)


    def predict_t(self, treatment: torch.Tensor, covariate: Optional[torch.Tensor]):  # base function taking in torch.Tensor
        treatment = preprocess.rescale_treatment(treatment, self.data_name)
        if covariate is None:
            return self.response_net(treatment)
        else:
            return self.response_net(treatment, covariate)


    def predict(self, treatment: np.ndarray, covariate: Optional[np.ndarray]):  # numpy wrapper
        treatment_t = torch.tensor(treatment, dtype=torch.float32)
        covariate_t = None
        if covariate is not None:
            covariate_t = torch.tensor(covariate, dtype=torch.float32)
        return self.predict_t(treatment_t, covariate_t).data.numpy()


    def evaluate_t(self, test_data: TestDataSetTorch):
        target = test_data.Y_struct
        with torch.no_grad():
            pred = self.predict_t(test_data.X_all, test_data.covariate)
        pred = preprocess.inv_rescale_outcome(pred, self.data_name)
        return float((torch.norm((target - pred)) ** 2) / target.size()[0]), pred.numpy()


    def evaluate(self, test_data: TestDataSet):
        return self.evaluate_t(TestDataSetTorch.from_numpy(test_data)).data.item()

