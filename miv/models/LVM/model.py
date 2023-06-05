from typing import Optional

import torch
from torch import nn
import numpy as np

from miv.data.data_class import TestDataSetTorch, TestDataSet


class LatentVariableModel:
    def __init__(self, response: nn.Module, data_name: str):
        self.response = response
        self.data_name = data_name


    def predict_t(self, treatment: torch.Tensor, covariate: Optional[torch.Tensor]):
        # base function taking in torch.Tensor
        # breakpoint()
        if covariate is None:
            return self.response(treatment)
        else:
            return self.response(torch.cat([treatment, covariate], dim=-1))


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
        return float((torch.norm((target - pred)) ** 2) / target.size()[0]), pred.numpy()


    def evaluate(self, test_data: TestDataSet):
        return self.evaluate_t(TestDataSetTorch.from_numpy(test_data)).data.item()

