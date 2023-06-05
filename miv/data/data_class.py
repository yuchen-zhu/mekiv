from typing import Optional
import numpy as np
import torch
from miv.util import dotdict


class TrainDataSet(dotdict):
    def __init__(self,
                 X_hidden: np.ndarray,
                 X_obs: Optional[np.ndarray],
                 covariate: Optional[np.ndarray],
                 Z: np.ndarray,
                 M: np.ndarray,
                 N: np.ndarray,
                 Y: np.ndarray,
                 Y_struct: np.ndarray):
        super(TrainDataSet, self).__init__()
        self.X_hidden = X_hidden
        self.X_obs = X_obs
        self.covariate = covariate
        self.Z = Z
        self.M = M
        self.N = N
        self.Y = Y
        self.Y_struct = Y_struct


class TestDataSet(dotdict):
    def __init__(self,
                 X_all: np.ndarray,
                 covariate: Optional[np.ndarray],
                 Y_struct: np.ndarray):
        super(TestDataSet, self).__init__()
        self.X_all = X_all
        self.covariate = covariate
        self.Y_struct = Y_struct


class ZTestDataSet(dotdict):
    def __init__(self,
                 Z: np.ndarray,
                 Y: np.ndarray):
        super(ZTestDataSet, self).__init__()
        self.Z = Z
        self.Y = Y


class TrainDataSetTorch(dotdict):
    def __int__(self,
                X_hidden: torch.Tensor,
                X_obs: Optional[torch.Tensor],
                covariate: Optional[torch.Tensor],
                Z: torch.Tensor,
                M: torch.Tensor,
                N: torch.Tensor,
                Y: torch.Tensor,
                Y_struct: torch.Tensor):

        super(TrainDataSetTorch, self).__init__()
        self.X_hidden = X_hidden
        self.X_obs = X_obs
        self.covariate = covariate
        self.Z = Z
        self.M = M
        self.N = N
        self.Y = Y
        self.Y_struct = Y_struct

    @classmethod
    def from_numpy(cls,
                   train_data: TrainDataSet):
        covariate, X_obs = None, None
        if train_data.covariate is not None:
            covariate = torch.tensor(train_data.covariate, dtype=torch.float64)
        if train_data.X_obs is not None:
            X_obs = torch.tensor(train_data.covariate, dtype=torch.float64)
        return cls(
            X_hidden=torch.tensor(train_data.X_hidden, dtype=torch.float64),
            X_obs=X_obs,
            covariate=covariate,
            Z=torch.tensor(train_data.Z, dtype=torch.float64),
            M=torch.tensor(train_data.M, dtype=torch.float64),
            N=torch.tensor(train_data.N, dtype=torch.float64),
            Y=torch.tensor(train_data.Y, dtype=torch.float64),
            Y_struct=torch.tensor(train_data.Y_struct, dtype=torch.float64)
        )

    def to_gpu(self):
        covariate, X_obs = None, None
        if self.covariate:
            covariate = self.covariate.cuda()
        if self.X_obs:
            X_obs = self.X_obs.cuda()
        self.X_hidden = self.X_hidden.cuda()
        self.X_obs = X_obs
        self.covariate = covariate
        self.Z = self.Z.cuda()
        self.M = self.M.cuda()
        self.N = self.N.cuda()
        self.Y = self.Y.cuda()
        self.Y_struct = self.Y_struct.cuda()


class TestDataSetTorch(dotdict):
    def __init__(self,
                 X_all: torch.Tensor,
                 covariate: torch.Tensor,
                 Y_struct: torch.Tensor):

        super(TestDataSetTorch, self).__init__()
        self.X_all = X_all
        self.covariate = covariate
        self.Y_struct = Y_struct

    @classmethod
    def from_numpy(cls,
                   test_data: TestDataSet):
        covariate = None
        if test_data.covariate is not None:
            covariate = torch.tensor(test_data.covariate, dtype=torch.float64)
        return cls(
            X_all=torch.tensor(test_data.X_all, dtype=torch.float64),
            covariate=covariate,
            Y_struct=torch.tensor(test_data.Y_struct, dtype=torch.float64)
        )

    def to_gpu(self):
        covariate = None
        if self.covariate:
            covariate = self.covariate.cuda()
        self.X_all = self.X_all.cuda()
        self.covariate = covariate
        self.Y_struct = self.Y_struct.cuda()


class ZTestDataSetTorch(dotdict):
    def __init__(self,
                 Z: torch.Tensor,
                 Y: torch.Tensor):

        super(ZTestDataSetTorch, self).__init__()
        self.Z = Z
        self.Y = Y

    @classmethod
    def from_numpy(cls,
                   z_test_data: TestDataSet):
        return cls(
            Z=torch.tensor(z_test_data.Z, dtype=torch.float32),
            Y=torch.tensor(z_test_data.Y_struct, dtype=torch.float32)
        )

    def to_gpu(self):
        self.Z = self.Z.cuda()
        self.Y = self.Y.cuda()



class StageMDataSet(dotdict):
    def __init__(self,
                 Chi: np.ndarray,
                 Z: Optional[np.ndarray],
                 labels: np.ndarray):
        super(StageMDataSet, self).__init__()
        self.Chi = Chi
        self.Z = Z
        self.labels = labels


class StageMDataSetTorch(dotdict):
    def __int__(self,
                Chi: torch.Tensor,
                Z: torch.Tensor,
                labels: torch.Tensor):

        super(StageMDataSetTorch, self).__init__()
        self.Chi = Chi
        self.Z = Z
        self.labels = labels

    @classmethod
    def from_numpy(cls,
                   train_data: StageMDataSet):
        # breakpoint()
        return cls(
            Chi=torch.tensor(train_data.Chi, dtype=torch.float64),
            Z=torch.tensor(train_data.Z, dtype=torch.float64),
            labels=torch.tensor(train_data.labels, dtype=torch.complex128)
        )

    def to_gpu(self):
        self.Z = self.Z.cuda()
        self.Chi = self.Chi.cuda()
        self.labels = self.labels.cuda()

