import numpy as np
from scipy import stats

from ..data.data_class import TrainDataSet, TestDataSet, ZTestDataSet
from ..data.merror_funcs import get_merror_func
from numpy.random import default_rng
import torch
from torch import nn
from miv.util import dotdict

import pickle

data = None


class YModel(torch.nn.Module):
    # Y = f(X) + g(U) + eps
    # p(y|x, u) = gaussian(f(x) + g(u), eps)
    # input dim = dim_x + dim_u = 1 + 2 + 5 = 8

    def __init__(self, data):
        super().__init__()

        num_u = 28
        num_z = 12
        num_y = 1
        num_x, num_m, num_n = 1, 1, 1

        self.fx = nn.Sequential(nn.Linear(1, 5), nn.ReLU(), nn.Linear(5, 1))

        self.gu = nn.Sequential(nn.Linear(num_u, 5), nn.ReLU(), nn.Linear(5, 1))

        self.y_noise_logscale = nn.Linear(1 + num_u, 1)

        self.data = data

    def forward(self, x, idx):
        u = self.data.U[idx]

        pyxu_mean, pyxu_scale = self.forward_(x, u)

        return pyxu_mean, pyxu_scale

    def forward_(self, x, u):
        fx = self.fx(x)
        gu = self.gu(u)

        pyxu_mean = fx + gu

        pyxu_scale = torch.exp(self.y_noise_logscale(torch.cat([x, u], axis=-1)))

        return pyxu_mean, pyxu_scale


with open('miv/data/dahl_lochner.pickle', 'rb') as handle:
    data = pickle.load(handle)

data_tf = dotdict({})

for key in data.keys():
    data_tf[key] = torch.tensor(data[key], dtype=torch.float32)


y_model = YModel(data=None)

y_model.load_state_dict(torch.load('miv/data/dahl_lochner_f.pt'))
y_model.eval()


def f(x: np.ndarray) -> np.ndarray:
    f = y_model.fx
    g = y_model.gu
    exp_gu = torch.mean(g(data_tf['U']))
    x_tf = torch.tensor(x, dtype=torch.float32).reshape(-1,1)
    fx = f(x_tf) + exp_gu

    return fx.detach().numpy()


def generate_test_dahl_lochner() -> TestDataSet:
    """
    Returns
    -------
    test_data : TestDataSet
        Uniformly sampled from x.
    """
    x = np.linspace(-1.5, 1.5, 100)
    y = f(x)

    test_data = TestDataSet(X_all=x[:, np.newaxis],
                            Y_struct=y,
                            covariate=None)
    return test_data


def generate_train_dahl_lochner(data_size: int,
                                  merror_func_str: str,
                                  m_scale: float,
                                  n_scale: float,
                                  bias: float,
                                  rand_seed: int = 42) -> TrainDataSet:
    """

    Parameters
    ----------
    data_size : int
        size of data
    merror_func_str: str
        parameter for choosing a measurement error mechanism
    m_scale: float
        chooses the error spread in M
    n_scale: float
        chooses the error spread in N
    bias: float
        chooses the bias level in N
    rand_seed : int
        random seed


    Returns
    -------
    train_data : TrainDataSet
    """
    structural = f(data['X'][:data_size])

    train_data = TrainDataSet(X_hidden=data['X'][:data_size],
                              X_obs=None,
                              covariate=None,
                              M=data['M'][:data_size],
                              N=data['N'][:data_size],
                              Z=data['Z'][:data_size],
                              Y_struct=structural,
                              Y=data['Y'][:data_size])
    return train_data

