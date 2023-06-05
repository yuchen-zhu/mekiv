import numpy as np
from miv.util import dotdict, fns
from scipy import stats
import torch
from miv.data import sim_dgp


class Datasets:
    def __init__(self):
        self.custom1 = None

        self.NP = None  # Newey Powell

        self.HLLT = None  # Hartford Lewis Leyton-Brown Taddy demand design

    def construct_custom1(self):
        custom1_design = dotdict({
            'name': 'custom1',
            'N_data': 10000,
            'noise_level': 0.,
            'noise_level_merror': 3.,
            'merror_dim': 0,
            'bias': 0.
        })
        custom1_design = dotdict(custom1_design)
        custom1_design.fu = lambda N_data: np.random.normal(0, 1.5, size=N_data).reshape(-1,1)
        custom1_design.fz = lambda N_data: np.random.normal(0, 1.5, size=N_data).reshape(-1,1)
        custom1_design.fx = lambda z, u, N_data: u + z + custom1_design.noise_level * np.random.normal(0, 1, N_data).reshape(-1,1)
        custom1_design.fy = lambda x, u, N_data: fns.sigmoid(x) + 0.05 * u + custom1_design.noise_level * np.random.normal(0, 1, N_data).reshape(-1,1)
        custom1_design.fm = lambda x, N_data: x + custom1_design.noise_level_merror * np.random.normal(0, 1, N_data).reshape(-1,1)
        custom1_design.fn = lambda x, N_data: x + custom1_design.noise_level_merror * np.random.normal(0, 1, N_data).reshape(-1,1) + custom1_design.bias
        custom1_design.ydox = lambda x, u, N_data: fns.sigmoid(x)

        self.custom1 = dotdict({})
        self.custom1.data = sim_dgp(custom1_design)
        self.custom1.design = custom1_design

    def construct_NP(self): # Newey-Powell 'sigmoid' design
        NP_design = dotdict({
            'name': 'NP',
            'N_data': 10000,
            'noise_level': 2.,
            'noise_level_merror': 0.5,
            'merror_dim': 0,
            'bias': 0.
        })
        NP_design = dotdict(NP_design)

        NP_design.MU = np.zeros((3,))
        NP_design.SIGMA = np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])

        NP_design.futw = lambda N_data: np.random.multivariate_normal(NP_design.MU, NP_design.SIGMA, size=N_data)
        NP_design.utw = NP_design.futw(NP_design.N_data)
        NP_design.fu = lambda N_data: NP_design.utw[:, 0].reshape(-1,1)
        NP_design.fz = lambda N_data: stats.norm.cdf(NP_design.utw[:, 2]).reshape(-1,1)
        NP_design.fx = lambda z, u, N_data: stats.norm.cdf((NP_design.utw[:, 1] + NP_design.utw[:, 2]) / np.sqrt(2)).reshape(-1,1)
        NP_design.fy = lambda x, u, N_data: np.log(np.abs(16 * x - 8) + 1) * np.sign(x - 0.5) + u
        NP_design.fm = lambda x, N_data: x + NP_design.noise_level_merror * np.random.normal(0, 1, N_data).reshape(-1,1)
        NP_design.fn = lambda x, N_data: x + NP_design.noise_level_merror * np.random.normal(0, 1, N_data).reshape(-1,1) + NP_design.bias
        NP_design.ydox = lambda x: np.log(np.abs(16 * x - 8) + 1) * np.sign(x - 0.5)
        data_NP = sim_dgp(NP_design)

        self.NP = dotdict({})
        self.NP.data = data_NP
        self.NP.design = NP_design

    def construct_HLLT(self):  # demand design
        def get_x(z, u, N_data):
            # t = HLLT_.ft(N_data)
            # s = HLLT_.fs(N_data)
            c = z[:, 0].reshape(-1,1)
            t = z[:, 1].reshape(-1,1)
            s = z[:, 2].reshape(-1,1)
            p = 25 + (c+3)*get_psi(t) + u

            return np.concatenate([p, t, s], axis=-1)

        def get_z(N_data):
            c = np.random.normal(0, 1, HLLT_.N_data).reshape(-1,1)
            t = HLLT_.ft(N_data)
            s = HLLT_.fs(N_data)
            # breakpoint()
            return np.concatenate([c, t, s], axis=-1)


        def get_psi(t):
            out = 2 * ((t - 5)**4/600 + np.exp(-4 * (t - 5) ** 2) + t / 10 - 2)
            return out

        HLLT_ = dotdict({
            'name': 'HLLT',
            'N_data': 10000,
            'noise_level': 2.,
            'noise_level_merror': 0.,
            'obs_dim': [1, 2],
            'merror_dim': 0,  # can only do 1 merror dim right now.
            'discrete_xdims': [2],
            'bias': 0.,
            'rho': 0.5
        })
        HLLT_ = dotdict(HLLT_)
        # HLLT_.fz = lambda: np.random.normal(0, 1, HLLT_.N_data).reshape(-1,1)
        # HLLT_.fv = lambda N_data: np.random.normal(0, 1, HLLT_.N_data).reshape(-1,1)  # this is u
        HLLT_.fs = lambda N_data: np.random.choice(7, HLLT_.N_data).reshape(-1,1) + 1
        HLLT_.ft = lambda N_data: np.random.uniform(0, 1, HLLT_.N_data).reshape(-1,1) * 10

        HLLT_.fu = lambda N_data: np.random.normal(0, 1, HLLT_.N_data).reshape(-1,1)  # noise price
        # HLLT_.fz = lambda N_data: np.random.normal(0, 1, HLLT_.N_data).reshape(-1,1)
        HLLT_.fz = lambda N_data: get_z(N_data)
        HLLT_.fx = lambda z, u, N_data: get_x(z, u, N_data)
        HLLT_.fm = lambda x, N_data: np.reshape(x[:, HLLT_.merror_dim] + HLLT_.noise_level_merror * np.random.normal(0, 1, N_data), newshape=(-1, 1))
        HLLT_.fn = lambda x, N_data: np.reshape(x[:, HLLT_.merror_dim] + HLLT_.noise_level_merror * np.random.normal(0, 1, N_data) + HLLT_.bias, newshape=(-1, 1))
        HLLT_.fy = lambda x, u, N_data: np.reshape(100 + (10 + x[:, 0]) * x[:, 2] * get_psi(x[:, 1]) - 2 * x[:, 0] \
                                        + np.random.normal(HLLT_.rho * u, np.sqrt(1 - HLLT_.rho ** 2), HLLT_.N_data), newshape=(-1,1))
        HLLT_.ydox = lambda x: 100 + (10 + x[:, 0]) * x[:, 2] * get_psi(x[:, 1]) - 2 * x[:, 0]

        self.HLLT = dotdict({})
        self.HLLT.data = sim_dgp(HLLT_)
        self.HLLT.design = HLLT_

    def construct_datasets(self):
        self.construct_custom1()
        self.construct_NP()
        self.construct_HLLT()


datasets = Datasets(); datasets.construct_datasets()