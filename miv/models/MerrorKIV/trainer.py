from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import logging
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
import torch
from torch import tensor
from numpy.random import default_rng


from miv.data import generate_train_data, generate_test_data
from miv.util import dotdict, make_dotdict
from miv.data.data_class import TrainDataSet, TrainDataSetTorch, StageMDataSetTorch
from miv.models.MerrorKIV.model import MerrorKIVModel
from miv.models.MerrorKIV.stage_m_utils import create_stage_M_raw_data, prepare_stage_M_data
from miv.models.MerrorKIV.stage_m import StageMModel, stage_m_train

# from miv.designs import datasets

logger = logging.getLogger()


def get_median(X) -> float:
    dist_mat = cdist(X, X, "sqeuclidean")
    res: float = np.median(a=dist_mat)
    return res


class MerrorKIVTrainer:

    def __init__(self, data_configs: dotdict, train_params: dotdict,
                 gpu_flg: bool = False, dump_folder: Optional[Path] = None):
        self.data_config = data_configs
        self.train_params = make_dotdict(train_params)

        self.lambda_mn = self.train_params["lambda_mn"]
        self.lambda_n = self.train_params["lambda_n"]
        self.xi = self.train_params["xi"]
        self.n_chi = self.train_params["n_chi"]
        self.split_ratio = self.train_params["split_ratio"]

    def split_train_data(self, train_data: TrainDataSet):
        n_data = train_data.X_hidden.shape[0]
        idx_train_1st, idx_train_2nd = train_test_split(np.arange(n_data), train_size=self.split_ratio)

        def get_data(data, idx):
            return data[idx] if data is not None else None

        train_1st_data, train_2nd_data = {}, {}
        for key in train_data.keys():
            train_1st_data[key], train_2nd_data[key] = get_data(train_data[key], idx_train_1st), get_data(train_data[key], idx_train_2nd)

        train_1st_data, train_2nd_data = TrainDataSet(**train_1st_data), TrainDataSet(**train_2nd_data)
        return train_1st_data, train_2nd_data

    def train(self, rand_seed: int = 42, verbose: int = 0):
        """

        Parameters
        ----------
        rand_seed: int
            random seed
        verbose : int
            Determine the level of logging
        Returns
        -------
        oos_result : float
            The performance of model evaluated by oos
        """

        train_data = generate_train_data(rand_seed=rand_seed, **self.data_config)
        test_data = generate_test_data(**self.data_config)
        train_1st_data, train_2nd_data = self.split_train_data(train_data)

        # get stage1 data
        N1, MN1 = train_1st_data.N, np.c_[train_1st_data.M, train_1st_data.N]
        N2, MN2 = train_2nd_data.N, np.c_[train_2nd_data.M, train_2nd_data.N]
        Z1 = train_1st_data.Z
        Y1 = train_1st_data.Y
        n = N1.shape[0]

        # get stageMerror data
        Z2 = train_2nd_data.Z

        # get stage2 data
        Y2 = train_2nd_data.Y
        m = Z2.shape[0]

        if verbose > 0:
            logger.info("start stage1")

        sigmaN = get_median(N1)
        sigmaMN = get_median(MN1)
        sigmaZ = get_median(Z1)
        KN1N1 = MerrorKIVModel.cal_gauss(N1, N1, sigmaN)
        KN1N2 = MerrorKIVModel.cal_gauss(N1, N2, sigmaN)
        KMN1MN1 = MerrorKIVModel.cal_gauss(MN1, MN1, sigmaMN)
        KMN1MN2 = MerrorKIVModel.cal_gauss(MN1, MN2, sigmaMN)
        KZ1Z1 = MerrorKIVModel.cal_gauss(Z1, Z1, sigmaZ)
        KZ1Z2 = MerrorKIVModel.cal_gauss(Z1, Z2, sigmaZ)
        # KX1X2 = MerrorKIVModel.cal_gauss(X1, X2, sigmaX)

        if isinstance(self.lambda_mn, list):
            lambda_mn = np.exp(np.linspace(self.lambda_mn[0], self.lambda_mn[1], 50))
            gamma_mn, lambda_mn = self.stage1_tuning(KMN1MN1, KMN1MN2, KZ1Z1, KZ1Z2, lambda_mn)
            self.lambda_mn = lambda_mn
        else:
            gamma_mn = np.linalg.solve(KZ1Z1 + n * self.lambda_mn * np.eye(n), KZ1Z2)

        if isinstance(self.lambda_n, list):
            lambda_n = np.exp(np.linspace(self.lambda_n[0], self.lambda_n[1], 50))
            gamma_n, lambda_n = self.stage1_tuning(KN1N1, KN1N2, KZ1Z1, KZ1Z2, lambda_n)
            self.lambda_n = lambda_n
        else:
            gamma_n = np.linalg.solve(KZ1Z1 + n * self.lambda_n * np.eye(n), KZ1Z2)

        if verbose > 0:
            logger.info("end stage 1")
            logger.info("start stage merror")

        # get stageM data
        M1 = train_1st_data.M
        stageM_data = create_stage_M_raw_data(self.n_chi, N1, M1, Z2, gamma_n, gamma_mn, sigmaN, KZ1Z2)
        stageM_data = prepare_stage_M_data(raw_data2=stageM_data, rand_seed=rand_seed)
        stage1_MNZ = dotdict({'M': M1, 'N': N1, 'Z': Z1, 'sigmaZ': sigmaZ})

        stage_m_out = self.stage_M_main(stageM_data=stageM_data, stage1_MNZ=stage1_MNZ, train_params=self.train_params)
        lambda_x, fitted_X = stage_m_out.lambda_x, stage_m_out.fitted_x


        print('------ fitted X / N / M / (M+N)/2  compared with ground truth X -------')
        print((np.sum((fitted_X - train_1st_data.X_hidden) ** 2) / fitted_X.shape[0]) ** 0.5 / np.std(
            train_1st_data.X_hidden))
        print((np.sum((train_1st_data.N - train_1st_data.X_hidden) ** 2) / fitted_X.shape[0]) ** 0.5 / np.std(
            train_1st_data.X_hidden))
        print((np.sum((train_1st_data.M - train_1st_data.X_hidden) ** 2) / fitted_X.shape[0]) ** 0.5 / np.std(
            train_1st_data.X_hidden))
        print((np.sum((1 / 2 * train_1st_data.M + 1 / 2 * train_1st_data.N - train_1st_data.X_hidden) ** 2) /
               fitted_X.shape[0]) ** 0.5 / np.std(train_1st_data.X_hidden))


        if train_1st_data.X_obs is not None:
            fitted_X = np.concatenate([fitted_X, train_1st_data.X_obs], axis=-1)
        if train_1st_data.covariate is not None:
            fitted_X = np.concatenate([fitted_X, train_1st_data.covariate], axis=-1)

        gamma_x = np.linalg.solve(KZ1Z1 + n * lambda_x * np.eye(n), KZ1Z2)
        sigmaX = get_median(fitted_X)
        KfittedX = MerrorKIVModel.cal_gauss(fitted_X, fitted_X, sigmaX)
        W = KfittedX.dot(gamma_x)
        if verbose > 0:
            logger.info("end stageM")
            logger.info("start stage2")

        if isinstance(self.xi, list):
            # breakpoint()
            self.xi = np.exp(np.linspace(self.xi[0], self.xi[1], 50))
            alpha, xi = self.stage2_tuning(W, KfittedX, Y1, Y2)
            self.xi = xi
        else:
            alpha = np.linalg.solve(W.dot(W.T) + m * self.xi * KfittedX, W.dot(Y2))

        if verbose > 0:
            logger.info("end stage2")

        mdl = MerrorKIVModel(fitted_X=fitted_X, alpha=alpha, sigma=sigmaX)
        test_input = test_data.X_all
        if test_data.covariate is not None:
            test_input = np.concatenate([test_input, test_data.covariate], axis=-1)
        Y_struct = test_data.Y_struct

        mse, preds = mdl.evaluate(test_data=test_data)
        return mse, test_input, preds, Y_struct

    def stage1_tuning(self, KX1X1, KX1X2, KZ1Z1, KZ1Z2, lambda_1):
        n = KX1X1.shape[0]
        gamma_list = [np.linalg.solve(KZ1Z1 + n * lam1 * np.eye(n), KZ1Z2) for lam1 in lambda_1]
        score = [np.trace(gamma.T.dot(KX1X1.dot(gamma)) - 2 * KX1X2.T.dot(gamma)) for gamma in gamma_list]
        lambda1 = lambda_1[np.argmin(score)]
        return gamma_list[np.argmin(score)], lambda1

    def stage2_tuning(self, W, KX1X1, Y1, Y2):
        M = W.shape[1]
        b = W.dot(Y2)
        A = W.dot(W.T)
        alpha_list = [np.linalg.solve(A + M * lam2 * KX1X1, b) for lam2 in self.xi]
        score = [np.linalg.norm(Y1 - KX1X1.dot(alpha)) for alpha in alpha_list]
        xi = self.xi[np.argmin(score)]
        return alpha_list[np.argmin(score)], xi


    def stage_M_main(self, stageM_data: StageMDataSetTorch, train_params: dotdict, stage1_MNZ: dotdict):
        model = StageMModel(stageM_data=stageM_data, train_params=train_params, stage1_MNZ=stage1_MNZ)
        model = stage_m_train(model, stageM_data=stageM_data, stageM_args=self.train_params)
        stage_M_out = dotdict({})

        stage_M_out.fitted_x = model.x.detach().numpy()
        assert stage_M_out.fitted_x.shape[0] == stage1_MNZ.Z.shape[0]

        if not train_params.lambda_x:
            lambda_x = np.exp(model.lambda_x.detach().numpy())  # todo: these are the worng syntax
            # breakpoint()
        else:
            lambda_x = model.lambda_x

        stage_M_out.lambda_x = lambda_x

        return stage_M_out


