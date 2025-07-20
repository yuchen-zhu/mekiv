from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import logging

from sklearn.model_selection import train_test_split
import torch
from torch import tensor
from numpy.random import default_rng


from miv.data import load_data, split_train_data
from miv.utils.util import dotdict, make_dotdict
from miv.data.data_class import TrainDataSet, TrainDataSetTorch, StageMDataSetTorch
from miv.models.MerrorKIV.model import MerrorKIVModel
from miv.models.MerrorKIV.stage_m_utils import (
    create_stage_2_raw_data,
    prepare_stage_2_data,
)
from miv.models.MerrorKIV.stage_m import StageMModel, stage_m_train
from miv.models.MerrorKIV.gaussian_gram import compute_gaussian_gram
from miv.models.MerrorKIV.get_median import get_median
from miv.models.MerrorKIV.stage_m_utils import log_stage2_results


logger = logging.getLogger(__name__)


class MerrorKIVTrainer:

    def __init__(
        self,
        data_configs: dotdict,
        train_params: dotdict
    ):
        self.data_config = data_configs
        self.train_params = make_dotdict(train_params)

        self.lambda_mn_inits = self.train_params["lambda_mn_inits"]
        self.lambda_n_inits = self.train_params["lambda_n_inits"]
        self.xi = self.train_params["xi"]
        self.n_chi = self.train_params["n_chi"]
        self.split_ratio = self.train_params["split_ratio"]


    def train(self, rand_seed: int = 42):
        """

        Parameters
        ----------
        rand_seed: int
            random seed

        Returns
        -------
        oos_result : float
            The performance of model evaluated by oos
        """
        train_data, test_data = load_data(
            data_config=self.data_config, rand_seed=rand_seed
        )

        train_1st_data, train_2nd_data = split_train_data(
            split_ratio=self.split_ratio, train_data=train_data
        )

        # get stage1 data
        N1 = train_1st_data.N
        MN1 = np.c_[train_1st_data.M, train_1st_data.N]
        N2 = train_2nd_data.N
        MN2 = np.c_[train_2nd_data.M, train_2nd_data.N]
        Z1 = train_1st_data.Z
        Y1 = train_1st_data.Y
        n = N1.shape[0]

        # get stage2 data
        M1 = train_1st_data.M
        Z2 = train_2nd_data.Z

        # get stage3 data
        Y2 = train_2nd_data.Y
        m = Z2.shape[0]

        gamma_mn, gamma_n = self.stage1(
            N1=N1,
            N2=N2,
            MN1=MN1,
            MN2=MN2,
            Z1=Z1,
            Z2=Z2
        )
        
        logger.info("start stage 2 (measurement error correction)")

        stageM_data = create_stage_2_raw_data(
            n_chi=self.n_chi, 
            N1=N1, 
            M1=M1, 
            Z2=Z2, 
            gamma_n=gamma_n, 
            gamma_mn=gamma_mn
        )
        stageM_data = prepare_stage_2_data(raw_data2=stageM_data, rand_seed=rand_seed)
        stage1_MNZ = dotdict(
            {
                "M": M1, 
                "N": N1, 
                "Z": Z1, 
                "sigmaZ": get_median(Z1)
            }
        )

        stage_2_out = self.stage_2_main(
            stageM_data=stageM_data,
            stage1_MNZ=stage1_MNZ,
            train_params=self.train_params,
        )
        lambda_x_final, fitted_X = stage_2_out.lambda_x_final, stage_2_out.fitted_x


        log_stage2_results(
            fitted_X=fitted_X,
            X_hidden=train_1st_data.X_hidden,
            M=train_1st_data.M,
            N=train_1st_data.N
        )

        if train_1st_data.X_obs is not None:
            fitted_X = np.concatenate([fitted_X, train_1st_data.X_obs], axis=-1)
        if train_1st_data.covariate is not None:
            fitted_X = np.concatenate([fitted_X, train_1st_data.covariate], axis=-1)

        assert hasattr(self, 'KZ1Z1') and hasattr(self, 'KZ1Z2')
        gamma_x = np.linalg.solve(self.KZ1Z1 + n * lambda_x_final * np.eye(n), self.KZ1Z2)
        sigmaX = get_median(fitted_X)
        KfittedX = compute_gaussian_gram(fitted_X, fitted_X, sigmaX)
        W = KfittedX.dot(gamma_x)

        logger.info("end stage 2 (measurement error correction)")

        logger.info("start stage3")

        if isinstance(self.xi, list):
            # breakpoint()
            self.xi = np.exp(np.linspace(self.xi[0], self.xi[1], 50))
            alpha, xi = self.stage3_tuning(W, KfittedX, Y1, Y2)
            self.xi = xi
        else:
            alpha = np.linalg.solve(W.dot(W.T) + m * self.xi * KfittedX, W.dot(Y2))

        logger.info("end stage3")

        logger.info("start evaluation")
        mdl = MerrorKIVModel(fitted_X=fitted_X, alpha=alpha, sigma=sigmaX)
        test_input = test_data.X_all
        if test_data.covariate is not None:
            test_input = np.concatenate([test_input, test_data.covariate], axis=-1)
        Y_struct = test_data.Y_struct

        mse, preds = mdl.evaluate(test_data=test_data)
        return mse, test_input, preds, Y_struct
    

    def stage1(self, N1, N2, MN1, MN2, Z1, Z2):
        """
        
        :return gamma_mn: 
        :return gamma_n:
        """

        logger.info("start stage1")

        n = N1.shape[0]
        
        sigmaN = get_median(N1)
        sigmaMN = get_median(MN1)
        sigmaZ = get_median(Z1)
        
        KN1N1 = compute_gaussian_gram(N1, N1, sigmaN)
        KN1N2 = compute_gaussian_gram(N1, N2, sigmaN)
        KMN1MN1 = compute_gaussian_gram(MN1, MN1, sigmaMN)
        KMN1MN2 = compute_gaussian_gram(MN1, MN2, sigmaMN)
        KZ1Z1 = compute_gaussian_gram(Z1, Z1, sigmaZ)
        KZ1Z2 = compute_gaussian_gram(Z1, Z2, sigmaZ)

        if isinstance(self.lambda_mn_inits, list):
            lambda_mn_inits = np.exp(np.linspace(self.lambda_mn_inits[0], self.lambda_mn_inits[1], 50))
            gamma_mn, lambda_mn_final = self.stage1_tuning(
                KMN1MN1, KMN1MN2, KZ1Z1, KZ1Z2, lambda_mn_inits
            )
            self.lambda_mn_final = lambda_mn_final
        else:
            self.lambda_mn_final = lambda_mn_inits
            gamma_mn = np.linalg.solve(KZ1Z1 + n * self.lambda_mn_final * np.eye(n), KZ1Z2)

        if isinstance(self.lambda_n_inits, list):
            lambda_n_inits = np.exp(np.linspace(self.lambda_n_inits[0], self.lambda_n_inits[1], 50))
            gamma_n, lambda_n_final = self.stage1_tuning(
                KN1N1, KN1N2, KZ1Z1, KZ1Z2, lambda_n_inits
            )
            self.lambda_n_final = lambda_n_final
        else:
            self.lambda_n_final = self.lambda_n_inits
            gamma_n = np.linalg.solve(KZ1Z1 + n * self.lambda_n_final * np.eye(n), KZ1Z2)
            
        logger.info("saving KZ1Z1 and KZ1Z2 for later use.")
        self.KZ1Z1 = KZ1Z1
        self.KZ1Z2 = KZ1Z2
        
        logger.info("end stage 1")
        
        return gamma_mn, gamma_n
    
        
    def stage1_tuning(self, KX1X1, KX1X2, KZ1Z1, KZ1Z2, lambda_1):
        n = KX1X1.shape[0]
        gamma_list = [
            np.linalg.solve(KZ1Z1 + n * lam1 * np.eye(n), KZ1Z2) for lam1 in lambda_1
        ]
        score = [
            np.trace(gamma.T.dot(KX1X1.dot(gamma)) - 2 * KX1X2.T.dot(gamma))
            for gamma in gamma_list
        ]
        lambda1 = lambda_1[np.argmin(score)]
        return gamma_list[np.argmin(score)], lambda1

    def stage_2_main(
        self,
        stageM_data: StageMDataSetTorch,
        train_params: dotdict,
        stage1_MNZ: dotdict,
    ):
        model = StageMModel(
            stageM_data=stageM_data, train_params=train_params, stage1_MNZ=stage1_MNZ
        )
        model = stage_m_train(
            model, stageM_data=stageM_data, stageM_args=self.train_params
        )
        stage_M_out = dotdict({})

        stage_M_out.fitted_x = model.x.detach().numpy()
        assert stage_M_out.fitted_x.shape[0] == stage1_MNZ.Z.shape[0]

        if not train_params.lambda_x:
            lambda_x = np.exp(
                model.lambda_x.detach().numpy()
            )  # todo: these are the worng syntax
            # breakpoint()
        else:
            lambda_x = model.lambda_x

        stage_M_out.lambda_x_final = lambda_x

        return stage_M_out

    def stage3_tuning(self, W, KX1X1, Y1, Y2):
        M = W.shape[1]
        b = W.dot(Y2)
        A = W.dot(W.T)
        alpha_list = [np.linalg.solve(A + M * lam2 * KX1X1, b) for lam2 in self.xi]
        score = [np.linalg.norm(Y1 - KX1X1.dot(alpha)) for alpha in alpha_list]
        xi = self.xi[np.argmin(score)]
        return alpha_list[np.argmin(score)], xi
