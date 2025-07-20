import abc
import logging
from typing import Optional

import numpy as np

from miv.data import load_data, split_train_data
from miv.models.KIV.model import KernelIVModel
from miv.utils import DotDict, compute_median_sq_dist, compute_gaussian_gram

logger = logging.getLogger(__name__)


class BaseKIVTrainer(abc.ABC):
    which_regressor: str = None

    def __init__(self, data_configs: DotDict, train_params: DotDict):
        if self.which_regressor is None:
            raise ValueError("which_regressor must be set in the subclass")
        self.data_config = data_configs
        self.train_params = train_params

        self.split_ratio = train_params["split_ratio"]

        # Initial hyperparameters
        self.lambd_inits = train_params["lambda_inits"]
        self.xi_inits = train_params["xi"]

        # Hyperparameter after tuning
        self.lambd_final: Optional = None
        self.xi_final: Optional = None
        self.alpha_final: Optional = None

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
        if self.which_regressor in {"X_hidden", "M", "N"}:
            X1 = train_1st_data[self.which_regressor]
        elif self.which_regressor == "avMN":
            X1 = 0.5 * train_1st_data["M"] + 0.5 * train_1st_data["N"]
        else:
            raise ValueError("which_regressor needs to be {X_hidden, M, N, avMN}")

        if train_data.X_obs:
            X1 = np.concatenate([X1, train_1st_data.X_obs], axis=-1)
        if train_1st_data.covariate is not None:
            X1 = np.concatenate([X1, train_1st_data.covariate], axis=-1)
        Z1 = train_1st_data.Z
        Y1 = train_1st_data.Y
        n = X1.shape[0]

        # get stage2 data
        if self.which_regressor in {"X_hidden", "M", "N"}:
            X2 = train_2nd_data[self.which_regressor]
        elif self.which_regressor == "avMN":
            X2 = 0.5 * train_2nd_data["M"] + 0.5 * train_2nd_data["N"]
        else:
            raise ValueError("which_regressor needs to be {X_hidden, M, N, avMN}")

        if train_2nd_data.X_obs:
            X2 = np.concatenate([X2, train_2nd_data.X_obs], axis=-1)
        if train_2nd_data.covariate is not None:
            X2 = np.concatenate([X2, train_2nd_data.covariate], axis=-1)
        Z2 = train_2nd_data.Z
        Y2 = train_2nd_data.Y
        m = X2.shape[0]

        logger.info("start stage1")

        sigmaX = compute_median_sq_dist(X1)
        sigmaZ = compute_median_sq_dist(Z1)
        KX1X1 = compute_gaussian_gram(X1, X1, sigmaX)
        KZ1Z1 = compute_gaussian_gram(Z1, Z1, sigmaZ)
        KZ1Z2 = compute_gaussian_gram(Z1, Z2, sigmaZ)
        KX1X2 = compute_gaussian_gram(X1, X2, sigmaX)

        if isinstance(self.lambd_inits, list):
            self.lambd_inits = np.exp(
                np.linspace(self.lambd_inits[0], self.lambd_inits[1], 50)
            )
            gamma, self.lambd_final = self.stage1_tuning(KX1X1, KX1X2, KZ1Z1, KZ1Z2)
        else:
            gamma = np.linalg.solve(KZ1Z1 + n * self.lambd_inits * np.eye(n), KZ1Z2)
            self.lambd_final = self.lambd_inits
        W = KX1X1.dot(gamma)

        logger.info("end stage1")
        logger.info("start stage2")

        if isinstance(self.xi_inits, list):
            self.xi_inits = 10 ** np.linspace(self.xi_inits[0], self.xi_inits[1], 50)
            self.alpha_final, self.xi_final = self.stage2_tuning(W, KX1X1, Y1, Y2)
        else:
            self.xi_final = self.xi_inits
            self.alpha_final = np.linalg.solve(
                W.dot(W.T) + m * self.xi_inits * KX1X1, W.dot(Y2)
            )

        logger.info("end stage2")

        mdl = KernelIVModel(
            X_train=X1,
            Z_train=Z1,
            alpha=self.alpha_final,
            z_brac=KZ1Z1 + n * self.lambd_final * np.eye(n),
            sigma_x=sigmaX,
            sigma_z=sigmaZ,
        )

        test_input = test_data.X_all
        if test_data.covariate is not None:
            test_input = np.concatenate([test_input, test_data.covariate], axis=-1)
        Y_struct = test_data.Y_struct

        mse, preds_x = mdl.evaluate(test_data=test_data)

        return mse, test_input, preds_x, Y_struct

    def stage1_tuning(self, KX1X1, KX1X2, KZ1Z1, KZ1Z2):
        N = KX1X1.shape[0]
        gamma_list = [
            np.linalg.solve(KZ1Z1 + N * lambd * np.eye(N), KZ1Z2)
            for lambd in self.lambd_inits
        ]
        score = [
            np.trace(gamma.T.dot(KX1X1.dot(gamma)) - 2 * KX1X2.T.dot(gamma))
            for gamma in gamma_list
        ]
        lambd_final = self.lambd_inits[np.argmin(score)]
        return gamma_list[np.argmin(score)], lambd_final

    def stage2_tuning(self, W, KX1X1, Y1, Y2):
        N = KX1X1.shape[0]
        M = W.shape[1]
        b = W.dot(Y2)
        A = W.dot(W.T)
        if self.data_config["data_name"] == "dahl_lochner":
            alpha_list = [
                np.linalg.solve(A + M * xi * KX1X1 + np.eye(N) * 1e-9, b)
                for xi in self.xi_inits
            ]
        else:
            alpha_list = [
                np.linalg.solve(A + M * xi * KX1X1, b) for xi in self.xi_inits
            ]
        score = [np.linalg.norm(Y1 - KX1X1.dot(alpha)) for alpha in alpha_list]
        xi = self.xi_inits[np.argmin(score)]
        return alpha_list[np.argmin(score)], xi
