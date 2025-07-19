from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import logging
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split


from miv.data import load_data, split_train_data
from miv.data.data_class import TrainDataSet, TrainDataSetTorch
from miv.models.base_KIV.model import KernelIVModel
from miv.models.base_KIV.gaussian_gram import compute_gaussian_gram
from miv.utils.util import dotdict, make_dotdict

logger = logging.getLogger()


def get_median(X) -> float:
    dist_mat = cdist(X, X, "sqeuclidean")
    res: float = np.median(dist_mat)
    return res


class BaseKIVTrainer:

    def __init__(
        self,
        data_configs: Dict[str, Any],
        train_params: Dict[str, Any]
    ):
        self.data_config = data_configs
        self.train_params = make_dotdict(train_params)

        self.lambd_inits = train_params["lambda_inits"]
        self.xi = train_params["xi"]
        self.split_ratio = train_params["split_ratio"]

    def _train(self, which_regressor: str, rand_seed: int = 42):
        """

        Parameters
        ----------
        which_regressor: str
            selects the variable to use as treatment, {X_hidden, M, N, avMN}
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
        X1 = None
        if which_regressor in {"X_hidden", "M", "N"}:
            X1 = train_1st_data[which_regressor]
        elif which_regressor == "avMN":
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
        X2 = None
        if which_regressor in {"X_hidden", "M", "N"}:
            X2 = train_2nd_data[which_regressor]
        elif which_regressor == "avMN":
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

        sigmaX = get_median(X1)
        sigmaZ = get_median(Z1)
        KX1X1 = compute_gaussian_gram(X1, X1, sigmaX)
        KZ1Z1 = compute_gaussian_gram(Z1, Z1, sigmaZ)
        KZ1Z2 = compute_gaussian_gram(Z1, Z2, sigmaZ)
        KX1X2 = compute_gaussian_gram(X1, X2, sigmaX)

        if isinstance(self.lambd_inits, list):
            self.lambd_inits = np.exp(np.linspace(self.lambd_inits[0], self.lambd_inits[1], 50))
            gamma = self.stage1_tuning(KX1X1, KX1X2, KZ1Z1, KZ1Z2)
        else:
            gamma = np.linalg.solve(KZ1Z1 + n * self.lambd_inits * np.eye(n), KZ1Z2)
        W = KX1X1.dot(gamma)

        logger.info("end stage1")
        logger.info("start stage2")

        if isinstance(self.xi, list):
            self.xi = 10 ** np.linspace(self.xi[0], self.xi[1], 50)
            alpha = self.stage2_tuning(W, KX1X1, Y1, Y2)
        else:
            alpha = np.linalg.solve(W.dot(W.T) + m * self.xi * KX1X1, W.dot(Y2))

        logger.info("end stage2")

        mdl = KernelIVModel(
            X_train=X1,
            Z_train=Z1,
            alpha=alpha,
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
        self.lambd_final = self.lambd_inits[np.argmin(score)]
        return gamma_list[np.argmin(score)]

    def stage2_tuning(self, W, KX1X1, Y1, Y2):
        N = KX1X1.shape[0]
        M = W.shape[1]
        b = W.dot(Y2)
        A = W.dot(W.T)
        if self.data_config["data_name"] == "dahl_lochner":
            alpha_list = [
                np.linalg.solve(A + M * xi * KX1X1 + np.eye(N) * 1e-9, b)
                for xi in self.xi
            ]
        else:
            alpha_list = [np.linalg.solve(A + M * xi * KX1X1, b) for xi in self.xi]
        score = [np.linalg.norm(Y1 - KX1X1.dot(alpha)) for alpha in alpha_list]
        self.xi = self.xi[np.argmin(score)]
        return alpha_list[np.argmin(score)]
