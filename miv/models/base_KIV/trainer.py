from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import logging
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split


from miv.data import generate_train_data, generate_test_data, generate_z_test_data
from miv.data.data_class import TrainDataSet, TrainDataSetTorch
from miv.models.base_KIV.model import KernelIVModel
from miv.util import dotdict, make_dotdict

logger = logging.getLogger()


def get_median(X) -> float:
    dist_mat = cdist(X, X, "sqeuclidean")
    res: float = np.median(dist_mat)
    return res


class BaseKIVTrainer:

    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any],
                 gpu_flg: bool = False, dump_folder: Optional[Path] = None):
        self.data_config = data_configs
        self.train_params = make_dotdict(train_params)

        self.lambd = train_params["lambda"]
        self.xi = train_params["xi"]
        self.split_ratio = train_params["split_ratio"]

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

    def _train(self, which_regressor: str, rand_seed: int = 42, verbose: int = 0):
        """

        Parameters
        ----------
        which_regressor: str
            selects the variable to use as treatment, {X_hidden, M, N, avMN}
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
        # z_test_data = generate_z_test_data(**self.data_config)
        train_1st_data, train_2nd_data = self.split_train_data(train_data)

        # get stage1 data
        X1 = None
        if which_regressor in {'X_hidden', 'M', 'N'}:
            X1 = train_1st_data[which_regressor]
        elif which_regressor == 'avMN':
            X1 = 0.5 * train_1st_data['M'] + 0.5 * train_1st_data['N']
        else:
            raise ValueError('which_regressor needs to be {X_hidden, M, N, avMN}')

        if train_data.X_obs:
            X1 = np.concatenate([X1, train_1st_data.X_obs], axis=-1)
        if train_1st_data.covariate is not None:
            X1 = np.concatenate([X1, train_1st_data.covariate], axis=-1)
        Z1 = train_1st_data.Z
        Y1 = train_1st_data.Y
        n = X1.shape[0]


        # get stage2 data
        X2 = None
        if which_regressor in {'X_hidden', 'M', 'N'}:
            X2 = train_2nd_data[which_regressor]
        elif which_regressor == 'avMN':
            X2 = 0.5 * train_2nd_data['M'] + 0.5 * train_2nd_data['N']
        else:
            raise ValueError('which_regressor needs to be {X_hidden, M, N, avMN}')

        if train_2nd_data.X_obs:
            X2 = np.concatenate([X2, train_2nd_data.X_obs], axis=-1)
        if train_2nd_data.covariate is not None:
            X2 = np.concatenate([X2, train_2nd_data.covariate], axis=-1)
        Z2 = train_2nd_data.Z
        Y2 = train_2nd_data.Y
        m = X2.shape[0]

        if verbose > 0:
            logger.info("start stage1")

        sigmaX = get_median(X1)
        sigmaZ = get_median(Z1)
        KX1X1 = KernelIVModel.cal_gauss(X1, X1, sigmaX)
        KZ1Z1 = KernelIVModel.cal_gauss(Z1, Z1, sigmaZ)
        KZ1Z2 = KernelIVModel.cal_gauss(Z1, Z2, sigmaZ)
        KX1X2 = KernelIVModel.cal_gauss(X1, X2, sigmaX)

        if isinstance(self.lambd, list):
            self.lambd = np.exp(np.linspace(self.lambd[0], self.lambd[1], 50))
            gamma = self.stage1_tuning(KX1X1, KX1X2, KZ1Z1, KZ1Z2)
        else:
            gamma = np.linalg.solve(KZ1Z1 + n * self.lambd * np.eye(n), KZ1Z2)
        W = KX1X1.dot(gamma)
        if verbose > 0:
            logger.info("end stage1")
            logger.info("start stage2")

        if isinstance(self.xi, list):
            self.xi = 10 ** np.linspace(self.xi[0], self.xi[1], 50)
            alpha = self.stage2_tuning(W, KX1X1, Y1, Y2)
        else:
            alpha = np.linalg.solve(W.dot(W.T) + m * self.xi * KX1X1, W.dot(Y2))

        if verbose > 0:
            logger.info("end stage2")

        mdl = KernelIVModel(X_train=X1, Z_train=Z1,
                            alpha=alpha, z_brac=KZ1Z1 + n * self.lambd * np.eye(n),
                            sigma_x=sigmaX, sigma_z=sigmaZ)

        # mdl = MerrorKIVModel(fitted_X=fitted_X, alpha=alpha, sigma=sigmaX)
        test_input = test_data.X_all
        if test_data.covariate is not None:
            test_input = np.concatenate([test_input, test_data.covariate], axis=-1)
        Y_struct = test_data.Y_struct

        mse, preds_x = mdl.evaluate(test_data=test_data)
        # mse_z, preds_z = mdl.evaluate_z(z_test_data=z_test_data)
        mse_z, preds_z = None, None

        # mses = dotdict({'x': mse, 'z': mse_z})
        # test_inputs = dotdict({'x': test_input, 'z': z_test_data.Z})
        # preds = dotdict({'x': preds_x, 'z': preds_z})
        # labels = dotdict({'x': Y_struct, 'z': z_test_data.Y})

        mses = {'x': mse, 'z': mse_z}
        # test_inputs = {'x': test_input, 'z': z_test_data.Z}
        test_inputs = {'x': test_input, 'z': None}
        preds = {'x': preds_x, 'z': preds_z}
        # labels = {'x': Y_struct, 'z': z_test_data.Y}
        labels = {'x': Y_struct, 'z': None}
        # breakpoint()

        return mses, test_inputs, preds, labels

    def stage1_tuning(self, KX1X1, KX1X2, KZ1Z1, KZ1Z2):
        N = KX1X1.shape[0]
        gamma_list = [np.linalg.solve(KZ1Z1 + N * lambd * np.eye(N), KZ1Z2) for lambd in self.lambd]
        score = [np.trace(gamma.T.dot(KX1X1.dot(gamma)) - 2 * KX1X2.T.dot(gamma)) for gamma in gamma_list]
        self.lambd = self.lambd[np.argmin(score)]
        return gamma_list[np.argmin(score)]

    def stage2_tuning(self, W, KX1X1, Y1, Y2):
        N = KX1X1.shape[0]
        M = W.shape[1]
        b = W.dot(Y2)
        A = W.dot(W.T)
        if self.data_config['data_name'] == 'dahl_lochner':
            alpha_list = [np.linalg.solve(A + M * xi * KX1X1 + np.eye(N) * 1e-9, b) for xi in self.xi]
        else:
            alpha_list = [np.linalg.solve(A + M * xi * KX1X1, b) for xi in self.xi]
        score = [np.linalg.norm(Y1 - KX1X1.dot(alpha)) for alpha in alpha_list]
        self.xi = self.xi[np.argmin(score)]
        return alpha_list[np.argmin(score)]
