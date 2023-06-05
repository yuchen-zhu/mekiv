from typing import Dict, Any, Optional
from pathlib import Path
import logging
from miv.util import dotdict, make_dotdict
from miv.models.base_KIV.trainer import BaseKIVTrainer

logger = logging.getLogger()



class KIV_MNTrainer(BaseKIVTrainer):

    def __init__(self, data_configs: dotdict, train_params: dotdict,
                 gpu_flg: bool = False, dump_folder: Optional[Path] = None):
        super(KIV_MNTrainer, self).__init__(data_configs, train_params)

        self.which_regressor = 'avMN'

    def train(self, rand_seed: int = 42, verbose: int = 0) -> float:
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

        # train_data = generate_train_data(rand_seed=rand_seed, **self.data_config)
        # test_data = generate_test_data(**self.data_config)
        # train_1st_data, train_2nd_data = self.split_train_data(train_data)
        #
        # # get stage1 data
        # X1 = train_1st_data.X_hidden
        # if train_data.X_obs:
        #     X1 = np.concatenate([train_1st_data.X1, train_1st_data.X_obs], axis=-1)
        # if train_1st_data.covariate is not None:
        #     X1 = np.concatenate([X1, train_1st_data.covariate], axis=-1)
        # Z1 = train_1st_data.Z
        # Y1 = train_1st_data.Y
        # N = X1.shape[0]
        #
        # # get stage2 data
        # X2 = train_2nd_data.X_hidden
        # if train_2nd_data.X_obs:
        #     X2 = np.concatenate([train_2nd_data.X2, train_1st_data.X_obs], axis=-1)
        # if train_2nd_data.covariate is not None:
        #     X2 = np.concatenate([X2, train_2nd_data.covariate], axis=-1)
        # Z2 = train_2nd_data.Z
        # Y2 = train_2nd_data.Y
        # M = X2.shape[0]
        #
        # if verbose > 0:
        #     logger.info("start stage1")
        #
        # sigmaN = get_median(N1)
        # sigmaMN = get_median(MN1)
        # sigmaZ = get_median(Z1)
        # KN1N1 = MerrorKIVModel.cal_gauss(N1, N1, sigmaN)
        # KN1N2 = MerrorKIVModel.cal_gauss(N1, N2, sigmaN)
        # KMN1MN1 = MerrorKIVModel.cal_gauss(MN1, MN1, sigmaMN)
        # KMN1MN2 = MerrorKIVModel.cal_gauss(MN1, MN2, sigmaMN)
        # KZ1Z1 = MerrorKIVModel.cal_gauss(Z1, Z1, sigmaZ)
        # KZ1Z2 = MerrorKIVModel.cal_gauss(Z1, Z2, sigmaZ)
        # # KX1X2 = MerrorKIVModel.cal_gauss(X1, X2, sigmaX)
        #
        # if isinstance(self.lambda_mn, list):
        #     lambda_mn = np.exp(np.linspace(self.lambda_mn[0], self.lambda_mn[1], 50))
        #     gamma_mn, lambda_mn = self.stage1_tuning(KMN1MN1, KMN1MN2, KZ1Z1, KZ1Z2, lambda_mn)
        #     self.lambda_mn = lambda_mn
        # else:
        #     gamma_mn = np.linalg.solve(KZ1Z1 + n * self.lambda_mn * np.eye(n), KZ1Z2)
        #
        # if isinstance(self.lambda_n, list):
        #     lambda_n = np.exp(np.linspace(self.lambda_n[0], self.lambda_n[1], 50))
        #     gamma_n, lambda_n = self.stage1_tuning(KN1N1, KN1N2, KZ1Z1, KZ1Z2, lambda_n)
        #     self.lambda_n = lambda_n
        # else:
        #     gamma_n = np.linalg.solve(KZ1Z1 + n * self.lambda_n * np.eye(n), KZ1Z2)
        #
        # if verbose > 0:
        #     logger.info("end stage 1")
        #     logger.info("start stage merror")
        #
        # # get stageM data
        # M1 = train_1st_data.M
        # stageM_data = create_stage_M_raw_data(self.n_chi, N1, M1, Z2, gamma_n, gamma_mn, sigmaN, KZ1Z2)
        # stageM_data = prepare_stage_M_data(raw_data2=stageM_data, rand_seed=rand_seed)
        # stage1_MNZ = dotdict({'M': M1, 'N': N1, 'Z': Z1, 'sigmaZ': sigmaZ})
        #
        # stage_m_out = self.stage_M_main(stageM_data=stageM_data, stage1_MNZ=stage1_MNZ, train_params=self.train_params)
        # lambda_x, fitted_X = stage_m_out.lambda_x, stage_m_out.fitted_x
        #
        #
        # print('------ fitted X / N / M / (M+N)/2  compared with ground truth X -------')
        # print((np.sum((fitted_X - train_1st_data.X_hidden) ** 2) / fitted_X.shape[0]) ** 0.5 / np.std(
        #     train_1st_data.X_hidden))
        # print((np.sum((train_1st_data.N - train_1st_data.X_hidden) ** 2) / fitted_X.shape[0]) ** 0.5 / np.std(
        #     train_1st_data.X_hidden))
        # print((np.sum((train_1st_data.M - train_1st_data.X_hidden) ** 2) / fitted_X.shape[0]) ** 0.5 / np.std(
        #     train_1st_data.X_hidden))
        # print((np.sum((1 / 2 * train_1st_data.M + 1 / 2 * train_1st_data.N - train_1st_data.X_hidden) ** 2) /
        #        fitted_X.shape[0]) ** 0.5 / np.std(train_1st_data.X_hidden))
        #
        #
        # if train_1st_data.X_obs is not None:
        #     fitted_X = np.concatenate([fitted_X, train_1st_data.X_obs], axis=-1)
        # if train_1st_data.covariate is not None:
        #     fitted_X = np.concatenate([fitted_X, train_1st_data.covariate], axis=-1)
        #
        # gamma_x = np.linalg.solve(KZ1Z1 + n * lambda_x * np.eye(n), KZ1Z2)
        # sigmaX = get_median(fitted_X)
        # KfittedX = MerrorKIVModel.cal_gauss(fitted_X, fitted_X, sigmaX)
        # W = KfittedX.dot(gamma_x)
        # if verbose > 0:
        #     logger.info("end stageM")
        #     logger.info("start stage2")
        #
        # if isinstance(self.xi, list):
        #     # breakpoint()
        #     self.xi = np.exp(np.linspace(self.xi[0], self.xi[1], 50))
        #     alpha, xi = self.stage2_tuning(W, KfittedX, Y1, Y2)
        #     self.xi = xi
        # else:
        #     alpha = np.linalg.solve(W.dot(W.T) + m * self.xi * KfittedX, W.dot(Y2))
        #
        # if verbose > 0:
        #     logger.info("end stage2")
        #
        # mdl = MerrorKIVModel(fitted_X=fitted_X, alpha=alpha, sigma=sigmaX)
        # # breakpoint()
        # return mdl.evaluate(test_data=test_data)
        return self._train(which_regressor=self.which_regressor, rand_seed=rand_seed, verbose=verbose)

