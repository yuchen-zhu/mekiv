from typing import Optional
import numpy as np
from scipy.spatial.distance import cdist

from miv.data.data_class import TrainDataSet, TestDataSet, ZTestDataSet


class KernelIVModel:

    def __init__(self, X_train: np.ndarray, Z_train: np.ndarray,
                 alpha: np.ndarray, z_brac: np.ndarray,
                 sigma_x: float, sigma_z: float):
        """

        Parameters
        ----------
        X_train: np.ndarray[n_stage1, dim_treatment]
            data for treatment
        alpha:  np.ndarray[n_stage1*n_stage2 ,dim_outcome]
            final weight for prediction
        sigma_x: gauss parameter
        """
        self.X_train = X_train
        self.alpha = alpha
        self.sigma_x = sigma_x
        self.sigma_z = sigma_z
        self.Z_train = Z_train
        self.z_brac = z_brac

    @staticmethod
    def cal_gauss(XA, XB, sigma: float = 1):
        """
        Returns gaussian kernel matrix
        Parameters
        ----------
        XA : np.ndarray[n_data1, n_dim]
        XB : np.ndarray[n_data2, n_dim]
        sigma : float

        Returns
        -------
        mat: np.ndarray[n_data1, n_data2]
        """
        dist_mat = cdist(XA, XB, "sqeuclidean")
        return np.exp(-dist_mat / 1 / sigma)

    def predict(self, treatment: np.ndarray, covariate: np.ndarray):
        X = np.array(treatment, copy=True)
        if covariate is not None:
            X = np.concatenate([X, covariate], axis=1)
        Kx = self.cal_gauss(X, self.X_train, self.sigma_x)
        return np.dot(Kx, self.alpha)

    def evaluate(self, test_data: TestDataSet):
        pred = self.predict(test_data.X_all, test_data.covariate)
        return np.mean((test_data.Y_struct - pred)**2), pred

    def predict_z(self, instrument: np.ndarray):
        Z = np.array(instrument, copy=True)
        KX1X1 = self.cal_gauss(self.X_train, self.X_train, self.sigma_x)
        KZ1z = self.cal_gauss(self.Z_train, Z, self.sigma_z)
        gamma_z = np.linalg.solve(self.z_brac, KZ1z)
        W_z = KX1X1.dot(gamma_z)
        pred_z = self.alpha.T.dot(W_z)
        return pred_z

    def evaluate_z(self, z_test_data: ZTestDataSet):
        pred_z = self.predict_z(z_test_data.Z)
        return np.mean((z_test_data.Y - pred_z)**2), pred_z

