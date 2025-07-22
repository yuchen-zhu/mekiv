import numpy as np

from miv.data.data_class import TestDataSet, ZTestDataSet
from miv.utils import compute_gaussian_gram


class KernelIVModel:

    def __init__(
        self,
        X_train: np.ndarray,
        Z_train: np.ndarray,
        alpha: np.ndarray,
        z_brac: np.ndarray,
        sigma_x: float,
        sigma_z: float,
    ):
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

    def predict(self, treatment: np.ndarray, covariate: np.ndarray):
        X = np.array(treatment, copy=True)
        if covariate is not None:
            X = np.concatenate([X, covariate], axis=1)
        Kx = compute_gaussian_gram(X, self.X_train, self.sigma_x)
        return np.dot(Kx, self.alpha)

    def evaluate(self, test_data: TestDataSet):
        pred = self.predict(test_data.X_all, test_data.covariate)
        return np.mean((test_data.Y_struct - pred) ** 2), pred

    def predict_z(self, instrument: np.ndarray):
        Z = np.array(instrument, copy=True)
        KX1X1 = compute_gaussian_gram(self.X_train, self.X_train, self.sigma_x)
        KZ1z = compute_gaussian_gram(self.Z_train, Z, self.sigma_z)
        gamma_z = np.linalg.solve(self.z_brac, KZ1z)
        W_z = KX1X1.dot(gamma_z)
        pred_z = self.alpha.T.dot(W_z)
        return pred_z

    def evaluate_z(self, z_test_data: ZTestDataSet):
        pred_z = self.predict_z(z_test_data.Z)
        return np.mean((z_test_data.Y - pred_z) ** 2), pred_z
