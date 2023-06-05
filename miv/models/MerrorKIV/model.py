import numpy as np
from scipy.spatial.distance import cdist
import torch

from miv.data.data_class import TrainDataSet, TestDataSet


class MerrorKIVModel:

    def __init__(self, fitted_X: np.ndarray, alpha: np.ndarray, sigma: float):
        """

        Parameters
        ----------
        fitted_X: np.ndarray[n_stage1, dim_treatment]
            data for treatment
        alpha:  np.ndarray[n_stage1*n_stage2 ,dim_outcome]
            final weight for prediction
        sigma: gauss parameter
        """
        self.fitted_X = fitted_X
        self.alpha = alpha
        self.sigma = sigma

    @staticmethod
    def cal_gauss(XA, XB, sigma: float = 1):
        """
        Returns gaussian kernel matrix
        Parameters
        ----------
        XA : np.ndarray[n_data1, n_dim] OR torch.Tensor[n_data1, n_dim]
        XB : np.ndarray[n_data2, n_dim] OR torch.Tensor[n_data2, n_dim]
        sigma : float

        Returns
        -------
        mat: np.ndarray[n_data1, n_data2] OR torch.Tensor[n_data1, n_dim]
        """
        if isinstance(XA, np.ndarray):
            dist_mat = cdist(XA, XB, "sqeuclidean")
            return np.exp(-dist_mat / 2 / float(sigma))
        elif isinstance(XA, torch.Tensor):
            dist_mat = torch.cdist(XA, XB) ** 2
            return torch.exp(-dist_mat / 2 / float(sigma))

    def predict(self, treatment: np.ndarray, covariate: np.ndarray):
        X = np.array(treatment, copy=True)
        if covariate is not None:
            # breakpoint()
            X = np.concatenate([X, covariate], axis=1)
        Kx = self.cal_gauss(X, self.fitted_X, self.sigma)
        return np.dot(Kx, self.alpha)

    def evaluate(self, test_data: TestDataSet):
        pred = self.predict(test_data.X_all, test_data.covariate)
        return np.mean((test_data.Y_struct - pred)**2), pred