import numpy as np

from miv.data.data_class import TestDataSet
from miv.utils import compute_gaussian_gram


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

    def predict(self, treatment: np.ndarray, covariate: np.ndarray):
        X = np.array(treatment, copy=True)
        if covariate is not None:
            X = np.concatenate([X, covariate], axis=1)
        Kx = compute_gaussian_gram(X, self.fitted_X, self.sigma)
        return np.dot(Kx, self.alpha)

    def evaluate(self, test_data: TestDataSet):
        pred = self.predict(test_data.X_all, test_data.covariate)
        return np.mean((test_data.Y_struct - pred) ** 2), pred
