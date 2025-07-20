import numpy as np
import torch
from scipy.spatial.distance import cdist

__all__ = ["compute_median_sq_dist", "compute_gaussian_gram"]


def compute_median_sq_dist(X) -> float:
    dist_mat = cdist(X, X, "sqeuclidean")
    return np.median(dist_mat).item()


def compute_gaussian_gram(XA, XB, sigma2: float = 1):
    """
    Returns Gaussian kernel matrix
    Parameters
    ----------
    XA : np.ndarray[n_data1, n_dim] OR torch.Tensor[n_data1, n_dim]
    XB : np.ndarray[n_data2, n_dim] OR torch.Tensor[n_data2, n_dim]
    sigma2 : float

    Returns
    -------
    mat: np.ndarray[n_data1, n_data2] OR torch.Tensor[n_data1, n_data2]
    """
    if isinstance(XA, np.ndarray):
        dist_mat = cdist(XA, XB, "sqeuclidean")
        return np.exp(-dist_mat / 2 / sigma2)
    elif isinstance(XA, torch.Tensor):
        dist_mat = torch.cdist(XA, XB) ** 2
        return torch.exp(-dist_mat / 2 / sigma2)
