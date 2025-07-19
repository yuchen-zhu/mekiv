import numpy as np
from scipy.spatial.distance import cdist
import torch

def compute_gaussian_gram(XA, XB, sigma: float = 1):
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