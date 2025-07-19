import numpy as np
from scipy.spatial.distance import cdist


def compute_gaussian_gram(XA, XB, sigma: float = 1):
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