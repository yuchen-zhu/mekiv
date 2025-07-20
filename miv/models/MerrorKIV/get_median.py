import numpy as np
from scipy.spatial.distance import cdist

def get_median(X) -> float:
    dist_mat = cdist(X, X, "sqeuclidean")
    res: float = np.median(a=dist_mat)
    return res