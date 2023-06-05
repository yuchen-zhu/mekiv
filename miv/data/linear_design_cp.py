import numpy as np
from scipy import stats

from ..data.data_class import TrainDataSet, TestDataSet, ZTestDataSet
from ..data.merror_funcs import get_merror_func
from numpy.random import default_rng


def f(x: np.ndarray) -> np.ndarray:
    return 4 * x - 2


def generate_test_linear_cp_design() -> TestDataSet:
    """
    Returns
    -------
    test_data : TestDataSet
        Uniformly sampled from x.
    """
    x = np.linspace(0, 1, 100)
    y = f(x)

    test_data = TestDataSet(X_all=x[:, np.newaxis],
                            Y_struct=y[:, np.newaxis],
                            covariate=None)
    return test_data


def generate_train_linear_cp_design(data_size: int,
                                  merror_func_str: str,
                                  m_scale: float,
                                  n_scale: float,
                                  bias: float,
                                  rand_seed: int = 42) -> TrainDataSet:
    """

    Parameters
    ----------
    data_size : int
        size of data
    merror_func_str: str
        parameter for choosing a measurement error mechanism
    m_scale: float
        chooses the error spread in M
    n_scale: float
        chooses the error spread in N
    bias: float
        chooses the bias level in N
    rand_seed : int
        random seed


    Returns
    -------
    train_data : TrainDataSet
    """
    merror_func = get_merror_func(merror_func_str)
    rng = default_rng(seed=rand_seed)
    mu = np.zeros((3,))
    sigma = np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])
    # sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # sigma = np.array([[1, 0.5, 0],
    #                   [0.5, 1, 0],
    #                     [0, 0, 1]])
    utw = rng.multivariate_normal(mu, sigma, size=data_size)
    u = utw[:, 0:1]
    z = stats.norm.cdf(utw[:, 2])[:, np.newaxis]
    # x = stats.norm.cdf(utw[:, 1] + utw[:, 2] / np.sqrt(2))[:, np.newaxis]
    x = z + rng.normal(0, 0.1, data_size)[:, np.newaxis]
    structural = f(x)
    outcome = f(x) + u
    # outcome = f(x) + 0.1*u
    M, N = merror_func(X_hidden=x, scale_m=m_scale, scale_n=n_scale, bias=bias)

    train_data = TrainDataSet(X_hidden=x,
                              X_obs=None,
                              covariate=None,
                              M=M,
                              N=N,
                              Z=z,
                              Y_struct=structural,
                              Y=outcome)
    return train_data


def generate_z_test_linear_cp_design() -> ZTestDataSet:
    """
    Returns
    -------
    z_test_data : ZTestDataSet
    """
    ivs = np.linspace(0, 1, 100)

    rng = default_rng(seed=42)
    mu = np.zeros((2,))
    sigma = np.array([[1, 0.5], [0.5, 1]])
    outcome = []
    for iv in ivs:
        ut = rng.multivariate_normal(mu, sigma)
        u = ut[0]
        t = ut[1]
        x = stats.norm.cdf(t + iv / np.sqrt(2))
        struct = f(x)
        out = struct + u
        outcome.append(out)
    Z = ivs[:, np.newaxis]
    outcomes = np.array(outcome)[:, np.newaxis]
    z_test_data = ZTestDataSet(Z=Z,
                               Y=outcomes)
    return z_test_data
