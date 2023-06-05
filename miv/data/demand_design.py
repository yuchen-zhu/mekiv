import numpy as np
from ..data.data_class import TrainDataSet, TestDataSet, ZTestDataSet
from ..data.merror_funcs import get_merror_func
from itertools import product
from numpy.random import default_rng


def psi(t: np.ndarray) -> np.ndarray:
    out = 2 * ((t - 5) ** 4 / 600 + np.exp(-4 * (t - 5) ** 2) + t / 10 - 2)
    return out


def f(p: np.ndarray, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    return 100 + (10 + p) * s * psi(t) - 2 * p


def generate_test_demand_design() -> TestDataSet:
    """
    Returns
    -------
    test_data : TestDataSet
        Uniformly sampled from (p,t,s).
    """
    price = np.linspace(10, 25, 20)
    time = np.linspace(0.0, 10, 20)
    emotion = np.array([1, 2, 3, 4, 5, 6, 7])
    data = []
    target = []
    for p, t, s in product(price, time, emotion):
        data.append([p, t, s])
        target.append(f(p, t, s))
    features = np.array(data)
    targets: np.ndarray = np.array(target)[:, np.newaxis]
    test_data = TestDataSet(X_all=features[:, 0:1],
                            Y_struct=targets,
                            covariate=features[:, 1:])
    # breakpoint()
    return test_data


def generate_train_demand_design(data_size: int,
                                 rho: float,
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
    rho : float
        parameter for confounding
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
    emotion = rng.choice(list(range(1, 8)), data_size)
    time = rng.uniform(0, 10, data_size)
    cost = rng.normal(0, 1.0, data_size)

    noise_price = rng.normal(0, 1.0, data_size)
    Z = np.c_[cost, time, emotion]


    noise_demand = rho * noise_price + rng.normal(0, np.sqrt(1 - rho ** 2), data_size)
    price = 25 + (cost + 3) * psi(time) + noise_price

    X_hidden = price[:, np.newaxis]
    X_obs = None
    covariate = np.c_[time, emotion]
    M, N = merror_func(X_hidden=X_hidden, scale_m=m_scale, scale_n=n_scale, bias=bias)


    structure: np.ndarray = f(price, time, emotion).astype(float)
    outcome: np.ndarray = (structure + noise_demand).astype(float)


    train_data = TrainDataSet(X_hidden=X_hidden,
                              X_obs=X_obs,
                              covariate=covariate,
                              M=M,
                              N=N,
                              Z=Z,
                              Y_struct=structure[:, np.newaxis],
                              Y=outcome[:, np.newaxis])

    return train_data


def generate_z_test_demand_design(rho) -> ZTestDataSet:
    """
    Returns
    -------
    test_data : TestDataSet
        Uniformly sampled from (p,t,s).
    """
    rng = default_rng(seed=42)
    cost = np.linspace(-2.0, 2.0, 20)
    emotion = np.array([1,2,3,4,5,6,7])
    time = np.linspace(0, 10, 20)

    iv = []
    outcome = []
    for c, t, s in product(cost, time, emotion):
        iv.append([c, t, s])
        noise_p = rng.normal(0, 1.0)
        p = 25 + (c + 3) * psi(t) + noise_p
        noise_d = rho * noise_p + rng.normal(0, np.sqrt(1-rho ** 2))
        outcome.append(f(p, t, s) + noise_d)
    Z = np.array(iv)
    outcomes: np.ndarray = np.array(outcome)[:, np.newaxis]
    z_test_data = ZTestDataSet(Z=Z, Y=outcomes)
    return z_test_data
