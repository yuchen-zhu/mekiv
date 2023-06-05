import numpy as np


def get_merror_func(merror_name: str):
    if merror_name == 'gaussian':
        return gaussian_merror
    elif merror_name == 'multi_gaussian':
        return multi_gaussian_merror
    elif merror_name == 'multi_gaussian_old':
        return multi_gaussian_merror_old
    elif merror_name == "uniform":
        return unif_merror
    else:
        raise ValueError(f"merror name {merror_name} is not implemented.")


def gaussian_merror(X_hidden: np.ndarray, scale_m: float, scale_n: float, bias: float = 0.0):
    data_size = X_hidden.shape[0]
    std_X = np.std(X_hidden)
    std_M, std_N = std_X * scale_m, std_X * scale_n
    M = X_hidden + std_M * np.random.normal(0, 1, data_size)[:, np.newaxis]
    N = X_hidden + std_N * np.random.normal(0, 1, data_size)[:, np.newaxis]
    return M, N


def unif_merror(X_hidden: np.ndarray, scale_m: float, scale_n: float, bias: float = 0.0):
    data_size = X_hidden.shape[0]
    std_X = np.std(X_hidden)
    std_M, std_N = std_X * scale_m, std_X * scale_n
    M = X_hidden + std_M * np.random.uniform(-1, 1, data_size)[:, np.newaxis]
    N = X_hidden + std_N * np.random.uniform(-1, 1, data_size)[:, np.newaxis]
    return M, N



def multi_gaussian_merror_old(X_hidden: np.ndarray, scale_m: float, scale_n: float, bias: float = 0.0):
    data_size = X_hidden.shape[0]
    std_X = np.std(X_hidden)
    gauss1_mean, gauss1_std_sc, w1 = -10, std_X, 0.45
    gauss2_mean, gauss2_std_sc, w2 = 0, std_X, 0.1
    gauss3_mean, gauss3_std_sc, w3 = 10, std_X, 0.45
    # gauss1_mean, gauss1_std_sc, w1 = -10, std_X, 0.25
    # gauss2_mean, gauss2_std_sc, w2 = 2, std_X, 0.5
    # gauss3_mean, gauss3_std_sc, w3 = 6, std_X, 0.25
    # gauss1_mean, gauss1_std_sc, w1 = -10, std_X, 0.5
    # gauss2_mean, gauss2_std_sc, w2 = 2, std_X, 0.
    # gauss3_mean, gauss3_std_sc, w3 = 10, std_X, 0.5
    # breakpoint()
    std_M, std_N = std_X * scale_m, std_X * scale_n
    # std_M, std_N = 0, 0
    # breakpoint()

    M_noises = np.c_[gauss1_mean + std_M * np.random.normal(0, 1, data_size),
                     gauss2_mean + std_M * np.random.normal(0, 1, data_size),
                     gauss3_mean + std_M * np.random.normal(0, 1, data_size)]

    N_noises = np.c_[gauss1_mean + std_N * np.random.normal(0, 1, data_size),
                     gauss2_mean + std_N * np.random.normal(0, 1, data_size),
                     gauss3_mean + std_N * np.random.normal(0, 1, data_size)]

    M = X_hidden + M_noises[np.arange(data_size), np.random.choice([0, 1, 2], data_size, p=[w1, w2, w3])][:, np.newaxis]
    N = X_hidden + N_noises[np.arange(data_size), np.random.choice([0, 1, 2], data_size, p=[w1, w2, w3])][:, np.newaxis]

    # M = M_noises[np.arange(data_size), np.random.choice([0, 1, 2], data_size, p=[w1, w2, w3])][:, np.newaxis]
    # N = N_noises[np.arange(data_size), np.random.choice([0, 1, 2], data_size, p=[w1, w2, w3])][:, np.newaxis]


    return M, N


def multi_gaussian_merror(X_hidden: np.ndarray, scale_m: float, scale_n: float, bias: float = 0.0):
    data_size = X_hidden.shape[0]
    std_X = np.std(X_hidden)
    gauss1_mean, gauss1_std_sc, w1 = -2*std_X, std_X, 0.45
    gauss2_mean, gauss2_std_sc, w2 = 0, std_X, 0.1
    gauss3_mean, gauss3_std_sc, w3 = 2*std_X, std_X, 0.45

    std_M, std_N = std_X * scale_m, std_X * scale_n

    M_noises = np.c_[gauss1_mean + std_M * np.random.normal(0, 1, data_size),
                     gauss2_mean + std_M * np.random.normal(0, 1, data_size),
                     gauss3_mean + std_M * np.random.normal(0, 1, data_size)]

    N_noises = np.c_[gauss1_mean + std_N * np.random.normal(0, 1, data_size),
                     gauss2_mean + std_N * np.random.normal(0, 1, data_size),
                     gauss3_mean + std_N * np.random.normal(0, 1, data_size)]

    M = X_hidden + M_noises[np.arange(data_size), np.random.choice([0, 1, 2], data_size, p=[w1, w2, w3])][:, np.newaxis]
    N = X_hidden + N_noises[np.arange(data_size), np.random.choice([0, 1, 2], data_size, p=[w1, w2, w3])][:, np.newaxis]

    return M, N