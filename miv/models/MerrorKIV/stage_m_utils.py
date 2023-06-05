import numpy as np
from numpy.random import default_rng
from miv.data.data_class import StageMDataSet, StageMDataSetTorch
import torch
from torch import tensor
from miv.util import dotdict


def sample_from_khat(sample_size, sigma):
    samples = np.random.normal(0, 1, sample_size)
    samples = samples / 2 / np.pi / sigma
    return samples


def shuffle_(raw_data, rand_seed):
    rng = default_rng(seed=rand_seed)
    data_idices = np.arange(raw_data['Z'].shape[0])

    rng.shuffle(data_idices)

    for key in raw_data.keys():
        raw_data[key] = raw_data[key][data_idices]


def throw_away_outliers_with_labelrealparts(data2, outlier_cutoff_param):

    label_real = np.real(data2.labels).flatten()
    label_imag = np.imag(data2.labels).flatten()
    idx_select = (label_real < np.mean(label_real) + outlier_cutoff_param * np.std(label_real)) * (
            label_real > np.mean(label_real) - outlier_cutoff_param * np.std(label_real)) \
                 * (label_imag < np.mean(label_imag) + outlier_cutoff_param * np.std(label_imag)) * (
                         label_imag > np.mean(label_imag) - outlier_cutoff_param * np.std(label_imag))
    # breakpoint()
    data2.labels = data2.labels[idx_select]
    data2.Chi = data2.Chi[idx_select]
    data2.Z = data2.Z[idx_select]


def prepare_stage_M_data(raw_data2, rand_seed):
    """
    s1_data_and_args: stage 2 data inherits stage 1 training data and cme's
    """

    throw_away_outliers_with_labelrealparts(raw_data2, 1.); shuffle_(raw_data2, rand_seed)

    stage_m_data = StageMDataSetTorch.from_numpy(raw_data2)
    return stage_m_data


def create_stage_M_raw_data(n_chi, N1, M1, Z2, gamma_n, gamma_mn, sigmaN, KZ1Z2):
    # input: (n_Chi, z) --> output: y
    Chi = sample_from_khat(n_chi, sigmaN ** 0.5).reshape(-1, 1)  # because the computed sigmaN is actually sigma^2N
    n, m = KZ1Z2.shape

    ### decompose e^{i\mathcal{X}n_i} ###
    # breakpoint()
    cos_term = np.cos(Chi @ N1.reshape(1,-1))  # shape: Chi.shape[0] x args.train.N.shape[0]
    sin_term = np.sin(Chi @ N1.reshape(1,-1))
    #####################################

    ### denominator ###
    denom = dotdict({})
    # using gamma to evaluate the charasteristic function value at a bunch of Chi's
    denom.cos_weighted = cos_term.dot(gamma_n)
    denom.sin_weighted = sin_term.dot(gamma_n)
    denom.value = denom.cos_weighted + denom.sin_weighted * 1j
    ###################

    ### numerator ###
    numer = dotdict({})
    numer.cos_weighted = cos_term.dot(gamma_mn * M1)  # shape: Chi.shape[0] x args.dev.Z.shape[0]
    numer.sin_weighted = sin_term.dot(gamma_mn * M1)
    numer.value = numer.cos_weighted + numer.sin_weighted * 1j
    #################

    train_labels_ = numer.value / denom.value
    train_labels = train_labels_.flatten().reshape(-1, 1)

    Chi_flat = np.repeat(Chi, m).reshape(-1, 1)
    z_dim = Z2.shape[1]
    z_dev_flat = np.repeat(Z2[np.newaxis, :, :], n_chi, axis=0).reshape(-1, z_dim)
    # breakpoint()
    raw_data2 = dotdict({})
    raw_data2.Chi = Chi_flat
    raw_data2.Z = z_dev_flat
    raw_data2.labels = train_labels

    stage_m_raw_data = StageMDataSet(**raw_data2)

    return stage_m_raw_data