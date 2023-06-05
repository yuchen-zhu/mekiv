import numpy as np
from numpy import concatenate as cat
import os
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
import pickle
import copy
import yaml
from miv.general_util import _sqdist, get_median_inter_mnist
from scipy.optimize import minimize, fmin_bfgs, fmin_cg, fmin_powell
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from torch import tensor

from typing import Dict, Any, Iterator, Tuple
from itertools import product

### kernel helpers ###

"""
function loss = KIV1_loss(df,lambda)
%stage 1 error of KIV
%hyp=(lambda,vx,vz)

n=length(df.y1);
m=length(df.y2);

brac=make_psd(df.K_ZZ)+lambda.*eye(n);
gamma=(brac)\df.K_Zz;

loss=trace(df.K_xx-2.*df.K_xX*gamma+gamma'*df.K_XX*gamma)./m;

end

function K = make_psd(K)
%for numerical stability, add a small ridge to a symmetric matrix
eps=1e-10;

[N,~]=size(K);
K=(K+K')./2+eps.*eye(N);


end

"""

#######################################
######### some basic functions ########
#######################################

class Functions:
    def __init__(self):
        pass


    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))


fns = Functions()
#######################################


def compute_rbf_kernel(x1, x2, sigma):
    sqdist = _sqdist(x1, x2)
    return np.exp(-1 * sqdist / sigma / sigma / 2)


def cme_tuning(K, args):  # returns the best hyperparameter lambd_star for CME regression
    if isinstance(args.log_lambd_0, list):
        n = K.XX.shape[0]
        lambd_list = np.exp(np.linspace(args.log_lambd_0[0], args.log_lambd_0[1], 50))
        gamma_list = [np.linalg.solve(K.ZZ + n * lambd * np.eye(n), K.Zz) for lambd in lambd_list]
        score = [np.trace(gamma.T.dot(K.XX.dot(gamma)) - 2 * K.xX.dot(gamma)) for gamma in gamma_list]
        lambd_star = lambd_list[np.argmin(score)]
        gamma_star = gamma_list[np.argmin(score)]
        return np.log(lambd_star)

    else:
        Nfeval = [1]

        def callbackF(log_lambd, Nfeval=Nfeval):
            print('{0}   {1}   {2}'.format(Nfeval[0], log_lambd, cme_loss(np.exp(log_lambd), K, args)))
            Nfeval[0] = Nfeval[0] + 1

        log_lambd_0 = args.log_lambd_0
        obj = lambda x: cme_loss(np.exp(x), K, args)
        # breakpoint()
        log_lambd_star = \
        fmin_bfgs(obj, x0=log_lambd_0, callback=callbackF, gtol=1e-4, maxiter=5000, full_output=True, retall=False)[0]
        return log_lambd_star


def cme_loss(lambd, K, args):  # objective for tuning lambda in KRR for CME, see KIV paper A.5.2
    n = args.n_train
    m = args.n_dev
    brac = make_psd(K.ZZ)+ n * lambd*np.eye(n)
    # gamma = np.linalg.inv(brac).dot(K.Zz)
    gamma = np.linalg.solve(K.ZZ + n * lambd * np.eye(n), K.Zz)

    loss = np.trace(K.xx - 2 * K.xX @ gamma + gamma.T @ K.XX @ gamma)/m
    return loss


def KIV_pred(f, stage, cme_Xall, args):
    K_X, xi = f.K, f.xi
    n = K_X.ZZ.shape[0]
    m = K_X.Zz.shape[1]

    lambd = cme_Xall.lambd_star
    brac = make_psd(K_X.ZZ) + n * lambd * np.eye(n)
    W = K_X.XX @ np.linalg.inv(brac) @ K_X.Zz
    brac2 = make_psd(W @ W.T) + m * xi * make_psd(K_X.XX)
    alpha = np.linalg.inv(brac2) @ (W @ args.stage_1.dev.Y)

    K_Xtest = None
    if stage == 3:
        K_Xtest = K_X.XX
    elif stage == 4:  # stage = 4 when testing
        K_Xtest = K_X.Xtest

    y_pred = (alpha.T @ K_Xtest)

    return y_pred.reshape(-1, 1)




def f_tuning(cme_X, all_args):

    if isinstance(all_args.stage_3.log_xi_0, list):
        n = cme_X.K.XX.shape[0]
        # lambd_list = np.exp(np.linspace(stage3_args.log_lambd_0[0], stage3_args.log_lambd_0[1], 50))
        # gamma_list = [np.linalg.solve(cme_X.K.ZZ + n * lambd * np.eye(n), K.Zz) for lambd in lambd_list]
        # score = [np.trace(gamma.T.dot(K.XX.dot(gamma)) - 2 * K.xX.dot(gamma)) for gamma in gamma_list]
        # lambd_star = lambd_list[np.argmin(score)]
        # gamma_star = gamma_list[np.argmin(score)]
        # return np.log(lambd_star)


    else:
        Nfeval = [1]

        def callbackF(log_xi, Nfeval=Nfeval):
            print('{0}   {1}   {2}'.format(Nfeval[0], log_xi, f_loss(np.exp(log_xi), cme_X, all_args)))
            Nfeval[0] = Nfeval[0] + 1

        log_xi_0 = all_args.stage_3.log_xi_0
        obj = lambda x: f_loss(np.exp(x), cme_X, all_args)
        # breakpoint()
        log_xi_star = fmin_bfgs(obj, x0=log_xi_0, callback=callbackF, gtol=1e-4, maxiter=5000, full_output=True, retall=False)[0]

        return log_xi_star


"""
def KIV_pred(f, stage, args):
    n = args.n_train
    m = args.n_dev"""

def f_loss(xi, cme_X, all_args):
    K_X = cme_X.K
    f = dotdict({'K': K_X, 'xi': xi})
    y_pred = KIV_pred(f, 3, cme_X, all_args)
    loss = np.sum((all_args.stage_1.train.Y - y_pred) ** 2) / all_args.stage_1.n_train
    return loss


# y_test_pred = KIV_pred(K_X, args.xi, stage=4, args=args)


def make_psd(K):  # add small ridge to K to ensure it is psd
    eps = 1e-10
    N = K.shape[0]
    K = (K+K.T)/2 + eps * np.eye(N)
    return K


def get_K(all_args, output, incl_Xobs):
    """
    Computes the kernel matrices for the regressor and target variables.

    Assumption: the mismeasured dimension is always continuous, because discrete variables cannot satisfy
    the conditions of identification.

    :param all_args:
    :param output: the target variable in CME regression. should be one of {M, N, MN, avMN, X}. This could be concatenated with the observed dimensions of X.
    :param incl_Xobs: False for stage 1 and 2, and True for stage 3.
    :return:
    """

    K = {}

    Z = all_args.stage_1.train.Z  # stage 1, can be used with stage 3 as well.
    z = all_args.stage_1.dev.Z

    sig_Z = get_median_inter_mnist(Z)
    # breakpoint()
    K['ZZ'] = compute_rbf_kernel(Z, Z, sig_Z)
    K['Zz'] = compute_rbf_kernel(Z, z, sig_Z)

    K['sig_Z'] = sig_Z


    if (output == 'N') or (output == 'M') or (output == '_X'):  # done in stage 1
        X = all_args.stage_1.train[output]
        x = all_args.stage_1.dev[output]
    elif output == 'MN':  # done in stage 1
        X = np.concatenate([all_args.stage_1.train.M, all_args.stage_1.train.N], axis=-1)
        x = np.concatenate([all_args.stage_1.dev.M, all_args.stage_1.dev.N], axis=-1)
    elif output == 'X':  # done in stage 3
        X = all_args.stage_3.train.fitted_X
        x = all_args.stage_3.dev._X  # unused
    elif output == 'avMN':
        X = 0.5 * (all_args.stage_1.train.M + all_args.stage_1.train.N)
        x = 0.5 * (all_args.stage_1.dev.M + all_args.stage_1.dev.N)
    else:
        raise ValueError('output is one of {M, N, MN, avMN, X}.')

    def swap_merror_dim(Xall, x_, merror_dim):
        assert (len(x_.shape) <= 2) and (len(x_.shape) == 1 or x_.shape[1] == 1)
        new = np.zeros(Xall.shape)
        obs_dims = np.arange(Xall.shape[-1]) != merror_dim
        new[:, merror_dim] = x_.flatten()
        new[:, obs_dims] = Xall[:, obs_dims]
        return new

    if incl_Xobs:  # stage 3
        X = swap_merror_dim(Xall=all_args.train._Xall, x_=X, merror_dim=all_args.merror_dim)
        x = swap_merror_dim(Xall=all_args.test._Xall, x_=x, merror_dim=all_args.merror_dim)

        X_test = all_args.test._Xall


    sig_X = get_median_inter_mnist(X)



    K['XX'] = compute_rbf_kernel(X, X, sig_X)
    K['xX'] = compute_rbf_kernel(x, X, sig_X)
    K['xx'] = compute_rbf_kernel(x, x, sig_X)
    if incl_Xobs:
        K['Xtest'] = compute_rbf_kernel(X, X_test, sig_X)  # only done in test time, which would include all observed dims of X as well.



    K['sig_X'] = sig_X

    K = dotdict(K)
    return K


#####################################
### visualisation helpers

def visualise_dataset(dataset, sample_size, observed_keys): # dataset is a dict object
    D_cols = []
    D_data = []
    O_cols = []
    O_data = []

    for key in dataset.keys():
        if dataset[key] is None:
            continue
        if len(dataset[key].shape) == 1 or (len(dataset[key].shape)==2 and dataset[key].shape[-1]==1):
            vis_key, vis_data = key, dataset[key][:sample_size].flatten()
            observed = vis_key in observed_keys
            D_cols.append(vis_key)
            D_data.append(vis_data)
            O_cols.append(vis_key) if observed else print('{} is latent.'.format(vis_key))
            O_data.append(vis_data) if observed else print('')
        elif len(dataset[key].shape) == 2 and dataset[key].shape[-1] > 1:
            vis_key = [key + str(i) for i in range(dataset[key].shape[-1])]
            vis_data = [dataset[key][:sample_size, i] for i in range(dataset[key].shape[-1])]
            D_cols = D_cols + vis_key
            D_data = D_data + vis_data
            for i in range(dataset[key].shape[-1]):
                if (key + str(i)) in observed_keys:
                    O_cols.append(key + str(i))
                    O_data.append(dataset[key][:sample_size, i])
                else:
                    print('{} is latent.'.format(key+str(i)))
        else:
            raise ValueError('data for each variable is either 1d or 2d numpy arrays.')

    D = pd.DataFrame(D_data).T
    D.columns = D_cols
    O = pd.DataFrame(O_data).T
    O.columns = O_cols

    for v in D_cols:
        sns.displot(D, x=v, label=v, kde=True), plt.show()

    sns.set_theme(font="tahoma", font_scale=2)
    plt.show()

    sns.pairplot(D, height=3)
    if len(observed_keys) > 0:
        sns.pairplot(O, height=1)
    plt.show()








def visualise_cme(size, target_vars, cme, args, design):
    # for a range of z's
    z_vals = np.linspace(args.train.Z.min(), args.train.Z.max(), size).reshape(-1, 1)
    K_Zz = compute_rbf_kernel(args.train.Z, z_vals, cme.K.sig_Z)
    x_dict = {}
    for var in target_vars:
        var_train = args.train[var]
        x_dict[var] = np.linspace(np.max(var_train), np.min(var_train), size)
    x_vals = np.vstack([x_dict[var] for var in target_vars]).T
    X_train = np.hstack([args.train[var] for var in target_vars])

    # evaluate the mean at a bunch of n's
    K_xX = compute_rbf_kernel(x_vals, X_train, cme.K.sig_X)
    ehat_K_xX = K_xX.dot(cme.brac_inv.dot(K_Zz))  # \hat{E}_X[k(x, X)]. shape: [size_x_vals, size_z]

    # evaluate the emipircal mean of the kernels by sampling.
    sample_size = 3000
    e_K_xX = np.zeros((size, size))
    for i, z_val in enumerate(z_vals):
        # regenerate a bunch of n by sampling
        u = design.fu(sample_size)
        z_val_reps = np.repeat(z_val, sample_size)
        x = design.fx(z_val_reps, u, sample_size)
        n = design.fn(x, sample_size)
        m = design.fm(x, sample_size)
        var_dict = {'u': u, 'z': z_val_reps, 'x': x, 'n': n, 'm': m}
        if len(target_vars.lower()) == 1:
            x_target = var_dict[target_vars.lower()].reshape(-1, 1)
        else:
            x_target = np.vstack([var_dict[var] for var in target_vars.lower()]).T
        K_x_vals_x = compute_rbf_kernel(x_vals, x_target, cme.K.sig_X)
        e_K_xX[:, i] = np.mean(K_x_vals_x, axis=1)

    # do a heat map of both ehat_K_nN and e_K_nN
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    g1 = sns.heatmap(ehat_K_xX, cmap="YlGnBu", cbar=False, ax=ax1)
    g1.set_ylabel(target_vars.lower())
    g1.set_xlabel('z')
    g1.set_title('ehat')
    g2 = sns.heatmap(e_K_xX, cmap="YlGnBu", cbar=False, ax=ax2)  # e_K_xX = E[k(x, X)]
    g2.set_ylabel(target_vars.lower())
    g2.set_xlabel('z')
    g2.set_title('e')
    g3 = sns.heatmap(np.abs(e_K_xX - ehat_K_xX), cmap="YlGnBu", ax=ax3)
    g3.set_ylabel(target_vars.lower())
    g3.set_xlabel('z')
    g3.set_title('diff')


def visualise_char_fun(size, target_vars, cme, args, design):
    curly_x = np.linspace(-.3, .3, size).reshape(-1, 1)
    #     z_vals = args.train.Z[np.random.choice(np.arange(args.n_train), size=size, replace=False)]
    z_vals = np.linspace(args.train.Z[:1000].min(), args.train.Z[:1000].max(), size).reshape(-1, 1)

    K_Zz = compute_rbf_kernel(args.train.Z, z_vals, cme.K.sig_Z)
    # gamma = cme.brac_inv.dot(K_Zz)
    n = cme.K.ZZ.shape[0]
    gamma = np.linalg.solve(cme.K.ZZ + n * cme.lambd_star * np.eye(n), K_Zz)

    N_train = args.train.N

    cos_term = np.cos(curly_x @ N_train.reshape(1, -1))
    sin_term = np.sin(curly_x @ N_train.reshape(1, -1))

    # using gamma to evaluate the charasteristic function value at a bunch of curly_x's
    charhat_cos = cos_term.dot(gamma)
    charhat_sin = sin_term.dot(gamma)

    # evaluate the emipircal mean of the kernels by sampling.
    sample_size = 3000
    char_cos, char_sin = np.zeros((size, size)), np.zeros((size, size))
    for i, z_val in enumerate(z_vals):
        # regenerate a bunch of n by sampling
        u = design.fu(sample_size)
        z_val_reps = np.repeat(z_val, sample_size)
        x = design.fx(z_val_reps, u, sample_size)
        n = design.fn(x, sample_size)
        m = design.fm(x, sample_size)

        cos_term_ = np.cos(curly_x @ n.reshape(1, -1))
        sin_term_ = np.sin(curly_x @ n.reshape(1, -1))
        char_cos[:, i] = np.mean(cos_term_, axis=1)
        char_sin[:, i] = np.mean(sin_term_, axis=1)

    # do a heat map of both ehat_K_nN and e_K_nN
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

    g1 = sns.heatmap(charhat_cos, cmap="YlGnBu", cbar=False, ax=ax1)
    g1.set_ylabel('$\mathcal{X}$')
    g1.set_xlabel('z')
    g1.set_title('$\hat{E}[cos(\mathcal{X}N)|z]$')

    g2 = sns.heatmap(char_cos, cmap="YlGnBu", cbar=False, ax=ax2)  # e_K_xX = E[k(x, X)]
    g2.set_ylabel('$\mathcal{X}$')
    g2.set_xlabel('z')
    g2.set_title('$E[cos(\mathcal{X}N)|z]$')

    g3 = sns.heatmap(np.abs(char_cos - charhat_cos), cmap="YlGnBu", ax=ax3)
    g3.set_ylabel('$\mathcal{X}$')
    g3.set_xlabel('z')
    g3.set_title('diff')

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

    g1 = sns.heatmap(charhat_sin, cmap="YlGnBu", cbar=False, ax=ax1)
    g1.set_ylabel('$\mathcal{X}$')
    g1.set_xlabel('z')
    g1.set_title('$\hat{E}[sin(\mathcal{X}N)|z]$')

    g2 = sns.heatmap(char_sin, cmap="YlGnBu", cbar=False, ax=ax2)  # e_K_xX = E[k(x, X)]
    g2.set_ylabel('$\mathcal{X}$')
    g2.set_xlabel('z')
    g2.set_title('$E[sin(\mathcal{X}N)|z]$')

    g3 = sns.heatmap(np.abs(char_sin - charhat_sin), cmap="YlGnBu", ax=ax3)
    g3.set_ylabel('$\mathcal{X}$')
    g3.set_xlabel('z')
    g3.set_title('diff')


#     breakpoint()


#######################

### discrete parameters
def gen_discrete_factor(pa_categories, ch_categories):
    """
    generate a probability table for discrete factor p(ch | pa)
    :param pa_categories: array of int.
           ch_categories: int
    :return: array of dimension (num_pa + 1) x num_categories.
    """
    unnormalised_probs = np.random.rand(*cat([pa_categories, [ch_categories]]))
    p = unnormalised_probs / np.expand_dims(np.sum(unnormalised_probs, axis=-1), -1)

    return p


def load_model(model_path):
    """
    load a model from a pickle file
    :param model_path: str
    :return: loaded model class
    """
    with open(model_path, 'r') as f:
        model = pickle.load(f)

    return model


def fill_in_args(config_path):
    """
    helper function to fill dotdict with
    values from configs files in .yml format.

    Args:
        config_path: path to config yaml file

    Returns:
        args (filled in dotdict object for function parameters)

    """
    with open(config_path, 'r') as stream:
        relevant_dict = yaml.safe_load(stream)

    args = make_dotdict(relevant_dict)

    if not args.stage_1.log_lambd_0:
        args.stage_1.log_lambd_0 = np.log(0.05)

    if not args.stage_3.log_xi_0:
        args.stage_3.log_xi_0 = np.log(0.05)

    for dct in []:
        for key in dct.keys():
            setattr(args, key, dct[key])

    return args


def curate_data(splitted_data_and_args, raw_data):
    """
    :param splitted_data_and_args: dotdict object for storing splitted data read for training,
           and args containing keys: {'n_train', 'n_dev'}
    :param raw_data: dotdict object for storing raw (i.e. not splitted into train and dev) data [Z, M, N, X, Y, U]
    """
    for key in raw_data.keys():
        _key = '_' + key if key in ['X', 'U', 'Xall'] else key
        splitted_data_and_args.train[_key] = raw_data[key][: splitted_data_and_args.n_train]
        splitted_data_and_args.dev[_key] = raw_data[key][splitted_data_and_args.n_train: splitted_data_and_args.n_train \
                                                                       + splitted_data_and_args.n_dev]
        splitted_data_and_args.test[_key] = raw_data[key][splitted_data_and_args.n_train + splitted_data_and_args.n_dev:
                                                                         splitted_data_and_args.n_train \
                                                                         + splitted_data_and_args.n_dev \
                                                                         + splitted_data_and_args.n_test]




def throw_away_outliers_with_labelrealparts(data2, outlier_cutoff_param):

    label_real = np.real(data2.labels).flatten()
    label_imag = np.imag(data2.labels).flatten()
    idx_select = (label_real < np.mean(label_real) + outlier_cutoff_param * np.std(label_real)) * (
                label_real > np.mean(label_real) - outlier_cutoff_param * np.std(label_real)) \
                 * (label_imag < np.mean(label_imag) + outlier_cutoff_param * np.std(label_imag)) * (
                             label_imag > np.mean(label_imag) - outlier_cutoff_param * np.std(label_imag))

    data2.labels = data2.labels[idx_select]
    data2.Chi = data2.Chi[idx_select]
    data2.Z = data2.Z[idx_select]


def shuffle_(raw_data):

    data_idices = None

    for key in raw_data.keys():
        data_idices = np.arange(raw_data[key].shape[0])
        break

    np.random.shuffle(data_idices)

    for key in raw_data.keys():
        raw_data[key] = raw_data[key][data_idices]


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def make_dotdict(dct: dict):
    def recursive_dotdict(dotdct: dotdict):
        for key in dotdct.keys():
            if type(dotdct[key]) is not dict:
                continue
            else:
                dotdct[key] = dotdict(dotdct[key])
                recursive_dotdict(dotdct[key])
        return dotdct
    dotdct = dotdict(dct)

    return recursive_dotdict(dotdct)


def grid_search_dict(org_params: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Iterate list in dict to do grid search.

    Examples
    --------
    >>> test_dict = dict(a=[1,2], b = [1,2,3], c = 4)
    >>> list(grid_search_dict(test_dict))
    [('a:1-b:1', {'c': 4, 'a': 1, 'b': 1}),
    ('a:1-b:2', {'c': 4, 'a': 1, 'b': 2}),
    ('a:1-b:3', {'c': 4, 'a': 1, 'b': 3}),
    ('a:2-b:1', {'c': 4, 'a': 2, 'b': 1}),
    ('a:2-b:2', {'c': 4, 'a': 2, 'b': 2}),
    ('a:2-b:3', {'c': 4, 'a': 2, 'b': 3})]
    >>> test_dict = dict(a=1, b = 2, c = 3)
    >>> list(grid_search_dict(test_dict))
    [('one', {'a': 1, 'b': 2, 'c': 3})]

    Parameters
    ----------
    org_params : Dict
        Dictionary to be grid searched

    Yields
    ------
    name : str
        Name that describes the parameter of the grid
    param: Dict[str, Any]
        Dictionary that contains the parameter at grid

    """
    search_keys = []
    non_search_keys = []
    for key in org_params.keys():
        if isinstance(org_params[key], list):
            search_keys.append(key)
        else:
            non_search_keys.append(key)
    if len(search_keys) == 0:
        yield "one", org_params
    else:
        param_generator = product(*[org_params[key] for key in search_keys])
        for one_param_set in param_generator:
            one_dict = {k: org_params[k] for k in non_search_keys}
            tmp = dict(list(zip(search_keys, one_param_set)))
            one_dict.update(tmp)
            one_name = "-".join([k + ":" + str(tmp[k]) for k in search_keys])
            yield one_name, one_dict
