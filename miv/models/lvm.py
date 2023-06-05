"""
Latent variable model for measurement error model for discrete M, N, X, Z.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from numpy import concatenate as cat
import pickle
from util import PROJECT_ROOT
import argparse
import os
import scipy
from sklearn.utils import shuffle
from scipy.optimize import minimize, Bounds, LinearConstraint
from statsmodels.base.model import GenericLikelihoodModel

# hyperparameters
sem_name = 'discrete'
sem_seed = 50
data_transform = None
log_interval = 10

parser = argparse.ArgumentParser(description='simulation parser')
parser.add_argument('--sem-name', type=str, default=sem_name, help='set sem name.')
parser.add_argument('--sem-seed', type=int, default=sem_seed, help='set sem seed.')
parser.add_argument('--data-transform', type=str, default=data_transform, help='set data transform.')
parser.add_argument('--log-interval', type=int, default=log_interval, help='set log interval for training.')
args = parser.parse_args()

categories = 10

lr = 0.1
weight_decay = 0.1

optimiser = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

observed_vars = np.arange(num_variables)[is_Z + is_M + is_N]

unobserved_vars = np.arange(num_variables)[is_X]


##### Torch util functions #####

class MerrorIVTorchDataset(Dataset):
    """
    Torch dataset object for the measurement-error-IV model.
    """

    def __init__(self, args, transform=None):
        args.sem_name = 'discrete'
        args.sem_seed = 50
        data_path = os.path.join(PROJECT_ROOT, 'data', args.sim_name, args.sim_seed, 'main.csv')
        data_df = pd.read_csv(data_path)

        self.transform = transform

        if self.transform is None:
            self.Z = torch.from_numpy(data_df['Z'].values)
            self.M = torch.from_numpy(data_df['M'].values)
            self.N = torch.from_numpy(data_df['N'].values)
            self.Y = torch.from_numpy(data_df['Y'].values)

        else:
            self.Z = self.transform(torch.from_numpy(data_df['Z'].values))
            self.M = self.transform(torch.from_numpy(data_df['M'].values))
            self.N = self.transform(torch.from_numpy(data_df['N'].values))
            self.Y = self.transform(torch.from_numpy(data_df['Y'].values))


    def __len__(self):
        return self.M.shape[0]


    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        samples = (self.Z[item],
                   self.M[item],
                   self.N[item],
                   self.Y[item])

        return samples


def get_traindevtest_torch_loaders(args):
    full_data = MerrorIVTorchDataset(args, transform=args.transform)

    train_size, dev_size = int(0.6 * len(full_data)), int(0.2 * len(full_data))
    test_size = len(full_data) - train_size - dev_size

    train, dev, test = torch.utils.data.random_split(full_data, [train_size, dev_size, test_size])

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=1)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=True, num_workers=1)

    return train_loader, dev_loader, test_loader

#####################################

##### Torch functions #####
def train_torch_model(model, epoch, train_loader, optimiser, device, args, dag):
    train_loss = 0
    for batch_idx, (Z, M, N, Y) in enumerate(train_loader):
        Z, M, N, Y = Z.to(device), M.to(device), N.to(device), Y.to(device)
        optimiser.zero_grad()
        loss = negloglik(observations=(Z, M, N, Y), p_factors=model.p_factors, args=args, dag=dag)
        loss.backward()
        train_loss += loss.item()
        optimiser.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(Z), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(Z)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    return train_loss


def create_np_batches(data_df, batch_size):
    """
    Create np batches for data_df
    :param data_df:
    :return: np array
    """
    data_df = shuffle(data_df).values  # random state shared with np.random
    N = len(data_df)
    batches = []

    for i in range(N // batch_size):
        batches.append(data_df[i*batch_size: (i+1)*batch_size])

    return batches



def train_np_model(args, data_df):
    train_loss = 0
    for epoch in range(args.num_epochs):
        batches = create_np_batches(data_df=data_df, batch_size=args.batch_size)
        for batch_idx, batch in enumerate(batches):
            pass
###########################




def initialise_discrete_factor(pa_categories, ch_categories):
    """
    generate a probability table for discrete factor p(ch | pa)
    :param pa_categories: array of int.
           ch_categories: int
    :return: array of dimension (num_pa + 1) x num_categories.
    """
    unnormalised_probs = torch.random.rand(*torch.cat([pa_categories, ch_categories - 1]))
    return unnormalised_probs


def main(args):
    # 1. load graph to get things like DAG_order, DAG_parents, is_Z, is_X, is_M, is_N
    # 2. load data; split data into train dev test.
    # 3. initialise latent variable model
    # 4. train latent variable model
    # 5. test latent variable model
    pass


def load_DAG(args):
    args.sem_name = 'discrete'
    args.sem_seed = 50
    dag_model_path = os.path.join(PROJECT_ROOT, 'data', args.sim_name, 'scm.pkl')
    dag = pickle.load(dag_model_path)

    return dag


def shape_param_helper(param_1d, dag):
    p_factors = []
    num_singleton_variables = dag.DAG_order.shape[0]
    params_used = 0
    for i in range(num_singleton_variables):
        if dag.is_Z[i]:
            p_factors.append(param_1d[: dag.k_ZXMN])
            params_used += dag.k_ZXMN
        elif dag.is_M[i]:
            p_factors.append(param_1d[dag.k_ZXMN: dag.k_ZXMN*(1+dag.k_ZXMN)].reshape(dag.k_ZXMN, dag.k_ZXMN))
            params_used += dag.k_ZXMN ** 2
        elif dag.is_N[i]:
            p_factors.append(param_1d[dag.k_ZXMN*(1+dag.k_ZXMN): dag.k_ZXMN*(2+dag.k_ZXMN)].reshape(dag.k_ZXMN, dag.k_ZXMN))
            params_used += dag.k_ZXMN ** 2
        elif dag.is_X[i]:
            p_factors.append(param_1d[dag.k_ZXMN*(2+dag.k_ZXMN): dag.k_ZXMN*(3+dag.k_ZXMN)].reshape(dag.k_ZXMN, dag.k_ZXMN))
            params_used += dag.k_ZXMN ** 2
    assert params_used == len(param_1d)
    return tuple(p_factors)


# class MerrorIVLVM(GenericLikelihoodModel):
#     def __init__(self, endog, exog, dag, args):
#         # Z is exog; M, N, Y are endog.
#         super(MerrorIVLVM, self).__init__(endog=endog, exog=exog)
#         self.dag = dag
#         self.args = args
#
#     def nloglikeobs(self, params):
#         p_factors = shape_param_helper(param_1d=params, dag=self.dag)
#         M_dat, N_dat, Y_dat = self.exog.T
#         nll = negloglik(Z_dat=self.endog, M_dat=M_dat, N_dat=N_dat, Y_dat=Y_dat,
#                         p_factors=p_factors, args=self.args, dag=self.dag)
#         return nll
#
#     def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):
#         if start_params == None:
#             start_params = np.ones(self.dag.k_ZXMN * (3 * self.dag.k_ZXMN + 1)) / self.dag.k_ZXMN
#         linear_constraints = get_constraint(self.dag.num_categories)
#         constraints = linear_constraints
#         method = 'minimize'
#
#         return super(MerrorIVLVM, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun,
#                                             method=method, constraints=constraints)
#
#     def grab_fitted_param(self):
#         model = self.df_model

class MerrorIVLVM:
    def __init__(self, obs_data, dag, args):
        self.dag = dag
        self.args = args
        self.obs_data = obs_data
        self.params = None

    def nll(self, params):
        p_factors = shape_param_helper(param_1d=params, dag=self.dag)
        Z_dat, M_dat, N_dat, Y_dat = self.obs_data.T
        loss = negloglik(Z_dat=Z_dat, M_dat=M_dat, N_dat=N_dat, Y_dat=Y_dat,
                         p_factors=p_factors, args=self.args, dag=self.dag)
        return loss

    def fit(self, start_params=None):
        num_params = self.dag.k_ZXMN * (3 * self.dag.k_ZXMN + 1)
        if start_params is None:
            start_params = np.ones(self.dag.k_ZXMN * (3 * self.dag.k_ZXMN + 1)) / self.dag.k_ZXMN
        linear_constraints = LinearConstraint(get_constraint(self.dag.num_categories))
        constraints = [linear_constraints]
        bounds = Bounds(*[[0, 1] for i in range(num_params)])
        res = minimize(self.nll, x0=start_params, method='trust-constr', constraints=constraints, bounds=bounds)
        self.params = res.x
        return res

    def grab_optimised_params(self):
        if self.params is None:
            raise AttributeError('Need to run fit method first.')
        else:
            params = shape_param_helper(self.params, dag=self.dag)
        return params


def get_constraint(num_categories):
    constraint_mat = np.eye(num_categories * 3 + 1)
    constraint_mat = np.repeat(constraint_mat, num_categories, axis=1)
    return constraint_mat


def negloglik(Z_dat, M_dat, N_dat, Y_dat, p_factors, args, dag):
    N_data = Z_dat.shape[0]

    # fill in unobserved variables
    Z, M, N, Y = Z_dat.reshape(N_data, -1), M_dat.reshape(N_data, -1), \
                 N_dat.reshape(N_data, -1), Y_dat.reshape(N_data, -1)

    obs_and_gaps = np.zeros((Z.shape[0], args.num_variables), dtype=int) - 1
    for i in range(args.num_variables):
        if dag.is_Z[i]:
            obs_and_gaps[:, i] = Z
            continue
        elif dag.is_M[i]:
            obs_and_gaps[:, i] = M
            continue
        elif dag.is_N[i]:
            obs_and_gaps[:, i] = N
            continue
        elif dag.is_Y[i]:
            obs_and_gaps[:, i] = Y
            continue
        else:
            continue

    idx = obs_and_gaps.T

    pz, pmx, pnx, pxz = None, None, None, None
    for i in range(args.num_variables):
        if dag.is_Z[i]:
            p_factor = p_factors[i]

            obs_pa = np.intersect1d(dag.DAG_parents[i], observed_vars)
            obs_pa_data = idx[obs_pa]

            obs_ch_data = idx[i:i+1]

            obs_data = cat([obs_pa_data, obs_ch_data], axis=0)

            pz = p_factor[obs_data]
        elif (dag.is_M[i] or dag.is_N[i]):
            p_factor = p_factors[i]

            obs_pa = np.intersect1d(dag.DAG_parents[i], observed_vars)
            obs_pa_data = idx[obs_pa]

            unobs_pa = np.intersect1d(dag.DAG_parents[i], unobserved_vars)

            pa_order = []
            obs_pa_suborder = []
            unobs_pa_suborder = []
            for j, pa in enumerate(dag.DAG_parents[i]):
                pa_order.append(j)
                if pa in obs_pa:
                    obs_pa_suborder.append(j)
                elif pa in unobs_pa:
                    unobs_pa_suborder.append(j)
                else:
                    raise ValueError('parents must be in either obs_pa or unobs_pa.')

            obs_ch_data = idx[i:i+1]
            obs_data = cat([obs_pa_data, obs_ch_data], axis=0)

            factor = np.moveaxis(p_factor, cat([obs_pa_suborder, [-1], unobs_pa_suborder]), np.arange(p_factor.ndims))

            if dag.is_M[i]:
                pmx = factor[obs_data]
            else:
                pnx = factor[obs_data]

        elif dag.is_X[i]:
            p_factor = p_factors[i]

            obs_pa = np.intersect1d(dag.DAG_parents[i], observed_vars)
            obs_pa_data = idx[obs_pa]

            unobs_pa = np.intersect1d(dag.DAG_parents[i], unobserved_vars)

            pa_order = []
            obs_pa_suborder = []
            unobs_pa_suborder = []
            for j, pa in enumerate(dag.DAG_parents[i]):
                pa_order.append(j)
                if pa in obs_pa:
                    obs_pa_suborder.append(j)
                elif pa in unobs_pa:
                    unobs_pa_suborder.append(j)
                else:
                    raise ValueError('parents must be in either obs_pa or unobs_pa.')

            obs_ch_data = idx[i:i+1]
            obs_data = cat([obs_pa_data, obs_ch_data], axis=0)

            pxz = p_factor[obs_data]

        else:
            continue

    assert pz is not None
    assert pmx is not None
    assert pnx is not None
    assert pxz is not None

    pz = np.tile(pz, [dag.k_ZXMN, 1]).T

    negloglike = -1 * np.sum(np.log(np.sum(pz * pxz * pnx * pmx, axis=-1)))

    return negloglike


# def get_traintestdev_np(args):
#     args.sem_name = 'discrete'
#     args.sem_seed = 50
#     data_path = os.path.join(PROJECT_ROOT, 'data', args.sim_name, args.sim_seed, 'main.csv')
#     data_df = pd.read_csv(data_path)
#











