"""
Author: Yuchen Zhu

Simulation for the causal graph:
Z --> X --> Y
X <-- U --> Y
X --> M
X --> N
"""
import numpy as np
from numpy import concatenate as cat
import pandas as pd
import os
from util import gen_discrete_factor, PROJECT_ROOT
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join as path_cat
import argparse
import pickle as pkl

# hyperparameters
data_size = 10000
sem_name = 'discrete'
do_save_scm = True

parser = argparse.ArgumentParser(description='simulation parser')
parser.add_argument('--sem-name', type=str, default=sem_name, help='set sem name.')
parser.add_argument('--data-size', type=int, default=data_size, help='set dataset size to generate.')
parser.add_argument('--do-save-scm', type=bool, default=do_save_scm, help='set dataset size to generate.')

args = parser.parse_args()

generate_random_discrete_factors = True
seed_discrete_factor = 50

is_ZXMN_discrete = True
is_U_discrete = True
is_Y_discrete = True

k_ZXMN = 5
k_U = 5
k_Y = 5

if not is_ZXMN_discrete:
    k_ZXMN = -1
if not is_U_discrete:
    k_U = -1
if not is_Y_discrete:
    k_Y = -1

Z_dim = 1
M_dim = 1
N_dim = 1
X_dim = 1
U_dim = 1
Y_dim = 1

dag_config = {
    'is_ZXMN_discrete': is_ZXMN_discrete,
    'is_U_discrete': is_U_discrete,
    'is_Y_discrete': is_Y_discrete,

    'k_ZXMN': k_ZXMN,
    'k_U': k_U,
    'k_Y': k_Y,

    'Z_dim': Z_dim,
    'M_dim': M_dim,
    'N_dim': N_dim,
    'X_dim': X_dim,
    'U_dim': U_dim,
    'Y_dim': Y_dim,

    'generate_random_discrete_factors': generate_random_discrete_factors,
    'seed_discrete_factor': seed_discrete_factor,


}


### continuous structural equations

## linear ones

# standard noise configurations
normal_noise_config = {'mean': 0, 'std': 1}
noise_config = normal_noise_config

def _sum_function(parents):
    return np.sum(parents, axis=-1)


def _normal_noise(mean, std, size=-1):
    if size == -1:
        raise ValueError('need to set number of data points as input into noise function.')
    return np.random.normal(loc=mean, scale=std, size=size)


def construct_noisy_scm(scm, scm_params_dict, noise_fn, noise_params_dict):
    def noisy_scm(parents):
        size = parents.shape[0]
        noise_params_dict['size'] = size
        return scm(parents=parents, **scm_params_dict) + noise_fn(**noise_params_dict)
    return noisy_scm



class MerrorIV:
    """
    Simulator for DAG
                 U
                /  \
               v    v
        Z --> X --> Y
             / \
            v  v
            M  N
    """
    def __init__(self, **config):
        self.k_ZXMN = config['k_ZXMN']
        self.k_U = config['k_U']
        self.k_Y = config['k_Y']
        self.is_ZXMN_discrete = config['is_ZXMN_discrete']
        self.is_U_discrete = config['is_U_discrete']
        self.is_Y_discrete = config['is_Y_discrete']
        self.Z_dim, self.M_dim, self.N_dim = config['Z_dim'], config['M_dim'], config['N_dim']
        self.X_dim, self.U_dim, self.Y_dim = config['X_dim'], config['U_dim'], config['Y_dim']
        self.num_semantic_variables = 6
        self.seed_discrete_factor = config['seed_discrete_factor']
        self.generate_random_discrete_factors = config['generate_random_discrete_factors']


    def _encode_variables(self):
        variable_codes = []
        dims = [self.Z_dim, self.U_dim, self.X_dim, self.M_dim, self.N_dim, self.Y_dim]
        for i in np.arange(self.num_semantic_variables):
            current_semantic_var = []
            for j in range(dims[i]):
                current_semantic_var.append(i + j)
            variable_codes.append(current_semantic_var)
        self.Z, self.U, self.X, self.M, self.N, self.Y = variable_codes


    def _build_graph(self):
        # encode variables with numbers
        self._encode_variables()

        self.DAG_order = cat([self.Z, self.U, self.X, self.M, self.N, self.Y],
                             axis=0)  # an array containing all variables ordered from source code(s) to sink nodes
        num_variables = self.DAG_order.shape[0]
        self.rev_DAG_order = np.zeros((num_variables,), dtype=int)  # an array which tells the position of the node in the graph
        for i in range(num_variables):
            self.rev_DAG_order[i] = np.where(self.DAG_order == i)[0][0]

        self.is_Z = np.repeat(False, num_variables)
        self.is_Z[self.rev_DAG_order[self.Z]] = True
        self.is_U = np.repeat(False, num_variables)
        self.is_U[self.rev_DAG_order[self.U]] = True
        self.is_X = np.repeat(False, num_variables)
        self.is_X[self.rev_DAG_order[self.X]] = True
        self.is_M = np.repeat(False, num_variables)
        self.is_M[self.rev_DAG_order[self.M]] = True
        self.is_N = np.repeat(False, num_variables)
        self.is_N[self.rev_DAG_order[self.N]] = True
        self.is_Y = np.repeat(False, num_variables)
        self.is_Y[self.rev_DAG_order[self.Y]] = True

        is_discrete = np.repeat(False, self.DAG_order.shape[0])
        is_discrete[self.U] = is_U_discrete
        is_discrete[self.Y] = is_Y_discrete
        for var in [self.Z, self.X, self.M, self.N]:
            is_discrete[var] = is_ZXMN_discrete
        self.is_discrete = is_discrete

        categories = np.zeros((self.DAG_order.shape[0],), dtype=int)
        categories[self.U] = k_U
        categories[self.Y] = k_Y
        for var in [self.Z, self.X, self.M, self.N]:
            categories[var] = k_ZXMN
        self.categories = categories

        DAG_parents = []
        for i in range(num_variables):
            if self.is_Z[i]:
                DAG_parents.append(np.intersect1d(self.DAG_order[:i], cat([self.Z])))
            elif self.is_U[i]:
                DAG_parents.append(np.intersect1d(self.DAG_order[:i], cat([self.U])))
            elif self.is_X[i]:
                DAG_parents.append(np.intersect1d(self.DAG_order[:i], cat([self.Z, self.U])))
            elif self.is_M[i]:
                DAG_parents.append(np.intersect1d(self.DAG_order[:i], cat([self.X, self.M])))
            elif self.is_N[i]:
                DAG_parents.append(np.intersect1d(self.DAG_order[:i], cat([self.X, self.N])))
            elif self.is_Y[i]:
                DAG_parents.append(np.intersect1d(self.DAG_order[:i], cat([self.U, self.X, self.Y])))
        self.DAG_parents = DAG_parents


    def _build_scms(self):
        self.scms = []
        num_singleton_variables = self.DAG_order.shape[0]
        if self.generate_random_discrete_factors:
            np.random.seed(self.seed_discrete_factor)
        for i in range(num_singleton_variables):
            if self.is_discrete[i]:
                if self.generate_random_discrete_factors:
                    ch_categories = self.categories[i]
                    pa_categories = self.categories[self.DAG_parents[i]]
                    self.scms.append(gen_discrete_factor(pa_categories=pa_categories, ch_categories=ch_categories))
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
                noisy_scm = construct_noisy_scm(scm=_sum_function, scm_params_dict={},
                                                noise_fn=_normal_noise, noise_params_dict=noise_config)
                self.scms.append(noisy_scm)

    def build(self):
        self._build_graph()
        self._build_scms()


    def generate(self, data_size):
        data = []
        num_singleton_variables = self.DAG_order.shape[0]
        for i in range(num_singleton_variables):
            pa = self.DAG_parents[i]
            pa_data = np.array([data[pa_i] for pa_i in pa]).T
            if self.is_discrete[i]:
                ch_categories, scm_i = self.categories[i], self.scms[i]
                sample_fn = lambda prob: np.random.choice(np.arange(ch_categories), p=prob)
                prob_matrix = scm_i[tuple(pa_data.T)] if len(pa) > 0 else np.tile(scm_i, [data_size, 1])
                assert np.abs(np.sum(prob_matrix[0]) - 1.) < 1e-7
                ch_data = np.apply_along_axis(func1d=sample_fn, axis=-1, arr=prob_matrix)
            else:
                scm_i = self.scms[i]
                ch_data = scm_i(parents=pa_data)
            data.append(ch_data)

        data_np = np.array(data).T
        data_df = pd.DataFrame(data_np)
        cols = []
        for i in range(num_singleton_variables):
            if self.is_Z[i]:
                var = 'Z'
            elif self.is_Y[i]:
                var = 'Y'
            elif self.is_X[i]:
                var = 'X'
            elif self.is_U[i]:
                var = 'U'
            elif self.is_M[i]:
                var = 'M'
            elif self.is_N[i]:
                var = 'N'
            else:
                raise ValueError('Variable must be one of Z, U, X, M, N, Y.')
            cols.append(var)

        data_df.columns = cols

        return data_df


def save_csv(data_df, model, sem_name):
    save_dir = path_cat(PROJECT_ROOT, 'data', sem_name, str(model.seed_discrete_factor))
    os.makedirs(save_dir, exist_ok=True)
    data_df.to_csv(path_cat(save_dir, 'main.csv'))


def save_scm(model, sem_name):
    save_dir = path_cat(PROJECT_ROOT, 'data', sem_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(path_cat(save_dir, 'scm.pkl'), 'wb') as model_out:
        pkl.dump(model, model_out, pkl.HIGHEST_PROTOCOL)


def gen_plots(data_df, model, sem_name):
    save_dir = path_cat(PROJECT_ROOT, 'data', sem_name, str(model.seed_discrete_factor))
    os.makedirs(save_dir, exist_ok=True)

    sns.set_theme(font="tahoma", font_scale=1)
    sns.pairplot(data_df, kind='hist'), plt.savefig(path_cat(save_dir, 'full_pairwise.png')), plt.close()


def main(config, args):
    model = MerrorIV(**config)
    model.build()
    data_df = model.generate(data_size=args.data_size)
    save_csv(data_df=data_df, model=model, sem_name=args.sem_name)
    if args.do_save_scm:
        save_scm(model=model, sem_name=sem_name)
    gen_plots(data_df=data_df, model=model, sem_name=args.sem_name)

if __name__ == "__main__":
    main(config=dag_config, args=args)














