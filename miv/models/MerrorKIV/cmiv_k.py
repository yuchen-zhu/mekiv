import torch.nn as nn
import torch
from torch import optim, tensor
import numpy as np
import matplotlib.pyplot as plt
# from miv.util import fill_in_args, compute_rbf_kernel, cme_tuning, curate_data, \
#     throw_away_outliers_with_labelrealparts, shuffle_, get_K
from miv.util import *
from miv.general_util import get_median_inter_mnist
import argparse


"""
# load data
prepare_pre_stage_1_args_and_data(all_args, raw_sim)

# Stage 1
prepare_stage_1_data_and_args(all_args)
compute_cme_s1_main(all_args)

# Stage 2
raw_data2 = create_stage_2_raw_data(all_args)
prepare_stage_2_data_and_args(all_args, raw_data2)
stage_two_out = stage_two_main(all_args.stage_2) # need to 1. add hyperparameter tuning routine, 2. also tune for lambda_x

# Stage 3
prepare_stage_3_data_and_args(all_args, stage_two_out)
compute_cme_s3_main(all_args)
compute_f_main(all_args, which_cme_Xall='cme_Xall')

Y_pred = KIV_pred(all_args.stage_3.f, 4, all_args.stage_3.cme_Xall, all_args)

"""


def compute_cme(all_args, output, incl_obs):
    K = get_K(all_args, output=output, incl_Xobs=incl_obs)  # because all staeg 3 data inherits from stage 1
    if output == 'X':
        lambd_star = all_args.stage_3.lambd_x if not incl_obs else np.exp(cme_tuning(K, all_args.stage_3))
    elif (output == 'MN') \
            or (output == 'N') \
            or (output == 'M') \
            or (output == 'avMN') \
            or (output == '_X'):
        lambd_star = np.exp(cme_tuning(K, all_args.stage_1))
    else:
        raise ValueError('Output is one of {X, MN, N, M, avMN}.')
    n = all_args.stage_1.n_train
    brac = make_psd(K.ZZ) + n * lambd_star*np.eye(all_args.stage_1.n_train)
    brac_inv = np.linalg.inv(brac)
    gamma_star = np.linalg.solve(K.ZZ + n * lambd_star * np.eye(n), K.Zz)
    cme = dotdict({})
    cme.lambd_star = lambd_star
    cme.brac_inv = brac_inv
    cme.brac = brac
    cme.gamma_star = gamma_star
    cme.K = K
    return cme


def compute_f(all_args, cme_X):
    K_X = cme_X.K
    xi_star = np.exp(f_tuning(cme_X, all_args))
    brac_inv = cme_X.brac_inv
    W_star = K_X.XX @ brac_inv @ K_X.Zz  # is it alright to use a inv here?
    brac2_inv = np.linalg.inv(make_psd(W_star @ W_star.T) + all_args.stage_1.n_dev * xi_star * make_psd(K_X.XX))
    alpha_star = brac2_inv @ W_star @ all_args.stage_1.dev.Y

    f = dotdict({})
    f.xi = xi_star
    f.W_star = W_star
    f.brac2_inv = brac2_inv
    f.alpha_star = alpha_star
    f.K = K_X

    return f


def compute_cme_s1_main(all_args):
    cme_N = compute_cme(all_args, 'N', incl_obs=False)
    cme_MN = compute_cme(all_args, 'MN', incl_obs=False)
    all_args.stage_1.cme_N = cme_N
    all_args.stage_1.cme_MN = cme_MN
    all_args.stage_1.lambd_n = cme_N.lambd_star
    all_args.stage_1.lambd_mn = cme_MN.lambd_star


def compute_cme_s3_main(all_args):
    cme_Xall = compute_cme(all_args, 'X', incl_obs=True)
    all_args.stage_3.cme_Xall = cme_Xall
    all_args.stage_3.lambd_xall = cme_Xall.lambd_star


def compute_f_main(all_args, which_cme_Xall):
    if which_cme_Xall == 'cme_Xall':
        f = compute_f(all_args, all_args.stage_3[which_cme_Xall])
        all_args.stage_3.f = f
        all_args.stage_3.xi = f.xi
    else:
        f = compute_f(all_args, all_args.stage_3[which_cme_Xall])  # if which_cme_Xall = cme_Mall or cme_Nall or cme_avMNall, that means we are using these as stand-in's for X.
        return f


####################################################
#### CREATE STAGE 2 SUPERVISED LEARNING DATASET ####
####################################################
def sample_from_khat(sample_size, sigma):
    samples = np.random.normal(0, 1, sample_size)
    samples = samples / 2 / np.pi / sigma
    return samples


def create_stage_2_raw_data(all_args):
    # input: (n_Chi, z) --> output: y
    # Chi = np.linspace(all_args.stage_2.Chi_lim[0], all_args.stage_2.Chi_lim[1], all_args.stage_2.n_Chi).reshape(-1,1)
    Chi = sample_from_khat(all_args.stage_2.n_Chi, all_args.stage_1.cme_N.K.sig_X).reshape(-1,1)

    ### gamma ###
    K_Zz = compute_rbf_kernel(all_args.stage_1.train.Z, all_args.stage_1.dev.Z, all_args.stage_1.cme_N.K.sig_Z)
    # gamma_N = all_args.stage_1.cme_N.brac_inv.dot(K_Zz)
    n = all_args.stage_1.cme_N.K.ZZ.shape[0]
    gamma_N = np.linalg.solve(all_args.stage_1.cme_N.K.ZZ + n * all_args.stage_1.cme_N.lambd_star * np.eye(n), K_Zz)
    # gamma_MN = all_args.stage_1.cme_MN.brac_inv.dot(K_Zz)
    gamma_MN = np.linalg.solve(all_args.stage_1.cme_MN.K.ZZ + n * all_args.stage_1.cme_MN.lambd_star * np.eye(n), K_Zz)
    #############

    ### decompose e^{i\mathcal{X}n_i} ###
    cos_term = np.cos(Chi @ all_args.stage_1.train.N.reshape(1, -1)) # shape: Chi.shape[0] x args.train.N.shape[0]
    sin_term = np.sin(Chi @ all_args.stage_1.train.N.reshape(1, -1))
    #####################################

    ### denominator ###
    denom = dotdict({})
    # using gamma to evaluate the charasteristic function value at a bunch of Chi's
    denom.cos_weighted = cos_term.dot(gamma_N)
    denom.sin_weighted = sin_term.dot(gamma_N)
    denom.value = denom.cos_weighted + denom.sin_weighted*1j
    ###################

    ### numerator ###
    numer = dotdict({})
    numer.cos_weighted = cos_term.dot(gamma_MN * all_args.stage_1.train.M.reshape(-1, 1))  # shape: Chi.shape[0] x args.dev.Z.shape[0]
    numer.sin_weighted = sin_term.dot(gamma_MN * all_args.stage_1.train.M.reshape(-1, 1))
    numer.value = numer.cos_weighted + numer.sin_weighted*1j
    #################

    train_labels_ = numer.value / denom.value
    train_labels = train_labels_.flatten().reshape(-1, 1)

    Chi_flat = np.repeat(Chi, all_args.stage_1.n_dev).reshape(-1, 1)
    # z_dev_flat = np.repeat(all_args.stage_1.dev.Z, all_args.stage_2.n_Chi, axis=1).T.flatten().reshape(-1, 1)
    z_dim = all_args.stage_1.dev.Z.shape[1]
    # breakpoint()
    Z2 = all_args.stage_1.dev.Z
    z_dev_flat = np.repeat(Z2[np.newaxis, :, :], all_args.stage_2.n_Chi, axis=0).reshape(-1, z_dim)

    raw_data2 = dotdict({})
    raw_data2.Chi = Chi_flat
    raw_data2.Z = z_dev_flat
    raw_data2.labels = train_labels

    return raw_data2


# def process_multi_dim_x(raw_data, x_latent_idx):
#     data1 = dotdict({})
#
#     for key in raw_data.keys():
#         if len(raw_data[key].shape) == 1 or (len(raw_data[key].shape)==2 and raw_data[key].shape[-1]==1):
#             data1[key] = raw_data[key]
#         elif len(raw_data[key].shape) == 2 and raw_data[key].shape[-1] > 1:
#             idx = 1
#             data1[key] = dotdict({})
#             for i in range(raw_data[key].shape[-1]):
#                 if i == x_latent_idx:
#                     data1[key]['0'] = raw_data[key][:, i]
#                 else:
#                     data1[key][str(idx)] = raw_data[key][:, i]
#                     idx += 1
#         else:
#             raise ValueError('data for each variable is either 1d or 2d numpy arrays.')
#     return data1
#
#
# def preprocess_data_main(raw_data):
#     """
#     raw_data is generated with keys 'X,Y,Z, U, M, N'. This function splits the multidim data into separate 1d data.
#
#     :param raw_data:
#     :return raw_data_sep: raw_data separated into 1d data.
#     """
#     if raw_data.merror_dim is None:
#         return raw_data
#     else:
#         return process_multi_dim_x(raw_data, raw_data.merror_dim)
#


# identify K_ZZ, K_Zz, K_XX, K_xx, K_xX

def _prepare_X_data(raw_sim):
    """
    Split up the data for X into the mismeasured dimension, still denoted 'X',
    and the remaining, observed, dimensions, denoted 'Xobs'.

    :param raw_sim: a dotdict object, with two keys: data and design.
    data contains the data for {X,Y,M,N,Z,U}
    and design contains the configuration of the simulation.
    :return: raw_data: raw data ready to be split into train dev test.
    """
    # select the dimensions in X with measurement error.
    # the following routine only works if we only have
    # one dimension in X with measurement error.
    merror_dim = raw_sim.design.merror_dim

    if merror_dim is not None:
        data0 = dotdict({})
        for key in raw_sim.data.keys():
            if key == 'X':
                data0[key] = raw_sim.data[key][:, merror_dim].reshape(-1,1)
                data0['Xall'] = raw_sim.data[key]
            else:
                data0[key] = raw_sim.data[key]
    else:
        data0 = raw_sim.data
        data0.Xobs = None
    out = dotdict({'data0': data0, 'merror_dim': merror_dim})
    return out


def prepare_pre_stage_1_args_and_data(all_args, raw_sim):
    if raw_sim.design.discrete_xdims:
        all_args.discrete_xdims = raw_sim.design.discrete_xdims
    else:
        all_args.discrete_xdims = []

    out = _prepare_X_data(raw_sim)
    # breakpoint()
    all_args.merror_dim = out.merror_dim
    curate_data(all_args, out.data0)
    # raise ValueError


def prepare_stage_1_data_and_args(all_args):
    """
    fill in stage_1_args with raw_data.
    :param s1_data_and_args: stage 1 data and arguments
    :param raw_sim: a dotdict object, with two keys: data and design
    :return: None
    """
    # stage 1 inherits all preset train and dev data
    for key in ['n_train', 'n_dev', 'train', 'dev']:
        all_args.stage_1[key] = all_args[key]


def prepare_stage_2_data_and_args(all_args, raw_data2):
    """
    s1_data_and_args: stage 2 data inherits stage 1 training data and cme's
    """

    throw_away_outliers_with_labelrealparts(raw_data2, all_args.stage_2.label_cutoff); shuffle_(raw_data2)

    curate_data(all_args.stage_2, raw_data2)

    # tensorise
    for key in raw_data2.keys():
        all_args.stage_2.train[key] = tensor(all_args.stage_2.train[key])
        all_args.stage_2.dev[key] = tensor(all_args.stage_2.dev[key])

    for key in all_args.stage_1.train.keys():
        key1 = key + '1'
        all_args.stage_2[key1] = tensor(all_args.stage_1.train[key])  # tensorise for stage 2 training

    all_args.stage_2.sig_Z1 = get_median_inter_mnist(all_args.stage_1.train.Z)
    all_args.stage_2.cme_N = all_args.stage_1.cme_N
    all_args.stage_2.cme_MN = all_args.stage_1.cme_MN


def prepare_stage_3_data_and_args(all_args, stage_two_out):
    # stage 3 inherits all keys from stage 1.
    for key in all_args.stage_1.keys():
        all_args.stage_3[key] = all_args.stage_1[key]

    if not all_args.stage_2.learn_lambd_x:
        assert stage_two_out.lambd_x is None
        all_args.stage_3.lambd_x = all_args.stage_3.lambd_n  # use lambd_n as heuristic if not learning lambd_x
    else:
        assert stage_two_out.lambd_x is not None
        all_args.stage_3.lambd_x = stage_two_out.lambd_x
    # breakpoint()
    all_args.stage_3.train.fitted_X = stage_two_out.fitted_x

    # breakpoint()



###########################################
############# STAGE 2 MODEL ###############
###########################################


class StageTwo(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        print('first 10 M: ', args.M1[:10])
        print('first 10 N: ', args.N1[:10])
        print('first 10 _X: ', args._X1[:10])
        self.x = torch.nn.Parameter((torch.tensor(args.M1) + torch.tensor(args.N1)) / 2)
        # self.x = torch.nn.Parameter((torch.tensor(args._X1) + torch.tensor(args._X1)) / 2)

        print('first 10 initialised: ', self.x[:10])
        self.args = args
        self.reg_param = args.reg_param

        if args.learn_lambd_x:
            self.cme_X.log_lambd_star = torch.nn.Parameter(torch.tensor(np.log(0.05)))
            self.cme_X.lambd_star = torch.exp(self.cme_X.log_lambd_star)
        self.cme_X = dotdict({})
        # self.cme_X.lambd_star = args.cme_N.lambd_star
        self.cme_X.lambd_star = 0.006888337
        self.cme_X.brac_inv = torch.inverse(torch.tensor(make_psd(args.cme_N.K.ZZ)) + args.M1.shape[0] * self.cme_X.lambd_star*torch.eye(args.M1.shape[0]))

    def forward(self, idx):
        ### gamma ###
        z = self.args.train.Z[idx]

        K_Zz = tensor(compute_rbf_kernel(self.args.Z1, z.numpy(), self.args.sig_Z1))
        gamma = self.cme_X.brac_inv.matmul(K_Zz)
        #############

        ### decompose e^{i\mathcal{X}n_i} ###
        cos_term = torch.cos(self.args.train.Chi[idx].matmul(self.x.reshape(1, -1)))
        sin_term = torch.sin(self.args.train.Chi[idx].matmul(self.x.reshape(1, -1)))
        #####################################

        ### denominator ###
        denom = dotdict({})
        # using gamma to evaluate the charasteristic function value at a bunch of curly_x's
        denom.cos_weighted = torch.sum(cos_term * gamma.T, dim=-1).reshape(-1, 1)
        denom.sin_weighted = torch.sum(sin_term * gamma.T, dim=-1).reshape(-1, 1)
        denom.value = denom.cos_weighted + denom.sin_weighted * 1j
        ###################

        ### numerator ###
        numer = dotdict({})
        numer.cos_weighted = torch.sum(cos_term * gamma.T * self.x.reshape(1, -1), dim=-1).reshape(-1, 1)
        numer.sin_weighted = torch.sum(sin_term * gamma.T * self.x.reshape(1, -1), dim=-1).reshape(-1, 1)
        numer.value = numer.cos_weighted + numer.sin_weighted * 1j
        #################

        return numer.value / denom.value

    def loss(self, preds, idx):
        labels = self.args.train.labels[idx]

        dim_label = labels.shape[-1]
        num_label = labels.shape[0]

        preds_as_real = torch.view_as_real(preds)
        labels_as_real = torch.view_as_real(labels)

        mse = torch.sum((labels_as_real - preds_as_real) ** 2) / num_label / dim_label
        # breakpoint()
        reg = torch.sum((self.x - ((self.args.M1 + self.args.N1) / 2)) ** 2)

        loss = mse + self.reg_param * reg

        #         if loss > 100:
        #             print(self.x[:10])
        #             breakpoint()

        return loss, mse, reg


def split_into_batches(args):
    batches_idxes = []
    idxes = np.arange(args.train.Chi.shape[0])
    print('num train data: ', len(idxes))
    np.random.shuffle(idxes)

    batch_i = 0
    while True:
        batches_idxes.append(torch.tensor(idxes[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]))
        batch_i += 1
        if batch_i * args.batch_size >= args.train.Chi.shape[0]:
            break
    return batches_idxes


def train(model, args):
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    losses = []

    early_stop = False
    step = 0
    for ep in range(args.num_epochs):
        if early_stop:
            break
        running_loss = 0.0
        batches_idx = split_into_batches(args)
        for i, batch_idx in enumerate(batches_idx):
            #             print('first 10 parameters: ', list(model.parameters())[0].detach().numpy()[:10])
            preds = model(batch_idx)
            # breakpoint()
            loss, mse, reg = model.loss(preds, batch_idx)
            # breakpoint()

            optimizer.zero_grad()

            loss.backward()
            #             print('grad values: ', model.x.grad[:10])
            optimizer.step()

            running_loss += loss.item()

            if i % 1 == 0:
                print('[epoch %d, batch %5d] loss: %.5f, mse: %.5f, reg: %.5f' % (
                ep + 1, i + 1, running_loss / 1, mse / 1, args.reg_param * reg / 1))
                # breakpoint()
                running_loss = 0.0

            losses.append(loss.item())

            if (step > 2) and np.abs(losses[-1] - losses[-2]) < 1e-8:
                early_stop = True
                break
            step += 1
    return model


def stage_two_main(args):
    model = StageTwo(args)
    model = train(model, args)

    stage_two_out = dotdict({})

    if args.learn_lambd_x:
        fitted_x = model.parameters()[:-1]
        lambd_x = model.parameters()[-1]  # todo: these are the worng syntax
    else:
        fitted_x = list(model.parameters())[0].detach().numpy()
        lambd_x = None

    assert fitted_x.shape[0] == args.Z1.shape[0]

    stage_two_out.fitted_x = fitted_x
    stage_two_out.lambd_x = lambd_x

    return stage_two_out




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='command line arguments for cmiv.py')
    parser.add_argument('--config-regression', type=str)
    command_args = parser.parse_args()

    s1_args = fill_in_args(command_args.config_path)
