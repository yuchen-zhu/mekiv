from typing import Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal, OneHotCategorical
from torch.autograd import detect_anomaly
import numpy as np
import itertools
import os


from miv.models.LVM.nn_structure import build_extractor
from miv.data import generate_train_data, generate_test_data
from miv.data import preprocess
from miv.data.data_class import TrainDataSet, TrainDataSetTorch, TestDataSetTorch
from miv.models.LVM.model_old import LatentVariableModel

logger = logging.getLogger()



def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)


class LVMTrainer:

    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any],
                 gpu_flg: bool=False, dump_folder: Optional[Path]=None):

        self.data_config = data_configs
        self.gpu_flg = gpu_flg and torch.cuda.is_available()
        if self.gpu_flg:
            logger.info("gpu mode")
        # configure training params
        self.n_epochs = train_params["n_epochs"]
        self.batch_size = train_params["batch_size"]
        self.n_sample = train_params["n_sample"]

        dropout_rate = min(1000. /(1000. + data_configs["data_size"]), 0.5)
        args = dict(dropout_rate=dropout_rate)
        networks = build_extractor(data_configs["data_name"], **args)
        self.iv_net_hidden = networks.iv_net_hidden
        self.iv_net_obs = networks.iv_net_obs
        self.m_decoder = networks.m_decoder
        self.n_decoder = networks.n_decoder
        self.encoder = networks.encoder
        self.response_net = networks.response_net

        if self.gpu_flg:
            self.iv_net_hidden.to("cuda:0")
            self.m_decoder.to("cuda:0")
            self.n_decoder.to("cuda:0")
            self.encoder.to("cuda:0")
            self.response_net.to("cuda:0")
            if self.iv_net_obs:
                self.iv_net_obs.to("cuda:0")



        if self.iv_net_obs:
            self.iv_net_obs.apply(init_weights)
            self.mdn_opt = torch.optim.Adam(self.iv_net_obs.parameters(),
                                            weight_decay=0.001)

        from miv.models.LVM.set_weights_biases import set_iv_net_hidden, set_mn_decoder, set_encoder
        self.iv_net_hidden.apply(init_weights)
        # set_iv_net_hidden(self.iv_net_hidden)
        self.m_decoder.apply(init_weights)
        # set_mn_decoder(self.m_decoder)
        self.n_decoder.apply(init_weights)
        # set_mn_decoder(self.n_decoder)
        self.encoder.apply(init_weights)
        # set_encoder(self.encoder)
        self.response_net.apply(init_weights)
        breakpoint()

        self.lvm_opt = torch.optim.Adam(itertools.chain(list(self.iv_net_hidden.parameters()),
                                                        list(self.m_decoder.parameters()),
                                                        list(self.n_decoder.parameters()),
                                                        list(self.encoder.parameters())),
                                                        weight_decay=0.0001)
        # self.lvm_opt = torch.optim.Adam(itertools.chain(list(self.iv_net_hidden.parameters()),
        #                                                 list(self.m_decoder.parameters()),
        #                                                 list(self.encoder.parameters())),
        #                                                 lr=1e-4,
        #                                                 weight_decay=0.0001)

        self.response_opt = torch.optim.Adam(self.response_net.parameters(), lr=1e-3,
                                             weight_decay=0.001)

        self.checkpt = None
        if train_params["from_checkpt"] is not None:
            checkpt = torch.load(os.path.join('runs', train_params["from_checkpt"]))
            self.checkpt = checkpt
            self.iv_net_hidden.load_state_dict(checkpt['iv_net_hidden_state_dict'])
            self.m_decoder.load_state_dict(checkpt['m_decoder_state_dict'])
            self.n_decoder.load_state_dict(checkpt['n_decoder_state_dict'])
            self.encoder.load_state_dict(checkpt['encoder_state_dict'])
            self.lvm_opt.load_state_dict(checkpt['lvm_opt_state_dict'])
            self.epoch = checkpt['epoch']

        self.writer = SummaryWriter()

    def train(self, rand_seed: int=42, verbose: int=0) -> float:

        # print("enter training")
        train_data = generate_train_data(rand_seed=rand_seed, **self.data_config)
        test_data = generate_test_data(**self.data_config)
        train_data_t = TrainDataSetTorch.from_numpy(train_data)
        test_data_t = TestDataSetTorch.from_numpy(test_data)
        if self.gpu_flg:
            train_data_t = train_data_t.to_gpu()
            test_data_t = test_data_t.to_gpu()

        try:
            self.update_stage1(train_data_t, verbose)
            # breakpoint()
            self.update_stage2(train_data_t, verbose)
        except RuntimeError:
            # breakpoint()
            return np.nan

        self.writer.close()

        self.response_net.train(False)
        if self.iv_net_obs:
            self.iv_net_obs.train(False)
        self.iv_net_hidden.train(False)

        test_input = test_data.X_all
        if test_data.covariate is not None:
            test_input = np.concatenate([test_input, test_data.covariate], axis=-1)
        Y_struct = test_data.Y_struct

        mdl = LatentVariableModel(self.response_net, self.iv_net_hidden, self.iv_net_obs, self.data_config["data_name"])
        if self.gpu_flg:
            torch.cuda.empty_cache()
        oos_loss, preds = mdl.evaluate_t(test_data_t)
        return oos_loss, test_input, preds, Y_struct

    @staticmethod
    def density_est_loss(cat: OneHotCategorical, norm: Normal, treatment: torch.Tensor) -> torch.Tensor:
        assert treatment.size() == norm.mean.size()[:2]
        loglik = norm.log_prob(treatment.unsqueeze(2).expand_as(norm.mean))
        loglik = torch.sum(loglik, dim=1)
        loglik = torch.clamp(loglik, min=-100)
        loss = -torch.logsumexp(cat.logits + loglik, dim=1)
        return torch.sum(loss)

    def update_stage1(self, train_data: TrainDataSetTorch, verbose: int):
        # breakpoint()

        if train_data.X_obs:
            data_set = TensorDataset(train_data.Z, train_data.M, train_data.N, train_data.X_obs)
        else:
            data_set = TensorDataset(train_data.Z, train_data.M, train_data.N)
        loss_val_lvm, loss_val_mdn = None, None

        hidden_stps, obs_stps = 0, 0
        if self.checkpt is not None:
            hidden_stps, obs_stps = self.checkpt["hidden_stps"], self.checkpt["obs_stps"]
        for t in range(self.n_epochs):
            # if t == 5 - 1:
            #     breakpoint()
            if t % 100 == 0:
                torch.save({
                    'epoch': t,
                    'hidden_stps': hidden_stps,
                    'obs_stps': obs_stps,
                    'iv_net_hidden_state_dict': self.iv_net_hidden.state_dict(),
                    'm_decoder_state_dict': self.m_decoder.state_dict(),
                    'n_decoder_state_dict': self.n_decoder.state_dict(),
                    'encoder_state_dict': self.encoder.state_dict(),
                    'lvm_opt_state_dict': self.lvm_opt.state_dict()},
                    'runs/model{}.pt'.format(str(datetime.now().strftime("%m-%d-%H-%M-%S"))))
                # breakpoint()
            if t == self.n_epochs - 1:
                breakpoint()
            data_loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=True)

            for data in data_loader:
                # instrumental, m_obs, n_obs
                iv, m_obs, n_obs = data[0], data[1], data[2]  # shape: B x 1, B x 1, B x 1
                sample_size = 100

                self.lvm_opt.zero_grad()
                # breakpoint()
                qxzmn = self.encoder(instrument=iv, m_obs=m_obs, n_obs=n_obs)

                # ELBO = - E_q[log(p(x,m,n|z)/q(x|m,n,z))] = E_q[q] - E_q[log(p(x|z))] - E_q[log(p(m|x))] - E_q(log(p(n|x)))

                # Z --> X p(z) p(x|z) likelihood(X)  Free energy = E_q[log(p_theta(x,z)/q(z|x))]

                x_samples_from_q = qxzmn.sample(sample_shape=(sample_size,)) # size: sample_size x B x 1
                m_dist = self.m_decoder(x_samples_from_q.reshape(-1,1))  # p(m|x)
                n_dist = self.n_decoder(x_samples_from_q.reshape(-1,1))
                m_obs_reps = m_obs.repeat((sample_size, 1)).reshape(-1,1)  # Todo: check this. repeat by first axis sample_size times.
                n_obs_reps = n_obs.repeat((sample_size, 1)).reshape(-1,1)

                # breakpoint()
                clamp_val = 1e16
                log_pmx = torch.clamp(m_dist.log_prob(m_obs_reps), min=-1*clamp_val, max=clamp_val)
                log_pnx = torch.clamp(n_dist.log_prob(n_obs_reps), min=-1*clamp_val, max=clamp_val)

                recon_m_prob = torch.mean(log_pmx)
                recon_n_prob = torch.mean(log_pnx)

                # kld = E[log(p(x|z)) - log(q(x|z,m,n))]_q
                # breakpoint()
                pxz = self.iv_net_hidden(iv)
                # print("2")
                log_pxz = torch.mean(torch.clamp(pxz.log_prob(x_samples_from_q), min=-1*clamp_val, max=clamp_val))
                # print("3")
                # log_qxzmn = torch.mean(torch.clamp(qxzmn.log_prob(x_samples_from_q), min=-1*clamp_val, max=clamp_val))
                h_q = torch.mean(0.5*torch.log((qxzmn.scale**2)*2*torch.pi))
                # print("4")
                # kld = log_pxz - log_qxzmn # make H[q] analytical
                kld = log_pxz + h_q
                # print("5")
                loss_val_lvm = (recon_m_prob + recon_n_prob + kld) * -1
                # loss_val_lvm = (recon_m_loss + kld) * -1
                # print("6")
                # breakpoint()

                ###################################
                ##### compute true likelihood #####
                ###################################
                # pmnz = \int p(m,n|x,z) p(x|z) dx = E_{p(x|z)}[p(m,n|x)]
                from scipy import stats
                iv_reps = iv.repeat((sample_size, 1)).reshape(-1, 1)
                # x_samples_from_true_pxz = stats.norm.cdf(stats.norm.ppf(iv_reps.numpy()) / np.sqrt(2) + np.random.normal(0, 1, size=iv_reps.shape[0])[:, np.newaxis])
                x_samples_from_true_pxz = iv_reps.numpy() + np.random.normal(0, 0.1, size=iv_reps.shape[0])[:, np.newaxis]

                std_X = torch.tensor(np.std(train_data.X_hidden.numpy()))

                true_pmx_dist = Normal(loc=torch.tensor(x_samples_from_true_pxz), scale=self.data_config['m_scale']*std_X.repeat((iv_reps.shape[0], 1)))
                true_pnx_dist = Normal(loc=torch.tensor(x_samples_from_true_pxz), scale=self.data_config['m_scale']*std_X.repeat((iv_reps.shape[0], 1)))
                true_log_pmx = torch.log(torch.mean(torch.exp(true_pmx_dist.log_prob(m_obs_reps)).reshape(sample_size, -1), dim=0))
                true_log_pnx = torch.log(torch.mean(torch.exp(true_pnx_dist.log_prob(n_obs_reps)).reshape(sample_size, -1), dim=0))

                true_loglikelihood = -1 * (torch.mean(true_log_pmx) + torch.mean(true_log_pnx))
                # breakpoint()

                # breakpoint()
                self.writer.add_scalar('stage1/hidden/true', true_loglikelihood, hidden_stps)

                try:
                    # breakpoint()
                    with detect_anomaly():
                        loss_val_lvm.backward()
                        # breakpoint()
                        # gradient_iv_mean = torch.mean(self.iv_net_hidden.weight.grad)
                        # gradient_m_mean = torch.mean(self.m_decoder.weight.grad)
                        # gradient_q_mean = torch.mean(self.encoder.weight.grad)
                        logger.info(f"managed backward")
                        # breakpoint()
                        self.lvm_opt.step()
                        logger.info(f"stage 1 hidden step taken, loss_val_lvm: {loss_val_lvm.item()}")
                        self.writer.add_scalar('stage1/hidden/train', loss_val_lvm.item(), hidden_stps)
                        # self.writer.add_scalar('stage1/hidden/log_qxzmn', log_qxzmn.item(), hidden_stps)
                        self.writer.add_scalar('stage1/hidden/log_pmx', recon_m_prob.item(), hidden_stps)
                        self.writer.add_scalar('stage1/hidden/log_pnx', recon_n_prob.item(), hidden_stps)
                        self.writer.add_scalar('stage1/hidden/log_pxz', log_pxz.item(), hidden_stps)
                        # self.writer.add_scalar('stage1/hidden/true', true_loglikelihood.item(), hidden_stps)
                        # if loss_val_lvm.item() < 2.9:
                        #     breakpoint()
                        # self.writer.add_scalar('stage1/hidden/testest', 1., hidden_stps)
                        # self.writer.add_scalar('stage1/hidden/grad_iv', gradient_iv_mean.item(), hidden_stps)
                        # self.writer.add_scalar('stage1/hidden/grad_m', gradient_m_mean.item(), hidden_stps)
                        # self.writer.add_scalar('stage1/hidden/grad_q', gradient_q_mean.item(), hidden_stps)

                        hidden_stps += 1
                except RuntimeError:
                    # breakpoint()
                    logger.info("NaN detected in stage 1 hidden, skipping batch")
                    # breakpoint()
            # breakpoint()

            if self.iv_net_obs:
                for data in data_loader:
                    iv, X_obs = data[0], data[3]
                    self.iv_net_obs.zero_grad()
                    norm, cat = self.iv_net_obs(iv)
                    X_obs = preprocess.rescale_treatment(X_obs, self.data_config["data_name"])
                    loss_val_mdn = self.density_est_loss(cat, norm, X_obs)
                    try:
                        with detect_anomaly():
                            loss_val_mdn.backward()
                            self.iv_net_obs.step()
                            logger.info(f"stage 1 obs step taken, loss_val_lvm: {loss_val_mdn.item()}")
                            self.writer.add_scalar('stage1/obs/train', loss_val_mdn.item(), obs_stps)
                            obs_stps += 1
                    except RuntimeError:
                        logger.info("NaN detected in stage 1 obs, skipping batch")

            if verbose >= 2:
                loss_val_lvm_item = loss_val_lvm.item() if loss_val_lvm else "NaN"
                loss_val_mdn_item = loss_val_mdn.item() if loss_val_mdn else "NaN"
                logger.info(f"stage1 learning. lvm: {loss_val_lvm_item}, mdn: {loss_val_mdn_item}")

    def update_stage2(self, train_data: TrainDataSetTorch, verbose: int):
        data_set = TensorDataset(train_data.Z, train_data.Y)
        loss = nn.MSELoss()
        if train_data.covariate is not None:
            data_set = TensorDataset(train_data.Z,
                                     train_data.covariate,
                                     train_data.Y)
        loss_val = None
        if self.iv_net_obs:
            self.iv_net_obs.train(False)
        self.iv_net_hidden.train(False)

        s2_stps = 0
        for t in range(self.n_epochs+300):
        # for t in range(self.n_epochs-100):
            data_loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=True)
            for data in data_loader:
                self.response_opt.zero_grad()
                instrumental = data[0]
                outcome = data[-1]
                covariate = None
                if train_data.covariate is not None:
                    covariate = data[1]
                outcome = preprocess.rescale_outcome(outcome, self.data_config["data_name"])
                outcome = outcome.repeat((self.n_sample, 1))
                norm_obs, cat_obs = None, None
                if self.iv_net_obs:
                    norm_obs, cat_obs = self.iv_net_obs(instrumental)
                    if torch.sum(torch.isnan(cat_obs.probs)):
                        raise RuntimeError("NaN prob detected")

                norm_hidden = self.iv_net_hidden(instrumental)

                pred = LatentVariableModel.sample_from_density(self.n_sample, self.response_net, norm_hidden,
                                                       norm_obs, cat_obs, covariate)
                loss_val = loss(pred, outcome)
                loss_val.backward()
                self.response_opt.step()
                self.writer.add_scalar('stage2/train', loss_val.item(), s2_stps)
                s2_stps += 1

            if verbose >= 2 and loss_val is not None:
                logger.info(f"stage2 learning: {loss_val.item()}")
