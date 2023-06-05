from typing import Dict, Any, Optional
from pathlib import Path
import logging

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Normal, OneHotCategorical
from torch.autograd import detect_anomaly
import numpy as np
import itertools


from src.models.DeepIV.nn_structure import build_extractor
from src.data import generate_train_data, generate_test_data
from src.data import preprocess
from src.data.data_class import TrainDataSet, TrainDataSetTorch, TestDataSetTorch
from src.models.DeepIV.model import DeepIVModel

logger = logging.getLogger()


class DeepIVTrainer(object):

    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any],
                 gpu_flg: bool = False, dump_folder: Optional[Path] = None):
        self.data_config = data_configs
        self.gpu_flg = gpu_flg and torch.cuda.is_available()
        if self.gpu_flg:
            logger.info("gpu mode")
        # configure training params
        self.n_epochs = train_params["n_epoch"]
        self.batch_size = train_params["batch_size"]
        self.n_sample = train_params["n_sample"]

        dropout_rate = min(1000. / (1000. + data_configs["data_size"]), 0.5)
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
            self.mdn_opt = torch.optim.Adam(self.iv_net_obs.parameters(),
                                            weight_decay=0.001)

        self.lvm_opt = torch.optim.Adam(itertools.chain(list(self.iv_net_hidden.parameters()),
                                                        list(self.m_decoder.parameters()),
                                                        list(self.n_decoder.parameters()),
                                                        list(self.encoder.parameters())),
                                                        weight_decay=0.0001)

        self.response_opt = torch.optim.Adam(self.response_net.parameters(),
                                             weight_decay=0.001)

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
            self.update_stage2(train_data_t, verbose)
        except RuntimeError:
            return np.nan

        self.response_net.train(False)
        if self.iv_net_obs:
            self.iv_net_obs.train(False)
        self.iv_net_hidden.train(False)
        mdl = LatentVariableModel(self.response_net, self.iv_net_hidden, self.iv_net_obs, self.data_config["data_name"])
        if self.gpu_flg:
            torch.cuda.empty_cache()
        oos_loss: float = mdl.evaluate_t(test_data_t).data.item()
        return oos_loss

    @staticmethod
    def density_est_loss(cat: OneHotCategorical, norm: Normal, treatment: torch.Tensor) -> torch.Tensor:
        assert treatment.size() == norm.mean.size()[:2]
        loglik = norm.log_prob(treatment.unsqueeze(2).expand_as(norm.mean))
        loglik = torch.sum(loglik, dim=1)
        loglik = torch.clamp(loglik, min=-40)
        loss = -torch.logsumexp(cat.logits + loglik, dim=1)
        return torch.sum(loss)

    def update_stage1(self, train_data: TrainDataSetTorch, verbose: int):
        # breakpoint()

        if train_data.X_obs:
            data_set = TensorDataset(train_data.Z, train_data.M, train_data.N, train_data.X_obs)
        else:
            data_set = TensorDataset(train_data.Z, train_data.M, train_data.N)
        loss_val_lvm, loss_val_mdn = None, None
        for t in range(self.n_epochs):
            data_loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=True)

            for data in data_loader:
                # instrumental, m_obs, n_obs
                iv, m_obs, n_obs = data[0], data[1], data[2]
                sample_size = 10

                self.lvm_opt.zero_grad()

                qxzmn = self.encoder(instrument=iv, m_obs=m_obs, n_obs=n_obs)
                x_samples_from_q = qxzmn.sample(sample_shape=(sample_size,)) # size: sample_size x B x 1
                m_dist = self.m_decoder(x_samples_from_q.reshape(-1,1)) # p(m|x)
                n_dist = self.n_decoder(x_samples_from_q.reshape(-1,1))
                m_obs_reps = m_obs.repeat((sample_size, 1)).reshape(-1,1) # repeat by first axis sample_size times.
                n_obs_reps = n_obs.repeat((sample_size, 1)).reshape(-1,1)

                # breakpoint()
                clamp_val = 40
                log_pmx = torch.clamp(m_dist.log_prob(m_obs_reps), min=-1*clamp_val, max=clamp_val)
                log_pnx = torch.clamp(n_dist.log_prob(n_obs_reps), min=-1*clamp_val, max=clamp_val)
                recon_m_loss = torch.mean(log_pmx)
                recon_n_loss = torch.mean(log_pnx)
                # print("1")
                # kld = E[log(p(x|z)) - log(q(x|z,m,n))]_q
                # breakpoint()
                pxz, qxzmn = self.iv_net_hidden(iv), self.encoder(iv, m_obs, n_obs)
                # print("2")
                log_pxz = torch.mean(torch.clamp(pxz.log_prob(x_samples_from_q), min=clamp_val))
                # print("3")
                log_qxzmn = torch.mean(torch.clamp(qxzmn.log_prob(x_samples_from_q), min=clamp_val))
                # print("4")
                kld = log_pxz - log_qxzmn
                # print("5")
                loss_val_lvm = (recon_m_loss + recon_n_loss + kld) * -1
                # print("6")
                # breakpoint()


                try:
                    with detect_anomaly():
                        loss_val_lvm.backward()
                        logger.info(f"managed backward")
                        self.lvm_opt.step()
                        logger.info(f"stage 1 hidden step taken, loss_val_lvm: {loss_val_lvm.item()}")
                except RuntimeError:
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
        for t in range(self.n_epochs):
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

            if verbose >= 2 and loss_val is not None:
                logger.info(f"stage2 learning: {loss_val.item()}")
