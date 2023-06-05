import numpy as np
import torch
from torch import optim
from logging import getLogger
from typing import Dict, Any, Optional
from pathlib import Path

from miv.models.LVM.nn_structure import build_extractor
from miv.data import generate_train_data, generate_test_data
from miv.data.data_class import TrainDataSet, TrainDataSetTorch, TestDataSetTorch
from miv.models.LVM.model import LatentVariableModel

logger = getLogger()


def split_into_batches(train_size, batch_size):
    batches_idxes = []
    idxes = np.arange(train_size)
    np.random.shuffle(idxes)
    batch_i = 0
    while True:
        batches_idxes.append(torch.tensor(idxes[batch_i * batch_size: (batch_i + 1) * batch_size]))
        batch_i += 1
        if batch_i * batch_size >= train_size:
            break
    return batches_idxes


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
        self.sample_size_from_pxz = train_params["sample_size_from_pxz"]



    def train(self, rand_seed: int=42, verbose: int=0):
        self.train_data = generate_train_data(rand_seed=rand_seed, **self.data_config)
        self.test_data = generate_test_data(**self.data_config)
        self.train_data_t = TrainDataSetTorch.from_numpy(train_data=self.train_data)
        self.test_data_t = TestDataSetTorch.from_numpy(test_data=self.test_data)

        self.lvm = build_extractor(self.data_config["data_name"], data=self.train_data_t,
                                   sample_size_from_pxz=self.sample_size_from_pxz)

        test_input = self.test_data.X_all
        if self.test_data.covariate is not None:
            test_input = np.concatenate([test_input, self.test_data.covariate], axis=-1)
        Y_struct = self.test_data.Y_struct

        try:
            self.train1(self.train_data_t, self.batch_size)
            self.train2(self.train_data_t, self.batch_size)
        except:
            # breakpoint()
            return np.nan, test_input, np.nan * np.ones((test_input.shape[0], 1)), Y_struct

        self.lvm.response.eval()
        response_model = self.lvm.response
        mdl = LatentVariableModel(response_model, self.data_config["data_name"])
        if self.gpu_flg:
            torch.cuda.empty_cache()
        oos_loss, preds = mdl.evaluate_t(self.test_data_t)
        return oos_loss, test_input, preds, Y_struct

    def train1(self, train_data, batch_size):
        self.lvm.train()
        self.lvm.double()
        optimizer = optim.Adam(self.lvm.parameters(), lr=1e-3)
        losses = []
        step = 0
        for ep in range(self.n_epochs):
            running_loss = 0.0
            batches_idx = split_into_batches(train_size=train_data.Z.shape[0], batch_size=batch_size)
            for i, batch_idx in enumerate(batches_idx):
                # breakpoint()
                loss = self.lvm.stage_1_loss(batch_idx)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if i % 1 == 0:
                    print('[epoch %d, batch %5d] loss: %.5f' % (
                        ep + 1, i + 1, running_loss / 1))
                    # breakpoint()
                    running_loss = 0.0

                losses.append(loss.item())

                step += 1


    def train2(self, train_data, batch_size):
        self.lvm.eval()
        self.lvm.response.train()
        self.lvm.double()
        optimizer = optim.Adam(self.lvm.response.parameters(), lr=1e-3)
        losses = []
        step = 0
        for ep in range(self.n_epochs):
            running_loss = 0.0
            batches_idx = split_into_batches(train_size=train_data.Z.shape[0], batch_size=batch_size)
            for i, batch_idx in enumerate(batches_idx):
                loss = self.lvm.stage_2_loss(batch_idx)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if i % 1 == 0:
                    print('[epoch %d, batch %5d] loss: %.5f' % (
                        ep + 1, i + 1, running_loss / 1))
                    running_loss = 0.0

                losses.append(loss.item())

                step += 1
