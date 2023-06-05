import torch
from torch import tensor, optim
from miv.util import dotdict
import numpy as np
from miv.data.data_class import StageMDataSetTorch, TrainDataSet
from miv.models.MerrorKIV.model import MerrorKIVModel

###########################################
############# STAGE 2 MODEL ###############
###########################################


class StageMModel(torch.nn.Module):
    def __init__(self, stageM_data: StageMDataSetTorch, train_params: dotdict, stage1_MNZ: dotdict,
                 gpu_flg: bool = False):
        super().__init__()
        print('first 10 M: ', stage1_MNZ.M[:10])
        print('first 10 N: ', stage1_MNZ.N[:10])
        # self.x = torch.nn.Parameter((torch.tensor(stage1_MNZ.M) + torch.tensor(stage1_MNZ.N)) / 2)  # stage1_size x 1
        # print('first 10 initialised: ', self.x[:10])
        self.stageM_data = stageM_data
        self.stage1_MNZ = stage1_MNZ
        # breakpoint()
        self.reg_param = train_params.reg_param
        self.x_initialiser = (torch.tensor(stage1_MNZ.M) + torch.tensor(stage1_MNZ.N)) / 2

        if not train_params["lambda_x"]:
            # self.cme_X.log_lambd_star = torch.nn.Parameter(torch.tensor(np.log(0.05)))
            # self.cme_X.lambd_star = torch.exp(self.cme_X.log_lambd_star)

            self.params = torch.nn.Parameter(torch.cat([self.x_initialiser.flatten(), torch.tensor([np.log(0.05)])], dim=0))  # stage1_size x 1
            self.x = self.params[:-1].reshape(-1, 1)
            self.lambda_x = self.params[-1]
        else:
            self.params = torch.nn.Parameter(self.x_initialiser.flatten())
            self.x = self.params.reshape(-1,1)
            self.lambda_x = train_params["lambda_x"]
        print('first 10 initialised: ', self.x[:10])
        self.train_params = train_params
        self.KZ1Z1 = tensor(MerrorKIVModel.cal_gauss(stage1_MNZ.Z, stage1_MNZ.Z, stage1_MNZ.sigmaZ))

    def forward(self, idx):
        ### gamma ###
        n = self.stage1_MNZ.Z.shape[0]
        z = self.stageM_data.Z[idx]
        K_Z1z = MerrorKIVModel.cal_gauss(torch.tensor(self.stage1_MNZ.Z), z, self.stage1_MNZ.sigmaZ)
        # gamma = self.cme_X.brac_inv.matmul(K_Zz)
        # breakpoint()
        if not self.train_params["lambda_x"]:
            gamma_x = torch.linalg.solve(self.KZ1Z1 + n * torch.exp(self.lambda_x) * torch.eye(n), K_Z1z)
            # gamma_x = torch.linalg.solve(self.KZ1Z1 + n * self.lambda_x * torch.eye(n), K_Z1z)
        else:
            gamma_x = torch.linalg.solve(self.KZ1Z1 + n * self.lambda_x * torch.eye(n), K_Z1z)

        # gamma_x = torch.linalg.solve(self.KZ1Z1 + n * self.other1 * torch.eye(n), K_Z1z)

        #############

        ### decompose e^{i\mathcal{X}n_i} ###
        cos_term = torch.cos(self.stageM_data.Chi[idx].matmul(self.x.reshape(1, -1)))
        sin_term = torch.sin(self.stageM_data.Chi[idx].matmul(self.x.reshape(1, -1)))
        #####################################

        ### denominator ###
        denom = dotdict({})
        # using gamma to evaluate the charasteristic function value at a bunch of curly_x's
        denom.cos_weighted = torch.sum(cos_term * gamma_x.t(), dim=-1).reshape(-1, 1)
        denom.sin_weighted = torch.sum(sin_term * gamma_x.t(), dim=-1).reshape(-1, 1)
        denom.value = denom.cos_weighted + denom.sin_weighted * 1j
        ###################

        ### numerator ###
        numer = dotdict({})
        numer.cos_weighted = torch.sum(cos_term * gamma_x.t() * self.x.reshape(1, -1), dim=-1).reshape(-1, 1)
        numer.sin_weighted = torch.sum(sin_term * gamma_x.t() * self.x.reshape(1, -1), dim=-1).reshape(-1, 1)
        numer.value = numer.cos_weighted + numer.sin_weighted * 1j
        #################

        return numer.value / denom.value

    def loss(self, preds, idx):
        labels = self.stageM_data.labels[idx]

        dim_label = labels.shape[-1]
        num_label = labels.shape[0]

        preds_as_real = torch.view_as_real(preds)
        labels_as_real = torch.view_as_real(labels)

        mse = torch.sum((labels_as_real - preds_as_real) ** 2) / num_label / dim_label
        # breakpoint()
        reg = torch.sum((self.x - (torch.tensor(self.stage1_MNZ.M + self.stage1_MNZ.N) / 2)) ** 2)

        loss = mse + self.reg_param * reg

        return loss, mse, reg


def split_into_batches(stageM_data: StageMDataSetTorch, stageM_args: dotdict):
    batches_idxes = []
    idxes = np.arange(stageM_data.Chi.shape[0])
    print('num train data: ', len(idxes))
    np.random.shuffle(idxes)

    batch_i = 0
    while True:
        batches_idxes.append(torch.tensor(idxes[batch_i * stageM_args.batch_size: (batch_i + 1) * stageM_args.batch_size]))
        batch_i += 1
        if batch_i * stageM_args.batch_size >= stageM_data.Chi.shape[0]:
            break
    return batches_idxes


def stage_m_train(model: StageMModel, stageM_data: StageMDataSetTorch, stageM_args: dotdict):
    model.train()
    # itertools.chain(list(self.iv_net_hidden.parameters()),
    #                 list(self.m_decoder.parameters()),
    #                 list(self.n_decoder.parameters()),
    #                 list(self.encoder.parameters()))
    optimizer = optim.Adam(model.parameters(), lr=stageM_args.lr)

    losses = []

    early_stop = False
    step = 0
    for ep in range(stageM_args.num_epochs):
        if early_stop:
            break
        running_loss = 0.0
        batches_idx = split_into_batches(stageM_data=stageM_data, stageM_args=stageM_args)
        for i, batch_idx in enumerate(batches_idx):
            #             print('first 10 parameters: ', list(model.parameters())[0].detach().numpy()[:10])
            preds = model(batch_idx)
            loss, mse, reg = model.loss(preds, batch_idx)

            optimizer.zero_grad()
            # breakpoint()
            # loss.backward(retain_graph=True)
            loss.backward()
            #             print('grad values: ', model.x.grad[:10])
            optimizer.step()

            running_loss += loss.item()

            if i % 1 == 0:
                print('[epoch %d, batch %5d] loss: %.5f, mse: %.5f, reg: %.5f' % (
                ep + 1, i + 1, running_loss / 1, mse / 1, stageM_args.reg_param * reg / 1))
                # breakpoint()
                running_loss = 0.0

            losses.append(loss.item())

            if step > 8000:
                break
            if (step > 2) and np.abs(losses[-1] - losses[-2]) < 1e-7:
                early_stop = True
                break
            step += 1
    return model



