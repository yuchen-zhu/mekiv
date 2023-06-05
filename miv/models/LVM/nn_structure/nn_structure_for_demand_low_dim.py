import torch
from torch import nn


# likelihood models
#
# X - {P, T, S}
# Z - {E, T, S}
# Y
# M
# N


class Encoder(torch.nn.Module):
    # q({x}|{m},{n},{z}) = gaussian(q0 + q1*{m} + q2*{n} + q3*{z},
    #                       exp(qq0 + qq1*{m} + qq2*{n} + qq3*{z})^2)
    # H[q] = -E_q[log(q)] = log(prod(sigma)*(2*pi*e)**(B*0.5))
    def __init__(self, data):
        super().__init__()
        self.q_nonlinear = nn.Sequential(nn.Linear(5, 10), nn.ReLU())
        #         self.q_mean_fc = nn.Linear(3, 1)
        #         self.q_logscale_fc = nn.Linear(3, 1)
        self.q_mean_fc = nn.Linear(10, 1)
        self.q_logscale_fc = nn.Linear(10, 1)
        self.data = data

    def forward(self, idx):
        B = idx.shape[0]
        mnz = torch.cat([self.data.M[idx], self.data.N[idx], self.data.Z[idx]], axis=1)
        q_feature = self.q_nonlinear(mnz.double())
        q_mean = self.q_mean_fc(q_feature)
        #         q_mean = self.q_mean_fc(mnz.double())
        q_logscale = self.q_logscale_fc(q_feature)
        #         q_logscale = self.q_logscale_fc(mnz)
        q_scale = torch.exp(q_logscale)

        H_q = torch.log(torch.prod(q_scale) * (2 * torch.pi * torch.e) ** (B * 0.5))

        return q_mean, q_scale, H_q


class MDecoder(torch.nn.Module):
    # p(m|x) = gaussian(x, exp(mm0 + mm1 * x)^2)
    def __init__(self, data):
        super().__init__()
        self.m_logscale_fc = nn.Linear(1, 1)
        #         self.mm1 = torch.nn.Parameter(torch.tensor([[0.]]).double())
        #         self.mm0 = torch.nn.Parameter(torch.log(torch.tensor([[0.4]])).double())
        self.data = data

    def forward(self, x):
        pmx_mean = x
        pmx_logscale = self.m_logscale_fc(x.double())
        #         pmx_logscale = self.mm0 + x.matmul(self.mm1)
        pmx_scale = torch.exp(pmx_logscale)

        return pmx_mean, pmx_scale


class NDecoder(torch.nn.Module):
    # p(n|x) = gaussian(x, exp(nn0 + nn1 * x)^2)
    def __init__(self, data):
        super().__init__()
        self.n_logscale_fc = nn.Linear(1, 1)
        #         self.nn1 = torch.nn.Parameter(torch.tensor([[0.]]).double())
        #         self.nn0 = torch.nn.Parameter(torch.log(torch.tensor([[0.4]])).double())
        self.data = data

    def forward(self, x):
        pnx_mean = x
        pnx_logscale = self.n_logscale_fc(x.double())
        #         pnx_logscale = self.nn0 + x.matmul(self.nn1)
        pnx_scale = torch.exp(pnx_logscale)

        return pnx_mean, pnx_scale


class IVModel(torch.nn.Module):
    # p(x|z) = gaussian(z, exp(zz0 + zz1 * x)^2)
    def __init__(self, data):
        super().__init__()
        self.z_feature = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 6), nn.ReLU())
        self.z_mean_fc = nn.Linear(6, 1)
        self.z_logscale_fc = nn.Linear(6, 1)
        #         self.zz1 = torch.nn.Parameter(torch.tensor([[3.]]).double())
        #         self.zz0 = torch.nn.Parameter(torch.log(torch.tensor([[1.]])).double())
        self.data = data

    def forward(self, idx):
        z = self.data.Z[idx]
        z_feature = self.z_feature(z.double())
        pxz_mean = self.z_mean_fc(z_feature)
        pxz_logscale = self.z_logscale_fc(z_feature)
        #         pxz_logscale = self.zz0 + z.matmul(self.zz1)
        pxz_scale = torch.exp(pxz_logscale)

        return pxz_mean, pxz_scale


class IVModelObs(torch.nn.Module):
    pass


class ResponseModel(torch.nn.Module):
    # y = f(x) + noise
    def __init__(self):
        super().__init__()
        self.response_net = nn.Sequential(nn.Linear(3, 3),
                                          nn.ReLU(),
                                          nn.Linear(3, 3),
                                          nn.ReLU(),
                                          nn.Linear(3, 3),
                                          nn.ReLU(),
                                          nn.Linear(3, 1))

    #         self.response_net = nn.Sequential(nn.Linear(1, 1))
    def forward(self, x):
        return self.response_net(x)


class LVM(torch.nn.Module):
    def __init__(self, data, sample_size_from_pxz):
        super().__init__()
        self.ivm = IVModel(data)
        self.ndecoder = NDecoder(data)
        self.mdecoder = MDecoder(data)
        self.encoder = Encoder(data)
        self.response = ResponseModel()
        self.data = data
        self.sample_size_from_pxz = sample_size_from_pxz

    def stage_1_loss(self, idx):
        ### Free energy = E_q[log(p({m},{n},{x}|{z})) - log(q)]
        ###             = E_q[log(p({m},{n},{x}|{z}))] + H_q({m},{n},{z})
        ###             = E_q[sum_i{log(p(m_i, n_i, x_i|z_i))}] + H_q({m}, {n}, {z})
        ###             = E_q[sum_i{log(p(m_i|x_i))} + sum_i{log(p(n_i|x_i))} + sum_i{log(p(x_i|z_i))}]
        ###               + H_q({m},{n},{z})
        ###             = sum_i{E_q[log(p(m_i|x_i))]}
        ###               + sum_i{E_q[log(p(n_i|x_i))]}
        ###               + sum_i{E_q[log(p(x_i|z_i))]}
        ###               + H_q({m},{n},{z})
        ### ELBO = - sum_i{E_q[log(p(m_i|x_i))]}
        ###        - sum_i{E_q[log(p(n_i|x_i))]}
        ###        - sum_i{E_q[log(p(x_i|z_i))]}
        ###        - H_q({m},{n},{z})

        q_mean, q_scale, H_q = self.encoder(idx)
        q_dist = torch.distributions.Normal(q_mean, q_scale)
        x_samples_from_q = q_dist.rsample()

        pmx_mean, pmx_scale = self.mdecoder(x_samples_from_q)
        pmx_dist = torch.distributions.Normal(pmx_mean, pmx_scale)
        sum_pmx = torch.sum(pmx_dist.log_prob(self.data.M[idx]))

        pnx_mean, pnx_scale = self.ndecoder(x_samples_from_q)
        pnx_dist = torch.distributions.Normal(pnx_mean, pnx_scale)
        sum_pnx = torch.sum(pnx_dist.log_prob(self.data.N[idx]))

        pxz_mean, pxz_scale = self.ivm(idx)
        pxz_dist = torch.distributions.Normal(pxz_mean, pxz_scale)
        sum_pxz = torch.sum(pxz_dist.log_prob(x_samples_from_q))

        loss = -sum_pmx - sum_pnx - sum_pxz - H_q

        return loss

    def stage_2_loss(self, idx):
        # {Z, Y}
        # E[Y|Z] = E[f(X)|Z] vs E[Y|X] = f(X)
        #
        with torch.no_grad():
            pxz_mean, pxz_scale = self.ivm(idx)
            pxz_dist = torch.distributions.Normal(pxz_mean, pxz_scale)
            #             breakpoint()
            x_samples_from_z = pxz_dist.rsample(sample_shape=(self.sample_size_from_pxz,))
            #             x_samples_from_z = pxz_dist.rsample()
            dim_x = x_samples_from_z.shape[-1]
        #             breakpoint()

        response_inp = torch.cat([x_samples_from_z.reshape(-1, dim_x),
                                  self.data.covariate[idx].repeat((self.sample_size_from_pxz, 1)).reshape(-1,
                                                                                                          self.data.covariate.shape[
                                                                                                              -1])],
                                 axis=-1)
        preds = torch.mean(self.response(response_inp).reshape(self.sample_size_from_pxz, -1), axis=0).reshape(-1, 1)
        #         preds = torch.mean(self.response(x_samples_from_z.reshape(-1,dim_x)).reshape(self.sample_size_from_pxz, -1), axis=0).reshape(-1,1)
        #         breakpoint()
        #         preds = self.response(x_samples_from_z)
        stage_2_loss = torch.mean((self.data.Y[idx] - preds) ** 2)
        #         breakpoint()

        return stage_2_loss


def build_net_for_demand_low_dim(data, sample_size_from_pxz):
    lvm = LVM(data=data, sample_size_from_pxz=sample_size_from_pxz)
    return lvm