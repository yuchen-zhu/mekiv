import torch
from torch import nn

from ..nn_structure.mixture_density_net import MixtureDensityNet
from torch.distributions import Normal
from miv.util import dotdict


class Feature(nn.Module):

    def __init__(self, dropout_ratio, n_input_dim):
        super(Feature, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input_dim, 1),
            # nn.ReLU(),
            # nn.BatchNorm1d(1),
            # nn.Dropout(dropout_ratio),
            # nn.Linear(2, 1),
            # nn.ReLU(),
            # nn.BatchNorm1d(1)
            # nn.Dropout(dropout_ratio)
        )

    def forward(self, data):
        # print(data.shape)
        # breakpoint()
        out = self.net(data)
        return out


class ResponseModel(nn.Module):  # f(x_all)

    def __init__(self, dropout_ratio):
        super(ResponseModel, self).__init__()
        self.net = nn.Sequential(Feature(dropout_ratio, n_input_dim=1),
                                 nn.Linear(1, 1))

    def forward(self, treatment):
        return self.net(treatment)


class Encoder(nn.Module):  # q(x_hidden|m,n,z)
    def __init__(self, dropout_ratio):
        super(Encoder, self).__init__()
        # self.feature_net = Feature(dropout_ratio, n_input_dim=3)
        # self.mu_linear = nn.Linear(1, 1)
        # self.logsigma_linear = nn.Linear(1, 1)

        self.mu_linear = nn.Linear(3, 1)
        self.logsigma_linear = nn.Linear(3, 1)

    def forward(self, instrument, m_obs, n_obs):
        # breakpoint()
        inp = torch.cat([instrument, m_obs, n_obs], dim=1)
        # feature = self.feature_net(inp)

        mu = self.mu_linear(inp)  # B x 1
        logsigma = self.logsigma_linear(inp)
        # print(torch.min(torch.exp(logsigma)))
        norm = Normal(loc=mu, scale=torch.exp(logsigma))

        return norm


class MDecoder(nn.Module):  # p(m|x_hidden)
    def __init__(self, dropout_ratio):
        super(MDecoder, self).__init__()
        # self.feature_net = Feature(dropout_ratio, n_input_dim=1)
        self.logsigma_linear = nn.Linear(1, 1)

    def forward(self, treatment):
        mu = treatment

        # feature = self.feature_net(treatment)
        # logsigma = self.logsigma_linear(feature)

        logsigma = self.logsigma_linear(treatment)
        # print(torch.min(torch.exp(logsigma)))
        norm = Normal(loc=mu, scale=torch.exp(logsigma))
        return norm


class NDecoder(nn.Module):  # p(n|x_hidden)
    def __init__(self, dropout_ratio):
        super(NDecoder, self).__init__()
        self.feature_net = Feature(dropout_ratio, n_input_dim=1)
        self.logsigma_linear = nn.Linear(1, 1)

    def forward(self, treatment):
        mu = treatment

        # feature = self.feature_net(treatment)
        # logsigma = self.logsigma_linear(feature)

        logsigma = self.logsigma_linear(treatment)

        # print(torch.min(torch.exp(logsigma)))
        norm = Normal(loc=mu, scale=torch.exp(logsigma))
        return norm


class InstrumentModelHidden(nn.Module):  # p(x_hidden|z)
    def __init__(self, dropout_ratio):
        super(InstrumentModelHidden, self).__init__()
        # self.feature_net = Feature(dropout_ratio, n_input_dim=1)
        self.logsigma_linear = nn.Linear(1, 1)
        # self.mu_linear = nn.Linear(1, 1)

    def forward(self, instrument):
        # feature = self.feature_net(instrument)
        # logsigma = self.logsigma_linear(feature)

        mu = instrument
        logsigma = self.logsigma_linear(instrument)

        norm = Normal(loc=mu, scale=torch.exp(logsigma))
        return norm


class InstrumentalModelObs(nn.Module):  # p(x_obs|z)
    pass


def build_net_for_linear_and_sigmoid(dropout_rate, **args):
    iv_net_hidden = InstrumentModelHidden(dropout_rate)
    iv_net_obs = None
    response_net = ResponseModel(dropout_rate)

    encoder = Encoder(dropout_rate)
    m_decoder = MDecoder(dropout_rate)
    n_decoder = NDecoder(dropout_rate)

    nets = dotdict({'iv_net_hidden': iv_net_hidden, 'iv_net_obs': iv_net_obs, 'response_net': response_net,
                   'encoder': encoder, 'm_decoder': m_decoder, 'n_decoder': n_decoder})
    return nets


