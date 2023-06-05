import torch
from torch import nn

from ..nn_structure.mixture_density_net import MixtureDensityNet
from torch.distributions import Normal
from miv.util import dotdict


class Feature(nn.Module):

    def __init__(self, dropout_ratio, n_input_dim):
        super(Feature, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_ratio)
        )

    def forward(self, data):
        print(data.shape)
        # breakpoint()
        out = self.net(data)


        return out


class ResponseModel(nn.Module):  # f(x_all, t, s)

    def __init__(self, dropout_ratio):
        super(ResponseModel, self).__init__()
        self.net = nn.Sequential(Feature(dropout_ratio, n_input_dim=3),
                                 nn.Linear(32, 1))

    def forward(self, treatment, covariate):
        feature = torch.cat([treatment, covariate], dim=1)
        return self.net(feature)


class Encoder(nn.Module):  # q(x_hidden|m,n,z)
    def __init__(self, dropout_ratio):
        super(Encoder, self).__init__()
        self.feature_net = Feature(dropout_ratio, n_input_dim=5)
        self.mu_linear = nn.Linear(32, 1)
        self.logsigma_linear = nn.Linear(32, 1)

    def forward(self, instrument, m_obs, n_obs):

        feature = self.feature_net(torch.cat([instrument, m_obs, n_obs], dim=1))

        mu = self.mu_linear(feature)  # B x 1
        logsigma = self.logsigma_linear(feature)
        print(torch.min(torch.exp(logsigma)))
        norm = Normal(loc=mu, scale=torch.exp(logsigma)+1e-8)

        return norm


class MDecoder(nn.Module):  # p(m|x_hidden)
    def __init__(self, dropout_ratio):
        super(MDecoder, self).__init__()
        self.feature_net = Feature(dropout_ratio, n_input_dim=1)
        self.logsigma_linear = nn.Linear(32, 1)

    def forward(self, treatment):
        mu = treatment
        feature = self.feature_net(treatment)
        logsigma = self.logsigma_linear(feature)
        print(torch.min(torch.exp(logsigma)))
        norm = Normal(loc=mu, scale=torch.exp(logsigma)+1e-8)
        return norm


class NDecoder(nn.Module):  # p(n|x_hidden)
    def __init__(self, dropout_ratio):
        super(NDecoder, self).__init__()
        self.feature_net = Feature(dropout_ratio, n_input_dim=1)
        self.logsigma_linear = nn.Linear(32, 1)

    def forward(self, treatment):
        mu = treatment
        feature = self.feature_net(treatment)
        logsigma = self.logsigma_linear(feature)
        print(torch.min(torch.exp(logsigma)))
        norm = Normal(loc=mu, scale=torch.exp(logsigma)+1e-8)
        return norm


class InstrumentModelHidden(nn.Module):  # p(x_hidden|z)
    def __init__(self, dropout_ratio):
        super(InstrumentModelHidden, self).__init__()
        self.feature_net = Feature(dropout_ratio, n_input_dim=3)
        self.logsigma_linear = nn.Linear(32, 1)
        self.mu_linear = nn.Linear(32, 1)

    def forward(self, instrument):
        feature = self.feature_net(instrument)
        logsigma = self.logsigma_linear(feature)
        mu = self.mu_linear(feature)

        norm = Normal(loc=mu, scale=torch.exp(logsigma))
        return norm


class InstrumentalModelObs(nn.Module):  # p(x_obs|z)
    pass


def build_net_for_demand_low_dim(dropout_rate, **args):
    iv_net_hidden = InstrumentModelHidden(dropout_rate)
    iv_net_obs = None
    response_net = ResponseModel(dropout_rate)

    encoder = Encoder(dropout_rate)
    m_decoder = MDecoder(dropout_rate)
    n_decoder = NDecoder(dropout_rate)

    nets = dotdict({'iv_net_hidden': iv_net_hidden, 'iv_net_obs': iv_net_obs, 'response_net': response_net,
                   'encoder': encoder, 'm_decoder': m_decoder, 'n_decoder': n_decoder})
    return nets


