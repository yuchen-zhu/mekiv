import torch


def set_iv_net_hidden(iv_net_hidden):
    iv_net_hidden.feature_net.net[0].weight = torch.nn.Parameter(torch.tensor([[1.]]))
    iv_net_hidden.feature_net.net[0].bias = torch.nn.Parameter(torch.tensor([0.]))
    # iv_net_hidden.feature_net.net[1].bias = torch.nn.Parameter(torch.tensor([0.]))
    # iv_net_hidden.feature_net.net[1].weight = torch.nn.Parameter(torch.tensor([[1., 1.]]))
    iv_net_hidden.mu_linear.weight = torch.nn.Parameter(torch.tensor([[1.]]))
    iv_net_hidden.mu_linear.bias = torch.nn.Parameter(torch.tensor([0.]))
    iv_net_hidden.logsigma_linear.weight = torch.nn.Parameter(torch.tensor([[0.]]))
    iv_net_hidden.logsigma_linear.bias = torch.nn.Parameter(torch.log(torch.tensor([0.1])))


def set_mn_decoder(m_decoder):
    # m_decoder.feature_net.net[0].weight = torch.nn.Parameter(torch.tensor([[1.], [0.]]))
    # m_decoder.feature_net.net[0].bias = torch.nn.Parameter(torch.tensor([0., 0.]))
    m_decoder.logsigma_linear.weight = torch.nn.Parameter(torch.tensor([[0.]]))
    m_decoder.logsigma_linear.bias = torch.nn.Parameter(torch.log(torch.tensor([0.4])))


def set_encoder(encoder):
    encoder.feature_net.net[0].weight = torch.nn.Parameter(torch.tensor([[1., 0., 0.]]))
    encoder.feature_net.net[0].bias = torch.nn.Parameter(torch.tensor([0.]))
    # encoder.feature_net.net[1].weight = torch.nn.Parameter(torch.tensor([[1., 0.]]))
    # encoder.feature_net.net[1].bias = torch.nn.Parameter(torch.tensor([0.]))
    encoder.mu_linear.weight = torch.nn.Parameter(torch.tensor([[1.]]))
    encoder.mu_linear.bias = torch.nn.Parameter(torch.tensor([0.]))
    encoder.logsigma_linear.weight = torch.nn.Parameter(torch.tensor([[0.]]))
    encoder.logsigma_linear.bias = torch.nn.Parameter(torch.log(torch.tensor([0.1])))
