# NN for Confounded Measurement Error regression with Instrumental Variable
# Yuchen Zhu

import torch.nn as nn
import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from miv.util import fill_in_args
import argparse

### an nn regression model for fitting E[Me^{ietan}|Z]

class SimpleNN(nn.Module):
    def __init__(self, number_of_basis_func):
        super(SimpleNN, self).__init__()
        self.n_hidden_dim = 4 * number_of_basis_func
        # self.fc0 = nn.Linear(1, 1)
        self.fc1 = nn.Linear(2, self.n_hidden_dim )
        self.fc2 = nn.Linear(self.n_hidden_dim, self.n_hidden_dim)
        self.fc3 = nn.Linear(self.n_hidden_dim, number_of_basis_func*2)

    def forward(self, z):
        z1 = z + torch.randn(z.shape[0], z.shape[1], dtype=torch.cfloat)
        # breakpoint()
        z2 = self.fc1(torch.view_as_real(z1).float())
        # breakpoint()
        z3 = torch.relu(z2)
        z4 = torch.relu(self.fc2(z3))
        z5 = self.fc3(z4)
        z5 = torch.view_as_complex(z5.reshape(z5.shape[0], -1, 2))
#         breakpoint()
        return z5


def loss_func(preds, labels):
    dim_label = labels.shape[-1]
    num_label = labels.shape[0]

    preds_as_real = torch.view_as_real(preds)
    labels_as_real = torch.view_as_real(labels)

    sq_diffs = torch.sum((preds_as_real - labels_as_real) ** 2, dim=-1)
    return torch.sum(sq_diffs) / num_label / dim_label


def split_into_batches(args):
    print('batch size = {}, train_data size = {} x {}'.format(args.batch_size, args.train_data.shape[0], args.train_data.shape[1]))
    batches_data = []

    idxes = np.arange(args.train_data.shape[0])
    np.random.shuffle(idxes)

    batch_i = 0
    while True:
        batches_data.append(torch.tensor(args.train_data[idxes[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]]))
        #         breakpoint()
        batch_i += 1
        if batch_i * args.batch_size >= args.train_data.shape[0]:
            break
    return batches_data


def train(model, args):
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for ep in range(args.num_epochs):
        print('train ep: {}/{}'.format(ep, args.num_epochs))
        running_loss = 0.0
        batches_data = split_into_batches(args)
        for i, batch_data in enumerate(batches_data):
            #             print('train batch: {}/{}'.format(i, len(batches_data)))
            batch_data_input, batch_data_output = batch_data[:, -1:], batch_data[:, :-1]
            preds = model(batch_data_input)
            loss = args.loss_func(preds, batch_data_output)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1 == 0:
                print('print first datum: ', batch_data_input[0:5])
                print('[epoch %d, batch %5d] loss: %.3f' % (ep + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
    return model


def visualize_fit_2d(model, pred_dim, input_gt, output_gt, args):
    model.eval()
    preds = model(torch.Tensor(input_gt))[:args.num_points_to_visualize, pred_dim].detach().numpy()
    fig1, ax1 = plt.subplots()
    ax1.plot(input_gt, preds.real, 'r.')
    ax1.plot(input_gt, output_gt.real, 'g.')
    fig2, ax2 = plt.subplots()
    ax2.plot(input_gt, preds.imag, 'r.')
    ax2.plot(input_gt, output_gt.imag, 'g.')
    plt.show()


def regression_main(args):
    # todo: reimplement using config files.
    number_of_basis_func = args.train_data.shape[-1] - 1
    print('number_of_basis_func: ', number_of_basis_func)
    model = SimpleNN(number_of_basis_func=number_of_basis_func)
    model = train(model, args)
    for pred_dim in range(number_of_basis_func):
        print('visualising dim {}'.format(pred_dim))
        visualize_fit_2d(model, pred_dim,
                         input_gt=args.test_data[:args.num_points_to_visualize, -1:],
                         output_gt=args.test_data[:args.num_points_to_visualize, pred_dim],
                         args=args)


def prepro_args(args):
    if args.loss_func == 'loss_func':
        args.loss_func = loss_func


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='command line arguments for cmiv.py')
    parser.add_argument('--config-path', type=str)
    command_args = parser.parse_args()

    args = fill_in_args(command_args.config_path)
    prepro_args(args)
    regression_main(args)




