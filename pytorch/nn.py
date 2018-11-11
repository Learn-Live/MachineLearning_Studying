# -*- coding: utf-8 -*-
r"""

"""

import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data as Data
from torch import nn, optim
from torch.utils.data import Dataset


def normalize_data(X, range_value=[-1, 1], eps=1e-5):  # down=-1, up=1

    new_X = np.copy(X)

    mins = new_X.min(axis=0)  # column
    maxs = new_X.max(axis=0)

    rng = maxs - mins
    for i in range(rng.shape[0]):
        if rng[i] == 0.0:
            rng[i] += eps

    new_X = (new_X - mins) / rng * (range_value[1] - range_value[0]) + range_value[0]

    return new_X


class TrafficDataset(Dataset):

    def __init__(self, X, y, transform=None, normalization_flg=False):
        self.X = X
        self.y = y
        cnt = 0
        # with open(input_file, 'r') as fid_in:
        #     line = fid_in.readline()
        #     while line:
        #         line_arr = line.split(',')
        #         value = list(map(lambda x: float(x), line_arr[:-1]))
        #         self.X.append(value)
        #         self.y.append(float(line_arr[-1].strip()))
        #         line = fid_in.readline()
        #         cnt += 1
        if normalization_flg:
            self.X = normalize_data(np.asarray(self.X, dtype=float), range_value=[-1, 1], eps=1e-5)
            # with open(input_file + '_normalized.csv', 'w') as fid_out:
            #     for i in range(self.X.shape[0]):
            #         # print('i', i.data.tolist())
            #         tmp = [str(j) for j in self.X[i]]
            #         fid_out.write(','.join(tmp) + ',' + str(int(self.y[i])) + '\n')

        self.transform = transform

    def __getitem__(self, index):

        value_x = self.X[index]
        value_y = self.y[index]
        if self.transform:
            value_x = self.transform(value_x)

        value_x = torch.from_numpy(np.asarray(value_x)).double()
        value_y = torch.from_numpy(np.asarray(value_y)).double()

        # X_train, X_test, y_train, y_test = train_test_split(value_x, value_y, train_size=0.7, shuffle=True)
        return value_x, value_y  # Dataset.__getitem__() should return a single sample and label, not the whole dataset.
        # return value_x.view([-1,1,-1,1]), value_y

    def __len__(self):
        return len(self.X)


def print_network(describe_str, net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(describe_str, net)
    print('Total number of parameters: %d' % num_params)


def generated_train_set(num):
    X = []
    y = []
    for i in range(num):
        yi = 0
        if i % 2 == 0:
            yi = 1
        rnd = np.random.random()
        rnd2 = np.random.random()
        X.append([1000 * rnd, i * 1 * rnd2])
        y.append(yi)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    return TrafficDataset(X, y, normalization_flg=False)


class NerualNetworkDome():
    r"""
        test
    """

    def __init__(self):
        self.in_dim = 2
        self.h_dim = 1
        self.out_dim = 1

        # network structure
        in_lay = nn.Linear(self.in_dim, self.h_dim * 20, bias=True)  # class initialization
        hid_lay = nn.Linear(self.h_dim * 20, self.h_dim * 20, bias=True)
        hid_lay_2 = nn.Linear(self.h_dim * 20, self.h_dim * 20, bias=True)
        out_lay = nn.Linear(self.h_dim * 20, self.out_dim, bias=True)
        self.net = nn.Sequential(in_lay,
                                 nn.Sigmoid(),
                                 hid_lay,
                                 nn.LeakyReLU(),
                                 hid_lay_2,
                                 nn.LeakyReLU(),
                                 out_lay)

        # evaluation standards
        self.criterion = nn.MSELoss()  # class initialization

        # optimizer
        self.optim = optim.Adam(self.net.parameters(), lr=1e-3, betas=(0.9, 0.99))

        # print network architecture
        print_network('demo', self.net)
        print_net_parameters(self.net, OrderedDict(), title='Initialization parameters')

    def forward(self, X):
        o1 = self.net(X)

        return o1

    def train(self, train_set):
        # X,y = train_set
        # train_set = (torch.from_numpy(X).double(), torch.from_numpy(y).double())
        train_loader = Data.DataLoader(train_set, 50, shuffle=True, num_workers=4)
        param_order_dict = OrderedDict()
        ith_layer_out_dict = OrderedDict()

        loss_lst = []
        epochs = 10
        for epoch in range(epochs):
            for i, (b_x, b_y) in enumerate(train_loader):
                b_x = b_x.view([b_x.shape[0], -1]).float()
                b_y = b_y.view(b_y.shape[0], 1).float()

                self.optim.zero_grad()
                b_y_preds = self.forward(b_x)
                loss = self.criterion(b_y_preds, b_y)
                print('%d/%d, batch_ith = %d, loss=%f' % (epoch, epochs, i, loss.data))
                loss.backward()
                self.optim.step()

                loss_lst.append(loss.data)
                # for idx, param in enumerate(self.net.parameters()):
                for name, param in self.net.named_parameters():
                    # print(name, param)  # even is weigh and bias, odd is activation function, it's no parameters.
                    if name not in param_order_dict.keys():
                        param_order_dict[name] = [copy.deepcopy(np.reshape(param.data.numpy(), (-1, 1)))]
                    else:
                        param_order_dict[name].append(copy.deepcopy(np.reshape(param.data.numpy(), (-1, 1))))

        print_net_parameters(self.net, param_order_dict,
                             title='All parameters (weights and bias) from \n begin to finish in training process phase.')

        print_net_parameters(self.net, OrderedDict(), title='Final parameters')

        # param_lst = np.asarray(param_lst, dtype=float)
        # print(param_lst)
        # show_figures(param_lst[:, 0], param_lst[:, 1])
        # show_figures(loss_lst, loss_lst)


def print_net_parameters(net, param_order_dict=OrderedDict(), title=''):
    num_figs = len(net) // 2 + 1
    print('subplots:(%dx%d):' % (num_figs, num_figs))
    j = 1
    print(title)
    if param_order_dict == {}:
        # for idx, param in enumerate(self.net.parameters()):
        for name, param in net.named_parameters():
            print(name, param)  # even is weigh and bias, odd is activation function, it's no parameters.
            if name not in param_order_dict.keys():
                param_order_dict[name] = [copy.deepcopy(np.reshape(param.data.numpy(), (-1, 1)))]
            else:
                param_order_dict[name].append(copy.deepcopy(np.reshape(param.data.numpy(), (-1, 1))))

    # fig = plt.subplots()
    # fig.suptitle(title)
    plt.suptitle(title, fontsize=8)
    for ith, (name, param) in enumerate(net.named_parameters()):
        # dynamic_plot(param_order_dict[name])
        plt.subplot(num_figs, 2, j)
        print('subplot_%dth' % j)
        num_bins = 10
        histogram(np.reshape(np.asarray(param_order_dict[name], dtype=float), (-1, 1)), num_bins=num_bins, title=name,
                  x_label='Values', y_label='Frequency')
        j += 1

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()


def histogram(x, num_bins=5, title='histogram', x_label='Values.', y_label='Frequency'):
    # x = [21, 22, 23, 4, 5, 6, 77, 8, 9, 10, 31, 32, 33, 34, 35, 36, 37, 18, 49, 50, 100]
    # num_bins = 5
    n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)


# from python3.tricks import load_data

# matplotlib.use("Agg")
import matplotlib.animation as manimation


def dynamic_plot(X, y):
    r"""
        must install ffmpeg, then pip3 install ffmpeg

        Note:
            pycharm cannot show animation. so it needs to save animation to local file.

    :param input_f:
    :return:
    """
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    fig = plt.figure()
    #
    # def update_figure(X, y):
    #     # plt.scatter(X, y)
    #     plt.plot(X,y,'k-o')
    #     plt.xlim(0,100)
    #     plt.ylim(0,100)

    with writer.saving(fig, "writer_test.mp4", dpi=100):
        for k in range(10):
            # Create a new plot object
            plt.scatter(range(X), range(y))
            # update_figure(X,y)
            writer.grab_frame()


def show_figures(D_loss, G_loss):
    import matplotlib.pyplot as plt
    plt.figure()

    plt.plot(D_loss, 'r', alpha=0.5, label='D_loss of real and fake sample')
    plt.plot(G_loss, 'g', alpha=0.5, label='D_loss of G generated fake sample')
    plt.legend(loc='upper right')
    plt.title('D\'s loss of real and fake sample.')
    plt.show()


import matplotlib.pyplot as plt
import numpy

hl, = plt.plot([], [])


def update_line(hl, new_data):
    hl.set_xdata(numpy.append(hl.get_xdata(), new_data))
    hl.set_ydata(numpy.append(hl.get_ydata(), new_data))
    plt.draw()


if __name__ == '__main__':
    train_set = generated_train_set(100)
    nn_demo = NerualNetworkDome()
    nn_demo.train(train_set)

    # dynamic_plot(input_f="/home/kun/PycharmProjects/MachineLearning_Studying/data/attack_demo.csv")
