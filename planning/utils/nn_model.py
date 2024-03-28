import torch
import numpy as np
import copy
from collections import OrderedDict
import pickle
import torch.nn.functional as functional

def check_sanity(weights):
    assert ((torch.isnan(weights).any() == False) and (torch.isinf(weights).any() == False))

def weights_print(m):
    if type(m) == torch.nn.Linear:
        print(m.weight.size())
        print(m.weight)
        try:
            print(m.bias.size())
            print(m.bias.data)
        except:
            return

class FCNet(torch.nn.Module):
    """ a fully-connected NN with ReLU activations """
    def __init__(self, shape, np_random=None, last_linear=False, nonlinearity='relu'):

        super(FCNet, self).__init__()

        def init_weights_rep_relu(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if np_random is not None:
                    m.bias.data.fill_(np_random.uniform(-1.1))
                else:
                    m.bias.data.fill_(0.)

        def init_weights_rep_sigmoid(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                if np_random is not None:
                    m.bias.data.fill_(np_random.uniform(-1.1))
                else:
                    m.bias.data.fill_(0.)

        weights = []
        for i in range(len(shape)-1):
            temp_linear = torch.nn.Linear(shape[i], shape[i+1])
            weights.append(temp_linear)
            if i==(len(shape)-2) and last_linear:
                continue
            if nonlinearity == 'relu':
                temp_act = torch.nn.ReLU()
            else:
                temp_act = torch.nn.Sigmoid()
            weights.append(temp_act)

        self.weights = torch.nn.Sequential(*weights)
        if nonlinearity == 'relu':
            # self.weights.apply(init_weights_rep_relu)
            self.weights.apply(init_weights_rep_sigmoid)
        else:
            self.weights.apply(init_weights_rep_sigmoid)

    def forward(self, x):
        return self.weights(x)

class LinearNet(torch.nn.Module):
    """ a fully-connected NN with ReLU activations """
    def __init__(self, shape, bias=True, np_random=None, use_uniform_init=False):

        super(LinearNet, self).__init__()
        self.weights = torch.nn.Linear(shape[0], shape[1], bias=bias)
        if use_uniform_init:
            torch.nn.init.uniform_(self.weights.weight, a=-0.003, b=0.003)
        else:
            torch.nn.init.kaiming_normal_(self.weights.weight, nonlinearity='relu')
        if bias:
            if np_random is not None:
                self.weights.bias.data.fill_(np_random.uniform(-1,1))
            else:
                self.weights.bias.data.fill_(0.0)

    def forward(self, x):
        return self.weights(x)


class LTA:
    def __init__(self, tiles, bound_low, bound_high, eta, input_dim):
        # 1 tiling, binning
        self.n_tilings = 1
        self.n_tiles = tiles
        self.bound_low, self.bound_high = bound_low, bound_high
        self.delta = (self.bound_high - self.bound_low) / self.n_tiles
        self.c_mat = torch.as_tensor(np.array([self.delta * i for i in range(self.n_tiles)]) + self.bound_low, dtype=torch.float32)
        self.eta = eta
        self.d = input_dim

    def __call__(self, reps):
        temp = reps
        temp = temp.reshape([-1, self.d, 1])
        onehots = 1.0 - self.i_plus_eta(self.sum_relu(self.c_mat, temp))
        out = torch.reshape(torch.reshape(onehots, [-1]), [-1, int(self.d * self.n_tiles * self.n_tilings)])
        return out

    def sum_relu(self, c, x):
        out = functional.relu(c - x) + functional.relu(x - self.delta - c)
        return out

    def i_plus_eta(self, x):
        if self.eta == 0:
            return torch.sign(x)
        out = (x <= self.eta).type(torch.float32) * x + (x > self.eta).type(torch.float32)
        return out
