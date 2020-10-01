from __future__ import print_function
import torch
import torch.nn as nn
from new_layers import *
from utils import get_flat_fts
from copy import deepcopy
import torch.nn.functional as F
import torch.nn.init as init

class L0LeNet5(nn.Module):
    def __init__(self, num_classes, input_size=(1, 28, 28), conv_dims=(20, 50), fc_dims=500,
                 N=50000, beta_ema=0., weight_decay=1, lambas=(1., 1., 1., 1.), local_rep=False,
                 temperature=2./3.):
        super(L0LeNet5, self).__init__()
        self.N = N
        assert(len(conv_dims) == 2)
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.beta_ema = beta_ema
        self.weight_decay = weight_decay

        convs = [L0Conv2d(input_size[0], conv_dims[0], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas[0], local_rep=local_rep),
                 nn.ReLU(), nn.MaxPool2d(2),
                 L0Conv2d(conv_dims[0], conv_dims[1], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas[1], local_rep=local_rep),
                 nn.ReLU(), nn.MaxPool2d(2)]
        self.convs = nn.Sequential(*convs)
        if torch.cuda.is_available():
            self.convs = self.convs.cuda()

        flat_fts = get_flat_fts(input_size, self.convs)
        fcs = [L0Dense(flat_fts, self.fc_dims, droprate_init=0.5, weight_decay=self.weight_decay,
                       lamba=lambas[2], local_rep=local_rep, temperature=temperature), nn.ReLU(),
               L0Dense(self.fc_dims, num_classes, droprate_init=0.5, weight_decay=self.weight_decay,
                       lamba=lambas[3], local_rep=local_rep, temperature=temperature)]
        self.fcs = nn.Sequential(*fcs)

        self.layers = []
        for m in self.modules():
            if isinstance(m, L0Dense) or isinstance(m, L0Conv2d):
                self.layers.append(m)

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x):
        o = self.convs(x)
        o = o.view(o.size(0), -1)
        return self.fcs(o)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_w_sparsity(self):
        sparsity_num = 0.
        sparsity_denom = 0.
        for layer in self.layers:
            sparsity_num += layer.count_zero_w()
            sparsity_denom += layer.count_weight()
        return sparsity_num/sparsity_denom

    def count_active_neuron(self):
        neuron = 0
        for layer in self.layers:
            neuron += layer.count_active_neuron()

        return neuron

    def count_total_neuron(self):
        neuron = 0
        for layer in self.layers:
            if isinstance(layer, L0Conv2d) or isinstance(layer, MAPConv2d) or isinstance(layer, L0Dense):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, L0Conv2d) or isinstance(layer,L0Dense):
                neuron+=layer.count_active_neuron()
                total+=layer.count_total_neuron()

        return 1-neuron/total

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            try:
                e_fl, e_l0 = layer.count_expected_flops_and_l0()
                expected_flops += e_fl
                expected_l0 += e_l0
            except:
                pass
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

class group_lasso_LeNet5(nn.Module):
    def __init__(self, num_classes, input_size=(1, 28, 28), conv_dims=(20, 50), fc_dims=500,
                 N=50000, beta_ema=0., weight_decay=1, lambas=(1., 1., 1., 1.), local_rep=False,
                 temperature=2./3.):
        super(group_lasso_LeNet5, self).__init__()
        self.N = N
        assert(len(conv_dims) == 2)
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.beta_ema = beta_ema
        self.weight_decay = weight_decay

        convs = [group_lasso_Conv2d(input_size[0], conv_dims[0], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas[0], local_rep=local_rep),
                 nn.ReLU(), nn.MaxPool2d(2),
                 group_lasso_Conv2d(conv_dims[0], conv_dims[1], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas[1], local_rep=local_rep),
                 nn.ReLU(), nn.MaxPool2d(2)]
        self.convs = nn.Sequential(*convs)
        if torch.cuda.is_available():
            self.convs = self.convs.cuda()

        flat_fts = get_flat_fts(input_size, self.convs)
        fcs = [group_lasso_Dense(flat_fts, self.fc_dims, droprate_init=0.5, weight_decay=self.weight_decay,
                       lamba=lambas[2], local_rep=local_rep, temperature=temperature), nn.ReLU(),
               group_lasso_Dense(self.fc_dims, num_classes, droprate_init=0.5, weight_decay=self.weight_decay,
                       lamba=lambas[3], local_rep=local_rep, temperature=temperature)]
        self.fcs = nn.Sequential(*fcs)

        self.layers = []
        for m in self.modules():
            if isinstance(m, group_lasso_Dense) or isinstance(m, group_lasso_Conv2d):
                self.layers.append(m)

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x):
        o = self.convs(x)
        o = o.view(o.size(0), -1)
        return self.fcs(o)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            e_fl, e_l0 = layer.count_expected_flops_and_l0()
            expected_flops += e_fl
            expected_l0 += e_l0
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    def get_w_sparsity(self):
        sparsity_num = 0.
        sparsity_denom = 0.
        for layer in self.layers:
            sparsity_num += layer.count_zero_w()
            sparsity_denom += layer.count_weight()
        return sparsity_num/sparsity_denom

    def count_active_neuron(self):
        neuron = 0
        for layer in self.layers:
            neuron += layer.count_active_neuron()

        return neuron


    def count_total_neuron(self):
        neuron = 0
        for layer in self.layers:
            if isinstance(layer, group_lasso_Conv2d) or isinstance(layer, MAPConv2d) or isinstance(layer, group_lasso_Dense):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, group_lasso_Conv2d) or isinstance(layer,group_lasso_Dense):
                neuron+=layer.count_active_neuron()
                total+=layer.count_total_neuron()

        return 1-neuron/total

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            try:
                e_fl, e_l0 = layer.count_expected_flops_and_l0()
                expected_flops += e_fl
                expected_l0 += e_l0
            except:
                pass
        return expected_flops, expected_l0

class CGES_LeNet5(nn.Module):
    def __init__(self, num_classes, input_size=(1, 28, 28), conv_dims=(20, 50), fc_dims=500,
                 N=50000, beta_ema=0., weight_decay=1, lambas=(1., 1., 1., 1.), mu =0.5, local_rep=False,
                 temperature=2./3.):
        super(CGES_LeNet5, self).__init__()
        self.N = N
        assert(len(conv_dims) == 2)
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.beta_ema = beta_ema
        self.weight_decay = weight_decay
        self.mu = mu

        convs = [CGES_Conv2d(input_size[0], conv_dims[0], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas[0], mu= self.mu, local_rep=local_rep),
                 nn.ReLU(), nn.MaxPool2d(2),
                 CGES_Conv2d(conv_dims[0], conv_dims[1], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas[1], mu = self.mu, local_rep=local_rep),
                 nn.ReLU(), nn.MaxPool2d(2)]
        self.convs = nn.Sequential(*convs)
        if torch.cuda.is_available():
            self.convs = self.convs.cuda()

        flat_fts = get_flat_fts(input_size, self.convs)
        fcs = [CGES_Dense(flat_fts, self.fc_dims, droprate_init=0.5, weight_decay=self.weight_decay,
                       lamba=lambas[2], mu = self.mu, local_rep=local_rep, temperature=temperature), nn.ReLU(),
               CGES_Dense(self.fc_dims, num_classes, droprate_init=0.5, weight_decay=self.weight_decay,
                       lamba=lambas[3], mu = self.mu, local_rep=local_rep, temperature=temperature)]
        self.fcs = nn.Sequential(*fcs)

        self.layers = []
        for m in self.modules():
            if isinstance(m, CGES_Dense) or isinstance(m, CGES_Conv2d):
                self.layers.append(m)

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x):
        o = self.convs(x)
        o = o.view(o.size(0), -1)
        return self.fcs(o)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            e_fl, e_l0 = layer.count_expected_flops_and_l0()
            expected_flops += e_fl
            expected_l0 += e_l0
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    def get_w_sparsity(self):
        sparsity_num = 0.
        sparsity_denom = 0.
        for layer in self.layers:
            sparsity_num += layer.count_zero_w()
            sparsity_denom += layer.count_weight()
        return sparsity_num/sparsity_denom

    def count_active_neuron(self):
        neuron = 0
        for layer in self.layers:
            neuron += layer.count_active_neuron()

        return neuron


    def count_total_neuron(self):
        neuron = 0
        for layer in self.layers:
            if isinstance(layer, CGES_Conv2d) or isinstance(layer, MAPConv2d) or isinstance(layer, CGES_Dense):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, CGES_Conv2d) or isinstance(layer,CGES_Dense):
                neuron+=layer.count_active_neuron()
                total+=layer.count_total_neuron()

        return 1-neuron/total

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            try:
                e_fl, e_l0 = layer.count_expected_flops_and_l0()
                expected_flops += e_fl
                expected_l0 += e_l0
            except:
                pass
        return expected_flops, expected_l0


class group_relaxed_L0LeNet5(nn.Module):
    def __init__(self, num_classes, input_size=(1, 28, 28), conv_dims=(20, 50), fc_dims=500,
                 N=50000, beta_ema=0., weight_decay=1, lambas=(1., 1., 1., 1.), beta=4, local_rep=False,
                 temperature=2./3.):
        super(group_relaxed_L0LeNet5, self).__init__()
        self.N = N
        assert(len(conv_dims) == 2)
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.beta_ema = beta_ema
        self.weight_decay = weight_decay

        convs = [group_relaxed_L0Conv2d(input_size[0], conv_dims[0], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas[0], beta=beta, local_rep=local_rep),
                 nn.ReLU(), nn.MaxPool2d(2),
                 group_relaxed_L0Conv2d(conv_dims[0], conv_dims[1], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas[1], beta=beta, local_rep=local_rep),
                 nn.ReLU(), nn.MaxPool2d(2)]
        self.convs = nn.Sequential(*convs)
        if torch.cuda.is_available():
            self.convs = self.convs.cuda()

        flat_fts = get_flat_fts(input_size, self.convs)
        fcs = [group_relaxed_L0Dense(flat_fts, self.fc_dims, droprate_init=0.5, weight_decay=self.weight_decay,
                       lamba=lambas[2], beta=beta, local_rep=local_rep, temperature=temperature), nn.ReLU(),
               group_relaxed_L0Dense(self.fc_dims, num_classes, droprate_init=0.5, weight_decay=self.weight_decay,
                       lamba=lambas[3], beta=beta, local_rep=local_rep, temperature=temperature)]
        self.fcs = nn.Sequential(*fcs)

        self.layers = []
        for m in self.modules():
            if isinstance(m, group_relaxed_L0Dense) or isinstance(m, group_relaxed_L0Conv2d):
                self.layers.append(m)

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x):
        o = self.convs(x)
        o = o.view(o.size(0), -1)
        return self.fcs(o)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            e_fl, e_l0 = layer.count_expected_flops_and_l0()
            expected_flops += e_fl
            expected_l0 += e_l0
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    def get_w_sparsity(self):
        sparsity_num = 0.
        sparsity_denom = 0.
        for layer in self.layers:
            sparsity_num += layer.count_zero_w()
            sparsity_denom += layer.count_weight()
        return sparsity_num/sparsity_denom

    def count_active_neuron(self):
        neuron = 0
        for layer in self.layers:
            neuron += layer.count_active_neuron()

        return neuron


    def count_total_neuron(self):
        neuron = 0
        for layer in self.layers:
            if isinstance(layer, group_relaxed_L0Dense) or isinstance(layer, MAPConv2d) or isinstance(layer, group_relaxed_L0Conv2d):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, group_relaxed_L0Conv2d) or isinstance(layer,group_relaxed_L0Dense):
                neuron+=layer.count_active_neuron()
                total+=layer.count_total_neuron()

        return 1-neuron/total

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            try:
                e_fl, e_l0 = layer.count_expected_flops_and_l0()
                expected_flops += e_fl
                expected_l0 += e_l0
            except:
                pass
        return expected_flops, expected_l0

class group_relaxed_L1LeNet5(nn.Module):
    def __init__(self, num_classes, input_size=(1, 28, 28), conv_dims=(20, 50), fc_dims=500,
                 N=50000, beta_ema=0., weight_decay=1, lambas=(1., 1., 1., 1.), beta=4, local_rep=False,
                 temperature=2./3.):
        super(group_relaxed_L1LeNet5, self).__init__()
        self.N = N
        assert(len(conv_dims) == 2)
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.beta_ema = beta_ema
        self.weight_decay = weight_decay

        convs = [group_relaxed_L1Conv2d(input_size[0], conv_dims[0], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas[0], beta=beta, local_rep=local_rep),
                 nn.ReLU(), nn.MaxPool2d(2),
                 group_relaxed_L1Conv2d(conv_dims[0], conv_dims[1], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas[1], beta=beta, local_rep=local_rep),
                 nn.ReLU(), nn.MaxPool2d(2)]
        self.convs = nn.Sequential(*convs)
        if torch.cuda.is_available():
            self.convs = self.convs.cuda()

        flat_fts = get_flat_fts(input_size, self.convs)
        fcs = [group_relaxed_L1Dense(flat_fts, self.fc_dims, droprate_init=0.5, weight_decay=self.weight_decay,
                       lamba=lambas[2], beta=beta, local_rep=local_rep, temperature=temperature), nn.ReLU(),
               group_relaxed_L1Dense(self.fc_dims, num_classes, droprate_init=0.5, weight_decay=self.weight_decay,
                       lamba=lambas[3], beta=beta, local_rep=local_rep, temperature=temperature)]
        self.fcs = nn.Sequential(*fcs)

        self.layers = []
        for m in self.modules():
            if isinstance(m, group_relaxed_L1Dense) or isinstance(m, group_relaxed_L1Conv2d):
                self.layers.append(m)

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x):
        o = self.convs(x)
        o = o.view(o.size(0), -1)
        return self.fcs(o)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            e_fl, e_l0 = layer.count_expected_flops_and_l0()
            expected_flops += e_fl
            expected_l0 += e_l0
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    def get_w_sparsity(self):
        sparsity_num = 0.
        sparsity_denom = 0.
        for layer in self.layers:
            sparsity_num += layer.count_zero_w()
            sparsity_denom += layer.count_weight()
        return sparsity_num/sparsity_denom

    def count_active_neuron(self):
        neuron = 0
        for layer in self.layers:
            neuron += layer.count_active_neuron()

        return neuron


    def count_total_neuron(self):
        neuron = 0
        for layer in self.layers:
            if isinstance(layer, group_relaxed_L1Dense) or isinstance(layer, MAPConv2d) or isinstance(layer, group_relaxed_L1Conv2d):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, group_relaxed_L1Conv2d) or isinstance(layer,group_relaxed_L1Dense):
                neuron+=layer.count_active_neuron()
                total+=layer.count_total_neuron()

        return 1-neuron/total

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            try:
                e_fl, e_l0 = layer.count_expected_flops_and_l0()
                expected_flops += e_fl
                expected_l0 += e_l0
            except:
                pass
        return expected_flops, expected_l0


class group_relaxed_L1L2LeNet5(nn.Module):
    def __init__(self, num_classes, input_size=(1, 28, 28), conv_dims=(20, 50), fc_dims=500,
                 N=50000, beta_ema=0., weight_decay=1, lambas=(1., 1., 1., 1.), beta=4, local_rep=False,
                 temperature=2./3.):
        super(group_relaxed_L1L2LeNet5, self).__init__()
        self.N = N
        assert(len(conv_dims) == 2)
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.beta_ema = beta_ema
        self.weight_decay = weight_decay

        convs = [group_relaxed_L1L2Conv2d(input_size[0], conv_dims[0], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas[0], beta=beta, local_rep=local_rep),
                 nn.ReLU(), nn.MaxPool2d(2),
                 group_relaxed_L1L2Conv2d(conv_dims[0], conv_dims[1], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas[1], beta=beta, local_rep=local_rep),
                 nn.ReLU(), nn.MaxPool2d(2)]
        self.convs = nn.Sequential(*convs)
        if torch.cuda.is_available():
            self.convs = self.convs.cuda()

        flat_fts = get_flat_fts(input_size, self.convs)
        fcs = [group_relaxed_L1L2Dense(flat_fts, self.fc_dims, droprate_init=0.5, weight_decay=self.weight_decay,
                       lamba=lambas[2], beta=beta, local_rep=local_rep, temperature=temperature), nn.ReLU(),
               group_relaxed_L1L2Dense(self.fc_dims, num_classes, droprate_init=0.5, weight_decay=self.weight_decay,
                       lamba=lambas[3], beta=beta, local_rep=local_rep, temperature=temperature)]
        self.fcs = nn.Sequential(*fcs)

        self.layers = []
        for m in self.modules():
            if isinstance(m, group_relaxed_L1L2Dense) or isinstance(m, group_relaxed_L1L2Conv2d):
                self.layers.append(m)

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x):
        o = self.convs(x)
        o = o.view(o.size(0), -1)
        return self.fcs(o)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            e_fl, e_l0 = layer.count_expected_flops_and_l0()
            expected_flops += e_fl
            expected_l0 += e_l0
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    def get_w_sparsity(self):
        sparsity_num = 0.
        sparsity_denom = 0.
        for layer in self.layers:
            sparsity_num += layer.count_zero_w()
            sparsity_denom += layer.count_weight()
        return sparsity_num/sparsity_denom

    def count_active_neuron(self):
        neuron = 0
        for layer in self.layers:
            neuron += layer.count_active_neuron()

        return neuron


    def count_total_neuron(self):
        neuron = 0
        for layer in self.layers:
            if isinstance(layer, group_relaxed_L1L2Dense) or isinstance(layer, MAPConv2d) or isinstance(layer, group_relaxed_L1L2Conv2d):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, group_relaxed_L1L2Conv2d) or isinstance(layer,group_relaxed_L1L2Dense):
                neuron+=layer.count_active_neuron()
                total+=layer.count_total_neuron()

        return 1-neuron/total

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            try:
                e_fl, e_l0 = layer.count_expected_flops_and_l0()
                expected_flops += e_fl
                expected_l0 += e_l0
            except:
                pass
        return expected_flops, expected_l0

class group_relaxed_TF1LeNet5(nn.Module):
    def __init__(self, num_classes, input_size=(1, 28, 28), conv_dims=(20, 50), fc_dims=500,
                 N=50000, beta_ema=0., weight_decay=1, lambas=(1., 1., 1., 1.), alpha = 1., beta=4, local_rep=False,
                 temperature=2./3.):
        super(group_relaxed_TF1LeNet5, self).__init__()
        self.N = N
        assert(len(conv_dims) == 2)
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.beta_ema = beta_ema
        self.weight_decay = weight_decay

        convs = [group_relaxed_TF1Conv2d(input_size[0], conv_dims[0], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas[0], alpha=alpha, beta=beta, local_rep=local_rep),
                 nn.ReLU(), nn.MaxPool2d(2),
                 group_relaxed_TF1Conv2d(conv_dims[0], conv_dims[1], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas[1], alpha = alpha, beta=beta, local_rep=local_rep),
                 nn.ReLU(), nn.MaxPool2d(2)]
        self.convs = nn.Sequential(*convs)
        if torch.cuda.is_available():
            self.convs = self.convs.cuda()

        flat_fts = get_flat_fts(input_size, self.convs)
        fcs = [group_relaxed_TF1Dense(flat_fts, self.fc_dims, droprate_init=0.5, weight_decay=self.weight_decay,
                       lamba=lambas[2], alpha = alpha, beta=beta, local_rep=local_rep, temperature=temperature), nn.ReLU(),
               group_relaxed_TF1Dense(self.fc_dims, num_classes, droprate_init=0.5, weight_decay=self.weight_decay,
                       lamba=lambas[3], alpha = alpha, beta=beta, local_rep=local_rep, temperature=temperature)]
        self.fcs = nn.Sequential(*fcs)

        self.layers = []
        for m in self.modules():
            if isinstance(m, group_relaxed_TF1Dense) or isinstance(m, group_relaxed_TF1Conv2d):
                self.layers.append(m)

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x):
        o = self.convs(x)
        o = o.view(o.size(0), -1)
        return self.fcs(o)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            e_fl, e_l0 = layer.count_expected_flops_and_l0()
            expected_flops += e_fl
            expected_l0 += e_l0
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    def get_w_sparsity(self):
        sparsity_num = 0.
        sparsity_denom = 0.
        for layer in self.layers:
            sparsity_num += layer.count_zero_w()
            sparsity_denom += layer.count_weight()
        return sparsity_num/sparsity_denom

    def count_active_neuron(self):
        neuron = 0
        for layer in self.layers:
            neuron += layer.count_active_neuron()

        return neuron


    def count_total_neuron(self):
        neuron = 0
        for layer in self.layers:
            if isinstance(layer, group_relaxed_TF1Dense) or isinstance(layer, MAPConv2d) or isinstance(layer, group_relaxed_TF1Conv2d):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, group_relaxed_TF1Conv2d) or isinstance(layer,group_relaxed_TF1Dense):
                neuron+=layer.count_active_neuron()
                total+=layer.count_total_neuron()

        return 1-neuron/total

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            try:
                e_fl, e_l0 = layer.count_expected_flops_and_l0()
                expected_flops += e_fl
                expected_l0 += e_l0
            except:
                pass
        return expected_flops, expected_l0

class group_relaxed_SCADLeNet5(nn.Module):
    def __init__(self, num_classes, input_size=(1, 28, 28), conv_dims=(20, 50), fc_dims=500,
                 N=50000, beta_ema=0., weight_decay=1, lambas=(1., 1., 1., 1.), beta=4, local_rep=False,
                 temperature=2./3.):
        super(group_relaxed_SCADLeNet5, self).__init__()
        self.N = N
        assert(len(conv_dims) == 2)
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.beta_ema = beta_ema
        self.weight_decay = weight_decay

        convs = [group_relaxed_SCAD_Conv2d(input_size[0], conv_dims[0], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas[0], beta=beta, local_rep=local_rep),
                 nn.ReLU(), nn.MaxPool2d(2),
                 group_relaxed_SCAD_Conv2d(conv_dims[0], conv_dims[1], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas[1], beta=beta, local_rep=local_rep),
                 nn.ReLU(), nn.MaxPool2d(2)]
        self.convs = nn.Sequential(*convs)
        if torch.cuda.is_available():
            self.convs = self.convs.cuda()

        flat_fts = get_flat_fts(input_size, self.convs)
        fcs = [group_relaxed_SCAD_Dense(flat_fts, self.fc_dims, droprate_init=0.5, weight_decay=self.weight_decay,
                       lamba=lambas[2], beta=beta, local_rep=local_rep, temperature=temperature), nn.ReLU(),
               group_relaxed_SCAD_Dense(self.fc_dims, num_classes, droprate_init=0.5, weight_decay=self.weight_decay,
                       lamba=lambas[3], beta=beta, local_rep=local_rep, temperature=temperature)]
        self.fcs = nn.Sequential(*fcs)

        self.layers = []
        for m in self.modules():
            if isinstance(m, group_relaxed_SCAD_Dense) or isinstance(m, group_relaxed_SCAD_Conv2d):
                self.layers.append(m)

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x):
        o = self.convs(x)
        o = o.view(o.size(0), -1)
        return self.fcs(o)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            e_fl, e_l0 = layer.count_expected_flops_and_l0()
            expected_flops += e_fl
            expected_l0 += e_l0
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    def get_w_sparsity(self):
        sparsity_num = 0.
        sparsity_denom = 0.
        for layer in self.layers:
            sparsity_num += layer.count_zero_w()
            sparsity_denom += layer.count_weight()
        return sparsity_num/sparsity_denom

    def count_active_neuron(self):
        neuron = 0
        for layer in self.layers:
            neuron += layer.count_active_neuron()

        return neuron


    def count_total_neuron(self):
        neuron = 0
        for layer in self.layers:
            if isinstance(layer, group_relaxed_SCAD_Dense) or isinstance(layer, MAPConv2d) or isinstance(layer, group_relaxed_SCAD_Conv2d):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, group_relaxed_SCAD_Conv2d) or isinstance(layer,group_relaxed_SCAD_Dense):
                neuron+=layer.count_active_neuron()
                total+=layer.count_total_neuron()

        return 1-neuron/total

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            try:
                e_fl, e_l0 = layer.count_expected_flops_and_l0()
                expected_flops += e_fl
                expected_l0 += e_l0
            except:
                pass
        return expected_flops, expected_l0
