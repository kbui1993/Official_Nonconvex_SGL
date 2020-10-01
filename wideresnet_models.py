from __future__ import print_function
import torch
import torch.nn as nn
from new_layers import *
from utils import get_flat_fts
from copy import deepcopy
import torch.nn.functional as F
import torch.nn.init as init

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, droprate_init=0.0, weight_decay=0., lamba=0.01, local_rep=False,
                 temperature=2./3.):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = L0Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False,
                              droprate_init=droprate_init, weight_decay=weight_decay / (1 - 0.3), local_rep=local_rep,
                              lamba=lamba, temperature=temperature)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = MAPConv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               weight_decay=weight_decay)

        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and \
                            MAPConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False,
                                      weight_decay=weight_decay) or None

    def forward(self, x):
        if not self.equalInOut:
            x = F.relu(self.bn1(x))
        else:
            out = F.relu(self.bn1(x))

        out = self.conv1(out if self.equalInOut else x)
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.add(out, x if self.equalInOut else self.convShortcut(x))

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, droprate_init=0.0, weight_decay=0., lamba=0.01,
                 local_rep=False, temperature=2./3.):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, droprate_init,
                                      weight_decay=weight_decay, lamba=lamba, local_rep=local_rep,
                                      temperature=temperature)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, droprate_init,
                    weight_decay=0., lamba=0.01, local_rep=False, temperature=2./3.):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1,
                                droprate_init, weight_decay, lamba, local_rep=local_rep, temperature=temperature))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class L0WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, droprate_init=0.3, N=50000, beta_ema=0.99,
                 weight_decay=5e-4, local_rep=False, lamba=0.01, temperature=2./3.):
        super(L0WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        self.n = (depth - 4) // 6
        self.N = N
        self.beta_ema = beta_ema
        block = BasicBlock

        self.weight_decay = N * weight_decay
        self.lamba = lamba

        # 1st conv before any network block
        self.conv1 = MAPConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False,
                               weight_decay=self.weight_decay)
        # 1st block
        self.block1 = NetworkBlock(self.n, nChannels[0], nChannels[1], block, 1, droprate_init, self.weight_decay,
                                   self.lamba, local_rep=local_rep, temperature=temperature)
        # 2nd block
        self.block2 = NetworkBlock(self.n, nChannels[1], nChannels[2], block, 2, droprate_init, self.weight_decay,
                                   self.lamba, local_rep=local_rep, temperature=temperature)
        # 3rd block
        self.block3 = NetworkBlock(self.n, nChannels[2], nChannels[3], block, 2, droprate_init, self.weight_decay,
                                   self.lamba, local_rep=local_rep, temperature=temperature)
        # bn, relu and classifier
        self.bn = nn.BatchNorm2d(nChannels[3])
        self.fcout = MAPDense(nChannels[3], num_classes, weight_decay=self.weight_decay)

        self.layers, self.bn_params = [], []
        for m in self.modules():
            if isinstance(m, MAPDense) or isinstance(m, MAPConv2d) or isinstance(m, L0Conv2d):
                self.layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

        print('Using weight decay: {}'.format(self.weight_decay))

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fcout(out)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        for bnw in self.bn_params:
            if self.weight_decay > 0:
                regularization += (self.weight_decay / self.N) * .5 * torch.sum(bnw.pow(2))
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
            if isinstance(layer, L0Conv2d) or isinstance(layer, MAPConv2d) or isinstance(layer,MAPDense):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, L0Conv2d):
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


class group_lasso_BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, droprate_init=0.0, weight_decay=0., lamba=0.01):
        super(group_lasso_BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = group_lasso_Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False,
                              lamba=lamba, weight_decay = weight_decay)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = MAPConv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               weight_decay=weight_decay)

        self.droprate_init = droprate_init

        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and \
                            MAPConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False,
                                      weight_decay=weight_decay) or None

    def forward(self, x):
        if not self.equalInOut:
            x = F.relu(self.bn1(x))
        else:
            out = F.relu(self.bn1(x))

        out = self.conv1(out if self.equalInOut else x)
        out = F.relu(self.bn2(out))
        if self.droprate_init > 0:
            out = F.dropout(out, p=self.droprate_init, training=self.training)
        out = self.conv2(out)
        return torch.add(out, x if self.equalInOut else self.convShortcut(x))

class group_lasso_NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, droprate_init=0.0, weight_decay=0., lamba=0.01):
        super(group_lasso_NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, droprate_init, weight_decay=weight_decay, lamba=lamba)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, droprate_init, weight_decay=0, lamba=0.01):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, droprate_init, weight_decay, lamba))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class group_lasso_WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, droprate_init=0.3, N=50000, beta_ema=0.99, weight_decay=5e-4, lamba=0.01, beta = 0.4):
        super(group_lasso_WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        self.n = (depth - 4) // 6
        self.N = N
        self.beta_ema = beta_ema
        block = group_lasso_BasicBlock

        self.weight_decay = N * weight_decay
        self.lamba = lamba
        self.beta = beta

        # 1st conv before any network block
        self.conv1 = MAPConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False,
                               weight_decay=self.weight_decay)
        # 1st block
        self.block1 = group_lasso_NetworkBlock(self.n, nChannels[0], nChannels[1], block, 1, droprate_init, self.weight_decay, self.lamba)
        # 2nd block
        self.block2 = group_lasso_NetworkBlock(self.n, nChannels[1], nChannels[2], block, 2, droprate_init, self.weight_decay, self.lamba)
        # 3rd block
        self.block3 = group_lasso_NetworkBlock(self.n, nChannels[2], nChannels[3], block, 2, droprate_init, self.weight_decay, self.lamba)
        # bn, relu and classifier
        self.bn = nn.BatchNorm2d(nChannels[3])
        self.fcout = MAPDense(nChannels[3], num_classes, weight_decay=self.weight_decay)

        self.layers, self.bn_params = [], []
        for m in self.modules():
            if isinstance(m, MAPDense) or isinstance(m, MAPConv2d) or isinstance(m, group_lasso_Conv2d):
                self.layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

        print('Using weight decay: {}'.format(self.weight_decay))

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fcout(out)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        for bnw in self.bn_params:
            if self.lamba > 0:
                regularization += (self.weight_decay / self.N) * .5 * torch.sum(bnw.pow(2))
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
            if isinstance(layer, group_lasso_Conv2d) or isinstance(layer, MAPConv2d) or isinstance(layer,MAPDense):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, group_lasso_Conv2d):
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

class group_relaxed_L0BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, droprate_init = 0.0, weight_decay=0., lamba=0.01, beta = 0.4):
        super(group_relaxed_L0BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = group_relaxed_L0Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False,
                              lamba=lamba, beta=beta, weight_decay = weight_decay)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = MAPConv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               weight_decay=weight_decay)

        self.droprate_init = droprate_init

        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and \
                            MAPConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False,
                                      weight_decay=weight_decay) or None

    def forward(self, x):
        if not self.equalInOut:
            x = F.relu(self.bn1(x))
        else:
            out = F.relu(self.bn1(x))

        out = self.conv1(out if self.equalInOut else x)
        out = F.relu(self.bn2(out))
        if self.droprate_init > 0:
            out = F.dropout(out, p=self.droprate_init, training=self.training)
        out = self.conv2(out)
        return torch.add(out, x if self.equalInOut else self.convShortcut(x))

class group_relaxed_L0NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, droprate_init=0.0, weight_decay=0., lamba=0.01, beta = 0.4):
        super(group_relaxed_L0NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, droprate_init, weight_decay=weight_decay, lamba=lamba, beta=beta)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, droprate_init, weight_decay=0, lamba=0.01, beta = 0.4):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, droprate_init, weight_decay, lamba, beta))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class group_relaxed_L0WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, droprate_init=0.3, N=50000, beta_ema=0.99, weight_decay=5e-4, lamba=0.01, beta = 0.4):
        super(group_relaxed_L0WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        self.n = (depth - 4) // 6
        self.N = N
        self.beta_ema = beta_ema
        block = group_relaxed_L0BasicBlock

        self.weight_decay = N * weight_decay
        self.lamba = lamba
        self.beta = beta

        # 1st conv before any network block
        self.conv1 = MAPConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False,
                               weight_decay=self.weight_decay)
        # 1st block
        self.block1 = group_relaxed_L0NetworkBlock(self.n, nChannels[0], nChannels[1], block, 1, droprate_init, self.weight_decay, self.lamba, self.beta)
        # 2nd block
        self.block2 = group_relaxed_L0NetworkBlock(self.n, nChannels[1], nChannels[2], block, 2, droprate_init, self.weight_decay, self.lamba, self.beta)
        # 3rd block
        self.block3 = group_relaxed_L0NetworkBlock(self.n, nChannels[2], nChannels[3], block, 2, droprate_init, self.weight_decay, self.lamba, self.beta)
        # bn, relu and classifier
        self.bn = nn.BatchNorm2d(nChannels[3])
        self.fcout = MAPDense(nChannels[3], num_classes, weight_decay=self.weight_decay)

        self.layers, self.bn_params = [], []
        for m in self.modules():
            if isinstance(m, MAPDense) or isinstance(m, MAPConv2d) or isinstance(m, group_relaxed_L0Conv2d):
                self.layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

        print('Using weight decay: {}'.format(self.weight_decay))

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fcout(out)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        for bnw in self.bn_params:
            if self.lamba > 0:
                regularization += (self.weight_decay / self.N) * .5 * torch.sum(bnw.pow(2))
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_u_sparsity(self):
        sparsity_num = 0.
        sparsity_denom = 0.
        for layer in self.layers:
            sparsity_num += layer.count_zero_u()
            sparsity_denom += layer.count_weight()
        return sparsity_num/sparsity_denom

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
            if isinstance(layer, group_relaxed_L0Conv2d) or isinstance(layer, MAPConv2d) or isinstance(layer,MAPDense):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, group_relaxed_L0Conv2d):
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

class group_relaxed_L1BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, droprate_init = 0.0, weight_decay=0., lamba=0.01, beta = 0.4):
        super(group_relaxed_L1BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = group_relaxed_L1Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False,
                              lamba=lamba, beta=beta, weight_decay = weight_decay)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = MAPConv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               weight_decay=weight_decay)

        self.droprate_init = droprate_init

        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and \
                            MAPConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False,
                                      weight_decay=weight_decay) or None

    def forward(self, x):
        if not self.equalInOut:
            x = F.relu(self.bn1(x))
        else:
            out = F.relu(self.bn1(x))

        out = self.conv1(out if self.equalInOut else x)
        out = F.relu(self.bn2(out))
        if self.droprate_init > 0:
            out = F.dropout(out, p=self.droprate_init, training=self.training)
        out = self.conv2(out)
        return torch.add(out, x if self.equalInOut else self.convShortcut(x))

class group_relaxed_L1NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, droprate_init=0.0, weight_decay=0., lamba=0.01, beta = 0.4):
        super(group_relaxed_L1NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, droprate_init, weight_decay=weight_decay, lamba=lamba, beta=beta)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, droprate_init, weight_decay=0, lamba=0.01, beta = 0.4):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, droprate_init, weight_decay, lamba, beta))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class group_relaxed_L1WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, droprate_init=0.3, N=50000, beta_ema=0.99, weight_decay=5e-4, lamba=0.01, beta = 0.4):
        super(group_relaxed_L1WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        self.n = (depth - 4) // 6
        self.N = N
        self.beta_ema = beta_ema
        block = group_relaxed_L1BasicBlock

        self.weight_decay = N * weight_decay
        self.lamba = lamba
        self.beta = beta

        # 1st conv before any network block
        self.conv1 = MAPConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False,
                               weight_decay=self.weight_decay)
        # 1st block
        self.block1 = group_relaxed_L1NetworkBlock(self.n, nChannels[0], nChannels[1], block, 1, droprate_init, self.weight_decay, self.lamba, self.beta)
        # 2nd block
        self.block2 = group_relaxed_L1NetworkBlock(self.n, nChannels[1], nChannels[2], block, 2, droprate_init, self.weight_decay, self.lamba, self.beta)
        # 3rd block
        self.block3 = group_relaxed_L1NetworkBlock(self.n, nChannels[2], nChannels[3], block, 2, droprate_init, self.weight_decay, self.lamba, self.beta)
        # bn, relu and classifier
        self.bn = nn.BatchNorm2d(nChannels[3])
        self.fcout = MAPDense(nChannels[3], num_classes, weight_decay=self.weight_decay)

        self.layers, self.bn_params = [], []
        for m in self.modules():
            if isinstance(m, MAPDense) or isinstance(m, MAPConv2d) or isinstance(m, group_relaxed_L1Conv2d):
                self.layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

        print('Using weight decay: {}'.format(self.weight_decay))

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fcout(out)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        for bnw in self.bn_params:
            if self.lamba > 0:
                regularization += (self.weight_decay / self.N) * .5 * torch.sum(bnw.pow(2))
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_u_sparsity(self):
        sparsity_num = 0.
        sparsity_denom = 0.
        for layer in self.layers:
            sparsity_num += layer.count_zero_u()
            sparsity_denom += layer.count_weight()
        return sparsity_num/sparsity_denom

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
            if isinstance(layer, group_relaxed_L1Conv2d) or isinstance(layer, MAPConv2d) or isinstance(layer,MAPDense):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, group_relaxed_L1Conv2d):
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

class group_relaxed_TF1BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, droprate_init = 0.0, weight_decay=0., lamba=0.01, alpha = 1., beta = 0.4):
        super(group_relaxed_TF1BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = group_relaxed_TF1Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False,
                              lamba=lamba, alpha = alpha, beta=beta, weight_decay = weight_decay)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = MAPConv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               weight_decay=weight_decay)

        self.droprate_init = droprate_init

        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and \
                            MAPConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False,
                                      weight_decay=weight_decay) or None

    def forward(self, x):
        if not self.equalInOut:
            x = F.relu(self.bn1(x))
        else:
            out = F.relu(self.bn1(x))

        out = self.conv1(out if self.equalInOut else x)
        out = F.relu(self.bn2(out))
        if self.droprate_init > 0:
            out = F.dropout(out, p=self.droprate_init, training=self.training)
        out = self.conv2(out)
        return torch.add(out, x if self.equalInOut else self.convShortcut(x))

class group_relaxed_TF1NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, droprate_init=0.0, weight_decay=0., lamba=0.01, alpha = 1., beta = 0.4):
        super(group_relaxed_TF1NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, droprate_init, weight_decay=weight_decay, lamba=lamba, alpha = alpha, beta=beta)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, droprate_init, weight_decay=0, lamba=0.01, alpha = 1., beta = 0.4):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, droprate_init, weight_decay, lamba, alpha, beta))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class group_relaxed_TF1WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, droprate_init=0.3, N=50000, beta_ema=0.99, weight_decay=5e-4, lamba=0.01, alpha = 1., beta = 0.4):
        super(group_relaxed_TF1WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        self.n = (depth - 4) // 6
        self.N = N
        self.beta_ema = beta_ema
        block = group_relaxed_TF1BasicBlock

        self.weight_decay = N * weight_decay
        self.lamba = lamba
        self.alpha = alpha
        self.beta = beta

        # 1st conv before any network block
        self.conv1 = MAPConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False,
                               weight_decay=self.weight_decay)
        # 1st block
        self.block1 = group_relaxed_TF1NetworkBlock(self.n, nChannels[0], nChannels[1], block, 1, droprate_init, self.weight_decay, self.lamba, self.alpha, self.beta)
        # 2nd block
        self.block2 = group_relaxed_TF1NetworkBlock(self.n, nChannels[1], nChannels[2], block, 2, droprate_init, self.weight_decay, self.lamba, self.alpha, self.beta)
        # 3rd block
        self.block3 = group_relaxed_TF1NetworkBlock(self.n, nChannels[2], nChannels[3], block, 2, droprate_init, self.weight_decay, self.lamba, self.alpha, self.beta)
        # bn, relu and classifier
        self.bn = nn.BatchNorm2d(nChannels[3])
        self.fcout = MAPDense(nChannels[3], num_classes, weight_decay=self.weight_decay)

        self.layers, self.bn_params = [], []
        for m in self.modules():
            if isinstance(m, MAPDense) or isinstance(m, MAPConv2d) or isinstance(m, group_relaxed_TF1Conv2d):
                self.layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

        print('Using weight decay: {}'.format(self.weight_decay))

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fcout(out)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        for bnw in self.bn_params:
            if self.lamba > 0:
                regularization += (self.weight_decay / self.N) * .5 * torch.sum(bnw.pow(2))
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_u_sparsity(self):
        sparsity_num = 0.
        sparsity_denom = 0.
        for layer in self.layers:
            sparsity_num += layer.count_zero_u()
            sparsity_denom += layer.count_weight()
        return sparsity_num/sparsity_denom

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
            if isinstance(layer, group_relaxed_TF1Conv2d) or isinstance(layer, MAPConv2d) or isinstance(layer,MAPDense):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, group_relaxed_TF1Conv2d):
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

class group_relaxed_L1L2BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, droprate_init = 0.0, weight_decay=0., lamba=0.01, alpha = 1., beta = 0.4):
        super(group_relaxed_L1L2BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = group_relaxed_L1L2Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False,
                              lamba=lamba, alpha = alpha, beta=beta, weight_decay = weight_decay)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = MAPConv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               weight_decay=weight_decay)

        self.droprate_init = droprate_init

        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and \
                            MAPConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False,
                                      weight_decay=weight_decay) or None

    def forward(self, x):
        if not self.equalInOut:
            x = F.relu(self.bn1(x))
        else:
            out = F.relu(self.bn1(x))

        out = self.conv1(out if self.equalInOut else x)
        out = F.relu(self.bn2(out))
        if self.droprate_init > 0:
            out = F.dropout(out, p=self.droprate_init, training=self.training)
        out = self.conv2(out)
        return torch.add(out, x if self.equalInOut else self.convShortcut(x))

class group_relaxed_L1L2NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, droprate_init=0.0, weight_decay=0., lamba=0.01, alpha = 1., beta = 0.4):
        super(group_relaxed_L1L2NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, droprate_init, weight_decay=weight_decay, lamba=lamba, alpha = alpha, beta=beta)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, droprate_init, weight_decay=0, lamba=0.01, alpha = 1., beta = 0.4):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, droprate_init, weight_decay, lamba, alpha, beta))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class group_relaxed_L1L2WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, droprate_init=0.3, N=50000, beta_ema=0.99, weight_decay=5e-4, lamba=0.01, alpha = 1., beta = 0.4):
        super(group_relaxed_L1L2WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        self.n = (depth - 4) // 6
        self.N = N
        self.beta_ema = beta_ema
        block = group_relaxed_L1L2BasicBlock

        self.weight_decay = N * weight_decay
        self.lamba = lamba
        self.alpha = alpha
        self.beta = beta

        # 1st conv before any network block
        self.conv1 = MAPConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False,
                               weight_decay=self.weight_decay)
        # 1st block
        self.block1 = group_relaxed_L1L2NetworkBlock(self.n, nChannels[0], nChannels[1], block, 1, droprate_init, self.weight_decay, self.lamba, self.alpha, self.beta)
        # 2nd block
        self.block2 = group_relaxed_L1L2NetworkBlock(self.n, nChannels[1], nChannels[2], block, 2, droprate_init, self.weight_decay, self.lamba, self.alpha, self.beta)
        # 3rd block
        self.block3 = group_relaxed_L1L2NetworkBlock(self.n, nChannels[2], nChannels[3], block, 2, droprate_init, self.weight_decay, self.lamba, self.alpha, self.beta)
        # bn, relu and classifier
        self.bn = nn.BatchNorm2d(nChannels[3])
        self.fcout = MAPDense(nChannels[3], num_classes, weight_decay=self.weight_decay)

        self.layers, self.bn_params = [], []
        for m in self.modules():
            if isinstance(m, MAPDense) or isinstance(m, MAPConv2d) or isinstance(m, group_relaxed_L1L2Conv2d):
                self.layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

        print('Using weight decay: {}'.format(self.weight_decay))

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fcout(out)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        for bnw in self.bn_params:
            if self.lamba > 0:
                regularization += (self.weight_decay / self.N) * .5 * torch.sum(bnw.pow(2))
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_u_sparsity(self):
        sparsity_num = 0.
        sparsity_denom = 0.
        for layer in self.layers:
            sparsity_num += layer.count_zero_u()
            sparsity_denom += layer.count_weight()
        return sparsity_num/sparsity_denom

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
            if isinstance(layer, group_relaxed_L1L2Conv2d) or isinstance(layer, MAPConv2d) or isinstance(layer,MAPDense):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, group_relaxed_L1L2Conv2d):
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


class group_relaxed_SCAD_BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, droprate_init = 0.0, weight_decay=0., lamba=0.01, alpha = 3.7, beta = 0.4):
        super(group_relaxed_SCAD_BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = group_relaxed_SCAD_Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False,
                              lamba=lamba, alpha = alpha, beta=beta, weight_decay = weight_decay)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = MAPConv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               weight_decay=weight_decay)

        self.droprate_init = droprate_init

        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and \
                            MAPConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False,
                                      weight_decay=weight_decay) or None

    def forward(self, x):
        if not self.equalInOut:
            x = F.relu(self.bn1(x))
        else:
            out = F.relu(self.bn1(x))

        out = self.conv1(out if self.equalInOut else x)
        out = F.relu(self.bn2(out))
        if self.droprate_init > 0:
            out = F.dropout(out, p=self.droprate_init, training=self.training)
        out = self.conv2(out)
        return torch.add(out, x if self.equalInOut else self.convShortcut(x))

class group_relaxed_SCAD_NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, droprate_init=0.0, weight_decay=0., lamba=0.01, alpha = 3.7, beta = 0.4):
        super(group_relaxed_SCAD_NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, droprate_init, weight_decay=weight_decay, lamba=lamba, alpha = alpha, beta=beta)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, droprate_init, weight_decay=0, lamba=0.01, alpha = 3.7, beta = 0.4):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, droprate_init, weight_decay, lamba, alpha, beta))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class group_relaxed_SCAD_WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, droprate_init=0.3, N=50000, beta_ema=0.99, weight_decay=5e-4, lamba=0.01, alpha = 3.7, beta = 0.4):
        super(group_relaxed_SCAD_WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        self.n = (depth - 4) // 6
        self.N = N
        self.beta_ema = beta_ema
        block = group_relaxed_SCAD_BasicBlock

        self.weight_decay = N * weight_decay
        self.lamba = lamba
        self.alpha = alpha
        self.beta = beta

        # 1st conv before any network block
        self.conv1 = MAPConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False,
                               weight_decay=self.weight_decay)
        # 1st block
        self.block1 = group_relaxed_SCAD_NetworkBlock(self.n, nChannels[0], nChannels[1], block, 1, droprate_init, self.weight_decay, self.lamba, self.alpha, self.beta)
        # 2nd block
        self.block2 = group_relaxed_SCAD_NetworkBlock(self.n, nChannels[1], nChannels[2], block, 2, droprate_init, self.weight_decay, self.lamba, self.alpha, self.beta)
        # 3rd block
        self.block3 = group_relaxed_SCAD_NetworkBlock(self.n, nChannels[2], nChannels[3], block, 2, droprate_init, self.weight_decay, self.lamba, self.alpha, self.beta)
        # bn, relu and classifier
        self.bn = nn.BatchNorm2d(nChannels[3])
        self.fcout = MAPDense(nChannels[3], num_classes, weight_decay=self.weight_decay)

        self.layers, self.bn_params = [], []
        for m in self.modules():
            if isinstance(m, MAPDense) or isinstance(m, MAPConv2d) or isinstance(m, group_relaxed_SCAD_Conv2d):
                self.layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

        print('Using weight decay: {}'.format(self.weight_decay))

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fcout(out)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        for bnw in self.bn_params:
            if self.lamba > 0:
                regularization += (self.weight_decay / self.N) * .5 * torch.sum(bnw.pow(2))
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_u_sparsity(self):
        sparsity_num = 0.
        sparsity_denom = 0.
        for layer in self.layers:
            sparsity_num += layer.count_zero_u()
            sparsity_denom += layer.count_weight()
        return sparsity_num/sparsity_denom

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
            if isinstance(layer, group_relaxed_SCAD_Conv2d) or isinstance(layer, MAPConv2d) or isinstance(layer,MAPDense):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, group_relaxed_SCAD_Conv2d):
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

class CGES_BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, droprate_init=0.0, weight_decay=0., lamba=0.01, mu = 0.5):
        super(CGES_BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = CGES_Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False,
                              lamba=lamba, weight_decay = weight_decay, mu = 0.5)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = MAPConv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               weight_decay=weight_decay)

        self.droprate_init = droprate_init

        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and \
                            MAPConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False,
                                      weight_decay=weight_decay) or None

    def forward(self, x):
        if not self.equalInOut:
            x = F.relu(self.bn1(x))
        else:
            out = F.relu(self.bn1(x))

        out = self.conv1(out if self.equalInOut else x)
        out = F.relu(self.bn2(out))
        if self.droprate_init > 0:
            out = F.dropout(out, p=self.droprate_init, training=self.training)
        out = self.conv2(out)
        return torch.add(out, x if self.equalInOut else self.convShortcut(x))

class CGES_NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, droprate_init=0.0, weight_decay=0., lamba=0.01, mu = 0.5):
        super(CGES_NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, droprate_init, weight_decay=weight_decay, lamba=lamba, mu = mu)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, droprate_init, weight_decay=0, lamba=0.01, mu = 0.5):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, droprate_init, weight_decay, lamba, mu))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class CGES_WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, droprate_init=0.3, N=50000, beta_ema=0.99, weight_decay=5e-4, lamba=0.01, mu = 0.5):
        super(CGES_WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        self.n = (depth - 4) // 6
        self.N = N
        self.beta_ema = beta_ema
        block = CGES_BasicBlock

        self.weight_decay = N * weight_decay
        self.lamba = lamba
        self.mu = mu
       

        # 1st conv before any network block
        self.conv1 = MAPConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False,
                               weight_decay=self.weight_decay)
        # 1st block
        self.block1 = CGES_NetworkBlock(self.n, nChannels[0], nChannels[1], block, 1, droprate_init, self.weight_decay, self.lamba, self.mu)
        # 2nd block
        self.block2 = CGES_NetworkBlock(self.n, nChannels[1], nChannels[2], block, 2, droprate_init, self.weight_decay, self.lamba, self.mu)
        # 3rd block
        self.block3 = CGES_NetworkBlock(self.n, nChannels[2], nChannels[3], block, 2, droprate_init, self.weight_decay, self.lamba)
        # bn, relu and classifier
        self.bn = nn.BatchNorm2d(nChannels[3])
        self.fcout = MAPDense(nChannels[3], num_classes, weight_decay=self.weight_decay)

        self.layers, self.bn_params = [], []
        for m in self.modules():
            if isinstance(m, MAPDense) or isinstance(m, MAPConv2d) or isinstance(m, CGES_Conv2d):
                self.layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

        print('Using weight decay: {}'.format(self.weight_decay))

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fcout(out)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        for bnw in self.bn_params:
            if self.lamba > 0:
                regularization += (self.weight_decay / self.N) * .5 * torch.sum(bnw.pow(2))
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
            if isinstance(layer, CGES_Conv2d) or isinstance(layer, MAPConv2d) or isinstance(layer,MAPDense):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, CGES_Conv2d):
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