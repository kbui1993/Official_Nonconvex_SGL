from __future__ import print_function
import torch
import torch.nn as nn
import new_layers
from utils import get_flat_fts
from copy import deepcopy
import torch.nn.functional as F
import torch.nn.init as init

class L0ModelCNN(nn.Module):

    def __init__(self, droprate_init=0.3, N=60000, weight_decay=5e-4, local_rep=False, lamba = 0.01, temperature=2./3.):
        super(L0ModelCNN, self).__init__()

        self.N = N
        self.lamba = lamba

        self.conv1 = new_layers.L0Conv2d(1,32,kernel_size = 5, bias= False, droprate_init = droprate_init, temperature = temperature,
            weight_decay = weight_decay, lamba = self.lamba, local_rep = local_rep)
        self.conv2 = new_layers.L0Conv2d(32,64,kernel_size = 5, bias= False, droprate_init = droprate_init, temperature = temperature,
            weight_decay = weight_decay, lamba = self.lamba, local_rep = local_rep)
        self.fc1 = new_layers.L0Dense(64*4*4, 1000, bias=True, weight_decay = weight_decay, droprate_init = droprate_init, temperature =temperature,
            lamba = self.lamba, local_rep = False)
        self.fc2 = new_layers.L0Dense(1000, 10, bias=True, weight_decay = weight_decay, droprate_init = droprate_init, temperature =temperature,
            lamba = self.lamba, local_rep = False)


        self.layers = []
        for m in self.modules():
            if isinstance(m,new_layers.L0Dense) or isinstance(m, new_layers.L0Conv2d):
                self.layers.append(m)

    def forward(self, input_):
        h1 = F.relu(F.max_pool2d(self.conv1(input_), 2))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(F.max_pool2d(self.conv2(h1), 2))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h2 = h2.view(-1, 4*4*64)
        h3 = F.relu(self.fc1(h2))
        h3 = F.dropout(h3, p=0.5, training=self.training)
        h4 = self.fc2(h3)
        return h4

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
            if isinstance(layer, new_layers.L0Conv2d) or isinstance(layer, new_layers.L0Dense):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, new_layers.L0Conv2d) or isinstance(layer, new_layers.L0Dense):
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

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

class CGESModelCNN(nn.Module):

    def __init__(self, N=60000, weight_decay=5e-4, lamba = 0.01):
        super(CGESModelCNN, self).__init__()

        self.N = N
        self.lamba = lamba

        self.conv1 = new_layers.CGES_Conv2d(1,32,kernel_size = 5, bias= False, weight_decay = weight_decay, lamba = self.lamba)
        self.conv2 = new_layers.CGES_Conv2d(32,64,kernel_size = 5, bias= False, weight_decay = weight_decay, lamba = self.lamba)
        self.fc1 = new_layers.CGES_Dense(64*4*4, 1000, bias=True, weight_decay = weight_decay, lamba = self.lamba)
        self.fc2 = new_layers.CGES_Dense(1000, 10, bias=True, weight_decay = weight_decay, lamba = self.lamba)


        self.layers = []
        for m in self.modules():
            if isinstance(m,new_layers.CGES_Dense) or isinstance(m, new_layers.CGES_Conv2d):
                self.layers.append(m)

    def forward(self, input_):
        h1 = F.relu(F.max_pool2d(self.conv1(input_), 2))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(F.max_pool2d(self.conv2(h1), 2))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h2 = h2.view(-1, 4*4*64)
        h3 = F.relu(self.fc1(h2))
        h3 = F.dropout(h3, p=0.5, training=self.training)
        h4 = self.fc2(h3)
        return h4

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
            if isinstance(layer, new_layers.CGES_Conv2d) or isinstance(layer, new_layers.CGES_Dense):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, new_layers.CGES_Conv2d) or isinstance(layer, new_layers.CGES_Dense):
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

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

class group_relaxed_L0ModelCNN(nn.Module):

    def __init__(self, N=60000, weight_decay=5e-4, lamba = 0.01, beta = 4.):
        super(group_relaxed_L0ModelCNN, self).__init__()

        self.N = N
        self.lamba = lamba
        self.beta = beta

        self.conv1 = new_layers.group_relaxed_L0Conv2d(1,32,kernel_size = 5, bias= False, weight_decay = weight_decay, lamba = self.lamba, beta = self.beta)
        self.conv2 = new_layers.group_relaxed_L0Conv2d(32,64,kernel_size = 5, bias= False, weight_decay = weight_decay, lamba = self.lamba, beta = self.beta)
        self.fc1 = new_layers.group_relaxed_L0Dense(64*4*4, 1000, bias=True, weight_decay = weight_decay, lamba = self.lamba, beta = self.beta)
        self.fc2 = new_layers.group_relaxed_L0Dense(1000, 10, bias=True, weight_decay = weight_decay, lamba = self.lamba, beta = self.beta)


        self.layers = []
        for m in self.modules():
            if isinstance(m,new_layers.group_relaxed_L0Dense) or isinstance(m, new_layers.group_relaxed_L0Conv2d):
                self.layers.append(m)

    def forward(self, input_):
        h1 = F.relu(F.max_pool2d(self.conv1(input_), 2))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(F.max_pool2d(self.conv2(h1), 2))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h2 = h2.view(-1, 4*4*64)
        h3 = F.relu(self.fc1(h2))
        h3 = F.dropout(h3, p=0.5, training=self.training)
        h4 = self.fc2(h3)
        return h4

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
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
            if isinstance(layer, new_layers.group_relaxed_L0Conv2d) or isinstance(layer, new_layers.group_relaxed_L0Dense):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, new_layers.group_relaxed_L0Conv2d) or isinstance(layer, new_layers.group_relaxed_L0Dense):
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

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

class group_relaxed_L1ModelCNN(nn.Module):

    def __init__(self, N=60000, weight_decay=5e-4, lamba = 0.01, beta = 4.):
        super(group_relaxed_L1ModelCNN, self).__init__()

        self.N = N
        self.lamba = lamba
        self.beta = beta

        self.conv1 = new_layers.group_relaxed_L1Conv2d(1,32,kernel_size = 5, bias= False, weight_decay = weight_decay, lamba = self.lamba, beta = self.beta)
        self.conv2 = new_layers.group_relaxed_L1Conv2d(32,64,kernel_size = 5, bias= False, weight_decay = weight_decay, lamba = self.lamba, beta =self.beta)
        self.fc1 = new_layers.group_relaxed_L1Dense(64*4*4, 1000, bias=True, weight_decay = weight_decay, lamba = self.lamba, beta = self.beta)
        self.fc2 = new_layers.group_relaxed_L1Dense(1000, 10, bias=True, weight_decay = weight_decay, lamba = self.lamba, beta = self.beta)


        self.layers = []
        for m in self.modules():
            if isinstance(m,new_layers.group_relaxed_L1Dense) or isinstance(m, new_layers.group_relaxed_L1Conv2d):
                self.layers.append(m)

    def forward(self, input_):
        h1 = F.relu(F.max_pool2d(self.conv1(input_), 2))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(F.max_pool2d(self.conv2(h1), 2))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h2 = h2.view(-1, 4*4*64)
        h3 = F.relu(self.fc1(h2))
        h3 = F.dropout(h3, p=0.5, training=self.training)
        h4 = self.fc2(h3)
        return h4

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
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
            if isinstance(layer, new_layers.group_relaxed_L1Conv2d) or isinstance(layer, new_layers.group_relaxed_L1Dense):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, new_layers.group_relaxed_L1Conv2d) or isinstance(layer, new_layers.group_relaxed_L1Dense):
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

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

class group_relaxed_L1L2ModelCNN(nn.Module):

    def __init__(self, N=60000, weight_decay=5e-4, lamba = 0.01, alpha = 1., beta = 4.):
        super(group_relaxed_L1L2ModelCNN, self).__init__()

        self.N = N
        self.lamba = lamba
        self.alpha = alpha
        self.beta = beta

        self.conv1 = new_layers.group_relaxed_L1L2Conv2d(1,32,kernel_size = 5, bias= False, weight_decay = weight_decay, lamba = self.lamba, alpha = self.alpha, beta = self.beta)
        self.conv2 = new_layers.group_relaxed_L1L2Conv2d(32,64,kernel_size = 5, bias= False, weight_decay = weight_decay, lamba = self.lamba, alpha = self.alpha, beta = self.beta)
        self.fc1 = new_layers.group_relaxed_L1L2Dense(64*4*4, 1000, bias=True, weight_decay = weight_decay, lamba = self.lamba, alpha = self.alpha, beta = self.beta)
        self.fc2 = new_layers.group_relaxed_L1L2Dense(1000, 10, bias=True, weight_decay = weight_decay, lamba = self.lamba, alpha = self.alpha, beta = self.beta)


        self.layers = []
        for m in self.modules():
            if isinstance(m,new_layers.group_relaxed_L1L2Dense) or isinstance(m, new_layers.group_relaxed_L1L2Conv2d):
                self.layers.append(m)

    def forward(self, input_):
        h1 = F.relu(F.max_pool2d(self.conv1(input_), 2))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(F.max_pool2d(self.conv2(h1), 2))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h2 = h2.view(-1, 4*4*64)
        h3 = F.relu(self.fc1(h2))
        h3 = F.dropout(h3, p=0.5, training=self.training)
        h4 = self.fc2(h3)
        return h4

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
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
            if isinstance(layer, new_layers.group_relaxed_L1L2Conv2d) or isinstance(layer, new_layers.group_relaxed_L1L2Dense):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, new_layers.group_relaxed_L1L2Conv2d) or isinstance(layer, new_layers.group_relaxed_L1L2Dense):
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

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

class group_relaxed_SCAD_ModelCNN(nn.Module):

    def __init__(self, N=60000, weight_decay=5e-4, lamba = 0.01, alpha = 3.7, beta = 4.):
        super(group_relaxed_SCAD_ModelCNN, self).__init__()

        self.N = N
        self.lamba = lamba
        self.alpha = alpha
        self.beta = beta

        self.conv1 = new_layers.group_relaxed_SCAD_Conv2d(1,32,kernel_size = 5, bias= False, weight_decay = weight_decay, lamba = self.lamba, alpha = self.alpha, beta = self.beta)
        self.conv2 = new_layers.group_relaxed_SCAD_Conv2d(32,64,kernel_size = 5, bias= False, weight_decay = weight_decay, lamba = self.lamba, alpha = self.alpha, beta = self.beta)
        self.fc1 = new_layers.group_relaxed_SCAD_Dense(64*4*4, 1000, bias=True, weight_decay = weight_decay, lamba = self.lamba, alpha = self.alpha, beta = self.beta)
        self.fc2 = new_layers.group_relaxed_SCAD_Dense(1000, 10, bias=True, weight_decay = weight_decay, lamba = self.lamba, alpha = self.alpha, beta = self.beta)


        self.layers = []
        for m in self.modules():
            if isinstance(m,new_layers.group_relaxed_SCAD_Dense) or isinstance(m, new_layers.group_relaxed_SCAD_Conv2d):
                self.layers.append(m)

    def forward(self, input_):
        h1 = F.relu(F.max_pool2d(self.conv1(input_), 2))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(F.max_pool2d(self.conv2(h1), 2))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h2 = h2.view(-1, 4*4*64)
        h3 = F.relu(self.fc1(h2))
        h3 = F.dropout(h3, p=0.5, training=self.training)
        h4 = self.fc2(h3)
        return h4

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
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
            if isinstance(layer, new_layers.group_relaxed_SCAD_Conv2d) or isinstance(layer, new_layers.group_relaxed_SCAD_Dense):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, new_layers.group_relaxed_SCAD_Conv2d) or isinstance(layer, new_layers.group_relaxed_SCAD_Dense):
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

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

class group_relaxed_TF1_ModelCNN(nn.Module):

    def __init__(self, N=60000, weight_decay=5e-4, lamba = 0.01, alpha = 1.0, beta = 4.):
        super(group_relaxed_TF1_ModelCNN, self).__init__()

        self.N = N
        self.lamba = lamba
        self.alpha = alpha
        self.beta = beta

        self.conv1 = new_layers.group_relaxed_TF1Conv2d(1,32,kernel_size = 5, bias= False, weight_decay = weight_decay, lamba = self.lamba, alpha = self.alpha, beta = self.beta)
        self.conv2 = new_layers.group_relaxed_TF1Conv2d(32,64,kernel_size = 5, bias= False, weight_decay = weight_decay, lamba = self.lamba, alpha = self.alpha, beta = self.beta)
        self.fc1 = new_layers.group_relaxed_TF1Dense(64*4*4, 1000, bias=True, weight_decay = weight_decay, lamba = self.lamba, alpha = self.alpha, beta = self.beta)
        self.fc2 = new_layers.group_relaxed_TF1Dense(1000, 10, bias=True, weight_decay = weight_decay, lamba = self.lamba, alpha = self.alpha, beta = self.beta)


        self.layers = []
        for m in self.modules():
            if isinstance(m,new_layers.group_relaxed_TF1Dense) or isinstance(m, new_layers.group_relaxed_TF1Conv2d):
                self.layers.append(m)

    def forward(self, input_):
        h1 = F.relu(F.max_pool2d(self.conv1(input_), 2))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(F.max_pool2d(self.conv2(h1), 2))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h2 = h2.view(-1, 4*4*64)
        h3 = F.relu(self.fc1(h2))
        h3 = F.dropout(h3, p=0.5, training=self.training)
        h4 = self.fc2(h3)
        return h4

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
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
            if isinstance(layer, new_layers.group_relaxed_TF1Conv2d) or isinstance(layer, new_layers.group_relaxed_TF1Dense):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, new_layers.group_relaxed_TF1Conv2d) or isinstance(layer, new_layers.group_relaxed_TF1Dense):
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

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

class group_lasso_ModelCNN(nn.Module):

    def __init__(self, N=60000, weight_decay=5e-4, lamba = 0.01):
        super(group_lasso_ModelCNN, self).__init__()

        self.N = N
        self.lamba = lamba


        self.conv1 = new_layers.group_lasso_Conv2d(1,32,kernel_size = 5, bias= False, weight_decay = weight_decay, lamba = self.lamba)
        self.conv2 = new_layers.group_lasso_Conv2d(32,64,kernel_size = 5, bias= False, weight_decay = weight_decay, lamba = self.lamba)
        self.fc1 = new_layers.group_lasso_Dense(64*4*4, 1000, bias=True, weight_decay = weight_decay, lamba = self.lamba)
        self.fc2 = new_layers.group_lasso_Dense(1000, 10, bias=True, weight_decay = weight_decay, lamba = self.lamba)


        self.layers = []
        for m in self.modules():
            if isinstance(m,new_layers.group_lasso_Dense) or isinstance(m, new_layers.group_lasso_Conv2d):
                self.layers.append(m)

    def forward(self, input_):
        h1 = F.relu(F.max_pool2d(self.conv1(input_), 2))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(F.max_pool2d(self.conv2(h1), 2))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h2 = h2.view(-1, 4*4*64)
        h3 = F.relu(self.fc1(h2))
        h3 = F.dropout(h3, p=0.5, training=self.training)
        h4 = self.fc2(h3)
        return h4

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
            if isinstance(layer, new_layers.group_lasso_Conv2d) or isinstance(layer, new_layers.group_lasso_Dense):
                neuron+=layer.count_total_neuron()

        return neuron


    def count_reg_neuron_sparsity(self):
        neuron = 0
        total = 0
        for layer in self.layers:
            if isinstance(layer, new_layers.group_lasso_Conv2d) or isinstance(layer, new_layers.group_lasso_Dense):
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

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params