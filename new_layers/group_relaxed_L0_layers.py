from __future__ import absolute_import
import torch
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair as pair
from torch.nn import init
from torch.nn import Hardshrink
import numpy as np


class group_relaxed_L0Dense(Module):
	"""Implementation of TFL regularization for the input units of a fully connected layer"""
	def __init__(self, in_features, out_features, bias=True, lamba=1., beta= 4., weight_decay=1., **kwargs):
		"""
		:param in_features: input dimensionality
		:param out_features: output dimensionality
		:param bias: whether we use bias
		:param lamba: strength of the TF1 regularization
		"""
		super(group_relaxed_L0Dense,self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.Tensor(in_features, out_features))
		self.u = torch.rand(in_features, out_features)
		self.u = self.u.to('cuda')
		if bias:
			self.bias = Parameter(torch.Tensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.lamba = lamba
		self.beta = beta
		self.weight_decay = weight_decay
		self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
		self.reset_parameters()
		print(self)

	def reset_parameters(self):
		init.kaiming_normal(self.weight, mode='fan_out')

		if self.bias is not None:
			self.bias.data.normal_(0,1e-2)

	def constrain_parameters(self, **kwargs):
		#self.weight.data = F.normalize(self.weight.data, p=2, dim=1)
		m = Hardshrink((2*self.lamba/self.beta)**(1/2))
		self.u.data = m(self.weight.data)

	def grow_beta(self, growth_factor):
		self.beta = self.beta*growth_factor

	def _reg_w(self, **kwargs):
		logpw = -self.beta*torch.sum(0.5*self.weight.add(-self.u).pow(2))-self.lamba*np.sqrt(self.out_features)*torch.sum(torch.pow(torch.sum(self.weight.pow(2),1),0.5))
		logpb = 0
		if self.bias is not None:
			logpb = - torch.sum(self.weight_decay * .5 * (self.bias.pow(2)))
		return logpw + logpb

	def regularization(self):
		return self._reg_w()

	def count_zero_u(self):
		total = np.prod(self.u.size())
		zero = total - self.u.nonzero().size(0)
		return zero

	def count_zero_w(self):
		return torch.sum((self.weight.abs()<1e-5).int()).item()

	def count_weight(self):
		return np.prod(self.u.size())

	def count_active_neuron(self):
		return torch.sum(torch.sum(self.weight.abs()/self.out_features,1)>1e-5).item() 

	def count_total_neuron(self):
		return self.in_features

	def count_expected_flops_and_l0(self):
		ppos = torch.sum(self.weight.abs()>0.000001).item()
		expected_flops = (2*ppos-1)*self.out_features
		expected_l0 = ppos*self.out_features
		if self.bias is not None:
			expected_flops += self.out_features
			expected_l0 += self.out_features
		return expected_flops, expected_l0

	def forward(self, input):
		output = input.mm(self.weight)
		if self.bias is not None:
			output.add_(self.bias.view(1, self.out_features).expand_as(output))
		return output

	def __repr__(self):
		return self.__class__.__name__+' (' \
			+ str(self.in_features) + ' -> ' \
			+ str(self.out_features) + ', lambda: ' \
			+ str(self.lamba) + ')'

class group_relaxed_L0Conv2d(Module):
	"""Implementation of TF1 regularization for the feature maps of a convolutional layer"""
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
		lamba=1., beta=4., weight_decay = 1., **kwargs):
		"""
		:param in_channels: Number of input channels
		:param out_channels: Number of output channels
		:param kernel_size: size of the kernel
		:param stride: stride for the convolution
		:param padding: padding for the convolution
		:param dilation: dilation factor for the convolution
		:param groups: how many groups we will assume in the convolution
		:param bias: whether we will use a bias
		:param lamba: strength of the TFL regularization
		"""
		super(group_relaxed_L0Conv2d, self).__init__()
		self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = pair(kernel_size)
		self.stride = pair(stride)
		self.padding = pair(padding)
		self.dilation = pair(dilation)
		self.output_padding = pair(0)
		self.groups = groups
		self.lamba = lamba
		self.beta = beta
		self.weight_decay = weight_decay
		self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
		self.u = torch.rand(out_channels, in_channels // groups, *self.kernel_size)
		self.u = self.u.to('cuda')
		if bias:
			self.bias = Parameter(torch.Tensor(out_channels))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()
		self.input_shape = None
		print(self)

	def reset_parameters(self):
		init.kaiming_normal(self.weight, mode='fan_in')
		

		if self.bias is not None:
			self.bias.data.normal_(0,1e-2)

	def constrain_parameters(self, thres_std=1.):
		#self.weight.data = F.normalize(self.weight.data, p=2, dim=1)
		#print(torch.sum(self.weight.pow(2)))
		m = Hardshrink((2*self.lamba/self.beta)**(1/2))
		self.u.data = m(self.weight.data)

	def grow_beta(self, growth_factor):
		self.beta = self.beta*growth_factor

	def _reg_w(self, **kwargs):
		logpw = -self.beta*torch.sum(0.5*self.weight.add(-self.u).pow(2))-self.lamba*np.sqrt(self.in_channels*self.kernel_size[0]*self.kernel_size[1])*torch.sum(torch.pow(torch.sum(self.weight.pow(2),3).sum(2).sum(1),0.5))
		logpb = 0
		if self.bias is not None:
			logpb = - torch.sum(self.weight_decay * .5 * (self.bias.pow(2)))
		return logpw+logpb

	def regularization(self):
		return self._reg_w()

	def count_zero_u(self):
		total = np.prod(self.u.size())
		zero = total - self.u.nonzero().size(0)
		return zero

	def count_zero_w(self):
		return torch.sum((self.weight.abs()<1e-5).int()).item()

	def count_active_neuron(self):
		return torch.sum((torch.sum(self.weight.abs(),3).sum(2).sum(1)/(self.in_channels*self.kernel_size[0]*self.kernel_size[1]))>1e-5).item()

	def count_total_neuron(self):
		return self.out_channels


	def count_weight(self):
		return np.prod(self.u.size())

	def count_expected_flops_and_l0(self):
		#ppos = self.out_channels
		ppos = torch.sum(torch.sum(self.weight.abs(),3).sum(2).sum(1)>0.001).item()
		n = self.kernel_size[0]*self.kernel_size[1]*self.in_channels
		flops_per_instance = n+(n-1)

		num_instances_per_filter = ((self.input_shape[1] -self.kernel_size[0]+2*self.padding[0])/self.stride[0]) + 1
		num_instances_per_filter *=((self.input_shape[2] - self.kernel_size[1]+2*self.padding[1])/self.stride[1]) + 1

		flops_per_filter = num_instances_per_filter * flops_per_instance
		expected_flops = flops_per_filter*ppos
		expected_l0 = n*ppos

		if self.bias is not None:
			expected_flops += num_instances_per_filter*ppos
			expected_l0 += ppos
		return expected_flops, expected_l0

	def forward(self, input_):
		if self.input_shape is None:
			self.input_shape = input_.size()
		output = F.conv2d(input_, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
		return output

	def __repr__(self):
		s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size} '
			', stride={stride}')
		if self.padding != (0,) * len(self.padding):
			s += ', padding={padding}'
		if self.dilation != (1,) * len(self.dilation):
			s += ', dilation={dilation}'
		if self.output_padding != (0,) * len(self.output_padding):
			s += ', output_padding={output_padding}'
		if self.groups != 1:
			s += ', groups={groups}'
		if self.bias is None:
			s += ', bias=False'
		s += ')'
		return s.format(name=self.__class__.__name__, **self.__dict__)
