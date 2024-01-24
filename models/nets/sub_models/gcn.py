""" Graph Convolution Network for completion"""

import math

import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
	"""
	Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
	"""

	def __init__(self, in_features, out_features, bias=True, visual='n'):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input_x, adj):
		support = torch.mm(input_x, self.weight)
		output = torch.spmm(adj, support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
				+ str(self.in_features) + ' -> ' \
				+ str(self.out_features) + ')'




class GCN(nn.Module):
	"""
	Graph convolutional network for data recover.
	"""
	def __init__(self, hidden_structure, activation='relu', visual='n'):
		"""
		Parameters:
		-----------
		feature_dim: int
			Feature length.
		"""
		super(GCN, self).__init__()
		self._activation = activation

		# First Layer
		self.gc1 = GraphConvolution(hidden_structure[0], hidden_structure[0], visual)

		# Later layers
		hidden_layers = []
		for i in range(len(hidden_structure) - 1):
			hidden_layers.append(nn.Linear(hidden_structure[i], hidden_structure[i + 1]))
			if i < len(hidden_structure) - 1:
				if self._activation == 'sigmoid':
					hidden_layers.append(nn.Sigmoid())
				elif self._activation == 'leakyrelu':
					hidden_layers.append(nn.LeakyReLU(0.2, inplace=True))
				elif self._activation == 'tanh':
					hidden_layers.append(nn.Tanh())
				elif self._activation == 'relu':
					hidden_layers.append(nn.ReLU())
				else:
					raise ValueError('Unknown activation function %s' % self._activation)
		self._hidden_part = nn.Sequential(*hidden_layers)


	def forward(self, x, adj):
		"""
		Parameters:
		-----------
		x: float tensor
			Input feature matrix.
		adj: sparse matrix
			Adjacency matrix.

		Returns:
		--------
		x: float tensor
			The recovered data.
		"""
		x = F.relu(self.gc1(x, adj))

		return self._hidden_part(x)












