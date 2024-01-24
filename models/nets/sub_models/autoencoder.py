"""Sub-model: AutoEncoder"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
	"""AutoEncoder model that maps original features into latent subspace."""

	def __init__(self, encoder_structure, activation='relu', batchnorm=True):
		"""
		Parameters:
		-----------
		encoder_structure: List
			A list of ints, hidden sizes of encoder network,
			the last int is the dimension of the latent subspace.
		activation: String
			The activation function includes "sigmoid", "tanh", "relu", and "leakyrelu".
		batchnorm: Boolean
			Whether to use the Batch Normalization layer in autoencoders.
		"""
		super(AutoEncoder, self).__init__()
		self._depth = len(encoder_structure) - 1
		self._activation = activation
		self._batchnorm = batchnorm

		# Encoder pipeline
		encoder_layers = []
		for i in range(self._depth):
			# Add hidden layer
			encoder_layers.append(nn.Linear(encoder_structure[i], encoder_structure[i+1]))
			if i < self._depth - 1:
				# Add BN layer
				if self._batchnorm:
					encoder_layers.append(nn.BatchNorm1d(encoder_structure[i+1]))
				# Add activation function
				if self._activation == 'sigmoid':
					encoder_layers.append(nn.Sigmoid())
				elif self._activation == 'leakyrelu':
					encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
				elif self._activation == 'tanh':
					encoder_layers.append(nn.Tanh())
				elif self._activation == 'relu':
					encoder_layers.append(nn.ReLU())
				else:
					raise ValueError('Unknown activation function %s' % self._activation)
		# encoder_layers.append(nn.Softmax(dim=1))
		self._encoder = nn.Sequential(*encoder_layers)

		# Decoder pipeline
		decoder_structure = [nodes for nodes in reversed(encoder_structure)]
		decoder_layers = []
		for i in range(self._depth):
			# Add hidden layer
			decoder_layers.append(nn.Linear(decoder_structure[i], decoder_structure[i+1]))
			if i < self._depth - 1:
				# Add BN layer
				if self._batchnorm:
					decoder_layers.append(nn.BatchNorm1d(decoder_structure[i+1]))
				# Add activation function
				if self._activation == 'sigmoid':
					decoder_layers.append(nn.Sigmoid())
				elif self._activation == 'leakyrelu':
					decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
				elif self._activation == 'tanh':
					decoder_layers.append(nn.Tanh())
				elif self._activation == 'relu':
					decoder_layers.append(nn.ReLU())
				else:
					raise ValueError('Unknown activation function %s' % self._activation)
		self._decoder = nn.Sequential(*decoder_layers)

	def encode(self, x):
		""" Encode features.
		Parameters:
		-----------
		x: [sample_num, feature_dim], float tensor
			Input features.

		Returns:
		--------
		latent_representation: [sample_num, latent_dim], float tensor
			The representation of the samples in the latent subspace.
		"""
		latent_representation = self._encoder(x)

		return latent_representation

	def decode(self, latent_representation):
		""" Decode features.
		Parameters:
		-----------
		latent_representation: [sample_num, latent_dim], float tensor
			The representation of the samples in the latent subspace.

		Returns:
		--------
		x_hat: [sample_num, feature_dim], float tensor
			Reconstructed x.

		"""
		x_hat = self._decoder(latent_representation)
		return x_hat

	def forward(self, x):
		latent_representation = self.encode(x)
		x_hat = self.decode(latent_representation)
		
		return x_hat, latent_representation


































