"""Fully connected layers based classifier"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Classifier(nn.Module):
	"""Classifier with fully connected layers"""
	def __init__(self, view_num, latent_dim, classes, activation='relu', batchnorm=True):
		"""Fully connected layers based classfifer.
		Parameters:
		-----------
		view_num: int
			Number of views.
		latent_dim: int
			Dimension of latent space.
		classes: int
			Number of classes.
		activation: string (default: 'relu'')
			Type of activation function.
		batchnorm: boolean (default: True)
			Whether apply Batch Normalization.
		"""

		super(Classifier, self).__init__()
		self._batchnorm = batchnorm
		self._activation = activation

		structure = [int(view_num * latent_dim), int(latent_dim / 2), int(classes)]
		classifier_layers = []
		for i in range(len(structure) - 1):
			classifier_layers.append(nn.Linear(structure[i], structure[i + 1]))
			if i < len(structure) - 2:
				if self._batchnorm:
					classifier_layers.append(nn.BatchNorm1d(structure[i + 1]))
				if self._activation == 'sigmoid':
					classifier_layers.append(nn.Sigmoid())
				elif self._activation == 'leakyrelu':
					classifier_layers.append(nn.LeakyReLU(0.2, inplace=True))
				elif self._activation == 'tanh':
					classifier_layers.append(nn.Tanh())
				elif self._activation == 'relu':
					classifier_layers.append(nn.ReLU())
				else:
					raise ValueError('Unknown activation function %s' % self._activation)
		classifier_layers.append(nn.Softmax(dim=1))
		self._classifier = nn.Sequential(*classifier_layers)

	def forward(self, concat_z):
		"""
		Parameters:
		-----------
		concat_z: float tensor (N, view_num * latent_dim) if use concatenate

		Return:
		-------
		label_pre: np.array, (N,)
			Array of the predicted labels.
		prob: float tensor, (N, classes)
			The probability matrix for clasifying and calculating classification loss (entropy loss).
		"""
		prob = self._classifier(concat_z)

		label_pre = np.argmax(prob.detach().cpu().numpy(), axis=1)

		return label_pre, prob


