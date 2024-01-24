"""Semi-supervised Incomplete Multi-view Classification


Notes: If you want to use more views input,
	please update Completion with GCN part (i.e. 'complete_with_gcn')
	and Representation part!!
"""

import torch
import torch.nn as nn

from sub_models.autoencoder import AutoEncoder
from sub_models.classifier import Classifier
from sub_models.gcn import GCN
from scipy.io import savemat


class Net(nn.Module):
	def __init__(self, config, method, classes=2, device=None):
		super(Net, self).__init__()

		# Same latent dimension
		assert (config['Autoencoder']['arch1'][-1] == config['Autoencoder']['arch2'][-1])

		self.config = config
		self.device = device
		self.method = method
		self.latent_dim = config['Autoencoder']['arch1'][-1]

		"""
		+---------------+
		|Completion part|
		+---------------+
			'complete_with_gcn': use graph convolution network to complete missing instances.
			'no_complete': no complete part, in other word, only complete instances will be used to train the model.
			'random_complete': use 'torch.rand' function to randomly generate vectors as the completed vectors.
			'average_complete': use an average vector of a within-view mini-batch as the completed vector.
		"""
		if method['Completion'] == 'complete_with_gcn':
			# GCN dictionary will be used to complete data
			# Build sub-GCNs
			self.gcn1 = GCN(config['GCN']['arch1'], config['GCN']['activation1'], method['Visual'])
			self.gcn2 = GCN(config['GCN']['arch2'], config['GCN']['activation2'], method['Visual'])

			# Put into a dictionary
			self.gcn_dic = {}
			self.gcn_dic['gcn1'] = self.gcn1
			self.gcn_dic['gcn2'] = self.gcn2

		# elif method['Completion'] in ['no_complete', 'random_complete', 'average_complete']:
			# Cross-view predictors (Note: a: view1; b: view2)
			# Dual Prediction part will be applied
			# self.predictor_a2b = Predictor(self.config['Prediction']['arch1'],
			# 							self.config['Prediction']['activation1'],
			# 							self.config['Prediction']['batchnorm'])

			# self.predictor_b2a = Predictor(self.config['Prediction']['arch2'],
			# 							self.config['Prediction']['activation2'],
			# 							self.config['Prediction']['batchnorm'])

		# else:
			# raise ValueError('Unknown completion method %s' % method['Completion'])


		"""
		+-------------------+
		|Representation part|
		+-------------------+
			'autoencoder': the autoencoder will be applied in each view.
			'shared_specific': the shared and specific model will be applied in all views.
		"""
		if method['Representation'] == 'autoencoder':
			# View specific autoencoders
			# Build sub-AEs
			self.ae1 = AutoEncoder(self.config['Autoencoder'][f'arch1'],
									self.config['Autoencoder'][f'activation1'],
									self.config['Autoencoder']['batchnorm'])

			self.ae2 = AutoEncoder(self.config['Autoencoder'][f'arch2'],
									self.config['Autoencoder'][f'activation2'],
									self.config['Autoencoder']['batchnorm'])

			# Put into a dictionary
			self.ae_dic = {}
			self.ae_dic['ae1'] = self.ae1
			self.ae_dic['ae2'] = self.ae2

		else:
			raise ValueError('Unknown representation method %s' % method['Representation'])


		"""
		+-------------------+
		|Classification part|
		+-------------------+
			Fully connected layer based.
		"""
		self.classifier = Classifier(self.config['view'], self.latent_dim, classes)



	def _train(self, X, incomplete_input_subadj=None):
		""" Prediction with training data (contain incomplete data) with Dual Prediction.
		Parameters:
		-----------
		X: Dictionary
			{'view1_x':[sample_num, feature_dim], float tensor,
			'view2_x':[sample_num, feature_dim], float tensor,
			'mask':[sample_num, view_num], int}
		"""

		output_dic = {}

		concatenate_z = torch.FloatTensor().to(self.device)

		# Within-view
		if self.method['Completion'] in ['no_complete', 'average_complete', 'random_complete']:
			for v in range(1, self.config['view']+1):
				existing_idx = X['mask'][:, v - 1] == 1
				# Construct container for representation vector z{v}
				output_dic[f'z{v}'] = torch.zeros(X[f'view{v}_x'].shape[0], self.latent_dim).to(self.device)
				output_dic[f'x{v}_recon'], output_dic[f'z{v}'][existing_idx] = self.ae_dic[f'ae{v}'](X[f'view{v}_x'][existing_idx])
				output_dic[f'view{v}_x'] = X[f'view{v}_x'][existing_idx].clone()

				concatenate_z = torch.cat([concatenate_z, output_dic[f'z{v}']], dim=1)

		elif self.method['Completion'] == 'complete_with_gcn':
			for v in range(1, self.config['view']+1):
				# GCN completion
				completed_input = self._gcn_complete(incomplete_input_subadj)
				missing_idx = X['mask'][:, v - 1] == 0
				output_dic[f'view{v}_x'] = X[f'view{v}_x'].clone() + torch.tensor(0.0).cuda()
				output_dic[f'view{v}_x'][missing_idx] = completed_input[f'completed_input{v}'][missing_idx].clone() + torch.tensor(0.0).cuda()

				output_dic[f'x{v}_recon'], output_dic[f'z{v}'] = self.ae_dic[f'ae{v}'](output_dic[f'view{v}_x'])
				concatenate_z = torch.cat([concatenate_z, output_dic[f'z{v}']], dim=1)

		else:
			raise ValueError('Unknown completion method %s' % self.method['Completion'])


		# Classification
		output_dic['label_c'], output_dic['prob'] = self.classifier(concatenate_z)

		return output_dic


	def _eval(self, X, incomplete_input_subadj=None, visual='n'):
		""" Prediction with evaluation data (contain incomplete data) with Dual Prediction.
		Parameters:
		-----------
		X: Dictionary
			{'view1_x':[sample_num, feature_dim], float tensor,
			'view2_x':[sample_num, feature_dim], float tensor,
			'mask':[sample_num, view_num], int}
		"""
		# x_v1 = X['view1_x']
		# x_v2 = X['view2_x']
		# mask = X['mask']

		if self.method['Completion'] in ['no_complete', 'average_complete', 'random_complete']:
			latent_code = torch.FloatTensor().to(self.device)
			for v in range(1, self.config['view']+1):
				# Encode
				_, temp_latent_code = self.ae_dic[f'ae{v}'](X[f'view{v}_x'])
				latent_code = torch.cat([latent_code, temp_latent_code], dim=1)

		elif self.method['Completion'] == 'complete_with_gcn':
			# Create latent vector container
			latent_code = torch.FloatTensor().to(self.device)
			for v in range(1, self.config['view']+1):
				# GCN completion
				completed_input = self._gcn_complete(incomplete_input_subadj)
				missing_idx = X['mask'][:, v - 1] == 0
				X[f'view{v}_x'][missing_idx] = completed_input[f'completed_input{v}'][missing_idx] + torch.tensor(0.0).cuda()
				# Encode
				_, temp_latent_code = self.ae_dic[f'ae{v}'](X[f'view{v}_x'])
				latent_code = torch.cat([latent_code, temp_latent_code], dim=1)

			if visual == 'y':
				matlab_dic = {'recovered_X1': X['view1_x'].cpu().numpy(), 'recovered_X2': X['view2_x'].cpu().numpy(), 'mask': X['mask'].cpu().numpy()}
				savemat('/root/jzq/DIMvLN_code/visual.mat', matlab_dic)

		else:
			raise ValueError('Unknown completion method %s' % self.method['Completion'])

		# Classification
		label_c, prob = self.classifier(latent_code)

		return {'label_c': label_c, 'prob': prob}


	def _gcn_complete(self, X):
		"""Completing original input data with transformed kNN adjacency matrix via GCN.
		Parameters:
		-----------
		X: dictionary
			Incomplete input data.

		Returns:
		--------
		completed_input: dictionary
			Complete data using GCN and transformed adjacency.
		"""
		completed_input = {}
		for v in range(1, self.config['view']+1):
			completed_input[f'completed_input{v}'] = self.gcn_dic[f'gcn{v}'](X[f'view{v}_x'], X['tfed_adj'][f'tf_adj{v}'])

		return completed_input



	def forward(self, X, mode='train', incomplete_input_subadj=None, visual='n'):
		""" Get the output with Complete data or Incomplete data.
		Parameters:
		-----------
		X: Dictionary
			{'view1_x':[sample_num, feature_dim], float tensor,
			'view2_x':[sample_num, feature_dim], float tensor,
			'mask':[sample_num, view_num], int,

			'tfed_adj': dictionary of the transformed kNN adjacency of views (if completion method is 'complete_with_gcn').}
		
		mode: String
			'train': Input is Complete data;
			'eval': Input is Incomplete data;
			'only_gcn': Use GCN to complete input data.
		"""

		if mode == 'train':
			return self._train(X, incomplete_input_subadj)
		
		elif mode == 'eval':
			return self._eval(X, incomplete_input_subadj, visual)
		
		elif mode == 'only_gcn':
			return self._gcn_complete(X)
		
		else:
			raise ValueError('Unknown mode %s' % mode)


































