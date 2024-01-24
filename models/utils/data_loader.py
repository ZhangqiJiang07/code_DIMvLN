"""Load data, Construct data, Split data, and Generate mini-batch"""

import os
import sys
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

import math
import torch
import random
import numpy as np

from scipy.io import loadmat, savemat
from numpy.random import randint, choice
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from graph_tools import construct_transformed_knn_adj

MAIN_DIR = '/root/jzq'
# MAIN_DIR = '/mnt/MultiView'

def normalize(x):
	""" Normalize """
	x = (x - np.min(x)) / (np.max(x) - np.min(x))
	return x




def load_data(dataset_name, main_dir=MAIN_DIR):
	"""Load data as list by dataset name.
	Parameters:
	-----------
	dataset_name: string
		Name of dataset.

	Returns:
	--------
	X_list: list, (N, view1_dimension + view2_dimension)
		Concatenated original inputs.
	Y_list: list, (N + N, )
		Corresponding lables.
	"""
	X_list = []
	Y_list = []

	if dataset_name == 'Caltech101-20':
		mat = loadmat(os.path.join(main_dir, 'data', 'Caltech101-20.mat'))
		X = mat['X'][0]
		# HOG and GIST will be used
		for view in [3, 4]:
			x = X[view].astype('float32')
			# x = normalize(x).astype('float32')
			y = np.squeeze(mat['Y']).astype('int')
			X_list.append(x)
			Y_list.append(y)

	elif dataset_name == 'Cub':
		mat = loadmat(os.path.join(main_dir, 'data', 'cub_googlenet_doc2vec_c10.mat'))
		X = mat['X'][0]
		for view in range(2):
			x = X[view].transpose().astype('float32')
			y = np.squeeze(mat['gt']).astype('int')
			X_list.append(x)
			Y_list.append(y)

	elif dataset_name == 'NUS-WIDE':
		mat = loadmat(os.path.join(main_dir, 'data', 'NUSWIDEOBJ.mat'))
		X = mat['X'][0]
		for view in range(5):
			x = X[view].astype('float32')
			y = np.squeeze(mat['Y']).astype('int')
			X_list.append(x)
			Y_list.append(y)

	elif dataset_name == 'Wiki':
		mat = loadmat(os.path.join(main_dir, 'data', 'Wikipedia.mat'))
		X = mat['data'][0]
		for view in range(2):
			x = X[view].transpose().astype('float32')
			y = np.squeeze(mat['Y']).astype('int')
			X_list.append(x)
			Y_list.append(y)

	elif dataset_name == 'HandWritten':
		mat = loadmat(os.path.join(main_dir, 'data', 'handwritten.mat'))
		X = mat['X'][0]
		for view in range(6):
			x = X[view].astype('float32')
			y = np.squeeze(mat['Y']).astype('int')
			X_list.append(x)
			Y_list.append(y)

	elif dataset_name == 'ALOI':
		mat = loadmat(os.path.join(main_dir, 'data', 'ALOI.mat'))
		X = mat['fea'][0]
		for view in range(4):
			x = X[view].astype('float32')
			y = np.squeeze(mat['gt']).astype('int')
			X_list.append(x)
			Y_list.append(y)

	elif dataset_name == 'Out-Scene':
		mat = loadmat(os.path.join(main_dir, 'data', 'OutScene.mat'))
		X = mat['fea'][0]
		for view in range(4):
			x = X[view].astype('float32')
			y = np.squeeze(mat['gt']).astype('int')
			X_list.append(x)
			Y_list.append(y)

	elif dataset_name == 'YouTubeFace':
		mat = loadmat(os.path.join(main_dir, 'data', 'YoutubeFace.mat'))
		X = mat['X']
		for view in range(5):
			x = X[view][0].astype('float32')
			y = np.squeeze(mat['Y']).astype('int')
			X_list.append(x)
			Y_list.append(y)

	elif dataset_name == 'Animal':
		mat = loadmat(os.path.join(main_dir, 'data', 'Animal.mat'))
		X = mat['X'][0]
		for view in range(4):
			x = X[view].astype('float32')
			y = np.squeeze(mat['Y']).astype('int')
			X_list.append(x)
			Y_list.append(y)

	elif dataset_name == 'STL10':
		mat = loadmat(os.path.join(main_dir, 'data', 'stl10.mat'))
		X = mat['X'][0]
		for view in range(3):
			x = X[view].astype('float32')
			y = np.squeeze(mat['Y']).astype('int')
			X_list.append(x)
			Y_list.append(y)

	elif dataset_name == 'NoisyMNIST':
		mat = loadmat(os.path.join(main_dir, 'data', 'NoisyMNIST.mat'))
		train = DataSet_NoisyMNIST(mat['X1'], mat['X2'], mat['trainLabel'])
		tune = DataSet_NoisyMNIST(mat['XV1'], mat['XV2'], mat['tuneLabel'])
		test = DataSet_NoisyMNIST(mat['XTe1'], mat['XTe2'], mat['testLabel'])
		X_list.append(np.concatenate([tune.images1, test.images1], axis=0))
		X_list.append(np.concatenate([tune.images1[:, ::-1], test.images1[:, ::-1]], axis=0))
		Y_list.append(np.concatenate([np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]))
		Y_list.append(np.concatenate([np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]))

	else:
		raise ValueError("Unkonwn dataset name: %s" % dataset_name)

	return X_list, Y_list



def make_label_mask(alldata_len, missing_rate=0.1, unlabeled_rate=0.1):
	"""Randomly mask labels.
	Parameters:
	-----------
	alldata_len: int
		Number of samples (N).
	missing_rate: float (default: 0.1)
		Number of incomplete samples / Number of all samples.
	unlabeled_rate: float (default: 0.1)
		Number of unlabeled samples / Number of all samples.

	Returns:
	--------
	label_mask: np.array, (N, 1)
		The indicator matrix of labels.
	"""
	complete_len = int(alldata_len * (1 - missing_rate))
	complete_labeled_len = int(complete_len * (1 - unlabeled_rate))
	complete_unlabeled_len = complete_len - complete_labeled_len

	incomplete_len = alldata_len - complete_len
	incomplete_labeled_len = int(incomplete_len * (1 - unlabeled_rate))
	incomplete_unlabeled_len = incomplete_len - incomplete_labeled_len


	complete_labeled = np.ones((complete_labeled_len, 1))
	complete_unlabeled = np.zeros((complete_unlabeled_len, 1))
	incomplete_labeled = np.ones((incomplete_labeled_len, 1))
	incomplete_unlabeled = np.zeros((incomplete_unlabeled_len, 1))

	label_mask = np.concatenate([complete_labeled, complete_unlabeled, incomplete_labeled, incomplete_unlabeled], axis=0)

	return label_mask



def get_mask(view_num, alldata_len, missing_rate=0.1, unlabeled_rate=0.1):
	"""Get data mask and label mask with certen missing rate and unlabeled rate."""
	# Make label mask
	label_matrix = make_label_mask(alldata_len, missing_rate, unlabeled_rate)

	# Complete data
	full_matrix = np.ones((int(alldata_len * (1 - missing_rate)), view_num))
	missing_data_len = alldata_len - int(alldata_len * (1 - missing_rate))
	missing_rate = 0.5
	if missing_data_len != 0:
		one_rate = 1.0 - missing_rate
		# view_num = 2
		if one_rate <= (1 / view_num):
			enc = OneHotEncoder()
			view_preserve = enc.fit_transform(randint(0, view_num, size=(missing_data_len, 1))).toarray()
			while view_preserve.shape[1] == 1:
				view_preserve = enc.fit_transform(randint(0, view_num, size=(missing_data_len, 1))).toarray()
			full_matrix = np.concatenate([full_matrix, view_preserve], axis=0)
			
			choice_idx = choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
			matrix = full_matrix[choice_idx]
			label_matrix = label_matrix[choice_idx]

			return matrix, label_matrix
		
		error = 1
		# view_num > 2
		while error >= 0.005:
			enc = OneHotEncoder()
			view_preserve = enc.fit_transform(randint(0, view_num, size=(missing_data_len, 1))).toarray()

			one_num = view_num * missing_data_len * one_rate - missing_data_len
			ratio = one_num / (view_num * missing_data_len)
			matrix_iter = (randint(0, 100, size=(missing_data_len, view_num)) < int(ratio * 100)).astype(np.int_)

			a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int_))
			one_num_iter = one_num / (1 - a / one_num)
			ratio = one_num_iter / (view_num * missing_data_len)
			matrix_iter = (randint(0, 100, size=(missing_data_len, view_num)) < int(ratio * 100)).astype(np.int_)
			
			matrix = ((matrix_iter + view_preserve) > 0).astype(np.int_)
			ratio = np.sum(matrix) / (view_num * missing_data_len)

			error = abs(one_rate - ratio)

		full_matrix = np.concatenate([full_matrix, matrix], axis=0)

	choice_idx = choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
	matrix = full_matrix[choice_idx]
	label_matrix = label_matrix[choice_idx]

	return matrix, label_matrix




def make_train_test_dataset(config, x_data_dic, label, method=None, missing_rate=0.5, unlabeled_rate=0.5, seed=7, split2mat=False):
	"""Divide dataset into training dataset, validation dataset, and test dataset.
	Parameters:
	-----------
	config: dictionary
	x_data_dic: dictionary. $X^{(v)}$: float tensor, (N, view_dimension_v)
		Input dataset ${X^{(v)}}_{i}^{V}$.
	label: long tensor, (N,)
		Corresponding labels.
	missing_rate: float (default: 0.5)
		Number of incomplete samples / Number of all samples.
	unlabeled_rate: float (default: 0.5)
		Number of unlabeled samples / Number of all samples.
	seed: int
		Random seed.

	Returns:
	--------
	train_val_dataset: dictionary
		Dataset for training and validation.
	test_dataset: dictionary
		Dataset for testing.
	classes: int
		Number of class.
	flag_gt: boolean
		Whether label begining from 1.
	"""
	np.random.seed(seed)

	view_num = config['view']
	x_len_list = []
	for v in range(1, view_num + 1):
		x_len_list.append(x_data_dic[f'x{v}_data'].shape[1])
		if v == 1:
			data = x_data_dic[f'x{v}_data']
		else:
			data = np.concatenate([data, x_data_dic[f'x{v}_data']], axis=1)

	# Train, validation, and test split
	X_train_val, X_test, y_train_val, y_test = train_test_split(data, label, test_size=0.2, random_state=seed, stratify=label)

	X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=seed, stratify=y_train_val)

	view_data = {}
	view_data['train'], view_data['val'], view_data['test'] = {}, {}, {}
	temp = 0
	if method['Visual'] == 'n':
		for v in range(1, view_num + 1):
			view_data['train'][f'input{v}'] = normalize(X_train[:, temp:temp + x_len_list[v - 1]])
			view_data['val'][f'input{v}'] = normalize(X_val[:, temp:temp + x_len_list[v - 1]])
			view_data['test'][f'input{v}'] = normalize(X_test[:, temp:temp + x_len_list[v - 1]])
			temp += x_len_list[v - 1]
	else:
		for v in range(1, view_num + 1):
			view_data['train'][f'input{v}'] = X_train[:, temp:temp + x_len_list[v - 1]]
			view_data['val'][f'input{v}'] = X_val[:, temp:temp + x_len_list[v - 1]]
			view_data['test'][f'input{v}'] = X_test[:, temp:temp + x_len_list[v - 1]]
			temp += x_len_list[v - 1]

	# Get train data mask
	mask_train, label_mask_train = get_mask(view_num, view_data['train']['input1'].shape[0], missing_rate, unlabeled_rate)
	mask_val, _ = get_mask(view_num, view_data['val']['input1'].shape[0], missing_rate, unlabeled_rate)
	mask_test, _ = get_mask(view_num, view_data['test']['input1'].shape[0], missing_rate, unlabeled_rate)

	if method['Visual'] == 'y':
		mat_dic = {'origin_X1': view_data['test']['input1'], 'origin_X2': view_data['test']['input2']}
		savemat('/root/jzq/DIMvLN_code/origin.mat', mat_dic)

	for v in range(1, view_num + 1):
		view_data['train'][f'input{v}'] = view_data['train'][f'input{v}'] * mask_train[:, v - 1][:, np.newaxis]
		view_data['val'][f'input{v}'] = view_data['val'][f'input{v}'] * mask_val[:, v - 1][:, np.newaxis]
		view_data['test'][f'input{v}'] = view_data['test'][f'input{v}'] * mask_test[:, v - 1][:, np.newaxis]

	if split2mat:
		# Claculate class number
		classes = np.unique(np.concatenate([y_train, y_val, y_test])).size
		if np.min(y_train) == 0:
			y_train += 1
			y_val += 1
			y_test += 1
		return view_data, mask_train, y_train, label_mask_train, mask_val, y_val, mask_test, y_test, classes

	# Set data type
	for v in range(1, view_num + 1):
		for phase in ['train', 'val', 'test']:
			view_data[phase][f'input{v}'] = torch.from_numpy(view_data[phase][f'input{v}']).float()

	mask_train = torch.from_numpy(mask_train).long()
	label_mask_train = torch.from_numpy(label_mask_train).long()

	mask_val = torch.from_numpy(mask_val).long()

	mask_test = torch.from_numpy(mask_test).long()

	y_train = torch.from_numpy(np.array(y_train)).long()
	y_val = torch.from_numpy(np.array(y_val)).long()
	y_test = torch.from_numpy(np.array(y_test)).long()

	# Claculate class number
	classes = np.unique(np.concatenate([y_train, y_val, y_test])).size

	# Let label begining from 0
	if torch.min(y_train) == 1:
		y_train -= 1
		y_val -= 1
		y_test -= 1

	# Train dataset
	train_dataset = {}
	if method['Completion'] == 'no_complete':
		# use the complete data to train
		train_complete_idx = torch.sum(mask_train, dim=1) == config['view']

		for v in range(1, view_num + 1):
			train_dataset[f'view{v}_x'] = view_data['train'][f'input{v}'][train_complete_idx]
		train_dataset['gt_label'] = y_train[train_complete_idx]
		train_dataset['label_mask'] = label_mask_train[train_complete_idx] # (N, 1)
		train_dataset['mask'] = mask_train[train_complete_idx]

	elif method['Completion'] == 'complete_with_gcn':
		for v in range(1, view_num + 1):
			train_dataset[f'view{v}_x'] = view_data['train'][f'input{v}']
		train_dataset['gt_label'] = y_train
		train_dataset['label_mask'] = label_mask_train # (N, 1)
		train_dataset['mask'] = mask_train

	elif method['Completion'] in ['random_complete', 'average_complete']:
		train_dataset['gt_label'] = y_train
		train_dataset['label_mask'] = label_mask_train # (N, 1)
		train_dataset['mask'] = torch.ones(mask_train.size()).long()

		for v in range(1, view_num + 1):
			complete_index = mask_train[:, v - 1] == 1
			train_dataset[f'view{v}_x'] = torch.zeros(view_data['train'][f'input{v}'].size())
		
			train_dataset[f'view{v}_x'][complete_index] = view_data['train'][f'input{v}'][complete_index]

			if method['Completion'] == 'random_complete':
				train_dataset[f'view{v}_x'][~complete_index] = torch.rand(view_data['train'][f'input{v}'][~complete_index].size()).float()

			elif method['Completion'] == 'average_complete':
				train_dataset[f'view{v}_x'][~complete_index] = torch.sum(view_data['train'][f'input{v}'][complete_index], dim=0) / (complete_index.int().sum() + 1e-6)



	# Validation dataset
	val_dataset = {}
	if method['Completion'] == 'complete_with_gcn':
		for v in range(1, view_num + 1):
			val_dataset[f'view{v}_x'] = view_data['val'][f'input{v}']
		val_dataset['gt_label'] = y_val
		val_dataset['mask'] = mask_val

	elif method['Completion'] in ['random_complete', 'average_complete']:
		val_dataset['gt_label'] = y_val
		val_dataset['mask'] = torch.ones(mask_val.size()).long()
		for v in range(1, view_num + 1):
			complete_index = mask_val[:, v - 1] == 1
			val_dataset[f'view{v}_x'] = torch.zeros(view_data['val'][f'input{v}'].size())
			val_dataset[f'view{v}_x'][complete_index] = view_data['val'][f'input{v}'][complete_index]

			if method['Completion'] == 'random_complete':
				val_dataset[f'view{v}_x'][~complete_index] = torch.rand(view_data['val'][f'input{v}'][~complete_index].size()).float()

			elif method['Completion'] == 'average_complete':
				val_dataset[f'view{v}_x'][~complete_index] = torch.sum(view_data['val'][f'input{v}'][complete_index], dim=0) / (complete_index.int().sum() + 1e-6)


	
	# Test dataset
	test_dataset = {}
	if method['Completion'] == 'complete_with_gcn':
		for v in range(1, view_num + 1):
			test_dataset[f'view{v}_x'] = view_data['test'][f'input{v}']
		test_dataset['gt_label'] = y_test
		test_dataset['mask'] = mask_test

	elif method['Completion'] in ['random_complete', 'average_complete']:
		test_dataset['gt_label'] = y_test
		test_dataset['mask'] = torch.ones(mask_test.size()).long()
		for v in range(1, view_num + 1):
			complete_index = mask_test[:, v - 1] == 1
			test_dataset[f'view{v}_x'] = torch.zeros(view_data['test'][f'input{v}'].size())
			test_dataset[f'view{v}_x'][complete_index] = view_data['test'][f'input{v}'][complete_index]

			if method['Completion'] == 'random_complete':
				test_dataset[f'view{v}_x'][~complete_index] = torch.rand(view_data['test'][f'input{v}'][~complete_index].size()).float()

			elif method['Completion'] == 'average_complete':
				test_dataset[f'view{v}_x'][~complete_index] = torch.sum(view_data['test'][f'input{v}'][complete_index], dim=0) / (complete_index.int().sum() + 1e-6)



	# Create transformed adjacency matrix as sparse tensor
	if method['Completion'] == 'complete_with_gcn':
		ktop = config['ktop']
		train_dataset['tfed_adj'], train_dataset['indices_dic'] = construct_transformed_knn_adj(view_data['train'], train_dataset['mask'], ktop)
		
		val_dataset['tfed_adj'], _ = construct_transformed_knn_adj(view_data['val'], val_dataset['mask'], ktop)
		
		test_dataset['tfed_adj'], _ = construct_transformed_knn_adj(view_data['test'], test_dataset['mask'], ktop)

	# Create dataset info dictionary
	complete_train_len = ((mask_train[:, 0] + mask_train[:, 1]) == 2).int().sum().item()
	complete_val_len = ((mask_val[:, 0] + mask_val[:, 1]) == 2).int().sum().item()
	complete_test_len = ((mask_test[:, 0] + mask_test[:, 1]) == 2).int().sum().item()

	dataset_info_dic = dict(
		train_len=mask_train.shape[0],
		complete_train_len=complete_train_len,
		val_len=mask_val.shape[0],
		complete_val_len=complete_val_len,
		test_len=mask_test.shape[0],
		complete_test_len=complete_test_len
		)

	# Create training and evaluation dataset
	train_val_dataset = dict(
		train=train_dataset,
		eval=val_dataset
		)

	return train_val_dataset, test_dataset, classes, dataset_info_dic









class DataSet_NoisyMNIST(object):

	def __init__(self, images1, images2, labels, fake_data=False, one_hot=False,
				dtype=np.float32):
		"""Construct a DataSet.
		one_hot arg is used only if fake_data is true.  `dtype` can be either
		`uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
		`[0, 1]`.
		"""
		if dtype not in (np.uint8, np.float32):
			raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

		if fake_data:
			self._num_examples = 10000
			self.one_hot = one_hot
		else:
			assert images1.shape[0] == labels.shape[0], (
					'images1.shape: %s labels.shape: %s' % (images1.shape,
															labels.shape))
			assert images2.shape[0] == labels.shape[0], (
					'images2.shape: %s labels.shape: %s' % (images2.shape,
															labels.shape))
			self._num_examples = images1.shape[0]
			# Convert shape from [num examples, rows, columns, depth]
			# to [num examples, rows*columns] (assuming depth == 1)
			# assert images.shape[3] == 1
			# images = images.reshape(images.shape[0],
			#                        images.shape[1] * images.shape[2])
			if dtype == np.float32 and images1.dtype != np.float32:
				# Convert from [0, 255] -> [0.0, 1.0].
				print("type conversion view 1")
				images1 = images1.astype(np.float32)

			if dtype == np.float32 and images2.dtype != np.float32:
				print("type conversion view 2")
				images2 = images2.astype(np.float32)

		self._images1 = images1
		self._images2 = images2
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def images1(self):
		return self._images1

	@property
	def images2(self):
		return self._images2

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size, fake_data=False):
		"""Return the next `batch_size` examples from this data set."""
		if fake_data:
			fake_image = [1] * 784
			if self.one_hot:
				fake_label = [1] + [0] * 9
			else:
				fake_label = 0
			return [fake_image for _ in range(batch_size)], [fake_image for _ in range(batch_size)], [fake_label for _
																										in range(batch_size)]

		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._images1 = self._images1[perm]
			self._images2 = self._images2[perm]
			self._labels = self._labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples

		end = self._index_in_epoch
		return self._images1[start:end], self._images2[start:end], self._labels[start:end]
































