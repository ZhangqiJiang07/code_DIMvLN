"""Split mini Batch for model training"""

import math


def next_batch(dataset, batch_size, view, phase, epoch, drop_last=False, completion=False):
	"""
	Parameters:
	-----------
	dataset: Dictionary
		{'view1_x': , 'view2_x': , ..., 'gt_label': , 'mask': }
	batch_size: int
		Size of Batch.
	view: int
		View number of dataset.
	phase: string
		'train' or 'eval'.
	"""
	tot = dataset['view1_x'].shape[0]
	# Fix the last batch
	batch_num = math.ceil(tot / batch_size)
	for i in range(int(batch_num)):
		data_batch = {}
		start_idx = i * batch_size
		end_index = min(tot, (i + 1) * batch_size)
		for v in range(1, view + 1):
			data_batch[f'view{v}_x'] = dataset[f'view{v}_x'][start_idx:end_index, ...]
		data_batch['gt_label'] = dataset['gt_label'][start_idx:end_index, ...]
		data_batch['mask'] = dataset['mask'][start_idx:end_index, ...]

		if completion:
			data_batch['idx'] = dataset['idx'][start_idx:end_index, ...]

		if phase == 'train':
			data_batch['label_mask'] = dataset['label_mask'][start_idx:end_index, ...]
			if epoch > 1:
				data_batch['pesudo_label'] = dataset['pesudo_label'][start_idx:end_index, ...]
				data_batch['pesudo_label_mask'] = dataset['pesudo_label_mask'][start_idx:end_index, ...]

		if drop_last and ((end_index - start_idx) <= 1):
			break

		yield data_batch



def pre_train_next_batch(dataset, batch_size, view, drop_last=False, completion=False):
	"""
	Parameters:
	-----------
	dataset: Dictionary
		{'view1_x': , 'view2_x': , 'gt_label': , 'mask': }
	batch_size: int
		Size of Batch.
	"""
	tot = dataset['view1_x'].shape[0]
	# Fix the last batch
	batch_num = math.ceil(tot / batch_size)
	for i in range(int(batch_num)):
		data_batch = {}
		start_idx = i * batch_size
		end_index = min(tot, (i + 1) * batch_size)
		for v in range(1, view + 1):
			data_batch[f'view{v}_x'] = dataset[f'view{v}_x'][start_idx:end_index, ...]
		data_batch['mask'] = dataset['mask'][start_idx:end_index, ...]

		if completion:
			data_batch['idx'] = dataset['idx'][start_idx:end_index, ...]

		if drop_last and ((end_index - start_idx) <= 1):
			break

		yield data_batch





