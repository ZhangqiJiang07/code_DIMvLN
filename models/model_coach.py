"""Model Coach to train the model"""

import os
import copy
import numpy as np
from scipy.io import savemat

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from numpy.random import choice

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

from utils.data_batch import next_batch
from utils.data_batch import pre_train_next_batch


SUB_HEADER = ' Epoch     Loss     Acc     Loss     Acc'
LOSS_METRICE = {'Training': ['L_rc', 'L_icl', 'L_ccl', 'L_cel'], 'Validation': ['Loss']}
PERFORMANCE = {'Training': ['ACC'], 'Validation': ['ACC']}
SPACE = '     '


def print_header(loss={'Training': ['Loss'], 'Validation': ['Loss']}, metric={'Training': ['ACC'], 'Validation': ['ACC']}):
    header = '           '
    mid_line = '           -'
    sub_header = ' Epoch '
    
    for phase in ['Training', 'Validation']:
        loss_idx = 0
        for sub_loss in loss[phase]:
            sub_header += SPACE
            sub_header += sub_loss
            mid_line += '-' * (len(sub_loss) + len(SPACE))

        metric_idx = 0
        for sub_metric in metric[phase]:
            sub_header += SPACE
            sub_header += sub_metric
            if metric_idx < len(metric[phase]) - 1:
                mid_line += '-' * (len(sub_metric) + len(SPACE))
                metric_idx += 1
            else:
                mid_line += '-' * (len(sub_metric) + 1)

        header = header + ' ' * int((len(mid_line) - len(header))/2 - 4) + phase
        header += ' ' * int(len(mid_line) - len(header))
        if phase == 'Training':
            header += SPACE
            mid_line += SPACE
            sub_header += ' ' * 2
    mid_line += '-'
    print('-' * (len(sub_header) + 2))
    print(header)
    print(mid_line)
    
    print(sub_header)
    print('-' * (len(sub_header) + 2))
    
    return len(sub_header)


def print_buttom_line(sub_header_len):
	print('*' * (sub_header_len + 2))



class ModelCoach:
	def __init__(self, views, model, dataset, optimizer, criterion, method, classes=2, device=None):
		"""
		Parameters:
		-----------
		views: int
			View number of the used dataset.
		model: class
			Proposed net.
		dataset: dictionary
			{'view1_x': X1, 'view2_x': X2, ..., 'gt_label': , 'mask': }.
		optimizer: class (default: Adam)
			Optimizer.
		criterion: class
			Loss function.
		method: dictionary
			Used methods conbination.
		classes: int
			Number of classes.
		device: class (default: gpu:0)
			GPU or CPU.
		"""

		self.views = views
		self.model = model
		self.dataset = dataset
		self.optimizer = optimizer
		self.criterion = criterion.to(device)
		self.classes = classes
		self.device = device
		self.method = method

		# Record the best score and weights
		self.best_perf = {'best_acc': 0.0}
		self.best_wts = {'best_acc_wts': None}
		self.current_perf = {'epoch a': 0.0}



	def _log_info(self, logger, phase, epoch, epoch_loss, epoch_acc):
		""" Write Logger info.
		Parameters:
		-----------
		logger: class
			SummaryWriter.
		phase: string
			'train' or 'eval'.
		epoch: int
			Epoch.
		epoch_loss: float
			Loss.
		epoch_acc: float
			Accuracy.
		"""
		if phase == 'train':
			info = {phase + '_loss': epoch_loss,
					phase + '_acc': epoch_acc}
		elif phase == 'eval':
			info = {phase + '_acc': epoch_acc}

		elif phase == 'pre_train_gcn':
			info = {phase + '_loss': epoch_loss}

		for tag, value in info.items():
			logger.add_scalar(tag, value, epoch)



	def _data2device(self, X, completion=False):
		"""Move data to device.
		Parameters:
		-----------
		X: dictionary
			Dataset on cpu.

		Returns:
		--------
		data: dictionary
			Dataset on device.
		"""
		data = {}
		for v in range(1, self.views + 1):
			data[f'view{v}_x'] = X[f'view{v}_x'].to(self.device)
			# data['view2_x'] = X['view2_x'].to(self.device)

		if not completion:
			data['mask'] = X['mask'].to(self.device)

		return data



	def _compute_loss(self, input_dic, output_dic, gt_label=None, label_mask=None, classes=2, mode='without_dual', device=None):
		"""Use loss function to calculate the emperical risk.
		Parameters:
		-----------
		input_dic: dictionary
			Input data
		output_dic: dictionary
			Output data produced by model including $\hat{x}^{(v)}_i$, $z^{(v)}_i$, Probability Matrix, etc.
		gt_label: long tensor, (N,)
			The ground truth labels.
		label_mask: long tensor, (N,)
			The mask of labels.
		mode: string
			Indicating the loss formulation.

		Returns:
		--------
		loss: tensor with gridient
		loss_dic: dictionary
			The item of the sub-losses.{'L_rc': , 'L_icl': ,'L_ccl': ,'L_cel': }
		"""
		loss_dic = {}

		if mode == 'pre_train_gcn':
			loss_dic['original_input'] = input_dic
			# loss_dic['mask'] = output_dic['mask']
			loss_dic['completed_input'] = output_dic
			loss_dic['indices_dic'] = input_dic['indices_dic']

		elif mode in ['with_dual', 'without_dual']:
			loss_dic = output_dic
			if self.method['Completion'] == 'complete_with_gcn':
				loss_dic['mask'] = input_dic['mask']
				loss_dic['indices_dic'] = input_dic['indices_dic']

		loss, loss_dic = self.criterion(output_dic=loss_dic, gt_label=gt_label, label_mask=label_mask,
							classes=classes, mode=mode, device=device)

		return loss, loss_dic



	def _shuffle_dataset(self, dataset, epoch):
		"""
		Parameters:
		-----------
		dataset: dictionary
			Dataset.
		epoch: int
			Present epoch.

		Returns:
		--------
		s_dataset: dictionary
			Shuffled dataset.
		"""
		s_dataset = {}
		s_dataset['train'] = {}
		s_dataset['eval'] = {}

		shuffled_idx_train = np.arange(dataset['train']['view1_x'].shape[0])
		shuffled_idx_eval = np.arange(dataset['eval']['view1_x'].shape[0])

		shuffled_idx_train = shuffle(shuffled_idx_train)
		shuffled_idx_eval = shuffle(shuffled_idx_eval)

		for v in range(1, self.views + 1):
			s_dataset['train'][f'view{v}_x'] = dataset['train'][f'view{v}_x'][shuffled_idx_train, ...]
			s_dataset['eval'][f'view{v}_x'] = dataset['eval'][f'view{v}_x'][shuffled_idx_eval, ...]

		# Shuffle training set
		if epoch > 1:
			for keyW in ['mask', 'gt_label', 'label_mask', 'pesudo_label', 'pesudo_label_mask']:
				s_dataset['train'][keyW] = dataset['train'][keyW][shuffled_idx_train, ...]

		else:
			for keyW in ['mask', 'gt_label', 'label_mask']:
				s_dataset['train'][keyW] = dataset['train'][keyW][shuffled_idx_train, ...]

		# Shuffle eval set
		for keyW in ['mask', 'gt_label']:
			s_dataset['eval'][keyW] = dataset['eval'][keyW][shuffled_idx_eval, ...]

		# Shuffle instance index for GCN Completion
		if self.method['Completion'] == 'complete_with_gcn':
			s_dataset['train']['idx'] = dataset['train']['idx'][shuffled_idx_train, ...]
			s_dataset['eval']['idx'] = dataset['eval']['idx'][shuffled_idx_eval, ...]


		return s_dataset



	def _assign_labels(self, latent_z_dic, gt_label, label_mask, classes=2):
		""" Semi-Supervised:
				classify the unlabeled samples using the similarity of the labeled samples.
		Parameters:
		-----------
		latent_z_dic: dictionary
			The low dimensional representations of the original inputs $Z^{(v)}$, float tensor. (N, latent_dim)
		gt_label: long tensor, (N,)
			The ground truth labels.
		label_mask: long tensor, (N,)
			The mask of labels.
		classes: int
			Number of classes.

		Returns:
		--------
		assigned_labels: long tensor
			The assigned labels of the unlabeled samples.
		"""
		labeled_idx = label_mask[:, 0] == 1
		unlabeled_idx = label_mask[:, 0] == 0

		labeled_sample_concat_z = torch.FloatTensor().to(self.device)
		unlabeled_sample_concat_z = torch.FloatTensor().to(self.device)
		for v in range(1, self.views + 1):
			labeled_sample_concat_z = torch.cat((labeled_sample_concat_z, latent_z_dic[f'z{v}'][labeled_idx]), dim=1)
			unlabeled_sample_concat_z = torch.cat((unlabeled_sample_concat_z, latent_z_dic[f'z{v}'][unlabeled_idx]), dim=1)

		labeled_sample_concat_z = labeled_sample_concat_z.cpu().numpy()
		unlabeled_sample_concat_z = unlabeled_sample_concat_z.cpu().numpy()

		# labeled_sample_concat_z = torch.cat((z1[labeled_idx], z2[labeled_idx]), dim=1).cpu().numpy() # (N1, D)
		# unlabeled_sample_concat_z = torch.cat((z1[unlabeled_idx], z2[unlabeled_idx]), dim=1).cpu().numpy() # (N2, D)

		labeled_sample_gt_labels = gt_label[labeled_idx]

		label_onehot = F.one_hot(labeled_sample_gt_labels, classes).float().cpu().numpy() # (N1, C)

		S = np.dot(unlabeled_sample_concat_z, np.transpose(labeled_sample_concat_z)) # (N2, N1)
		label_num = np.sum(label_onehot, axis=0) + 1 # avoid zero division
		S_sum = np.dot(S, label_onehot) # (N2, C)
		S_mean = S_sum / label_num

		psd_labels = np.argmax(S_mean, axis=1)

		gt_label_copy = gt_label.cpu().numpy()
		gt_label_copy[unlabeled_idx] = psd_labels

		return torch.from_numpy(gt_label_copy).long()



	def _compare(self, gt_labels, pre_labels, assigned_labels, labels_mask):
		"""Semi-Supervised:
				compare the assigned labels (from similarity) and the predicted labels (from classifier).
		Parameters:
		-----------
		gt_labels: long tensor, (N,)
			The ground truth labels.
		pre_labels: long tensor, (N,)
			Predicted labels by classifier.
		assigned_labels: long tensor, (N,)
			Assigned labels based on similarity.
		labels_mask:long tensor, (N,)
			The mask of labels.

		Returns:
		--------
		labels_mask: long tensor, (N,)
			New mask of labels, if the predicted label and the assigned label are same,
			then the unlabeled samples will convert to labeled samples with the predicted label.
		gt_labels_with_pesudo_labels: long tensor, (N,)
			New label matrix with the pesudo labels.
		"""
		# new_labels_mask = labels_mask.clone()
		unlabeled_idx = (labels_mask[:, 0] == 0).numpy().astype(int)
		same_pesudo_label_idx = (pre_labels == assigned_labels).astype(int)

		# The index of the instances which satisfy the following constraints:
		# 1) unlabeled data, and 2) assigned label equal to predicted label.
		used_idx = (unlabeled_idx * same_pesudo_label_idx) == 1

		labels_mask[used_idx] = 1
		gt_labels[used_idx] = pre_labels[used_idx]

		return labels_mask, torch.from_numpy(gt_labels).long()



	def _pre_train_gcn(self, X, pre_train_epoch, batch_size, logger=None, info='y'):
		"""Completion:
				pre-train GCN with reconstruction loss.
		Parameters:
		-----------
		X: dictionary
			Contain view-input, view-transformed adjacency, and mask.
		pre_train_epoch: int
			The maximum iteration of pre-train phase.
		batch_size: int
			The mini-batch size during the pre-train phase.
		"""
		if info =='y':
			print('Start pre-train GCN...')
		data = self._data2device(X)

		data['indices_dic'] = {}
		incomplete_data_batch = {}
		for v in range(1, self.views + 1):
			data['indices_dic'][f'indices{v}'] = X['indices_dic'][f'indices{v}'].to(self.device)
			incomplete_data_batch[f'view{v}_x'] = data[f'view{v}_x']
		data['idx'] = X['idx'].to(self.device)
		
		incomplete_data_batch['tfed_adj'] = {}
		# Pre-train view-specific Graph Neural Networks.
		for epoch in range(pre_train_epoch):
			for data_batch in pre_train_next_batch(data, batch_size, self.views, completion=True):
				for v in range(1, self.views + 1):
					incomplete_data_batch['tfed_adj'][f'tf_adj{v}'] = torch.index_select(X['tfed_adj'][f'tf_adj{v}'].to(self.device), dim=0,
																			index=data_batch['idx'].long())

				output_dic = self.model(incomplete_data_batch, mode='only_gcn')
				output_dic['sub_mask'] = data_batch['mask']
				output_dic['sub_idx'] = data_batch['idx']
				loss, _ = self._compute_loss(data, output_dic, mode='pre_train_gcn', device=self.device)

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

			if logger is not None:
				self._log_info(logger, 'pre_train_gcn', epoch, loss.item(), 0)
			if info == 'y':
				print(f'pre_train_loss in {epoch}: {loss.item():.4f}')



	def _process_data_batch(self, data_batch, phase, epoch, classes=2, mode='without_dual', incomplete_dic=None, tf_adj_dic=None):
		"""Train model using mini-batch approch.
		Parameters:
		-----------
		data_batch: dictionary
			Mini-batch dataset.
		phase: string
			'train' or 'eval'.
		epoch: int
			Present epoch.
		classes: int
			Number of classes.
		mode: string
			Indicating the loss formulation and training way.
		incomplete_dic: dictionary (default: None)
			The incomplete input multi-view data.
		tf_adj_dic: dictionary (default: None)
			The transfered graphs $S^{(v)}_i$ of the missing instances in current mini-batch.

		Returns:
		--------
		loss: float tensor with gradient
			Mini-batch emperical risk (loss).
		output_dic: dictionary
			The dictionary of middle outputs and final outputs.
		gt_label: long tensor, (N,)
			The ground truth labels of samples.
		"""
		data = self._data2device(data_batch)
		gt_label = data_batch['gt_label'].to(self.device)
		loss = torch.tensor(0.0).to(self.device)
		loss_dic = {'Val': 0}

		with torch.set_grad_enabled(phase == 'train'):
			if incomplete_dic is None:
				output_dic = self.model(X=data, mode=phase)

			else:
				incomplete_input_subadj_dic = {}
				incomplete_input_subadj_dic['tfed_adj'] = {}
				for v in range(1, len(incomplete_dic) + 1):
					incomplete_input_subadj_dic[f'view{v}_x'] = incomplete_dic[f'view{v}_x']
					incomplete_input_subadj_dic['tfed_adj'][f'tf_adj{v}'] = torch.index_select(tf_adj_dic[f'tf_adj{v}'], dim=0,
																				index=data_batch['idx'].long()).to(self.device)
				output_dic = self.model(X=data, mode=phase,
									incomplete_input_subadj=incomplete_input_subadj_dic)

			if phase == 'train':
				if incomplete_dic is not None:
					data['indices_dic'] = {}
					for v in range(1, len(incomplete_dic) + 1):
						data['indices_dic'][f'indices{v}'] = self.dataset['train']['indices_dic'][f'indices{v}'].to(self.device)

				if epoch > 1:
					pesudo_label = data_batch['pesudo_label'].to(self.device)
					label_mask = data_batch['pesudo_label_mask'].to(self.device)
				else:
					pesudo_label = gt_label
					label_mask = data_batch['label_mask'].to(self.device)

				loss, loss_dic = self._compute_loss(input_dic=data, output_dic=output_dic,
											gt_label=pesudo_label, label_mask=label_mask,
											classes=classes, mode=mode, device=self.device)

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

		return loss, output_dic, gt_label, loss_dic


	def _note_ratio(self, ratio_list, path='./ratio.txt'):
		with open(path, 'w') as ratio_file:
			for itr in range(len(ratio_list)):
				ratio_file.write(f"{ratio_list[itr]}\n")

		ratio_file.close()



	def _run_training_loop(self, num_epochs, start_dual_pred_epoch, pre_train_epoch, recomplete_freq, batch_size, log_dir, info_freq):
		"""
		Parameters:
		-----------
		num_epochs: int
			Number of training epochs.
		start_dual_pred_epoch: int
			The epoch of starting the dual prediction part.
		pre_train_epoch: int
			Number of epoch to pre-train GCNs.
		recomplete_freq: int
			Gap between two completing operations.
		batch_size: int
			Batch size.
		log_dir: string
			The direction of logger.
		info_freq: int
			The frequency of printing training (evaluation) information.
		"""
		logger = SummaryWriter(log_dir)

		# Pre-train GCN
		completion = False
		if self.method['Completion'] == 'complete_with_gcn':
			completion = True

			# Insert index from sparse matrix slicing.
			self.dataset['train']['idx'] = torch.tensor(range(self.dataset['train']['view1_x'].shape[0]))
			self.dataset['eval']['idx'] = torch.tensor(range(self.dataset['eval']['view1_x'].shape[0]))

			incomplete_input_dic = {}
			incomplete_input_dic['train'] = self._data2device(self.dataset['train'], completion)
			incomplete_input_dic['eval'] = self._data2device(self.dataset['eval'], completion)

			self._pre_train_gcn(self.dataset['train'], pre_train_epoch, batch_size, logger)

		# Print header
		if info_freq is not None:
			print()
			print(' Epoch ')
			print_buttom_line(20)


		mode = 'without_dual'
		for epoch in range(1, num_epochs + 1):
			
			if info_freq is None:
				print_info = False
			else:
				print_info = (epoch == 1) or (epoch % info_freq == 0)

			# Shuffle dataset
			if epoch == 1:
				shuffled_dataset_dic = self._shuffle_dataset(self.dataset, epoch)
			else:
				shuffled_dataset_dic = self._shuffle_dataset(shuffled_dataset_dic, epoch)

			for phase in ['train', 'eval']:
				# Completion setup
				if self.method['Completion'] in ['no_complete', 'random_complete', 'average_complete']:
					# Whether use Dual Prediction for training?
					if mode == 'without_dual' and epoch > start_dual_pred_epoch:
						print('Start Dual Prediction ...')
						mode = 'with_dual'

				if phase == 'train':
					self.model.train()
				else:
					self.model.eval()

				# Create middle output containers
				running_loss = []
				running_loss_dic = {'L_rc': [], 'L_icl': [], 'L_ccl': [], 'L_cel': []}
				running_gt_labels = torch.LongTensor().to(self.device)
				running_pre_labels = torch.LongTensor()
				running_latent_z_dic = {}
				for v in range(1, self.views + 1):
					running_latent_z_dic[f'z{v}'] = torch.FloatTensor().to(self.device)

				# Training (evaluation) with mini-batch way
				# batch_id = 0
				for data_batch in next_batch(shuffled_dataset_dic[phase], batch_size, self.views, phase, epoch, drop_last=True, completion=completion):
					if self.method['Completion'] in ['no_complete', 'random_complete', 'average_complete']:
						loss, output_dic, gt_label, loss_dic = self._process_data_batch(data_batch=data_batch, phase=phase, epoch=epoch,
																		classes=self.classes, mode=mode)

					elif self.method['Completion'] == 'complete_with_gcn':
						loss, output_dic, gt_label, loss_dic = self._process_data_batch(data_batch=data_batch, phase=phase, epoch=epoch,
																		classes=self.classes, mode=mode, incomplete_dic=incomplete_input_dic[phase],
																		tf_adj_dic=self.dataset[phase]['tfed_adj'])


					running_loss.append(loss.item())
					running_gt_labels = torch.cat((running_gt_labels, gt_label.data.long()), dim=0)
					running_pre_labels = torch.cat((running_pre_labels, torch.from_numpy(output_dic['label_c']).long()), dim=0)

					# Semi-Supervised part (record output)
					if phase == 'train':
						for metrice in LOSS_METRICE['Training']:
							running_loss_dic[metrice].append(loss_dic[metrice])
						if self.method['Semi_supervised'] == 'similarity_based':
							for v in range(1, self.views + 1):
								running_latent_z_dic[f'z{v}'] = torch.cat((running_latent_z_dic[f'z{v}'], output_dic[f'z{v}'].detach()), dim=0)
						else:
							raise ValueError('Unknown Semi-supervised method %s' % self.method['Semi_supervised'])

				epoch_loss = torch.mean(torch.tensor(running_loss))
				epoch_acc = accuracy_score(running_gt_labels.cpu().numpy(), running_pre_labels.numpy())

				# Semi-Supervised part (assign pesudo-labels)
				if phase == 'train':
					print_loss_dic = {}
					for metrice in LOSS_METRICE['Training']:
						print_loss_dic[metrice] = torch.mean(torch.tensor(running_loss_dic[metrice]))

					if self.method['Semi_supervised'] == 'similarity_based':
						if epoch % 25 == 1:
							running_assigned_labels = self._assign_labels(running_latent_z_dic, running_gt_labels,
																		shuffled_dataset_dic[phase]['label_mask'],
																		self.classes)

							shuffled_dataset_dic[phase]['pesudo_label_mask'],\
							shuffled_dataset_dic[phase]['pesudo_label'] = self._compare(running_gt_labels.cpu().numpy(),
																				running_pre_labels.numpy(),
																				running_assigned_labels.numpy(),
																				shuffled_dataset_dic[phase]['label_mask'])

						else:
							shuffled_dataset_dic[phase]['pesudo_label_mask'],\
							shuffled_dataset_dic[phase]['pesudo_label'] = shuffled_dataset_dic[phase]['label_mask'], running_gt_labels

					else:
						raise ValueError('Unknown Semi-supervised method %s' % self.method['Semi_supervised'])

				if print_info:
					if phase == 'train':
						message = f' {epoch}/{num_epochs}'

						space = 10 - len(message)
						message += ' ' * space
						message += 'Train: '
						for metrice in LOSS_METRICE['Training']:
							message += f'{metrice}==>{print_loss_dic[metrice]:.4f}'
							message += ' '
						message += f'ACC==>{epoch_acc:.3f}'

					if phase == 'eval':
						message += ' || Validation: '
						message += f'Loss==>{epoch_loss:.4f}'
						message += ' '
						message += f'ACC==>{epoch_acc:.3f}'
						print(message)

				self._log_info(logger=logger, phase=phase, epoch=epoch,
							epoch_loss=epoch_loss, epoch_acc=epoch_acc)

				if phase == 'eval':
					# Record current performance
					k = list(self.current_perf.keys())[0]
					self.current_perf['epoch' + str(epoch)] = self.current_perf.pop(k)
					self.current_perf['epoch' + str(epoch)] = epoch_acc

					# Record top best model
					if epoch_acc > self.best_perf['best_acc']:
						self.best_perf['best_acc'] = epoch_acc
						self.best_wts['best_acc_wts'] = copy.deepcopy(self.model.state_dict())



	def train(self, num_epochs, start_dual_pred_epoch, pre_train_epoch, recomplete_freq, batch_size, log_dir, info_freq):
		"""Training / Evaluation function.
		Parameters:
		-----------
		num_epochs: int
			Number of training epochs.
		start_dual_pred_epoch: int
			The epoch of starting the dual prediction part.
		pre_train_epoch: int
			Number of epoch to pre-train GCNs.
		recomplete_freq: int
			Gap between two completing operations.
		batch_size: int
			Batch size.
		log_dir: string
			The direction of logger.
		info_freq: int
			The frequency of printing training (evaluation) information.
		"""

		self._run_training_loop(num_epochs, start_dual_pred_epoch, pre_train_epoch,
							recomplete_freq, batch_size, log_dir, info_freq)

		print_buttom_line(20)
		print('>>>>> Best validation Accuracy score:')
		for k, v in self.best_perf.items():
			print(f'     {v} {k}')
























