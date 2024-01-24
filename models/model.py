"""Abstract model with 2~6 views"""

import torch
import torch.nn as nn
from torch.optim import Adam
from scipy.io import savemat

from loss import Loss
from model_coach import ModelCoach

class Model:
	def __init__(self, config, method, device=None):
		self.config = config
		self.device = device
		self.method = method

		self.optimizer = Adam
		self.loss = Loss(method, config['view'], config['Training']['alpha'], config['Training']['lambda1'], config['Training']['lambda2'])



	def _instantiate_model(self, classes, move2device=True):
		print(f"Instantiate model with {self.config['view']} views...")
		if self.config['view'] == 2:
			from nets.net_2view import Net

		elif self.config['view'] == 3:
			from nets.net_3view import Net

		elif self.config['view'] == 4:
			from nets.net_4view import Net

		elif self.config['view'] == 5:
			from nets.net_5view import Net

		elif self.config['view'] == 6:
			from nets.net_6view import Net

		else:
			raise ValueError(f"You need to update Net structure for {self.config['view']} views input.")

		self.model = Net(config=self.config, method=self.method, classes=classes, device=self.device)

		if move2device:
			self.model = self.model.to(self.device)



	def fit(self, dataset, classes, log_dir, info_freq, test_time):
		"""Train model.
		Parameters:
		-----------
		dataset: dictionary
			Dataset.
		classes: int
			Classes.
		log_dir: string
			Logger path.
		info_freq: int
			The info_frequency of priniting training (evaluating) information.
		test_time: int
			Test times.
		"""
		self._instantiate_model(classes=classes)

		# Print model structure
		if test_time == 1:
			print(self.model)

		optimizer = self.optimizer(self.model.parameters(), lr=self.config['Training']['lr'])

		model_coach = ModelCoach(views=self.config['view'], model=self.model, dataset=dataset,
								optimizer=optimizer, criterion=self.loss, method=self.method,
								classes=classes, device=self.device)

		# Model train
		if self.method['Completion'] == 'complete_with_gcn':
			model_coach.train(num_epochs=self.config['Training']['epoch'],
							start_dual_pred_epoch=0,
							pre_train_epoch=self.config['Training']['pre_train_epoch'],
							recomplete_freq=self.config['Training']['recomplete_freq'],
							batch_size=self.config['Training']['batch_size'],
							log_dir=log_dir, info_freq=info_freq)
		else:
			model_coach.train(num_epochs=self.config['Training']['epoch'],
							start_dual_pred_epoch=self.config['Training']['start_dual_prediction'],
							pre_train_epoch=0,
							recomplete_freq=0,
							batch_size=self.config['Training']['batch_size'],
							log_dir=log_dir, info_freq=info_freq)

		# Equip the best weights after training
		self.model = model_coach.model
		self.best_model_weights = model_coach.best_wts
		self.best_perf = model_coach.best_perf
		self.current_perf = model_coach.current_perf



	def save_weights(self, saved_epoch, prefix, weight_dir):
		"""Save model weights."""
		print('Saving model weights to file:')
		if saved_epoch == 'current':
			# Save the current model weights
			epoch = list(self.current_perf.keys())[0]
			value = self.current_perf[epoch]
			file_name = os.path.join(
				weight_dir,
				f'{prefix}_{epoch}_acc{value:.2f}.pth')
		else:
			# Save the best weights
			file_name = os.path.join(
				weight_dir,
				f'{prefix}_{saved_epoch}_' + \
				f'acc{self.best_perf[saved_epoch]:.2f}.pth')
			self.model.load_state_dict(self.best_model_weights[saved_epoch + '_wts'])

		torch.save(self.model.stat_dict(), file_name)
		print(' ', file_name)



	def test(self):
		"""Load model with the weights of the best accuracy model on validation set."""
		self.model.load_state_dict(self.best_model_weights['best_acc_wts'])
		self.model = self.model.to(self.device)
		self.model.eval()



	def load_weights(self, path):
		"""Load weights with a certen weights in path."""
		print('Loadding model weights:')
		print(path)
		self.model.load_state_dict(torch.load(path))
		self.model = self.model.to(self.device)



	def predict(self, dataset):
		"""Using the trained model to predict outputs on dataset."""
		# prepare dataset
		with torch.no_grad():
			for key in list(dataset.keys()):
				if (self.method['Completion'] == 'complete_with_gcn') and (key == 'tfed_adj'):
					for sub_key in list(dataset[key].keys()):
						dataset[key][sub_key] = dataset[key][sub_key].to(self.device)

				else:
					dataset[key] = dataset[key].to(self.device)

			if self.method['Completion'] in ['no_complete', 'average_complete', 'random_complete']:
				return self.model(dataset, mode='eval')

			elif self.method['Completion'] == 'complete_with_gcn':
				if self.method['Visual'] == 'y':
					# mat_dic = {'origin_X1': dataset['view1_x'].cpu().numpy(), 'origin_X2': dataset['view2_x'].cpu().numpy()}
					# savemat('/root/jzq/DIMvLN_code/origin.mat', mat_dic)
					return self.model(dataset, mode='eval', incomplete_input_subadj=dataset, visual='y')
				else:
					return self.model(dataset, mode='eval', incomplete_input_subadj=dataset)


























