"""X views classification"""
import os
import sys
import time
current_path = os.getcwd()

sys.path.append(os.path.dirname(current_path))
sys.path.append(current_path + '/models')
sys.path.append(current_path + '/models/nets')
sys.path.append(current_path + '/config')
sys.path.append(current_path + '/models/utils')

import warnings
warnings.filterwarnings("ignore")

import torch
import argparse
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from models.utils.data_loader import load_data, make_train_test_dataset
from models.utils.info_tools import print_dataset_info
from models.utils.info_tools import print_test_score_info, print_test_mean_score_info
from models.model import Model
from config.fetch import fetch_config


# torch.autograd.set_detect_anomaly(True)

DATASET = {
	0: 'Caltech101-20',
	1: 'Cub',
	2: 'NUS-WIDE',
	3: 'Wiki',
	4: 'HandWritten',
	5: 'ALOI',
	6: 'Out-Scene',
	7: 'YouTubeFace',
	8: 'Animal',
	9: 'STL10',
	10: 'NoisyMNIST'
}

COMPLETION = {
	0: 'no_complete',
	1: 'complete_with_gcn',
	2: 'average_complete',
	3: 'random_complete'
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='0,1,2', help='dataset id')
parser.add_argument('--device', type=str, default='0', help='gpu device id')
parser.add_argument('--info_freq', type=int, default='50', help='frequency of information print')
parser.add_argument('--test_time', type=int, default='5', help='number of test times')
parser.add_argument('--missing_rate', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9', help='missing sample rate')
parser.add_argument('--unlabeled_rate', type=str, default='0.95,0.90,0.85,0.80,0.75,0.70,0.65', help='unlabeled data rate')
parser.add_argument('--save_result', type=str, default='y', help='whether save the model performance on testing set')
parser.add_argument('--completion', type=int, default='1', help='Completion method')
parser.add_argument('--visual', type=str, default='n', help='Visualization')

args = parser.parse_args()


# dataset_name = DATASET[args.dataset]
param_dataset_id = args.dataset.split(',')
param_missing_rate = args.missing_rate.split(',')
param_unlabeled_rate = args.unlabeled_rate.split(',')
dataset_list = list(map(int, param_dataset_id))
missing_rate_list = list(map(float, param_missing_rate))
unlabeled_rate_list = list(map(float, param_unlabeled_rate))


method = {
		'Completion': COMPLETION[args.completion],
		'Representation': 'autoencoder',
		'Semi_supervised': 'similarity_based',
		'Visual': args.visual
		}



# Set device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')


for dataset_i in dataset_list:
	dataset_name = DATASET[dataset_i]
	config = fetch_config(method, dataset_name)
	for missing_rate_i in missing_rate_list:
		for unlabeled_rate_i in unlabeled_rate_list:
			# Load data as list
			X_list, Y_list = load_data(dataset_name, main_dir=os.path.dirname(current_path))
			x_data_dics = {}
			for v in range(1, config['view'] + 1):
				x_data_dics[f'x{v}_data'] = X_list[v - 1]

			label = Y_list[0]


			test_acc_list = []
			test_precision_list = []
			test_f1_list = []
			test_auc_list = []
			training_time_list = []
			inference_time_list = []

			for test_time in range(1, args.test_time + 1):
				# train_test_split
				train_val_dataset, test_dataset,\
				classes, dataset_info_dic = make_train_test_dataset(config, x_data_dics, label, method=method,
																	missing_rate=missing_rate_i, unlabeled_rate=unlabeled_rate_i,
																	seed=test_time)

				# Set random seeds
				np.random.seed(config['seed'])
				torch.manual_seed(config['seed'] + 2)
				torch.cuda.manual_seed(config['seed'] + 3)

				# Initialize model
				model = Model(config=config, method=method, device=device)

				# Train and validation
				fit_args = {
					'dataset': train_val_dataset,
					'classes': classes,
					'log_dir': './training_logger/' + dataset_name +\
						f"/{method['Completion']}_{method['Representation']}_{method['Semi_supervised']}" +\
						f"/mr{missing_rate_i}_ur{unlabeled_rate_i}_" +\
						f"lr{config['Training']['lr']}_epoch{config['Training']['epoch']}" +\
						f"/test{test_time}",

					'info_freq': args.info_freq,
					'test_time': test_time
				}
				start = time.time()
				model.fit(**fit_args)
				end = time.time()
				training_time_list.append(end - start)


				# Test
				model.test()
				start_test = time.time()
				output_dic = model.predict(test_dataset)
				end_test = time.time()
				inference_time_list.append(end_test - start_test)
				pre_labels = output_dic['label_c']
				test_labels = test_dataset['gt_label'].cpu().numpy()

				# Claculate Accuracy, macro-Precision, and macro-F1
				test_acc_score = accuracy_score(test_labels, pre_labels)
				test_precision_score = precision_score(test_labels, pre_labels, average='macro')
				test_f1_score = f1_score(test_labels, pre_labels, average='macro')
				test_auc_score = roc_auc_score(test_labels, output_dic['prob'].detach().cpu().numpy(), multi_class='ovo')
				
				# print test info
				print_test_score_info(test_acc_score, test_precision_score, test_f1_score, test_time, args.test_time)

				test_acc_list.append(test_acc_score)
				test_precision_list.append(test_precision_score)
				test_f1_list.append(test_f1_score)
				test_auc_list.append(test_auc_score)

				torch.cuda.empty_cache()

			# Print summary info
			print_dataset_info(train_val_dataset, test_dataset, missing_rate_i, unlabeled_rate_i)
			print_test_mean_score_info(test_acc_list, test_precision_list, test_f1_list, args.test_time)


			# Save results as txt document
			if args.save_result == 'y':

				saved_path = f"./test_results/{dataset_name}/Unsemi_{method['Completion']}_{method['Representation']}_{method['Semi_supervised']}/"
				if not os.path.exists(saved_path):
					os.makedirs(saved_path)

				saved_path = saved_path + f"mr{missing_rate_i}_ur{unlabeled_rate_i}_lr{config['Training']['lr']}_epoch{config['Training']['epoch']}.txt"

				with open(saved_path, 'w') as file:
					file.write('============================ Summary Board ============================\n')
					
					file.write('>>>>>>>> Experimental Setup:\n')
					file.write(f"Missing Rate: {missing_rate_i} | Unlabeled Rate: {unlabeled_rate_i}\n")
					file.write(f"Completion: {method['Completion']} | Representation: {method['Representation']} | Semi-Supervised: {method['Semi_supervised']}\n")
					
					file.write('>>>>>>>> Dataset Info:\n')
					file.write(f'Dataset name: {dataset_name}\n')
					file.write(f"# of train(complete): {dataset_info_dic['train_len']} ({dataset_info_dic['complete_train_len']})\n")
					file.write(f"# of validation(complete): {dataset_info_dic['val_len']} ({dataset_info_dic['complete_val_len']})\n")
					file.write(f"# of test(complete): {dataset_info_dic['test_len']} ({dataset_info_dic['complete_test_len']})\n")

					file.write('>>>>>>>> Experiment Results:\n')
					for i in range(1, len(test_acc_list) + 1):
						file.write(f'Test{i}:  ')
						file.write('Accuracy:{0:.4f} | Precision:{1:.4f} | F1:{2:.4f} | AUC:{3:.4f}\n'.format(test_acc_list[i - 1], test_precision_list[i - 1], test_f1_list[i - 1], test_auc_list[i - 1]))

					file.write('>>>>>>>> Mean Score (standard deviation):\n')
					file.write('Accuracy: {0:.4f} ({1:.4f})\n'.format(np.mean(test_acc_list), np.std(test_acc_list)))
					file.write('Precision: {0:.4f} ({1:.4f})\n'.format(np.mean(test_precision_list), np.std(test_precision_list)))
					file.write('F1: {0:.4f} ({1:.4f})\n'.format(np.mean(test_f1_list), np.std(test_f1_list)))
					file.write('AUC: {0:.4f} ({1:.4f})\n'.format(np.mean(test_auc_list), np.std(test_auc_list)))

					file.write('>>>>>>>> Mean Training Time (standard deviation)(s):{0:.4f} ({1:.4f})\n'.format(np.mean(training_time_list), np.std(training_time_list)))
					for itr in range(len(training_time_list)):
						file.write(f"{itr}: {training_time_list[itr]}\n")

					file.write('>>>>>>>> Mean Inference Time (standard deviation)(s):{0:.4f} ({1:.4f})\n'.format(np.mean(inference_time_list), np.std(inference_time_list)))
					for itr in range(len(inference_time_list)):
						file.write(f"{itr}: {inference_time_list[itr]}\n")

				file.close()




















