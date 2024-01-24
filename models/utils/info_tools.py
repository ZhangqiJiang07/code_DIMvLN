"""Print information about final scores."""

import numpy as np

def print_dataset_info(train_val_dataset, test_dataset, missing_rate, unlabeled_rate):
	print()
	print('>>>>> Datasets Summary <<<<<')
	print(f'Missing Rate: {missing_rate}, Unlabeled Rate: {unlabeled_rate}')
	print('Train: ', train_val_dataset['train']['view1_x'].shape[0])
	print('Validation: ', train_val_dataset['eval']['view1_x'].shape[0])
	print('Test: ', test_dataset['view1_x'].shape[0])



def print_test_score_info(acc, pr, f1, test_time, total_time):
	print(f'>>>>> Test({test_time}/{total_time}) Report with best validation Accuracy model parameters:')
	print('Accuracy score: ', acc)
	print('Macro-Precision score: ', pr)
	print('Macro-F1 score: ', f1)
	print()



def print_test_mean_score_info(test_acc_list, test_precision_list, test_f1_list, total_time):
	print()
	print(f'>>>>> Test Score Summary after {total_time} testings <<<<<')
	print('Mean Acc score: ', np.mean(test_acc_list))
	print('Mean Macro Precision score: ', np.mean(test_precision_list))
	print('Mean mean Macro F1 score: ', np.mean(test_f1_list))
