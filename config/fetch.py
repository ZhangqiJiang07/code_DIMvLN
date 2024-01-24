
from config_without_completion import make_config_without_completion
from config_with_gcn_completion import make_config_with_gcn_completion

def fetch_config(method, data_name):
	"""Fetch config depende on method and dataset name.
	Parameters:
	-----------
	method: dict
		Describe the used method.
	data_name: str
		Dataset name.

	Returns:
	--------
	config: dict
		The early definded config about models.
	"""
	if method['Completion'] in ['no_complete', 'average_complete', 'random_complete']:
		return make_config_without_completion(data_name)

	elif method['Completion'] == 'complete_with_gcn':
		return make_config_with_gcn_completion(data_name)

	else:
		raise ValueError('Unknown method %s' % method['Completion'])