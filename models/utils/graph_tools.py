"""Construct kNN graph and transformed kNN graph for all views."""

import torch

from torch_cluster import knn_graph

import numpy as np
import scipy.sparse as sp

def construct_knn_edges(input_dic, mask, view_num, k):
	"""Construct view-specific kNN graph edges.
	Parameters:
	-----------
	input_dic: dict(float tensor)
		{'input1': [N, feature_dim_view1], 'input2': [N, feature_dim_view2], ...}
	mask: long tensor (N, view_num)
	view_num: int
		Construct the knn graph edges on the v-th view.
	k: int
		The Hyperparameter of kNN algorithm.

	Returns:
	--------
	edges_dic: dict(np.array)
		Dictionary of the edges of kNN graph on views.
		{'edges1': [2, N*k], 'edges2': [2, N*k], ...}, where 'edges{v}' represent the kNN edges of v-th view.
	exist_idx_dic: dict(np.array)
		Dictionary of the exist index of instances.
	"""

	edges_dic = {}

	for v in range(view_num):
		# Get the existing instances index in view
		exist_idx_bool = mask[:, v] == 1
		exist_idx = np.array(list(range(exist_idx_bool.size()[0])))[exist_idx_bool]
		idx_map = {i: j for i, j in enumerate(exist_idx)}
		# Construct kNN graph by torch_cluster.knn_graph
		edges_unordered = knn_graph(input_dic[f'input{v + 1}'][exist_idx_bool], k=k)
		edges_dic[f'edges{v + 1}'] = np.array(list(map(idx_map.get, edges_unordered.numpy().flatten())),
						dtype=np.int32).reshape(edges_unordered.shape)

	return edges_dic


def edges2adj(edges_dic, sample_num, view_num):
	"""Convert edge pairs to adjacency matrix."""
	# view_num = len(edges_dic)

	adj_dic = {}

	for v in range(view_num):
		adj_dic[f'adj{v + 1}'] = sp.coo_matrix((np.ones(edges_dic[f'edges{v + 1}'].shape[1]), (edges_dic[f'edges{v + 1}'][1, :], edges_dic[f'edges{v + 1}'][0, :])),
												shape=(sample_num, sample_num),
												dtype=np.float32)

	return adj_dic


def transform_adj(adj_dic, mask, view_num):
	"""Get the transformed adjacency matrix from the other views."""

	tf_adj_dic = {}

	for v in range(view_num):
		indicator = 1
		for k in range(view_num):
			if k == v:
				continue
			if indicator:
				indicator = 0
				sum_adj = adj_dic[f'adj{k + 1}']
			else:
				sum_adj += adj_dic[f'adj{k + 1}']
		
		# Dropout the index of missing instances in the transformed matrix in a view
		diag_m = sp.diags(np.array(mask[:, v]), dtype=np.float32)
		tf_adj_dic[f'tf_adj{v + 1}'] = sum_adj.dot(diag_m)

	return tf_adj_dic


def normalize(mx):
	"""Row-normalize sparse matrix"""
	rowsum = np.array(mx.sum(axis=1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)

	return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
	"""Convert a scipy sparse matrix to a torch sparse tensor."""
	sparse_mx = sparse_mx.tocoo().astype(np.float32)
	indices = torch.from_numpy(
		np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
	values = torch.from_numpy(sparse_mx.data)
	shape = torch.Size(sparse_mx.shape)

	return torch.sparse.FloatTensor(indices, values, shape), indices



def construct_transformed_knn_adj(input_dic, mask, ktop):
	""" Construct transformed knn adjacency matrix.
	Parameters:
	-----------
	input_dic: dict
		{'input1': [N, feature_dim_view1], 'input2': [N, feature_dim_view2], ...}
	mask: long tensor (N, view_num)

	ktop: int
		The hyperparameter of kNN algorithm.

	Returns:
	--------
	tf_adj_tensor_dic: dict (tensor)
		The dictionary of the normalized transformed kNN adjacency.
	"""
	view_num = len(input_dic)
	sample_num = input_dic[list(input_dic.keys())[0]].shape[0]

	edges_dic = construct_knn_edges(input_dic, mask, view_num, ktop)
	adj_dic = edges2adj(edges_dic, sample_num, view_num)
	tf_adj_dic = transform_adj(adj_dic, mask, view_num)

	tf_adj_tensor_dic = {}
	tf_indices_dic = {}
	# Normalize the adjacency matrix and convert the sparse matrix to tensor form.
	for v in range(view_num):
		tf_adj_tensor_dic[f'tf_adj{v + 1}'], tf_indices_dic[f'indices{v + 1}'] = sparse_mx_to_torch_sparse_tensor(
			normalize(tf_adj_dic[f'tf_adj{v + 1}']))

	return tf_adj_tensor_dic, tf_indices_dic











